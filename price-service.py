import os
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv
import pytz

import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class PriceFeed:
    """Data class for price feed information."""
    id: str
    description: str

@dataclass
class PriceData:
    """Data class for price information."""
    id: str
    price: float
    confidence_interval: float
    description: str
    time: str

class HermesClient:
    """Client for interacting with the Hermes API."""
    
    def __init__(self, base_url: str):
        self.BASE_URL = base_url
        self.session = requests.Session()    
    
    def get_price_feeds(self) -> List[Dict]:
        """Fetch all available price feeds."""
        response = self.session.get(f"{self.BASE_URL}/v2/price_feeds")
        response.raise_for_status()
        return response.json()
    
    def get_latest_prices(self, feed_ids: List[str]) -> List[Dict]:
        url = f"{self.BASE_URL}/v2/updates/price/latest"
        query_params = [('ids[]', id_) for id_ in feed_ids]
        
        try:
            response = self.session.get(
                url, 
                params=query_params,
                timeout=30  # Add timeout parameter
            )
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, dict) or 'parsed' not in data:
                raise ValueError("Unexpected API response format")
                
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out while fetching prices for feeds: {feed_ids}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error while fetching prices: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {response.status_code} while fetching prices: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid response format from API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching prices: {e}")
            raise

class PriceService:
    """Service for processing price-related requests."""
    
    def __init__(self):
        self.hermes_client = HermesClient(base_url=os.getenv('HERMES_API_BASE_URL'))
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.price_feeds = self.hermes_client.get_price_feeds()
    
    def process_tickers(self, tickers: List[str]) -> List[PriceFeed]:
        """Process tickers and return corresponding price feeds."""
        if not tickers:
            return []
        
        price_feeds = []
        for ticker in tickers:
            feeds = [
                PriceFeed(
                    id=item["id"],
                    description=item["attributes"]["description"]
                )
                for item in self.price_feeds
                if "base" in item["attributes"] 
                and item["attributes"]["base"] == ticker
            ]
            price_feeds.extend(feeds)
        return price_feeds

    def fetch_prices(self, tickers: List[str]) -> List[PriceData]:
        """Fetch prices for given tickers."""
        try:
            price_feeds = self.process_tickers(tickers)
            if not price_feeds:
                logger.warning(f"No price feeds found for tickers: {tickers}")
                return []

            feed_ids = [feed.id for feed in price_feeds]
            response_data = self.hermes_client.get_latest_prices(feed_ids)
            
            return [
                PriceData(
                    id=price["id"],
                    price=float(price["price"]["price"]) * 10 ** float(price["price"]["expo"]),
                    confidence_interval=float(price["price"]["conf"]) * 10 ** float(price["price"]["expo"]),
                    description=next(
                        feed.description 
                        for feed in price_feeds 
                        if feed.id == price["id"]
                    ),
                    time=datetime.fromtimestamp(
                        price["price"]["publish_time"],
                        tz=pytz.UTC
                    ).astimezone(pytz.timezone('America/New_York')).strftime("%B %d, %Y %I:%M %p %Z")
                )
                for price in response_data["parsed"]
            ]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching prices: {e}")
            raise

    def analyze_message(self, message: str) -> Dict[str, Union[bool, List[str], str]]:
        """Analyze message for price requests using GPT."""
        prompt = f"""

        Your name is Pythia. You are the price oracle of the Pyth Network. You speak truth and only truth.
        You are a helpful assistant that can answer questions and provide information.
        For now you tell the prices if you feel they have been asked.

        Analyze the following message:
        1. Determine if it's asking for a price. Reply in True/False. Please make sure with capital first letter.
        2. If yes, extract the symbols/tickers mentioned and convert them to proper format.
        3. If no, provide a friendly response to continue the conversation.

        Message: "{message}"

        Response format:
        {{
            "is_price_request": True/False,
            "tickers": ["symbol1", "symbol2"],
            "chat_response": "Response if not a price request"
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            # print("response.choices[0].message.content: ", response.choices[0].message.content)
            return eval(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error processing message with GPT: {e}")
            raise

    def handle_message(self, message: str) -> Union[List[PriceData], str]:
        """Handle user message and return appropriate response."""
        try:
            analysis = self.analyze_message(message)
            logger.info(f"Message analysis: {analysis}")
            
            if analysis["is_price_request"]:
                # print("analysis['tickers']: ", analysis["tickers"])
                return self.fetch_prices(analysis["tickers"])
            else:
                return analysis["chat_response"]
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return f"An error occurred: {str(e)}"

def main():
    """Main entry point for the application."""
    service = PriceService()
    
    print("Hello! I am Pythia.")
    print("I am the price oracle of the Pyth Network. I speak truth and only truth.")
    while True:
        try:
            user_message = input("\nEnter your message: ")
            if user_message.lower() == 'exit':
                print("Goodbye!")
                break
                
            response = service.handle_message(user_message)
            
            # Pretty print the response if it's a list of PriceData
            if isinstance(response, list) and all(isinstance(x, PriceData) for x in response):
                for price in response:
                    print(f"\n{price.description}")
                    print("-" * len(price.description))
                    print(f"Price: ${price.price:,.2f}")
                    print(f"Confidence Interval: {price.confidence_interval}")
                    print(f"Time: {price.time}")
            else:
                print("\nHere you go:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
