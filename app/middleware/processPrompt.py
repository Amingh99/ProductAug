import os 
import json
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient, login
from typing import Dict, Any, List
import random
import re
import time
import requests

# Load Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_API_KEY")

# Log in to Hugging Face API using the token
login(token=HF_TOKEN)

class HuggingFaceModel:
    def __init__(self, repo_id: str, timeout: int = 1200):
        self.repo_id = repo_id
        self.client = InferenceClient(model=repo_id, timeout=timeout)

    async def call_llm(self, prompt: str) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.post,
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 200},
                    "task": "text-generation",
                }
            )
            return json.loads(response.decode())[0]["generated_text"]
        except Exception as e:
            logging.error(f"Error calling LLM: {str(e)}")
            raise

class WebScraper:
    def __init__(self, search_url: str):
        self.search_url = search_url
        self._session = None

    def get_top_search_results(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        headers = {
            "User-Agent": self._get_random_user_agent()
        }

        
        retries = 2
        for attempt in range(retries):
            response = requests.get(self.search_url, params={"q": query}, headers=headers)
            
            if response.status_code == 429:
                sleep_time = random.randint(5, 10)  # Random wait time between retries
                print(f"Rate limit exceeded, retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                response.raise_for_status()  # Raise an error for any other bad responses
                break
        else:
            raise Exception("Exceeded maximum retry attempts due to rate limiting.")

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for idx, result in enumerate(soup.select("div.g"), 1):
            title = result.select_one("h3")
            if title and idx <= num_results:
                link = result.a["href"]
                snippet = result.select_one(".IsZvec").text if result.select_one(".IsZvec") else ""
                results.append({
                    "title": title.get_text(),
                    "link": link,
                    "snippet": snippet
                })

            if len(results) >= num_results:
                break

        return results if results else [{"message": "No additional information available."}]

    async def scrape_text_from_link(self, url: str) -> str:
        if not self._session:
            self._session = aiohttp.ClientSession()

        headers = {
            "User-Agent": self._get_random_user_agent()
        }

        try:
            async with self._session.get(url, headers=headers) as response:
                response.raise_for_status()
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                paragraphs = soup.find_all("p")
                return "\n".join([para.get_text() for para in paragraphs])[:1000]
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return "Could not retrieve content."

    async def fetch_all_text(self, links: List[str]) -> List[str]:
        tasks = [self.scrape_text_from_link(link) for link in links]
        return await asyncio.gather(*tasks)

    def _get_random_user_agent(self) -> str:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
        ]
        return random.choice(user_agents)

    async def close(self):
        if self._session:
            await self._session.close()

    def get_product_price(self, product_name: str) -> str:
        search_query = f"{product_name} price"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        }

        response = requests.get(self.search_url, params={"q": search_query}, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        prices = []
        for result in soup.select("div.g"):
            snippet_text = result.select_one(".IsZvec").text if result.select_one(".IsZvec") else ""
            price_matches = re.findall(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", snippet_text)
            if price_matches:
                prices.extend(price_matches)

        return prices[0] if prices else "Price not found"

    def get_product_weight(self, product_name: str) -> str:
        search_query = f"{product_name} weight"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
        }

        response = requests.get(self.search_url, params={"q": search_query}, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        weights = []
        for result in soup.select("div.g"):
            snippet_text = result.select_one(".IsZvec").text if result.select_one(".IsZvec") else ""
            weight_matches = re.findall(r"(\d+(\.\d+)?\s*(g|kg|oz|lbs))", snippet_text)
            if weight_matches:
                weights.extend([match[0] for match in weight_matches])

        return weights[0] if weights else "Weight not found"

class ProductEnhancer:
    def __init__(self, model: HuggingFaceModel, web_scraper: WebScraper):
        self.model = model
        self.web_scraper = web_scraper

    def generate_prompt(self, product: Dict[str, Any]) -> str:
        return (
            f"Description: {product.get('description', '')}\n"
            f"Product Details: {json.dumps(product.get('product_details', {}))}\n"
            f"Taxonomy: {json.dumps(product.get('taxonomy', {}))}\n"
            "Can you provide additional details about this product?"
        )

    async def enhance_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self.generate_prompt(product)
            
            # Parallel execution of model call and web scraping
            model_response, search_results = await asyncio.gather(
                self.model.call_llm(prompt),
                self.web_scraper.get_top_search_results(product.get('description', ''))
            )
            
            # Get and process search results
            if search_results:
                links = [result["link"] for result in search_results]
                additional_texts = await asyncio.gather(
                    *[self.web_scraper.scrape_text_from_link(link) for link in links]
                )
                
                # Add scraped text to search results
                for result, text in zip(search_results, additional_texts):
                    result["additional_text"] = text

            return {
                "original_product": product,
                "enhanced_data": {
                    "model_response": model_response,
                    "web_search_results": search_results
                }
            }
        except Exception as e:
            logging.error(f"Error enhancing product: {str(e)}")
            return {
                "original_product": product,
                "enhanced_data": {
                    "error": str(e)
                }
            }

async def promptResponse(json_string: str) -> Dict[str, Any]:
    """
    Process the input JSON string and return enhanced product information.
    
    Args:
        json_string (str): JSON string containing product data
        
    Returns:
        Dict[str, Any]: Enhanced product information
    """
    
    try:
        # Parse the JSON data
        json_data = json.loads(json_string)
        
        # Ensure json_data is a list
        if not isinstance(json_data, list):
            json_data = [json_data]
        
        # Initialize services
        model = HuggingFaceModel(repo_id="microsoft/Phi-3.5-mini-instruct")
        web_scraper = WebScraper("https://www.google.com/search")
        product_enhancer = ProductEnhancer(model, web_scraper)
        
        try:
            # Process all products concurrently
            enhanced_products = await asyncio.gather(
                *[product_enhancer.enhance_product(product) for product in json_data]
            )
            
            return {"enhanced_products": enhanced_products}
        
        finally:
            # Ensure the web scraper session is closed
            await web_scraper.close()

    except Exception as e:
        logging.error(f"Error processing JSON input: {str(e)}")
        return {"error": str(e)}