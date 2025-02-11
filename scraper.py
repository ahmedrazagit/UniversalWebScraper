import os
import random
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
import streamlit as st

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from openai import OpenAI
import google.generativeai as genai
from groq import Groq

from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import WebDriverException
from webdriver_manager.firefox import GeckoDriverManager
import shutil

from api_management import get_api_key
from assets import USER_AGENTS, PRICING, SYSTEM_MESSAGE, USER_MESSAGE, LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME
load_dotenv()

def setup_selenium(attended_mode=False):
    """
    Set up Chrome WebDriver with headless mode and additional configurations for robust scraping
    """
    options = Options()
    
    # Always run in headless mode
    options.add_argument('--headless=new')
    
    # Add essential arguments for stable headless operation
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    
    # Add additional arguments for better performance and stability
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-notifications')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-popup-blocking')
    
    # Set user agent to avoid detection
    options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
    
    # Additional preferences for performance
    options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)
    
    return driver

def fetch_html_selenium(url, attended_mode=False, driver=None):
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
    else:
        should_quit = False
    
    try:
        driver.set_page_load_timeout(30)
        driver.get(url)
        time.sleep(random.uniform(2, 4))
        
        total_height = int(driver.execute_script("return document.body.scrollHeight"))
        for i in range(3):
            scroll_height = total_height * (i + 1) // 3
            driver.execute_script(f"window.scrollTo(0, {scroll_height});")
            time.sleep(random.uniform(1, 2))
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        return driver.page_source
        
    except Exception as e:
        print(f"Error fetching URL {url}: {str(e)}")
        return None
        
    finally:
        if should_quit and driver is not None:
            driver.quit()

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup.find_all(['header', 'footer']):
        element.decompose()
    return str(soup)

def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    return markdown_converter.handle(cleaned_html)

def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    field_definitions = {field: (str, ...) for field in field_names}
    return create_model('DynamicListingModel', **field_definitions)

def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))

def trim_to_token_limit(text, model, max_tokens=120000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        return encoder.decode(tokens[:max_tokens])
    return text

def generate_system_message(listing_model: BaseModel) -> str:
    schema_info = listing_model.model_json_schema()
    field_descriptions = [
        f'"{field_name}": "{field_info["type"]}"'
        for field_name, field_info in schema_info["properties"].items()
    ]
    schema_structure = ",\n".join(field_descriptions)
    
    return f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
    from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
    with no additional commentary, explanations, or extraneous information. 
    You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
    Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }}"""

def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}
    
    if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
        client = OpenAI(api_key=get_api_key('OPENAI_API_KEY'))
        completion = client.beta.chat.completions.parse(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE + data},
            ],
            response_format=DynamicListingsContainer
        )
        encoder = tiktoken.encoding_for_model(selected_model)
        token_counts = {
            "input_tokens": len(encoder.encode(USER_MESSAGE + data)),
            "output_tokens": len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
        }
        return completion.choices[0].message.parsed, token_counts

    elif selected_model == "gemini-1.5-flash":
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": DynamicListingsContainer
                })
        prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
        completion = model.generate_content(prompt)
        usage_metadata = completion.usage_metadata
        token_counts = {
            "input_tokens": usage_metadata.prompt_token_count,
            "output_tokens": usage_metadata.candidates_token_count
        }
        return completion.text, token_counts
    
    elif selected_model == "Llama3.1 8B":
        sys_message = generate_system_message(DynamicListingModel)
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        completion = client.chat.completions.create(
            model=LLAMA_MODEL_FULLNAME,
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": USER_MESSAGE + data}
            ],
            temperature=0.7,
        )
        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }
        return parsed_response, token_counts

    elif selected_model == "Groq Llama3.1 70b":
        sys_message = generate_system_message(DynamicListingModel)
        client = Groq(api_key=get_api_key("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": USER_MESSAGE + data}
            ],
            model=GROQ_LLAMA_MODEL_FULLNAME,
        )
        response_content = completion.choices[0].message.content
        parsed_response = json.loads(response_content)
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }
        return parsed_response, token_counts

    else:
        raise ValueError(f"Unsupported model: {selected_model}")

def calculate_price(token_counts, model):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    return input_token_count, output_token_count, total_cost

def scrape_url(url: str, fields: List[str], selected_model: str, markdown: str):
    """Scrape a single URL and return the processed data."""
    try:
        DynamicListingModel = create_dynamic_listing_model(fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, selected_model)
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, selected_model)
        return input_tokens, output_tokens, total_cost, formatted_data

    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return 0, 0, 0, None