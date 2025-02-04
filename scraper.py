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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

from openai import OpenAI
import google.generativeai as genai
from groq import Groq

from api_management import get_api_key
from assets import USER_AGENTS, PRICING, HEADLESS_OPTIONS, SYSTEM_MESSAGE, USER_MESSAGE, LLAMA_MODEL_FULLNAME, GROQ_LLAMA_MODEL_FULLNAME, HEADLESS_OPTIONS_DOCKER

load_dotenv()

def is_running_in_docker():
    """
    Detect if the app is running inside a Docker container.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception:
        return False

def setup_selenium():
    """
    Set up Selenium WebDriver with appropriate options for different environments.
    """
    options = Options()
    
    # Basic options for headless mode
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    
    # Add random user agent
    options.add_argument(f'user-agent={random.choice(USER_AGENTS)}')
    
    # Additional options for stability
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-automation")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    try:
        if is_running_in_docker():
            # Docker-specific setup
            driver = webdriver.Chrome(options=options)
        else:
            # Regular setup with ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        
        return driver
    except Exception as e:
        st.error(f"Failed to initialize Chrome driver: {str(e)}")
        raise

def fetch_html_selenium(url, attended_mode=False, max_retries=3):
    """
    Fetch HTML content using Selenium with retry mechanism and better error handling.
    """
    retry_count = 0
    while retry_count < max_retries:
        driver = None
        try:
            driver = setup_selenium()
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            # Navigate to URL with wait
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Random delay to avoid detection
            time.sleep(random.uniform(2, 4))
            
            if not attended_mode:
                # Scroll with error handling
                try:
                    # Smooth scroll
                    total_height = int(driver.execute_script("return document.body.scrollHeight"))
                    for i in range(1, total_height, 100):
                        driver.execute_script(f"window.scrollTo(0, {i});")
                        time.sleep(0.1)
                except Exception as scroll_error:
                    st.warning(f"Scrolling encountered an error: {str(scroll_error)}")
            
            # Get page source
            page_source = driver.page_source
            return page_source
            
        except TimeoutException:
            retry_count += 1
            st.warning(f"Timeout while loading page. Attempt {retry_count} of {max_retries}")
            if retry_count == max_retries:
                raise
                
        except WebDriverException as e:
            retry_count += 1
            st.warning(f"WebDriver error. Attempt {retry_count} of {max_retries}. Error: {str(e)}")
            if retry_count == max_retries:
                raise
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            raise
            
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                    
        # Wait before retry
        if retry_count < max_retries:
            time.sleep(random.uniform(2, 5))

def clean_html(html_content):
    """
    Clean HTML content with improved error handling.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
            
        # Remove empty elements
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0:
                element.decompose()
                
        return str(soup)
    except Exception as e:
        st.error(f"Error cleaning HTML: {str(e)}")
        return html_content

# [Rest of your code remains the same...]
def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    return markdown_converter.handle(cleaned_html)

def save_raw_data(raw_data: str, output_folder: str, file_name: str):
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, file_name)
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    return raw_output_path


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Create field definitions using aliases for Field parameters
    field_definitions = {field: (str, ...) for field in field_names}
    # Dynamically create the model with all field
    return create_model('DynamicListingModel', **field_definitions)


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))




def trim_to_token_limit(text, model, max_tokens=120000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text

def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.
    """
    # Use the model_json_schema() method to introspect the Pydantic model
    schema_info = listing_model.model_json_schema()

    # Extract field descriptions from the schema
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Get the field type from the schema info
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Create the JSON schema structure for the listings
    schema_structure = ",\n".join(field_descriptions)

    # Generate the system message dynamically
    system_message = f"""
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
    }} """

    return system_message



def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}
    
    if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
        # Use OpenAI API
        client = OpenAI(api_key=get_api_key('OPENAI_API_KEY'))
        completion = client.beta.chat.completions.parse(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE + data},
            ],
            response_format=DynamicListingsContainer
        )
        # Calculate tokens using tiktoken
        encoder = tiktoken.encoding_for_model(selected_model)
        input_token_count = len(encoder.encode(USER_MESSAGE + data))
        output_token_count = len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
        token_counts = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count
        }
        return completion.choices[0].message.parsed, token_counts

    elif selected_model == "gemini-1.5-flash":
        # Use Google Gemini API
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": DynamicListingsContainer
                })
        prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
        # Count input tokens using Gemini's method
        input_tokens = model.count_tokens(prompt)
        completion = model.generate_content(prompt)
        # Extract token counts from usage_metadata
        usage_metadata = completion.usage_metadata
        token_counts = {
            "input_tokens": usage_metadata.prompt_token_count,
            "output_tokens": usage_metadata.candidates_token_count
        }
        return completion.text, token_counts
    
    elif selected_model == "Llama3.1 8B":

        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        # print(SYSTEM_MESSAGE)
        # Point to the local server
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        completion = client.chat.completions.create(
            model=LLAMA_MODEL_FULLNAME, #change this if needed (use a better model)
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": USER_MESSAGE + data}
            ],
            temperature=0.7,
            
        )

        # Extract the content from the response
        response_content = completion.choices[0].message.content
        print(response_content)
        # Convert the content from JSON string to a Python dictionary
        parsed_response = json.loads(response_content)
        
        # Extract token usage
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }

        return parsed_response, token_counts
    elif selected_model== "Groq Llama3.1 70b":
        
        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        # print(SYSTEM_MESSAGE)
        # Point to the local server
        client = Groq(api_key=get_api_key("GROQ_API_KEY"),)

        completion = client.chat.completions.create(
        messages=[
            {"role": "system","content": sys_message},
            {"role": "user","content": USER_MESSAGE + data}
        ],
        model=GROQ_LLAMA_MODEL_FULLNAME,
    )

        # Extract the content from the response
        response_content = completion.choices[0].message.content
        
        # Convert the content from JSON string to a Python dictionary
        parsed_response = json.loads(response_content)
        
        # completion.usage
        token_counts = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens
        }

        return parsed_response, token_counts
    else:
        raise ValueError(f"Unsupported model: {selected_model}")



def save_formatted_data(formatted_data, output_folder: str, json_file_name: str, excel_file_name: str):
    """Save formatted data as JSON and Excel in the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse the formatted data if it's a JSON string (from Gemini API)
    if isinstance(formatted_data, str):
        try:
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            raise ValueError("The provided formatted data is a string but not valid JSON.")
    else:
        # Handle data from OpenAI or other sources
        formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Save the formatted data as JSON
    json_output_path = os.path.join(output_folder, json_file_name)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, excel_file_name)
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None

def calculate_price(token_counts, model):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)
    
    # Calculate the costs
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    
    return input_token_count, output_token_count, total_cost


def generate_unique_folder_name(url):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    url_name = re.sub(r'\W+', '_', url.split('//')[1].split('/')[0])  # Extract domain name and replace non-alphanumeric characters
    return f"{url_name}_{timestamp}"


def scrape_url(url: str, fields: List[str], selected_model: str, output_folder: str, file_number: int, markdown: str):
    """Scrape a single URL and save the results."""
    try:
        # Save raw data
        save_raw_data(markdown, output_folder, f'rawData_{file_number}.md')

        # Create the dynamic listing model
        DynamicListingModel = create_dynamic_listing_model(fields)

        # Create the container model that holds a list of the dynamic listing models
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        # Format data
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, selected_model)
        
        # Save formatted data
        save_formatted_data(formatted_data, output_folder, f'sorted_data_{file_number}.json', f'sorted_data_{file_number}.xlsx')

        # Calculate and return token usage and cost
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, selected_model)
        return input_tokens, output_tokens, total_cost, formatted_data

    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return 0, 0, 0, None

# Remove the main execution block if it's not needed for testing purposes
        