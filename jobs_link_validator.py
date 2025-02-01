import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse

class JobLinkValidator:
    def __init__(self, timeout=10, verbose=True):
        """
        Initialize JobLinkValidator with configurable settings.
        
        Args:
            timeout (int): Maximum wait time for page load in seconds
            verbose (bool): Print detailed validation information
        """
        self.timeout = timeout
        self.verbose = verbose

    def _create_driver(self):
        """
        Create a new WebDriver instance.
        
        Returns:
            WebDriver: Configured Chrome WebDriver
        """
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.get_chrome_options())

    def get_chrome_options(self):
        """
        Configure Chrome options for headless browsing.
        
        Returns:
            Options: Configured Chrome options
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        return chrome_options

    def validate_job_link(self, url):
        """
        Validate a job link by attempting to load it in a browser.
        
        Args:
            url (str): URL to validate
        
        Returns:
            tuple: (is_valid, reason)
        """
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return False, "Invalid URL format"

        driver = None
        try:
            driver = self._create_driver()
            driver.set_page_load_timeout(self.timeout)
            driver.get(url)
            
            WebDriverWait(driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            current_url = driver.current_url
            if self.verbose:
                print(f"Checking URL: {url}")
                print(f"Current URL after navigation: {current_url}")

            original_domain = urlparse(url).netloc
            current_domain = urlparse(current_url).netloc
            if original_domain != current_domain:
                return False, f"Redirected from {original_domain} to {current_domain}"
            
            return True, "Valid job link"
        
        except Exception as e:
            if self.verbose:
                print(f"Error validating {url}: {e}")
            return False, str(e)
        
        finally:
            if driver:
                driver.quit()

def validate_job_links(csv_path, output_path=None, max_workers=10, verbose=True):
    """
    Validate job links in a CSV file.
    
    Args:
        csv_path (str): Path to input CSV
        output_path (str, optional): Path to save filtered CSV
        max_workers (int): Maximum concurrent link checks
        verbose (bool): Print detailed validation information
    
    Returns:
        pd.DataFrame: DataFrame with valid job links
    """
    validator = JobLinkValidator(verbose=verbose)
    df = pd.read_csv(csv_path)
    
    if 'ApplicationLink' not in df.columns:
        raise ValueError("CSV must contain an 'ApplicationLink' column")
    
    valid_links = []
    link_statuses = {}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(validator.validate_job_link, row['ApplicationLink']): index 
                for index, row in df.iterrows()
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    is_valid, reason = future.result()
                    link_statuses[index] = (is_valid, reason)
                    if is_valid:
                        valid_links.append(index)
                    
                    if verbose:
                        print(f"Link {index}: {'Valid' if is_valid else 'Invalid'} - {reason}")
                    
                    time.sleep(0.5)  # Delay to avoid overwhelming servers
                except Exception as exc:
                    print(f"Error processing link at index {index}: {exc}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")
        filtered_df = df.loc[valid_links].reset_index(drop=True)
        if output_path:
            filtered_df.to_csv(output_path, index=False)
        return filtered_df

    filtered_df = df.loc[valid_links].reset_index(drop=True)
    
    if verbose:
        print("\nLink Validation Summary:")
        for index, (is_valid, reason) in link_statuses.items():
            print(f"Link {index}: {'Valid' if is_valid else 'Invalid'} - {reason}")
    
    if output_path:
        filtered_df.to_csv(output_path, index=False)
    
    return filtered_df

def main():
    input_csv = 'jobs.csv'
    output_csv = 'valid_jobs.csv'
    
    try:
        valid_jobs = validate_job_links(
            input_csv, 
            output_path=output_csv, 
            max_workers=10,  # Increase concurrency
            verbose=True
        )
        
        print(f"\nTotal jobs before validation: {len(pd.read_csv(input_csv))}")
        print(f"Total valid jobs: {len(valid_jobs)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
