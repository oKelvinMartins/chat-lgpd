import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

URL = "http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "lgpd_raw.html")

def fetch_lgpd() -> None:
    """Downloads the official LGPD text and saves it locally."""
    logging.info(f"Downloading LGPD from {URL}...")
    
    # Headers to mimic a real browser request to avoid basic blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Ensure the response is treated as ISO-8859-1 since Planalto uses it usually
        response.encoding = 'windows-1252' 
        
        content = response.text
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
            
        logging.info(f"LGPD HTML successfully saved to {OUTPUT_FILE}")
        
    except requests.RequestException as e:
        logging.error(f"Failed to fetch LGPD: {e}")
        raise

if __name__ == "__main__":
    fetch_lgpd()
