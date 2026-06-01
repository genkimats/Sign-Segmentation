import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

# --- Configuration ---
# 1. Dynamically resolve paths relative to this script
# Get the exact directory where this python script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level (parent directory), then point to a new "raw_data" folder
BASE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "raw_data")

# Create the specific subdirectories inside the "raw_data" folder
ANNOTATIONS_DIR = os.path.join(BASE_PATH, "annotations")
VIDEOS_DIR = os.path.join(BASE_PATH, "videos")

# 2. Set the HTML file path (assuming it's next to this script) and base URL
HTML_FILE = os.path.join(SCRIPT_DIR, "index.html")
BASE_URL = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/" 

# --- Setup ---
# Ensure the target directories exist (creates them if they don't)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

def download_file(url, target_path, max_retries=5):
    """Streams a file from a URL with built-in timeout and retry logic."""
    if os.path.exists(target_path):
        if os.path.getsize(target_path) > 0:
            print(f"⏩ Skipping (Already Exists): {os.path.basename(target_path)}")
            return

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✅ Success: Saved to {target_path}")
            return 
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"⚠️ Attempt [{attempt}/{max_retries}] failed for {os.path.basename(target_path)}: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5  
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Failed to download {url} after {max_retries} attempts.")

def main():
    print(f"Parsing {HTML_FILE}...")
    try:
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except FileNotFoundError:
        print(f"❌ Error: Could not find {HTML_FILE}. Make sure index.html is in the same folder as this script.")
        return

    links = soup.find_all('a', href=True)
    download_count = 0
    skipped_links = []

    for link in links:
        href = link['href']
        href_lower = href.lower()
        link_text = link.get_text(strip=True).lower()
        
        target_dir = None
        
        # 1. Match Annotations (ELAN / .eaf)
        if '.eaf' in href_lower or 'elan' in link_text:
            target_dir = ANNOTATIONS_DIR
            
        # 2. Match Videos 
        elif '.mp4' in href_lower or 'video a' in link_text or 'video b' in link_text:
            target_dir = VIDEOS_DIR
            
        else:
            skipped_links.append((link_text, href))

        if target_dir:
            download_count += 1
            full_url = urljoin(BASE_URL, href)
            
            file_name = unquote(full_url.split('/')[-1])
            
            if '?' in file_name:
                file_name = file_name.split('?')[0]
                
            if target_dir == VIDEOS_DIR and not file_name.endswith('.mp4'):
                file_name = f"video_{download_count}.mp4"
                
            file_path = os.path.join(target_dir, file_name)
            
            print(f"Downloading [{download_count}] {file_name} into {os.path.basename(target_dir)}/...")
            download_file(full_url, file_path)

    print(f"\nFinished processing. Attempted to download {download_count} target files.")
    
    print("\n--- Diagnostic: First 10 Skipped Links ---")
    for text, h in skipped_links[:10]:
        print(f"Skipped -> Text: '{text}' | Link: {h}")

if __name__ == "__main__":
    main()