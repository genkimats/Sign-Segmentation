import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import time

# --- Configuration ---
# 1. Dynamically set base paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "data")
ANNOTATIONS_DIR = os.path.join(BASE_PATH, "annotations")
VIDEOS_DIR = os.path.join(BASE_PATH, "videos")

# 2. Set the HTML file path and correct base URL for the DGS corpus
# This assumes index.html is in the exact same folder as this Python script
HTML_FILE = os.path.join(SCRIPT_DIR, "index.html")
BASE_URL = "https://www.sign-lang.uni-hamburg.de/meinedgs/ling/" 

# --- Setup ---
# Ensure the target directories exist
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

def download_file(url, target_path, max_retries=5):
    """Streams a file from a URL with built-in timeout and retry logic."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
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
            return # Exit function on success
            
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"⚠️ Attempt [{attempt}/{max_retries}] failed for {os.path.basename(target_path)}: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5  # Incremental backoff (5s, 10s, 15s...)
                print(f"Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                print(f"❌ Failed to download {url} after {max_retries} attempts.")

def main():
    # Read the uploaded HTML file
    print(f"Parsing {HTML_FILE}...")
    
    # Failsafe: Check if index.html actually exists in this directory
    if not os.path.exists(HTML_FILE):
        print(f"❌ Error: Could not find 'index.html' at {HTML_FILE}.")
        print("Please ensure your HTML file is in the same directory as this script.")
        return

    with open(HTML_FILE, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find all anchor tags with an href attribute
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
            
        # 2. Match Videos (Broadened check)
        elif '.mp4' in href_lower or 'video a' in link_text or 'video b' in link_text:
            target_dir = VIDEOS_DIR
            
        else:
            skipped_links.append((link_text, href))

        # If the link matches our target files, execute the download
        if target_dir:
            download_count += 1
            full_url = urljoin(BASE_URL, href)
            
            file_name = unquote(full_url.split('/')[-1])
            
            if '?' in file_name:
                file_name = file_name.split('?')[0]
                
            if target_dir == VIDEOS_DIR and not file_name.endswith('.mp4'):
                file_name = f"video_{download_count}.mp4"
                
            file_path = os.path.join(target_dir, file_name)
            
            print(f"Downloading [{download_count}] {file_name}...")
            download_file(full_url, file_path)

    print(f"\nFinished processing. Attempted to download {download_count} target files.")
    
    # 3. Debug Print
    print("\n--- Diagnostic: First 10 Skipped Links ---")
    for text, h in skipped_links[:10]:
        print(f"Skipped -> Text: '{text}' | Link: {h}")

if __name__ == "__main__":
    main()