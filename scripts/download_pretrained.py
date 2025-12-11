"""
This script downloads the karpathy/nanochat-d34 pre-trained model from Hugging Face
and places the files in the correct directories as described in the README.
"""

import os
import requests
import shutil
from tqdm import tqdm

# The base directory for nano-embed artifacts
# Note: The user mentioned ~/.cache/nanochat, but since we renamed the project to nano-embed,
# we should probably use ~/.cache/nano-embed. I will use the original path for now as requested.
BASE_DIR = os.path.expanduser("~/.cache/nanochat")

# Files to download and their destinations
FILES_TO_DOWNLOAD = {
    "token_bytes.pt": "tokenizer",
    "tokenizer.pkl": "tokenizer",
    "meta_169150.json": "base_checkpoints/d34",
    "model_169150.pt": "base_checkpoints/d34",
}

# Hugging Face repository and base URL
REPO_ID = "karpathy/nanochat-d34"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

def download_file(url, destination_path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination_path, 'wb') as file, tqdm(
        desc=os.path.basename(destination_path),
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
            
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def main():
    print(f"Base directory set to: {BASE_DIR}")

    for filename, dest_subdir in FILES_TO_DOWNLOAD.items():
        # Construct destination directory path
        destination_dir = os.path.join(BASE_DIR, dest_subdir)
        os.makedirs(destination_dir, exist_ok=True)
        
        # Construct full destination path
        destination_path = os.path.join(destination_dir, filename)
        
        # Construct download URL
        url = f"{BASE_URL}/{filename}"
        
        if os.path.exists(destination_path):
            print(f"'{filename}' already exists in '{destination_dir}'. Skipping download.")
            continue
            
        print(f"Downloading '{filename}' to '{destination_dir}'...")
        try:
            download_file(url, destination_path)
            print(f"Successfully downloaded '{filename}'.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading '{filename}': {e}")
            # Clean up partially downloaded file if it exists
            if os.path.exists(destination_path):
                os.remove(destination_path)

    print("\nAll files downloaded and placed in their respective directories.")

if __name__ == "__main__":
    main()
