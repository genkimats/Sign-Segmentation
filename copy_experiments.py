import shutil
import os
from pathlib import Path

def copy_experiments_filtered(src_dir, dst_dir):
    """
    Copy experiments directory excluding PNG files and final_decoded_metrics.json files.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate through all items in source directory
    for item in src_path.iterdir():
        src_item = src_path / item.name
        dst_item = dst_path / item.name
        
        if src_item.is_dir():
            # Recursively copy directories
            copy_experiments_filtered(src_item, dst_item)
        elif src_item.is_file():
            # Skip PNG files and final_decoded_metrics.json
            if src_item.suffix.lower() == '.png' or src_item.name == 'final_decoded_metrics.json':
                print(f"Skipping: {src_item}")
            else:
                # Copy the file
                dst_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, dst_item)
                print(f"Copied: {src_item} -> {dst_item}")

if __name__ == "__main__":
    src = "/mnt/d/Genki_GR/Sign-Segmentation/experiments"
    dst = "/mnt/d/Genki_GR/Sign-Segmentation/experiments_filtered"
    
    print(f"Copying {src} to {dst}...")
    print("Excluding: .png files and final_decoded_metrics.json\n")
    
    copy_experiments_filtered(src, dst)
    
    print("\n✓ Copy complete!")
