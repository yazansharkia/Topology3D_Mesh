# rename_frames.py
# Renames frame_*.jpg files to img_*.jpg in the specified directory

import os
import glob
import sys

def rename_frames(directory):
    # Get all frame_*.jpg files in the directory
    frame_files = glob.glob(os.path.join(directory, "frame_*.jpg"))
    
    if not frame_files:
        print(f"❌ No frame_*.jpg files found in {directory}")
        return
    
    print(f"📁 Found {len(frame_files)} files to rename in {directory}")
    
    # Rename each file
    for old_path in frame_files:
        # Get the filename without the directory
        filename = os.path.basename(old_path)
        # Create new filename by replacing 'frame_' with 'img_'
        new_filename = filename.replace('frame_', 'img_')
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {filename} -> {new_filename}")
    
    print(f"✨ Successfully renamed {len(frame_files)} files")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_frames.py <directory_path>")
        print("Example: python rename_frames.py ../output/run_1")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"❌ Directory not found: {directory}")
        sys.exit(1)
    
    rename_frames(directory) 