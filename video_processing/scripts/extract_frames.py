# extract_frames.py
# Converts a video into individual images and stores them in a folder

import cv2
import os
import sys
import glob

# === Configuration ===
VIDEO_PATH = "../input/James_rec.mp4"  # Path to video in video_processing/input/
OUTPUT_BASE = "../output"  # Base output folder in video_processing/output/
FRAME_INTERVAL = 1  # Save every Nth frame

# === Create unique output directory ===
def get_next_run_number():
    # Get all existing run directories
    existing_runs = glob.glob(os.path.join(OUTPUT_BASE, "run_*"))
    if not existing_runs:
        return 1
    
    # Extract numbers from existing run directories and find the max
    run_numbers = [int(os.path.basename(d).split('_')[1]) for d in existing_runs]
    return max(run_numbers) + 1

# Create base output directory if it doesn't exist
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Create unique run directory
run_number = get_next_run_number()
OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, f"run_{run_number}")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"📁 Creating new run directory: {OUTPUT_FOLDER}")

# === Load video ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    sys.exit(1)

frame_count = 0
saved_count = 0
print("📽️ Extracting frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"img_{saved_count:03d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done. Extracted {saved_count} frames to '{OUTPUT_FOLDER}/'")
