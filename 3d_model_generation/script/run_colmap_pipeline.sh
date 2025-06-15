#!/bin/bash

# === COLMAP Full Pipeline Script ===
# Automates sparse + dense reconstruction for a folder of images
# Make sure colmap is installed (brew install colmap)

set -e  # Exit on error

# === CONFIG ===
IMAGE_DIR="colmap_test/images"
WORKSPACE="colmap_test"
DB_PATH="$WORKSPACE/database.db"
SPARSE_DIR="$WORKSPACE/sparse"
DENSE_DIR="$WORKSPACE/dense"

# === CREATE FOLDERS ===
echo "📁 Creating workspace folders..."
mkdir -p "$IMAGE_DIR" "$SPARSE_DIR" "$DENSE_DIR"

# === STEP 1: Feature extraction ===
echo "🔍 Running feature extraction..."
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR"

# === STEP 2: Feature matching ===
echo "🔗 Running exhaustive matcher..."
colmap exhaustive_matcher \
    --database_path "$DB_PATH"

# === STEP 3: Sparse reconstruction ===
echo "🧠 Running sparse mapper..."
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_DIR"

# === STEP 4: Undistort images for dense reconstruction ===
echo "📸 Undistorting images..."
colmap image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size 2000

# === STEP 5: Patch Match Stereo ===
echo "🔬 Running dense stereo matching..."
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# === STEP 6: Stereo Fusion ===
echo "🌀 Fusing depth maps into final point cloud..."
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply"

# === DONE ===
echo "✅ Done! Dense 3D model saved to $DENSE_DIR/fused.ply"
