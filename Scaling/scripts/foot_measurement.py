#!/usr/bin/env python3

import os
import sys
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FootMeasurement:
    def __init__(self, input_dir: str, output_dir: str, params_dir: str):
        """
        Initialize the FootMeasurement processor.
        
        Args:
            input_dir (str): Path to the input directory containing foot images
            output_dir (str): Path to the output directory for measurements
            params_dir (str): Path to the directory containing A4 paper parameters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.params_dir = Path(params_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load parameters from the first processed image
        self.paper_params = self.load_first_params()
        if self.paper_params is None:
            logger.error("Could not load parameters from any image. Please run scaling_processor.py first.")
            sys.exit(1)
            
        logger.info(f"Initialized FootMeasurement with input_dir: {input_dir}")
        logger.info(f"Output will be saved to: {output_dir}")
        logger.info(f"Using parameters from first processed image")

    def load_first_params(self):
        """
        Load parameters from the first processed image.
        
        Returns:
            dict: Paper parameters if found, None otherwise
        """
        # Look for any measurements file in the params directory
        for params_file in self.params_dir.glob('measurements_*.txt'):
            try:
                with open(params_file, 'r') as f:
                    content = f.read()
                    
                # Parse the parameters
                params = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if 'aspect ratio' in key:
                            params['aspect_ratio'] = float(value)
                        elif 'distortion factor' in key:
                            params['distortion'] = float(value)
                        elif 'angles' in key:
                            # Parse the angles string into a list of floats, handling np.float64(...)
                            angles_str = value.strip('[]')
                            angle_list = []
                            for angle in angles_str.split(','):
                                angle = angle.strip()
                                if angle.startswith('np.float64('):
                                    angle = angle[len('np.float64('):-1]
                                if angle:
                                    angle_list.append(float(angle))
                            params['angles'] = angle_list
                
                logger.info(f"Loaded parameters from {params_file}")
                return params
            except Exception as e:
                logger.error(f"Error loading parameters from {params_file}: {str(e)}")
                continue
        return None

    def correct_image(self, image):
        """
        Apply distortion correction to the image.
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Corrected image
        """
        # Get the image dimensions
        height, width = image.shape[:2]
        
        # Create a simple radial distortion correction
        # We'll use the distortion factor to create a map
        distortion = self.paper_params.get('distortion', 1.0)
        
        # Create coordinate matrices
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate the distance from the center
        center_x = width / 2
        center_y = height / 2
        r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Apply the distortion correction
        # Scale the radius by the distortion factor
        r_corrected = r * distortion
        
        # Calculate the correction factors
        if r.max() > 0:  # Avoid division by zero
            scale = r_corrected / r
            scale[r == 0] = 1.0  # Keep center point unchanged
        else:
            scale = np.ones_like(r)
        
        # Apply the correction
        X_corrected = (X - center_x) * scale + center_x
        Y_corrected = (Y - center_y) * scale + center_y
        
        # Create the remap
        map_x = X_corrected.astype(np.float32)
        map_y = Y_corrected.astype(np.float32)
        
        # Apply the remap
        corrected_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return corrected_image

    def process_image(self, image_path):
        """
        Process a single image to correct it for manual foot measurement.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            numpy.ndarray: Corrected image if successful, None otherwise
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        # Correct the image
        corrected_image = self.correct_image(image)
        
        return corrected_image

    def process(self):
        """
        Main processing method.
        Process all images in the input directory.
        """
        logger.info("Starting foot measurement process...")
        
        # Process all images in the input directory
        for image_path in self.input_dir.glob('*.jpg'):
            logger.info(f"Processing image: {image_path.name}")
            
            corrected_image = self.process_image(image_path)
            if corrected_image is None:
                continue
                
            # Save the corrected image with original quality
            corrected_path = self.output_dir / f"corrected_{image_path.name}"
            cv2.imwrite(str(corrected_path), corrected_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        logger.info("Foot measurement process completed.")

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define input and output directories
    input_dir = project_root / "Scaling" / "input_foot_detection"
    output_dir = project_root / "Scaling" / "output"
    params_dir = project_root / "Scaling" / "output"
    
    # Create and run the processor
    processor = FootMeasurement(str(input_dir), str(output_dir), str(params_dir))
    processor.process()

if __name__ == "__main__":
    main() 