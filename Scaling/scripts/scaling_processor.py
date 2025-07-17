#!/usr/bin/env python3

"""
A4 Paper Detection and Parameter Extraction Script

This script processes images containing A4 paper to:
1. Detect the A4 paper in the image
2. Calculate distortion parameters
3. Save the parameters for use in foot measurement

The output parameters are used by the foot_measurement.py script to calculate
accurate foot measurements.
"""

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

class ScalingProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the ScalingProcessor for A4 paper detection.
        
        Args:
            input_dir (str): Path to the input directory containing images with A4 paper
            output_dir (str): Path to the output directory for parameters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # A4 paper dimensions in mm
        self.A4_WIDTH = 210
        self.A4_HEIGHT = 297
        
        # A4 paper aspect ratio (height/width)
        self.A4_ASPECT_RATIO = self.A4_HEIGHT / self.A4_WIDTH
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized A4 Paper Detection with input_dir: {input_dir}")
        logger.info(f"Parameters will be saved to: {output_dir}")

    def preprocess_image(self, image):
        """
        Preprocess the image to enhance white paper detection.
        
        Args:
            image: Input image
            
        Returns:
            tuple: (preprocessed_image, debug_image) containing the preprocessed image and a debug image showing the steps
        """
        # Create a copy for debug visualization
        debug_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding to separate white paper from background
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Create debug visualization
        debug_steps = np.hstack((
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        ))
        
        # Add labels to debug image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_steps, "Grayscale", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_steps, "Blurred", (gray.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_steps, "Binary", (2*gray.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_steps, "Final", (3*gray.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        return closing, debug_steps

    def find_largest_contour(self, binary_image):
        """
        Find the largest continuous contour in the binary image.
        
        Args:
            binary_image: Binary image
            
        Returns:
            tuple: (contour, debug_image) containing the largest contour and a debug image
        """
        # Find all contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create debug image
        debug_image = np.zeros_like(binary_image)
        cv2.drawContours(debug_image, [largest_contour], -1, 255, 2)
        
        return largest_contour, debug_image

    def analyze_contour_lines(self, contour):
        """
        Analyze the lines of the contour to find parallel lines and calculate distortion.
        
        Args:
            contour: The contour to analyze
            
        Returns:
            tuple: (lines, angles, debug_image) containing the lines, their angles, and a debug image
        """
        # Approximate the contour to get a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            return None, None, None
            
        # Get the lines
        lines = []
        for i in range(4):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 4][0]
            lines.append((pt1, pt2))
            
        # Calculate angles between lines
        angles = []
        for i in range(4):
            line1 = lines[i]
            line2 = lines[(i + 1) % 4]
            
            # Calculate vectors
            vec1 = np.array(line1[1]) - np.array(line1[0])
            vec2 = np.array(line2[1]) - np.array(line2[0])
            
            # Calculate angle
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angles.append(np.degrees(angle))
            
        # Create debug image
        debug_image = np.zeros((1000, 1000), dtype=np.uint8)
        for line in lines:
            cv2.line(debug_image, tuple(line[0]), tuple(line[1]), 255, 2)
            
        return lines, angles, debug_image

    def calculate_distortion(self, lines, angles):
        """
        Calculate the distortion parameters based on the lines and angles.
        
        Args:
            lines: List of lines
            angles: List of angles
            
        Returns:
            dict: Distortion parameters
        """
        # Calculate line lengths
        lengths = [np.linalg.norm(np.array(line[1]) - np.array(line[0])) for line in lines]
        
        # Find the longest and shortest lines
        max_length = max(lengths)
        min_length = min(lengths)
        
        # Calculate aspect ratio
        aspect_ratio = max_length / min_length
        
        # Calculate distortion
        distortion = aspect_ratio / self.A4_ASPECT_RATIO
        
        return {
            'aspect_ratio': aspect_ratio,
            'distortion': distortion,
            'angles': angles
        }

    def detect_a4_paper(self, image):
        """
        Detect A4 paper in the image and calculate distortion.
        
        Args:
            image: Input image
            
        Returns:
            tuple: (contour, lines, distortion_params, debug_steps) if found, None otherwise
        """
        # Preprocess the image
        preprocessed, debug_steps = self.preprocess_image(image)
        
        # Find the largest contour
        contour, contour_debug = self.find_largest_contour(preprocessed)
        if contour is None:
            return None, None, None, debug_steps
            
        # Analyze the contour lines
        lines, angles, lines_debug = self.analyze_contour_lines(contour)
        if lines is None:
            return None, None, None, debug_steps
            
        # Calculate distortion
        distortion_params = self.calculate_distortion(lines, angles)
        
        return contour, lines, distortion_params, debug_steps

    def detect_foot(self, image, paper_contour):
        """
        Detect the foot in the image.
        
        Args:
            image: Input image
            paper_contour: Contour of the A4 paper
            
        Returns:
            tuple: (foot_contour, debug_image) containing the foot contour and a debug image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to separate foot from background
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Create a mask for the paper area
        paper_mask = np.zeros_like(binary)
        cv2.drawContours(paper_mask, [paper_contour], -1, 255, -1)
        
        # Apply the mask to keep only the paper area
        masked_binary = cv2.bitwise_and(binary, paper_mask)
        
        # Find contours
        contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Find the largest contour that's not the paper
        foot_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area and area < cv2.contourArea(paper_contour) * 0.9:
                max_area = area
                foot_contour = contour
        
        if foot_contour is None:
            return None, None
            
        # Create debug image
        debug_image = image.copy()
        cv2.drawContours(debug_image, [foot_contour], -1, (0, 255, 0), 2)
        
        return foot_contour, debug_image

    def calculate_foot_measurements(self, foot_contour, distortion_params):
        """
        Calculate the actual foot measurements in millimeters.
        
        Args:
            foot_contour: Contour of the foot
            distortion_params: Distortion parameters from A4 paper
            
        Returns:
            dict: Foot measurements in millimeters
        """
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(foot_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Calculate the width and height in pixels
        width_px = rect[1][0]
        height_px = rect[1][1]
        
        # Get the rotation angle
        angle = rect[2]
        
        # Calculate the scaling factor based on A4 paper dimensions
        # We use the shorter dimension of A4 (210mm) as reference
        scaling_factor = self.A4_WIDTH / min(width_px, height_px)
        
        # Apply distortion correction
        corrected_width = width_px * scaling_factor / distortion_params['distortion']
        corrected_height = height_px * scaling_factor / distortion_params['distortion']
        
        # Calculate the actual length and width
        if width_px > height_px:
            length_mm = corrected_width
            width_mm = corrected_height
        else:
            length_mm = corrected_height
            width_mm = corrected_width
        
        return {
            'length_mm': length_mm,
            'width_mm': width_mm,
            'angle_degrees': angle,
            'scaling_factor': scaling_factor
        }

    def process_image(self, image_path):
        """
        Process a single image to detect and scale the foot.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            tuple: (measurements, debug_image) if successful, None otherwise
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        # Detect A4 paper
        paper_detection = self.detect_a4_paper(image)
        if paper_detection[0] is None:
            logger.error(f"Could not detect A4 paper in image: {image_path}")
            return None
            
        contour, lines, distortion_params, debug_steps = paper_detection
        
        # Detect foot
        foot_contour, foot_debug = self.detect_foot(image, contour)
        if foot_contour is None:
            logger.error(f"Could not detect foot in image: {image_path}")
            return None
            
        # Calculate foot measurements
        foot_measurements = self.calculate_foot_measurements(foot_contour, distortion_params)
        
        # Create debug image
        debug_image = image.copy()
        cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)  # Paper contour
        cv2.drawContours(debug_image, [foot_contour], -1, (0, 0, 255), 2)  # Foot contour
        
        # Draw lines
        for line in lines:
            cv2.line(debug_image, tuple(line[0]), tuple(line[1]), (255, 0, 0), 2)
            
        return {
            'distortion_params': distortion_params,
            'foot_measurements': foot_measurements,
            'lines': lines
        }, debug_image, debug_steps

    def process(self):
        """
        Main processing method.
        Process all images in the input directory.
        """
        logger.info("Starting scaling process...")
        
        # Process all images in the input directory
        for image_path in self.input_dir.glob('*.jpg'):
            logger.info(f"Processing image: {image_path.name}")
            
            result = self.process_image(image_path)
            if result is None:
                continue
                
            measurements, debug_image, debug_steps = result
            
            # Save debug images
            debug_path = self.output_dir / f"debug_{image_path.name}"
            cv2.imwrite(str(debug_path), debug_image)
            
            # Save preprocessing steps
            steps_path = self.output_dir / f"steps_{image_path.name}"
            cv2.imwrite(str(steps_path), debug_steps)
            
            # Save measurements
            measurements_path = self.output_dir / f"measurements_{image_path.stem}.txt"
            with open(measurements_path, 'w') as f:
                f.write(f"Distortion parameters:\n")
                f.write(f"Aspect ratio: {measurements['distortion_params']['aspect_ratio']:.2f}\n")
                f.write(f"Distortion factor: {measurements['distortion_params']['distortion']:.2f}\n")
                f.write(f"Angles: {measurements['distortion_params']['angles']}\n\n")
                
                f.write(f"Foot measurements:\n")
                f.write(f"Length: {measurements['foot_measurements']['length_mm']:.1f} mm\n")
                f.write(f"Width: {measurements['foot_measurements']['width_mm']:.1f} mm\n")
                f.write(f"Angle: {measurements['foot_measurements']['angle_degrees']:.1f} degrees\n")
                f.write(f"Scaling factor: {measurements['foot_measurements']['scaling_factor']:.3f} mm/pixel\n")
        
        logger.info("Scaling process completed.")

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define input and output directories
    input_dir = project_root / "Scaling" / "input"  # Directory for A4 paper detection
    output_dir = project_root / "Scaling" / "output"
    
    # Create and run the processor
    processor = ScalingProcessor(str(input_dir), str(output_dir))
    processor.process()

if __name__ == "__main__":
    main() 