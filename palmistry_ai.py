#!/usr/bin/env python3
"""
AI Palmistry Reader - Advanced Palm Analysis System
==================================================
A comprehensive palmistry analysis tool that combines computer vision with traditional
palm reading wisdom to provide detailed insights into personality and destiny.

Author: AI Engineer & Palmistry Scholar
Version: 1.0
"""
def create_annotated_image(self, original_image, classified_lines):
        """
        Create an annotated image with colored palm lines and labels.
        Enhanced for clear visualization of detected lines.
        
        Args:
            original_image (np.ndarray): Original palm image
            classified_lines (dict): Dictionary of classified lines
            
        Returns:
            np.ndarray: Annotated image
        """
        print("🎨 Creating annotated palm visualization...")
        
        # Create a copy for annotation
        annotated = original_image.copy()
        
        # Draw each classified line with enhanced visibility
        line_thickness = 4
        label_font_scale = 0.8
        
        for line_type, line_data in classified_lines.items():
            color = self.colors.get(line_type, (255, 255, 255))
            contour = line_data['contour']
            
            # Draw the line with enhanced thickness
            cv2.drawContours(annotated, [contour], -1, color, line_thickness)
            
            # Add semi-transparent overlay for better visibility
            overlay = annotated.copy()
            cv2.drawContours(overlay, [contour], -1, color, line_thickness + 2)
            cv2.addWeighted(annotated, 0.8, overlay, 0.2, 0, annotated)
            
            # Add label with improved positioning
            mid_point = line_data['mid']
            label = line_type.replace('_', ' ').title()
            
            # Smart label positioning to avoid overlap
            label_offset_x = 15
            label_offset_y = -10
            
            # Adjust label position based on line type
            if line_type == 'heart_line':
                label_offset_y = -25
            elif line_type == 'head_line':
                label_offset_y = 20
            elif line_type == 'life_line':
                label_offset_x = -100
                label_offset_y = 0
            elif line_type == 'fate_line':
                label_offset_x = 20
                label_offset_y = 0
            
            label_pos = (mid_point[0] + label_offset_x, mid_point[1] + label_offset_y)
            
            # Create text with enhanced background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, font, label_font_scale, thickness)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import morphology, measure, filters
from skimage.feature import canny
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import os
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PalmistryAI:
    def __init__(self):
        """Initialize the Palmistry AI system with color mappings and palmistry knowledge."""
        self.colors = {
            'life_line': (0, 255, 0),      # Green
            'head_line': (255, 0, 0),      # Red  
            'heart_line': (255, 0, 255),   # Magenta
            'fate_line': (0, 255, 255),    # Cyan
            'mount_venus': (255, 165, 0),   # Orange
            'mount_jupiter': (128, 0, 128)  # Purple
        }
        
        # Traditional palmistry knowledge base
        self.palmistry_rules = {
            'life_line': {
                'long': "A long, well-defined life line suggests vitality, robust health, and a zest for life that will carry you through many adventures.",
                'short': "A shorter life line indicates intense living - you pack more experience into each moment than most souls do in a lifetime.",
                'curved': "The graceful curve of your life line reveals a warm, generous nature that draws others like moths to flame.",
                'straight': "Your straight life line speaks of determination and unwavering focus in pursuing your life's purpose.",
                'broken': "Breaks in the life line suggest major life transitions - periods of profound transformation that will reshape your destiny.",
                'deep': "The depth of your life line indicates strong life force and the ability to overcome any obstacle through sheer willpower.",
                'faint': "A faint life line suggests a more spiritual, introspective nature - one who finds strength in quiet contemplation."
            },
            'head_line': {
                'long': "Your extensive head line reveals an analytical mind capable of penetrating the deepest mysteries of existence.",
                'short': "A concise head line indicates quick, decisive thinking - you grasp concepts instantly and act with precision.",
                'curved': "The curve in your head line shows creative intelligence and the ability to see solutions others miss.",
                'straight': "Your straight head line reveals logical, methodical thinking - a mind that builds understanding brick by brick.",
                'forked': "The fork at the end of your head line indicates versatility - you can adapt your thinking to any situation.",
                'deep': "The depth of your head line suggests intense concentration and the ability to focus completely on your goals.",
                'rising': "An upward-sloping head line indicates optimism and the power to turn dreams into reality through positive thinking."
            },
            'heart_line': {
                'long': "Your extensive heart line reveals a generous, loving nature capable of deep, lasting emotional connections.",
                'short': "A shorter heart line indicates selective but intense emotional bonds - you love deeply but choose carefully.",
                'curved': "The curve of your heart line shows emotional warmth and the ability to empathize deeply with others' feelings.",
                'straight': "Your straight heart line reveals controlled emotions and the wisdom to think with your heart while feeling with your mind.",
                'high': "A high heart line indicates idealistic love - you seek soulmate connections that transcend the ordinary.",
                'low': "A lower heart line suggests passionate, physical love - you experience romance with all your senses.",
                'chained': "Chains in your heart line reveal complex emotional experiences that have taught you profound lessons about love.",
                'deep': "The depth of your heart line indicates the intensity of your emotional nature - you feel everything profoundly."
            },
            'fate_line': {
                'present': "The presence of a fate line marks you as one guided by destiny - external forces shape your path in meaningful ways.",
                'absent': "The absence of a fate line indicates self-determination - you are the master of your own destiny, creating your path through will alone.",
                'straight': "Your straight fate line reveals a clear life purpose and the ability to follow your destined path without deviation.",
                'wavy': "Undulations in your fate line suggest a dynamic life with many changes - each twist brings new opportunities for growth.",
                'deep': "A deep fate line indicates strong karmic influences - past actions continue to shape your present circumstances.",
                'starting_low': "Your fate line beginning near the wrist suggests early awareness of your life purpose and destiny.",
                'starting_high': "A fate line starting higher indicates that your true calling will emerge later in life, bringing unexpected fulfillment."
            }
        }

    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess the palm image for analysis.
        
        Args:
            image_path (str): Path to the palm image
            
        Returns:
            tuple: (original_image, preprocessed_image, gray_image)
        """
        print(f"🔍 Loading and preprocessing image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (maintain aspect ratio)
        height, width = original.shape[:2]
        if width > 1024 or height > 1024:
            scale = min(1024/width, 1024/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_rgb = cv2.resize(original_rgb, (new_width, new_height))
            print(f"📏 Resized image to {new_width}x{new_height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        print("✅ Image preprocessing complete")
        return original_rgb, enhanced, gray

    def detect_palm_region(self, image):
        """
        Detect and isolate the palm region from the hand image.
        Optimized for high-quality palm images with clear backgrounds.
        
        Args:
            image (np.ndarray): Preprocessed grayscale image
            
        Returns:
            tuple: (palm_mask, palm_center, palm_radius)
        """
        print("🖐️ Detecting palm region...")
        
        h, w = image.shape
        
        # For high-quality palm images like the sample, use adaptive approach
        # Apply Gaussian blur to smooth variations
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Use Otsu thresholding to separate hand from background
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create refined mask
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            # Calculate palm center more accurately for this type of image
            # The palm center is typically in the lower-middle region
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # Adjust center towards palm area (lower part of hand)
                palm_center = (cx, min(cy + int(h * 0.1), h - 50))
            else:
                palm_center = (w//2, int(h * 0.6))
            
            # Calculate palm radius based on hand size
            palm_radius = int(np.sqrt(cv2.contourArea(largest_contour)) * 0.25)
            
        else:
            # Fallback for edge cases
            palm_center = (w//2, int(h * 0.6))
            palm_radius = min(w, h) // 4
            mask = np.zeros_like(image)
            cv2.circle(mask, palm_center, int(palm_radius * 1.5), 255, -1)
        
        print(f"✅ Palm region detected - Center: {palm_center}, Radius: {palm_radius}")
        return mask, palm_center, palm_radius

    def extract_palm_lines(self, image, palm_mask):
        """
        Extract and enhance palm lines using advanced edge detection.
        Optimized for high-contrast palm images with clear line definition.
        
        Args:
            image (np.ndarray): Preprocessed grayscale image
            palm_mask (np.ndarray): Binary mask of palm region
            
        Returns:
            tuple: (line_image, skeleton_image)
        """
        print("📏 Extracting palm lines...")
        
        # Apply mask to focus on palm region
        masked_image = cv2.bitwise_and(image, palm_mask)
        
        # For high-quality images, use multiple enhancement techniques
        
        # 1. Enhance line contrast using morphological operations
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Vertical lines
        enhanced_v = cv2.morphologyEx(masked_image, cv2.MORPH_BLACKHAT, kernel_line)
        
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal lines
        enhanced_h = cv2.morphologyEx(masked_image, cv2.MORPH_BLACKHAT, kernel_line)
        
        # Combine directional enhancements
        enhanced = cv2.add(enhanced_v, enhanced_h)
        
        # 2. Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(masked_image, 9, 80, 80)
        
        # 3. Multi-scale edge detection with optimized parameters for palm lines
        edges1 = canny(bilateral, sigma=0.8, low_threshold=30, high_threshold=80)
        edges2 = canny(bilateral, sigma=1.5, low_threshold=20, high_threshold=60)
        edges3 = canny(enhanced, sigma=1.0, low_threshold=25, high_threshold=70)
        
        # Combine all edge maps
        combined_edges = np.logical_or(np.logical_or(edges1, edges2), edges3).astype(np.uint8) * 255
        
        # 4. Morphological operations to connect line segments
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        connected_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_connect)
        
        # 5. Apply advanced thinning for precise line extraction
        skeleton = morphology.skeletonize(connected_edges > 0).astype(np.uint8) * 255
        
        # 6. Clean up skeleton - remove noise and short segments
        # Remove small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)
        
        cleaned_skeleton = np.zeros_like(skeleton)
        min_area = 10  # Decreased minimum pixels for a valid line segment
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_skeleton[labels == i] = 255
        
        # 7. Final smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final_skeleton = cv2.morphologyEx(cleaned_skeleton, cv2.MORPH_OPEN, kernel_smooth)
        
        print("✅ Palm lines extracted and refined")
        return connected_edges, final_skeleton

    def classify_palm_lines(self, skeleton, palm_center, palm_radius):
        """
        Classify detected lines into major palm lines using heuristics.
        Enhanced for high-quality palm images with clearly visible lines.
        
        Args:
            skeleton (np.ndarray): Skeletonized line image
            palm_center (tuple): Center coordinates of palm
            palm_radius (int): Estimated palm radius
            
        Returns:
            dict: Dictionary containing classified lines
        """
        print("🔮 Classifying palm lines using ancient wisdom...")
        
        # Find line contours with better connectivity
        # Use RETR_EXTERNAL to get main line segments
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️ No lines detected")
            return {}
        
        # For high-quality images, use more refined filtering
        min_length = palm_radius * 0.05  # Much more permissive for line detection
        valid_contours = []
        
        for cnt in contours:
            arc_length = cv2.arcLength(cnt, False)
            if arc_length > min_length:
                # Additional filtering based on line characteristics
                # Remove very small or circular contours
                area = cv2.contourArea(cnt)
                if area < arc_length * 0.8:  # Avoid thick blobs
                    valid_contours.append(cnt)
        
        if not valid_contours:
            print("⚠️ No significant lines found")
            return {}
        
        h, w = skeleton.shape
        cx, cy = palm_center
        
        classified_lines = {}
        
        # Sort contours by length (longest first) and analyze top candidates
        valid_contours.sort(key=lambda x: cv2.arcLength(x, False), reverse=True)
        
        # For high-quality images, we can analyze more lines
        max_lines_to_analyze = min(10, len(valid_contours))
        
        for i, contour in enumerate(valid_contours[:max_lines_to_analyze]):
            # Calculate line properties with higher precision
            length = cv2.arcLength(contour, False)
            
            # Get line endpoints and calculate multiple points for better analysis
            contour_points = contour.reshape(-1, 2)
            start_point = contour_points[0]
            end_point = contour_points[-1]
            
            # Use multiple points for better curve analysis
            quarter_idx = len(contour_points) // 4
            mid_idx = len(contour_points) // 2
            three_quarter_idx = 3 * len(contour_points) // 4
            
            quarter_point = contour_points[quarter_idx] if quarter_idx < len(contour_points) else start_point
            mid_point = contour_points[mid_idx]
            three_quarter_point = contour_points[three_quarter_idx] if three_quarter_idx < len(contour_points) else end_point
            
            # Calculate relative positions
            rel_start_x = (start_point[0] - cx) / palm_radius
            rel_start_y = (start_point[1] - cy) / palm_radius
            rel_end_x = (end_point[0] - cx) / palm_radius
            rel_end_y = (end_point[1] - cy) / palm_radius
            rel_mid_x = (mid_point[0] - cx) / palm_radius
            rel_mid_y = (mid_point[1] - cy) / palm_radius
            
            # Calculate angle and curvature with better precision
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = math.degrees(math.atan2(dy, dx)) % 360
            
            # Enhanced curvature calculation using multiple points
            straight_dist = np.sqrt(dx**2 + dy**2)
            curvature = length / max(straight_dist, 1) if straight_dist > 0 else 1
            
            # Additional curvature measure using midpoint deviation
            if straight_dist > 0:
                # Distance from midpoint to straight line between start and end
                line_to_mid = abs((end_point[1] - start_point[1]) * mid_point[0] - 
                                 (end_point[0] - start_point[0]) * mid_point[1] + 
                                 end_point[0] * start_point[1] - end_point[1] * start_point[0]) / straight_dist
                curvature_alt = 1 + (line_to_mid / (straight_dist / 4))  # Normalized curvature
                curvature = max(curvature, curvature_alt)
            
            # Classification with enhanced heuristics
            line_type = self._classify_single_line(
                start_point, end_point, mid_point, cx, cy, palm_radius,
                angle, curvature, length, rel_start_y, rel_end_y, rel_mid_x, rel_mid_y
            )
            
            if line_type and line_type not in classified_lines:
                classified_lines[line_type] = {
                    'contour': contour,
                    'length': length,
                    'angle': angle,
                    'curvature': curvature,
                    'start': start_point,
                    'end': end_point,
                    'mid': mid_point,
                    'quarter': quarter_point,
                    'three_quarter': three_quarter_point,
                    'properties': self._analyze_line_properties(contour, line_type, length, angle, curvature)
                }
        
        # Post-processing: ensure we have the major lines and resolve conflicts
        classified_lines = self._resolve_line_conflicts(classified_lines, cx, cy, palm_radius)
        
        # If no lines classified, fallback: show all as 'unknown_line' for debugging
        if not classified_lines and valid_contours:
            print("[DEBUG] No lines classified, showing all detected as 'unknown_line'.")
            for i, contour in enumerate(valid_contours[:max_lines_to_analyze]):
                length = cv2.arcLength(contour, False)
                contour_points = contour.reshape(-1, 2)
                mid_point = contour_points[len(contour_points)//2]
                classified_lines[f'unknown_line_{i+1}'] = {
                    'contour': contour,
                    'length': length,
                    'angle': 0,
                    'curvature': 1,
                    'start': contour_points[0],
                    'end': contour_points[-1],
                    'mid': mid_point,
                    'quarter': contour_points[len(contour_points)//4] if len(contour_points) > 4 else contour_points[0],
                    'three_quarter': contour_points[3*len(contour_points)//4] if len(contour_points) > 4 else contour_points[-1],
                    'properties': []
                }
        
        print(f"✅ Classified {len(classified_lines)} major palm lines")
        return classified_lines
    
    def _resolve_line_conflicts(self, classified_lines, cx, cy, palm_radius):
        """
        Resolve conflicts in line classification and ensure major lines are properly identified.
        """
        # If we have multiple candidates for the same line type, keep the best one
        resolved_lines = {}
        
        for line_type, line_data in classified_lines.items():
            if line_type not in resolved_lines:
                resolved_lines[line_type] = line_data
            else:
                # Keep the longer line
                if line_data['length'] > resolved_lines[line_type]['length']:
                    resolved_lines[line_type] = line_data
        
        # Ensure logical ordering of heart line vs head line (heart should be higher)
        if 'heart_line' in resolved_lines and 'head_line' in resolved_lines:
            heart_y = resolved_lines['heart_line']['mid'][1]
            head_y = resolved_lines['head_line']['mid'][1]
            
            # If head line is above heart line, swap them
            if head_y < heart_y:
                temp = resolved_lines['heart_line']
                resolved_lines['heart_line'] = resolved_lines['head_line']
                resolved_lines['head_line'] = temp
                
                # Update their properties
                resolved_lines['heart_line']['properties'] = self._analyze_line_properties(
                    resolved_lines['heart_line']['contour'], 'heart_line', 
                    resolved_lines['heart_line']['length'], resolved_lines['heart_line']['angle'], 
                    resolved_lines['heart_line']['curvature'])
                
                resolved_lines['head_line']['properties'] = self._analyze_line_properties(
                    resolved_lines['head_line']['contour'], 'head_line', 
                    resolved_lines['head_line']['length'], resolved_lines['head_line']['angle'], 
                    resolved_lines['head_line']['curvature'])
        
        return resolved_lines

    def _classify_single_line(self, start, end, mid, cx, cy, radius, angle, curvature, length, rel_start_y, rel_end_y, rel_mid_x, rel_mid_y):
        """
        Helper method to classify a single line based on position and characteristics.
        Updated for the specific palm image type with clear major lines.
        """
        
        # Calculate more precise relative positions
        rel_start_x = (start[0] - cx) / radius
        rel_end_x = (end[0] - cx) / radius
        
        # Life Line: Curves around thumb area, typically starts from index finger area
        # and curves down towards wrist on the thumb side
        if (rel_start_x > -0.9 and rel_start_x < 0.3 and  # Starting position (less strict)
            rel_end_x < -0.2 and  # Ending towards thumb side (less strict)
            rel_start_y < 0.1 and rel_end_y > 0.2 and  # Top to bottom movement (less strict)
            curvature > 1.05 and length > radius * 0.3):  # Curved and substantial (less strict curvature and length)
            return 'life_line'
        
        # Heart Line: Upper horizontal line, runs across the upper palm
        # Usually the topmost major horizontal line
        if (rel_start_y < -0.1 and rel_end_y < 0.2 and  # Upper region of palm (less strict)
            abs(rel_end_y - rel_start_y) < 0.5 and  # Relatively horizontal (less strict)
            length > radius * 0.5 and  # Substantial length (less strict)
            abs(rel_mid_y) < 0.4):  # Middle point in reasonable range (less strict)
            return 'heart_line'
        
        # Head Line: Middle horizontal line, often straighter than heart line
        # Usually below the heart line, runs across palm
        if (rel_start_y > -0.2 and rel_start_y < 0.5 and  # Middle region (less strict)
            rel_end_y > -0.3 and rel_end_y < 0.6 and  # Stays in middle region (less strict)
            abs(rel_end_y - rel_start_y) < 0.4 and  # More horizontal (less strict)
            length > radius * 0.4 and  # Good length (less strict)
            abs(rel_mid_y) < 0.5):  # Reasonable middle position (less strict)
            return 'head_line'
        
        # Fate Line: Vertical or near-vertical line through palm center
        # Runs from bottom to top of palm, often through the middle
        if (abs(rel_mid_x) < 0.5 and  # Near center horizontally (less strict)
            length > radius * 0.4 and  # Substantial length (less strict)
            abs(rel_start_y - rel_end_y) > 0.5 and  # Significant vertical span (less strict)
            (abs(angle - 90) < 30 or abs(angle - 270) < 30)):  # Near vertical (less strict angle)
            return 'fate_line'
        
        # Additional check for secondary lines that might be fate line
        # Sometimes fate line is not perfectly vertical
        if (abs(rel_mid_x) < 0.6 and  # Reasonably centered (less strict)
            length > radius * 0.3 and # Less strict length
            abs(rel_start_y - rel_end_y) > 0.4 and  # Good vertical span (less strict)
            curvature < 1.2):  # Relatively straight (less strict curvature)
            return 'fate_line'
        
        return None

    def _analyze_line_properties(self, contour, line_type, length, angle, curvature):
        """Analyze specific properties of a classified line for palmistry interpretation."""
        properties = []
        
        # Length analysis
        if length > 200:
            properties.append('long')
        elif length < 100:
            properties.append('short')
        
        # Curvature analysis
        if curvature > 1.2:
            properties.append('curved')
        elif curvature < 1.05:
            properties.append('straight')
        
        # Line-specific properties
        if line_type == 'heart_line':
            # Analyze height and depth
            contour_points = contour.reshape(-1, 2)
            y_coords = contour_points[:, 1]
            if np.std(y_coords) < 5:  # Very straight
                properties.append('straight')
            if np.mean(y_coords) < 150:  # Assuming image height > 300
                properties.append('high')
            else:
                properties.append('low')
        
        elif line_type == 'head_line':
            # Check for forks or branches
            if curvature > 1.15:
                properties.append('forked')
            # Check slope
            if angle > 0 and angle < 15:
                properties.append('rising')
        
        elif line_type == 'life_line':
            # Analyze depth and continuity
            if len(contour) > 100:  # More points suggest deeper line
                properties.append('deep')
            else:
                properties.append('faint')
        
        elif line_type == 'fate_line':
            properties.append('present')
            contour_points = contour.reshape(-1, 2)
            start_y = min(contour_points[:, 1])
            if start_y > 200:  # Starting from middle of palm
                properties.append('starting_high')
            else:
                properties.append('starting_low')
        
        return properties

    def create_annotated_image(self, original_image, classified_lines):
        """
        Create an annotated image with colored palm lines and labels.
        
        Args:
            original_image (np.ndarray): Original palm image
            classified_lines (dict): Dictionary of classified lines
            
        Returns:
            np.ndarray: Annotated image
        """
        print("🎨 Creating annotated palm visualization...")
        
        # Create a copy for annotation
        annotated = original_image.copy()
        
        # Draw each classified line
        for line_type, line_data in classified_lines.items():
            color = self.colors.get(line_type, (255, 255, 255))
            contour = line_data['contour']
            
            # Draw the line with thickness
            cv2.drawContours(annotated, [contour], -1, color, 3)
            
            # Add label near the line
            mid_point = line_data['mid']
            label = line_type.replace('_', ' ').title()
            
            # Create text with background for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(annotated, 
                         (mid_point[0] - 5, mid_point[1] - text_height - 5),
                         (mid_point[0] + text_width + 5, mid_point[1] + baseline + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(annotated, label, 
                       (mid_point[0], mid_point[1]), 
                       font, font_scale, color, thickness)
        
        # Add timestamp and title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_text = f"Palmistry Analysis - {timestamp}"
        cv2.putText(annotated, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        print("✅ Annotated image created")
        return annotated

    def generate_palmistry_report(self, classified_lines):
        """
        Generate a comprehensive palmistry reading based on detected lines.
        
        Args:
            classified_lines (dict): Dictionary of classified lines with properties
            
        Returns:
            str: Formatted palmistry report
        """
        print("📜 Generating mystical palmistry report...")
        
        report_lines = []
        report_lines.append("🔮 MYSTICAL PALMISTRY ANALYSIS 🔮")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        report_lines.append(f"Reading performed on: {timestamp}")
        report_lines.append("")
        
        # Introduction
        report_lines.append("The ancient art of palmistry reveals the secrets written in the lines of your hand.")
        report_lines.append("Each line tells a story of your past, present, and the destiny that awaits...")
        report_lines.append("")
        
        # Analyze each major line
        major_lines = ['life_line', 'head_line', 'heart_line', 'fate_line']
        
        for line_type in major_lines:
            if line_type in classified_lines:
                line_data = classified_lines[line_type]
                properties = line_data['properties']
                
                # Line title
                line_title = line_type.replace('_', ' ').title()
                report_lines.append(f"📏 {line_title.upper()}")
                report_lines.append("-" * 20)
                
                # Generate interpretations based on properties
                interpretations = []
                for prop in properties:
                    if prop in self.palmistry_rules[line_type]:
                        interpretations.append(self.palmistry_rules[line_type][prop])
                
                if interpretations:
                    for interpretation in interpretations[:2]:  # Max 2 per line
                        report_lines.append(interpretation)
                        report_lines.append("")
                else:
                    # Default interpretation
                    report_lines.append(f"Your {line_title.lower()} shows unique characteristics that speak to your individual journey through life.")
                    report_lines.append("")
                
                # Add technical details
                length_desc = "long and prominent" if line_data['length'] > 150 else "concise but meaningful"
                curve_desc = "gracefully curved" if line_data['curvature'] > 1.1 else "remarkably straight"
                report_lines.append(f"Technical observation: This {length_desc} line appears {curve_desc}, suggesting specific traits in your character.")
                report_lines.append("")
            
            else:
                # Handle missing lines
                if line_type == 'fate_line':
                    report_lines.append("📏 FATE LINE")
                    report_lines.append("-" * 20)
                    report_lines.append(self.palmistry_rules['fate_line']['absent'])
                    report_lines.append("")
        
        # Overall synthesis
        report_lines.append("🌟 SYNTHESIS OF YOUR PALM")
        report_lines.append("-" * 30)
        
        # Count line characteristics for overall reading
        total_lines = len(classified_lines)
        
        if total_lines >= 3:
            report_lines.append("Your palm reveals a rich tapestry of lines, indicating a complex and multifaceted personality. ")
            report_lines.append("You are someone who experiences life fully, with deep emotional connections, intellectual curiosity, ")
            report_lines.append("and a strong sense of purpose guiding your path.")
        elif total_lines == 2:
            report_lines.append("Your palm shows clear, defined lines that speak to a focused individual who knows their priorities. ")
            report_lines.append("You prefer quality over quantity in all aspects of life, from relationships to pursuits.")
        else:
            report_lines.append("Your palm displays unique line patterns that suggest an independent spirit who creates their own ")
            report_lines.append("destiny through determination and personal will.")
        
        report_lines.append("")
        report_lines.append("Remember, palmistry is an ancient art of guidance, not predetermination. Your choices and actions ")
        report_lines.append("shape your destiny more than any line upon your palm. Use this insight as a tool for self-reflection ")
        report_lines.append("and personal growth on your journey through life.")
        
        report_lines.append("")
        report_lines.append("🔮 End of Reading 🔮")
        report_lines.append("=" * 50)
        
        full_report = "\n".join(report_lines)
        print("✅ Mystical report generated")
        return full_report

    def save_results(self, annotated_image, report, output_image_path="palm_with_lines_overlay.png", output_text_path="palmistry_report.txt"):
        """
        Save the annotated image and report to files.
        
        Args:
            annotated_image (np.ndarray): Annotated palm image
            report (str): Generated palmistry report
            output_image_path (str): Path for output image
            output_text_path (str): Path for output text report
        """
        print("💾 Saving results...")
        
        # Save annotated image
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title('AI Palmistry Analysis - Annotated Palm Lines', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save report
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Results saved:")
        print(f"   📸 Annotated image: {output_image_path}")
        print(f"   📜 Report: {output_text_path}")

    def analyze_palm(self, image_path):
        """
        Complete palm analysis pipeline.
        
        Args:
            image_path (str): Path to the palm image
            
        Returns:
            tuple: (annotated_image, palmistry_report)
        """
        print("🚀 Starting AI Palmistry Analysis...")
        print("=" * 50)
        
        try:
            # Step 1: Load and preprocess image
            original_image, processed_image, gray_image = self.load_and_preprocess_image(image_path)
            
            # Step 2: Detect palm region
            palm_mask, palm_center, palm_radius = self.detect_palm_region(processed_image)
            
            # Step 3: Extract palm lines
            line_image, skeleton_image = self.extract_palm_lines(processed_image, palm_mask)
            
            # Step 4: Classify palm lines
            classified_lines = self.classify_palm_lines(skeleton_image, palm_center, palm_radius)
            
            if not classified_lines:
                print("⚠️ Warning: No major palm lines could be reliably classified.")
                print("This might be due to image quality, lighting, or hand positioning.")
                # Create a basic report for undetected lines
                basic_report = self._generate_basic_report()
                return original_image, basic_report
            
            # Step 5: Create annotated visualization
            annotated_image = self.create_annotated_image(original_image, classified_lines)
            
            # Step 6: Generate palmistry report
            palmistry_report = self.generate_palmistry_report(classified_lines)
            
            # Step 7: Save results
            self.save_results(annotated_image, palmistry_report)
            
            print("🎉 Analysis complete! The secrets of your palm have been revealed.")
            return annotated_image, palmistry_report
            
        except Exception as e:
            error_msg = f"❌ Error during palm analysis: {str(e)}"
            print(error_msg)
            return None, error_msg

    def _generate_basic_report(self):
        """Generate a basic report when no lines are detected."""
        report_lines = [
            "🔮 MYSTICAL PALMISTRY ANALYSIS 🔮",
            "=" * 50,
            "",
            f"Reading performed on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            "",
            "🌟 UNIQUE PALM READING",
            "-" * 25,
            "",
            "Your palm presents a fascinating challenge to traditional reading methods. Sometimes,",
            "the most interesting palms are those that don't conform to conventional patterns.",
            "",
            "This could indicate:",
            "• A free spirit who creates their own destiny",
            "• Someone whose fate is still being written",
            "• A person whose true nature transcends physical manifestation",
            "",
            "Consider this an invitation to look deeper within yourself for answers.",
            "Your destiny lies not in the lines of your palm, but in the choices you make",
            "and the paths you choose to walk.",
            "",
            "🔮 End of Reading 🔮",
            "=" * 50
        ]
        return "\n".join(report_lines)


def main():
    """Main function to demonstrate the palmistry AI system."""
    
    # Initialize the AI system
    palmistry_ai = PalmistryAI()
    
    # Image path (replace with your image)
    image_path = "IMG_0015.jpeg"  # Replace with actual image path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("Please ensure the image file exists in the current directory.")
        print("\n🎯 To test this system:")
        print("1. Place a high-resolution palm image in the same directory as this script")
        print("2. Update the 'image_path' variable with your image filename")
        print("3. Run the script again")
        return
    
    # Perform analysis
    annotated_image, report = palmistry_ai.analyze_palm(image_path)
    
    # Display results
    if annotated_image is not None:
        print("\n" + "="*60)
        print("📜 PALMISTRY REPORT")
        print("="*60)
        print(report)
        
        # Show annotated image
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title('Your Palm Analysis - Lines of Destiny Revealed', fontsize=16, pad=20)
        plt.show()
    
    print("\n🌟 Thank you for using the AI Palmistry Reader!")
    print("May the wisdom of your palm guide you on your journey through life.")


# Optional: Streamlit Web Interface
def create_streamlit_app():
    """
    Create a Streamlit web interface for the palmistry AI.
    Uncomment and run with: streamlit run palmistry_ai.py
    """
    try:
        import streamlit as st
        from PIL import Image as PILImage
        import io

        # --- REMOVE API Key Gate: Let users upload image directly ---
        # (API key input and validation removed)

        # --- Custom CSS for background and aesthetics ---
        st.markdown(
            """
            <style>
            body {
                background: linear-gradient(135deg, #1e1e2f 0%, #3a2c4d 100%) !important;
                color: #f3e9ff;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                background: rgba(30, 30, 47, 0.95);
                border-radius: 18px;
                box-shadow: 0 4px 32px 0 rgba(80, 0, 80, 0.15);
            }
            .stButton>button {
                background: linear-gradient(90deg, #a4508b 0%, #5f0a87 100%);
                color: white;
                border-radius: 8px;
                font-size: 1.1rem;
                font-weight: bold;
                padding: 0.6em 2em;
                margin-top: 1em;
            }
            .stTextArea textarea {
                background: #2d223a;
                color: #f3e9ff;
                border-radius: 10px;
                font-size: 1.1rem;
            }
            .stDownloadButton>button {
                background: linear-gradient(90deg, #5f0a87 0%, #a4508b 100%);
                color: white;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: bold;
            }
            .stExpanderHeader {
                color: #e0b3ff !important;
                font-weight: bold;
            }
            .stSidebar {
                background: #2d223a !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # --- Optional: Banner or header image ---
        st.markdown(
            '<div style="text-align:center; margin-bottom:1.5em;">'
            '<span style="font-size:3rem;">🔮</span>'
            '<span style="font-size:2.2rem; font-family:serif; color:#e0b3ff;"> AI Palmistry Reader </span>'
            '<span style="font-size:3rem;">✨</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<hr style="border:1px solid #a4508b; margin-bottom:1.5em;">', unsafe_allow_html=True)

        st.markdown('<div style="text-align:center; color:#e0b3ff; font-size:1.2rem; margin-bottom:1.5em;">Unlock the secrets written in the lines of your palm</div>', unsafe_allow_html=True)

        # Sidebar for instructions in an expander
        with st.sidebar:
            st.markdown('<div style="font-size:1.3rem; color:#e0b3ff;">📋 Instructions</div>', unsafe_allow_html=True)
            with st.expander("How to use the Palmistry Reader", expanded=True):
                st.markdown("""
                1. Upload a clear photo of your palm
                2. Ensure good lighting and contrast
                3. Keep fingers spread apart
                4. Click 'Analyze My Palm' and wait for the AI
                5. Receive your mystical reading!
                """)
            st.markdown('<hr style="border:1px solid #a4508b;">', unsafe_allow_html=True)
            st.markdown('<div style="color:#e0b3ff; font-size:1.1rem;">✨ Tip: For best results, use a high-resolution image with a plain background.</div>', unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a palm image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a high-quality image of your palm with fingers spread"
        )

        if uploaded_file is not None:
            image = PILImage.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("📸 Your Palm Image")
                st.image(image, caption="Uploaded palm image", use_container_width=True)
            temp_path = f"temp_palm_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if st.button("🔮 Analyze My Palm", type="primary"):
                with st.spinner("🔍 Analyzing your palm... The spirits are consulting..."):
                    palmistry_ai = PalmistryAI()
                    original_image, processed_image, gray_image = palmistry_ai.load_and_preprocess_image(temp_path)
                    palm_mask, palm_center, palm_radius = palmistry_ai.detect_palm_region(processed_image)
                    line_image, skeleton_image = palmistry_ai.extract_palm_lines(processed_image, palm_mask)
                    classified_lines = palmistry_ai.classify_palm_lines(skeleton_image, palm_center, palm_radius)
                    annotated_image = palmistry_ai.create_annotated_image(original_image, classified_lines)
                    report = palmistry_ai.generate_palmistry_report(classified_lines)
                try:
                    os.remove(temp_path)
                except:
                    pass
                if annotated_image is not None:
                    with col2:
                        st.subheader("✨ Annotated Analysis")
                        st.image(annotated_image, caption="Your palm with detected lines", use_container_width=True)
                        with st.expander("Show detected palm lines (skeleton/edge image)"):
                            st.image(skeleton_image, caption="Detected palm lines (skeleton)", use_container_width=True)
                    st.subheader("📜 Your Palmistry Reading")
                    st.text_area("Your Mystical Reading", value=report, height=400, disabled=True)
                    col3, col4 = st.columns([1, 1])
                    with col3:
                        img_buffer = io.BytesIO()
                        annotated_pil = PILImage.fromarray(annotated_image)
                        annotated_pil.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        st.download_button(
                            label="📸 Download Annotated Image",
                            data=img_buffer.getvalue(),
                            file_name="palm_analysis.png",
                            mime="image/png"
                        )
                    with col4:
                        st.download_button(
                            label="📜 Download Report",
                            data=report,
                            file_name="palmistry_report.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("❌ Analysis failed. Please try with a different image.")
        st.markdown('<hr style="border:1px solid #a4508b; margin-top:2em;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; color:#e0b3ff; font-size:1.1rem; margin-top:1em;">✨ Created with mystical AI technology and ancient palmistry wisdom ✨<br><span style="font-size:1.3rem;">"The lines of your palm are the map of your soul."</span></div>', unsafe_allow_html=True)
    except ImportError:
        print("Streamlit not installed. To use the web interface, install with: pip install streamlit")


# Advanced Features and Enhancements
class AdvancedPalmistryAI(PalmistryAI):
    """
    Advanced version with additional features like mount detection and ML integration.
    """
    
    def __init__(self):
        super().__init__()
        self.mount_regions = {
            'venus': {'position': (-0.6, 0.3), 'size': 0.3},
            'jupiter': {'position': (0.4, -0.7), 'size': 0.2},
            'saturn': {'position': (0.1, -0.8), 'size': 0.2},
            'apollo': {'position': (-0.2, -0.8), 'size': 0.2},
            'mercury': {'position': (-0.5, -0.7), 'size': 0.2},
            'mars_positive': {'position': (0.2, 0.1), 'size': 0.2},
            'mars_negative': {'position': (-0.3, -0.2), 'size': 0.2},
            'luna': {'position': (-0.7, -0.2), 'size': 0.3}
        }
    
    def detect_palm_mounts(self, image, palm_center, palm_radius):
        """
        Detect and analyze palm mounts (raised areas) for advanced reading.
        
        Args:
            image (np.ndarray): Processed palm image
            palm_center (tuple): Center coordinates of palm
            palm_radius (int): Palm radius
            
        Returns:
            dict: Detected mounts with their characteristics
        """
        print("🏔️ Detecting palm mounts...")
        
        # Apply Gaussian blur to highlight raised areas
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        
        # Use morphological operations to detect elevated regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        detected_mounts = {}
        cx, cy = palm_center
        
        for mount_name, mount_info in self.mount_regions.items():
            rel_x, rel_y = mount_info['position']
            size = mount_info['size']
            
            # Calculate actual position
            mount_x = int(cx + rel_x * palm_radius)
            mount_y = int(cy + rel_y * palm_radius)
            mount_size = int(size * palm_radius)
            
            # Extract region
            x1 = max(0, mount_x - mount_size)
            y1 = max(0, mount_y - mount_size)
            x2 = min(image.shape[1], mount_x + mount_size)
            y2 = min(image.shape[0], mount_y + mount_size)
            
            if x2 > x1 and y2 > y1:
                region = tophat[y1:y2, x1:x2]
                
                # Analyze mount prominence
                prominence = np.mean(region)
                if prominence > 5:  # Threshold for significant mount
                    detected_mounts[mount_name] = {
                        'position': (mount_x, mount_y),
                        'prominence': prominence,
                        'size': mount_size
                    }
        
        print(f"✅ Detected {len(detected_mounts)} prominent mounts")
        return detected_mounts
    
    def generate_advanced_report(self, classified_lines, detected_mounts):
        """
        Generate an advanced palmistry report including mount analysis.
        
        Args:
            classified_lines (dict): Classified palm lines
            detected_mounts (dict): Detected palm mounts
            
        Returns:
            str: Advanced palmistry report
        """
        # Get basic report
        basic_report = self.generate_palmistry_report(classified_lines)
        
        # Add mount analysis
        mount_analysis = [
            "",
            "🏔️ MOUNT ANALYSIS - THE HILLS OF DESTINY",
            "-" * 45,
            "",
            "The mounts of your palm represent different aspects of your personality and potential:",
            ""
        ]
        
        mount_meanings = {
            'venus': "Mount of Venus - Love, passion, and vitality flow strongly through your being.",
            'jupiter': "Mount of Jupiter - Leadership qualities and ambition mark your character.",
            'saturn': "Mount of Saturn - Wisdom, discipline, and deep thinking guide your path.",
            'apollo': "Mount of Apollo - Creativity, artistic talent, and charisma shine within you.",
            'mercury': "Mount of Mercury - Communication skills and business acumen are your gifts.",
            'mars_positive': "Mount of Mars (Positive) - Courage and determination drive your actions.",
            'mars_negative': "Mount of Mars (Negative) - Patience and strategic thinking are your strengths.",
            'luna': "Mount of Luna - Intuition, imagination, and mystical awareness illuminate your soul."
        }
        
        if detected_mounts:
            for mount_name, mount_data in detected_mounts.items():
                if mount_name in mount_meanings:
                    prominence_desc = "highly developed" if mount_data['prominence'] > 10 else "moderately developed"
                    mount_analysis.append(f"🔸 {mount_meanings[mount_name]}")
                    mount_analysis.append(f"   This mount appears {prominence_desc}, indicating strong influence in your life.")
                    mount_analysis.append("")
        else:
            mount_analysis.append("Your palm shows a balanced mount structure, suggesting harmony between different")
            mount_analysis.append("aspects of your personality. No single trait dominates, creating a well-rounded character.")
            mount_analysis.append("")
        
        # Combine reports
        full_report = basic_report + "\n".join(mount_analysis)
        return full_report


# Example usage and testing functions
def test_with_sample_data():
    """
    Test the palmistry AI with synthetic data when no real image is available.
    """
    print("🧪 Running test with synthetic palm data...")
    
    # Create a synthetic palm image for testing
    test_image = np.ones((400, 300, 3), dtype=np.uint8) * 200
    
    # Draw synthetic palm lines
    # Life line (curved)
    cv2.ellipse(test_image, (100, 200), (80, 120), 45, 0, 180, (180, 180, 180), 3)
    # Head line (horizontal)
    cv2.line(test_image, (50, 150), (250, 170), (180, 180, 180), 3)
    # Heart line (horizontal, higher)
    cv2.line(test_image, (60, 100), (240, 110), (180, 180, 180), 3)
    # Fate line (vertical)
    cv2.line(test_image, (150, 50), (160, 300), (180, 180, 180), 2)
    
    # Save test image
    cv2.imwrite("test_palm.png", test_image)
    
    # Analyze test image
    palmistry_ai = PalmistryAI()
    annotated_image, report = palmistry_ai.analyze_palm("test_palm.png")
    
    if annotated_image is not None:
        print("✅ Test completed successfully!")
        print("\n" + "="*50)
        print("TEST REPORT:")
        print("="*50)
        print(report)
    else:
        print("❌ Test failed")
    
    # Clean up
    try:
        os.remove("test_palm.png")
    except:
        pass


# Performance optimization utilities
def optimize_image_processing(image):
    """
    Optimize image processing for better performance on large images.
    
    Args:
        image (np.ndarray): Input image
        
    Returns:
        np.ndarray: Optimized image
    """
    # Resize if too large
    h, w = image.shape[:2]
    max_size = 800
    
    if max(h, w) > max_size:
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


# --- Streamlit auto-run block ---
import streamlit as st
create_streamlit_app()