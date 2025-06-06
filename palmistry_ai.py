#!/usr/bin/env python3
"""
AI Palmistry Reader - Advanced Palm Analysis System
==================================================
A comprehensive palmistry analysis tool that combines computer vision with traditional
palm reading wisdom to provide detailed insights into personality and destiny.

Author: AI Engineer & Palmistry Scholar
Version: 1.0
"""
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
import random
import requests
import json
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# Try to import tiktoken, but don't fail if it's not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("tiktoken not available, using fallback token estimation")

# Optional: QR code for localhost
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Groq configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    "llama3-70b-8192",
    "mistral-7b-8192",
    "llama2-70b-4096"
]

# Model token limits (leaving room for response)
MODEL_TOKEN_LIMITS = {
    "llama3-70b-8192": 5500,
    "mistral-7b-8192": 5500,
    "llama2-70b-4096": 3500
}

# Try to load API keys from environment variables first
ENV_API_KEY = os.environ.get("GROQ_API_KEY")

# Session-specific API keys as fallback
SESSION_API_KEYS = []

# Add environment API key if available
if ENV_API_KEY:
    SESSION_API_KEYS.append(ENV_API_KEY)
    logger.info("Using API key from environment variables")

# Load keys from config.json if available
try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            if "api_keys" in config and config["api_keys"]:
                SESSION_API_KEYS.extend(config["api_keys"])
                logger.info(f"Loaded {len(config['api_keys'])} API keys from config.json")
except Exception as e:
    logger.warning(f"Failed to load API keys from config.json: {str(e)}")

# Fallback API keys (use only if no environment or config keys available)
if not SESSION_API_KEYS:
    SESSION_API_KEYS = [
        "AIzaSyB_xPYF3IAO9j6m3GARZjyMQE3GIz7I2s0",
        "AIzaSyCPK-TcfngXIZXoInz8nKkUei7hVYvBKRo",
        "AIzaSyAB113h5ALxnUiZRv6cxE_58AqObSyrJA4"
    ]
    logger.warning("Using fallback API keys. Consider setting GROQ_API_KEY in environment variables for better security.")

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string using tiktoken or fallback method.
    
    Args:
        text (str): Input text to estimate tokens for
        
    Returns:
        int: Estimated number of tokens
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to estimate tokens using tiktoken: {str(e)}")
    
    # Fallback to rough estimation
    # Average English word is ~4 chars, and tokens are roughly word-based
    # Add 20% buffer for safety
    estimated_tokens = int(len(text) / 4 * 1.2)
    logger.info(f"Using fallback token estimation: {estimated_tokens} tokens")
    return estimated_tokens

def truncate_palm_data(palm_data: dict, max_tokens: int) -> dict:
    """
    Truncate palm data to fit within token limits while preserving essential information.
    
    Args:
        palm_data (dict): Original palm data
        max_tokens (int): Maximum allowed tokens
        
    Returns:
        dict: Truncated palm data
    """
    # Start with essential data
    truncated_data = {
        "timestamp": palm_data["timestamp"]
    }
    
    # Process lines
    if "lines" in palm_data:
        truncated_data["lines"] = []
        for line in palm_data["lines"]:
            # Keep only essential line information
            truncated_line = {
                "type": line["type"],
                "start": line["start"],
                "end": line["end"],
                "length": line["length"]
            }
            truncated_data["lines"].append(truncated_line)
    
    # Process landmarks
    if "landmarks" in palm_data:
        truncated_data["landmarks"] = []
        for landmark in palm_data["landmarks"]:
            # Keep only essential landmark information
            truncated_landmark = {
                "type": landmark["type"],
                "position": landmark["position"]
            }
            truncated_data["landmarks"].append(truncated_landmark)
    
    # Check token count and further truncate if needed
    data_str = json.dumps(truncated_data)
    token_count = estimate_tokens(data_str)
    
    if token_count > max_tokens:
        # If still too large, reduce number of landmarks and lines
        if "landmarks" in truncated_data:
            truncated_data["landmarks"] = truncated_data["landmarks"][:5]
        if "lines" in truncated_data:
            truncated_data["lines"] = truncated_data["lines"][:3]
        
        # Check again after truncation
        data_str = json.dumps(truncated_data)
        token_count = estimate_tokens(data_str)
        logger.info(f"Token count after truncation: {token_count}")
    
    return truncated_data

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Any object that might contain NumPy types
        
    Returns:
        Object with all NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

class PalmistryAI:
    def __init__(self):
        """Initialize the Palmistry AI system with color mappings and palmistry knowledge."""
        # Configure logging
        self.logger = logger
        self.logger.setLevel(logging.INFO)
        
        # Color mappings for palm lines
        self.line_colors = {
            'heart': (0, 0, 255),    # Red
            'head': (0, 255, 0),     # Green
            'life': (255, 0, 0),     # Blue
            'fate': (255, 255, 0),   # Cyan
            'sun': (0, 255, 255),    # Yellow
            'mercury': (255, 0, 255) # Magenta
        }
        
        # Set hand cascade to None - haarcascade_hand.xml is not standard in OpenCV
        self.logger.info("Using contour-based palm detection for better compatibility.")
        self.hand_cascade = None
        
        # Groq API configuration with multiple keys and fallback options
        self.groq_api_keys = SESSION_API_KEYS
        self.current_key_index = 0
        self.groq_api_url = GROQ_API_URL
        
        # Available models with fallback options (ordered by preference)
        self.available_models = [
            "llama2-70b-4096",    # Primary model - most powerful
            "llama2-13b-4096",    # Fallback 1 - good balance
            "mistral-7b-instruct", # Fallback 2 - reliable
            "gemma-7b-it"         # Fallback 3 - lightweight
        ]
        self.current_model_index = 0
        
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

    def ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in BGR format for OpenCV processing.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image in BGR format
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:
            return image
        else:
            raise ValueError(f"Unsupported image format with {image.shape[2]} channels")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for better palm detection and line extraction."""
        try:
            image = self.ensure_bgr(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def detect_palm_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect the palm region using contour analysis."""
        try:
            image = self.ensure_bgr(image)
            processed = self.preprocess_image(image)
            
            # Try cascade classifier first if available
            if self.hand_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hands = self.hand_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                if len(hands) > 0:
                    x, y, w, h = hands[0]
                    result = image.copy()
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    return result, (x, y, w, h)
            
            # Fallback to contour-based detection
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.warning("No contours found in the image")
                return image, (0, 0, image.shape[1], image.shape[0])
            
            # Find the largest contour (assumed to be the palm)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw rectangle on the image
            result = image.copy()
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            return result, (x, y, w, h)
        except Exception as e:
            self.logger.error(f"Error in palm detection: {str(e)}")
            raise

    def extract_palm_lines(self, image: np.ndarray, palm_region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Extract palm lines using edge detection and Hough transform.
        
        Args:
            image (numpy.ndarray): Input image
            palm_region (tuple): Palm region coordinates (x, y, w, h)
            
        Returns:
            List[Dict[str, Any]]: List of detected palm lines with their properties
        """
        try:
            x, y, w, h = palm_region
            palm_roi = image[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Line detection using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return []
            
            palm_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Classify line based on position and angle
                line_type = self.classify_palm_line(x1, y1, x2, y2, angle, w, h)
                
                if line_type:
                    palm_lines.append({
                        'type': line_type,
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'length': float(length),
                        'angle': float(angle)
                    })
            
            return palm_lines
            
        except Exception as e:
            self.logger.error(f"Error in extract_palm_lines: {str(e)}")
            return []

    def classify_palm_line(self, x1: int, y1: int, x2: int, y2: int, angle: float, 
                          width: int, height: int) -> Optional[str]:
        """
        Classify a palm line based on its position and angle.
        
        Args:
            x1, y1 (int): Start point coordinates
            x2, y2 (int): End point coordinates
            angle (float): Line angle in degrees
            width, height (int): Palm region dimensions
            
        Returns:
            Optional[str]: Line type or None if not a significant palm line
        """
        try:
            # Normalize coordinates
            x1_norm = x1 / width
            y1_norm = y1 / height
            x2_norm = x2 / width
            y2_norm = y2 / height
            
            # Calculate line center
            center_x = (x1_norm + x2_norm) / 2
            center_y = (y1_norm + y2_norm) / 2
            
            # Classify based on position and angle
            if 0.3 <= center_x <= 0.7 and 0.2 <= center_y <= 0.4:
                return 'heart'
            elif 0.3 <= center_x <= 0.7 and 0.4 <= center_y <= 0.6:
                return 'head'
            elif 0.2 <= center_x <= 0.4 and 0.3 <= center_y <= 0.7:
                return 'life'
            elif 0.5 <= center_x <= 0.7 and 0.3 <= center_y <= 0.7:
                return 'fate'
            elif 0.4 <= center_x <= 0.6 and 0.6 <= center_y <= 0.8:
                return 'sun'
            elif 0.6 <= center_x <= 0.8 and 0.6 <= center_y <= 0.8:
                return 'mercury'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in classify_palm_line: {str(e)}")
            return None

    def extract_hand_landmarks(self, image: np.ndarray, palm_region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Extract hand landmarks using contour analysis.
        
        Args:
            image (numpy.ndarray): Input image
            palm_region (tuple): Palm region coordinates (x, y, w, h)
            
        Returns:
            List[Dict[str, Any]]: List of detected hand landmarks
        """
        try:
            x, y, w, h = palm_region
            palm_roi = image[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            landmarks = []
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 100:  # Filter small contours
                    continue
                    
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Classify landmark based on position
                    landmark_type = self.classify_landmark(cx, cy, w, h)
                    if landmark_type:
                        landmarks.append({
                            'type': landmark_type,
                            'position': (int(cx), int(cy)),
                            'area': float(area)
                        })
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error in extract_hand_landmarks: {str(e)}")
            return []

    def classify_landmark(self, x: int, y: int, width: int, height: int) -> Optional[str]:
        """
        Classify a landmark based on its position.
        
        Args:
            x, y (int): Landmark coordinates
            width, height (int): Palm region dimensions
            
        Returns:
            Optional[str]: Landmark type or None if not significant
        """
        try:
            # Normalize coordinates
            x_norm = x / width
            y_norm = y / height
            
            # Classify based on position
            if 0.2 <= x_norm <= 0.3 and 0.1 <= y_norm <= 0.3:
                return 'thumb_base'
            elif 0.3 <= x_norm <= 0.4 and 0.1 <= y_norm <= 0.3:
                return 'index_base'
            elif 0.4 <= x_norm <= 0.5 and 0.1 <= y_norm <= 0.3:
                return 'middle_base'
            elif 0.5 <= x_norm <= 0.6 and 0.1 <= y_norm <= 0.3:
                return 'ring_base'
            elif 0.6 <= x_norm <= 0.7 and 0.1 <= y_norm <= 0.3:
                return 'pinky_base'
            elif 0.4 <= x_norm <= 0.6 and 0.4 <= y_norm <= 0.6:
                return 'palm_center'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in classify_landmark: {str(e)}")
            return None

    def classify_palm_lines(self, lines, landmarks):
        """
        Classify palm lines based on position and characteristics.
        
        Args:
            lines (list): List of detected palm lines
            landmarks (dict): Hand landmark measurements
            
        Returns:
            dict: Classified palm lines with measurements
        """
        classified_lines = {}
        
        if not lines:
            return classified_lines
            
        # Sort lines by length
        sorted_lines = sorted(lines, key=lambda x: x['length'], reverse=True)
        
        # Classify the longest lines
        for i, line in enumerate(sorted_lines[:5]):
            angle = line['angle']
            length = line['length']
            
            # Basic classification based on angle and position
            if 30 <= angle <= 60:
                classified_lines['life_line'] = line
            elif 60 < angle <= 90:
                classified_lines['head_line'] = line
            elif 90 < angle <= 120:
                classified_lines['heart_line'] = line
            elif 120 < angle <= 150:
                classified_lines['fate_line'] = line
            else:
                classified_lines[f'unknown_line_{i+1}'] = line
        
        return classified_lines

    def prepare_palm_data_for_ai(self, palm_data, classified_lines, landmarks):
        """
        Prepare structured palm data for AI analysis.
        
        Args:
            palm_data (dict): Dictionary of palm line properties
            classified_lines (dict): Dictionary of classified palm lines
            landmarks (dict): Hand landmark measurements
            
        Returns:
            dict: Structured palm data for AI analysis
        """
        structured_data = {
            "palm_lines": {},
            "mounts": {},
            "fingers": {},
            "overall_characteristics": {}
        }
        
        # Process palm lines
        for line_type, line_data in classified_lines.items():
            structured_data["palm_lines"][line_type] = {
                "length": line_data['length'],
                "angle": line_data['angle'],
                "properties": palm_data.get(line_type, []),
                "start_point": line_data['start'],
                "end_point": line_data['end']
            }
        
        # Add finger measurements if available
        if landmarks:
            structured_data["fingers"] = {
                "lengths": landmarks['fingers'],
                "palm_width": landmarks['palm_width'],
                "palm_height": landmarks['palm_height']
            }
        
        # Calculate overall characteristics
        total_lines = len(classified_lines)
        structured_data["overall_characteristics"] = {
            "total_lines": total_lines,
            "complexity": "high" if total_lines >= 3 else "medium" if total_lines == 2 else "low",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return structured_data

    def generate_ai_palmistry_report(self, palm_data: dict) -> str:
        """
        Generate a palmistry report using Groq AI with key rotation and fallback.
        
        Args:
            palm_data (dict): Dictionary containing palm analysis data
            
        Returns:
            str: AI-generated palmistry report or error message
        """
        # Sanitize palm_data for JSON serialization
        sanitized_data = sanitize_for_json(palm_data)
        
        prompt = f"""You are an expert palm reader. Here is the palm data: {json.dumps(sanitized_data)}. 
        Give a detailed, unique palmistry report that includes:
        1. A mystical introduction
        2. Detailed analysis of each detected line (life, head, heart, fate)
        3. Interpretation of line properties and measurements
        4. Overall synthesis of the palm reading
        5. A mystical closing
        
        Make the reading unique, personal, and insightful. Use mystical and poetic language while maintaining professionalism."""
        
        # Estimate tokens for the prompt
        prompt_tokens = estimate_tokens(prompt)
        self.logger.info(f"Estimated prompt tokens: {prompt_tokens}")
        
        for model in GROQ_MODELS:
            # Skip models that can't handle the token count
            if prompt_tokens > MODEL_TOKEN_LIMITS[model]:
                self.logger.warning(f"Skipping model {model} due to token limit ({prompt_tokens} > {MODEL_TOKEN_LIMITS[model]})")
                continue
                
            for key_index, api_key in enumerate(SESSION_API_KEYS):
                try:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are an expert palm reader with deep knowledge of palmistry."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                    
                    self.logger.info(f"Attempting API call - Model: {model}, Key Index: {key_index}, Estimated Tokens: {prompt_tokens}")
                    
                    response = requests.post(
                        GROQ_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    self.logger.info(f"API Response - Status: {response.status_code}, Model: {model}, Key Index: {key_index}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        report = result['choices'][0]['message']['content']
                        self.logger.info(f"Successfully generated report using model: {model}")
                        return report
                    elif response.status_code == 401:
                        self.logger.warning(f"Authentication failed for key index {key_index}")
                        continue
                    elif response.status_code == 404:
                        self.logger.warning(f"Model {model} not found, trying next model")
                        break
                    elif response.status_code == 413:
                        self.logger.warning(f"Payload too large for model {model}, trying next model")
                        break
                    else:
                        self.logger.warning(f"API error: {response.status_code} - {response.text}")
                        continue
                        
                except requests.exceptions.Timeout:
                    self.logger.error(f"Timeout with model {model}, key index {key_index}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error: {str(e)}")
                    continue
        
        return "Unable to generate AI palmistry report. Please try again later or use local analysis."

    def process_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Process a palm image and generate a palmistry report.
        
        Args:
            image (numpy.ndarray): Input palm image
            
        Returns:
            tuple: (annotated_image, palmistry_report)
        """
        try:
            # Convert to BGR once at the start
            image = self.ensure_bgr(image)
            self.logger.info(f"Processing image with shape: {image.shape}")
            
            # Detect palm region
            annotated_image, palm_region = self.detect_palm_region(image)
            if palm_region is None:
                return None, "No palm region detected. Please ensure the palm is clearly visible in the image."
            
            # Extract palm lines and landmarks
            palm_lines = self.extract_palm_lines(image, palm_region)
            landmarks = self.extract_hand_landmarks(image, palm_region)
            
            if not palm_lines and not landmarks:
                return None, "No significant palm features detected. Please ensure the palm is clearly visible and well-lit."
            
            # Convert all NumPy types to native Python types
            palm_data = {
                "lines": sanitize_for_json(palm_lines),
                "landmarks": sanitize_for_json(landmarks),
                "timestamp": datetime.now().isoformat()
            }
            
            # Truncate palm data to fit within token limits
            truncated_data = truncate_palm_data(palm_data, MODEL_TOKEN_LIMITS["llama3-70b-8192"])
            
            # Generate AI report
            report = self.generate_ai_palmistry_report(truncated_data)
            
            # Draw lines on annotated image
            for line in palm_lines:
                color = self.line_colors.get(line['type'], (255, 255, 255))
                x1, y1 = line['start']
                x2, y2 = line['end']
                cv2.line(annotated_image, (x1, y1), (x2, y2), color, 2)
                
            return annotated_image, report
            
        except Exception as e:
            self.logger.error(f"Error in process_image: {str(e)}")
            return None, f"Image processing failed: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Palmistry Analysis",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 AI Palmistry Analysis")
    st.markdown("""
    Upload a clear image of your palm to receive an AI-powered palmistry reading.
    The analysis will include:
    - Palm lines (Heart, Head, Life, Fate, Sun, Mercury)
    - Finger analysis
    - Overall palm characteristics
    """)
    
    # Show QR code for localhost
    if QR_AVAILABLE and st.experimental_get_query_params() == {}:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        url = f"http://{local_ip}:8501"
        qr = qrcode.make(url)
        st.image(np.array(qr), caption="Scan to open on mobile", use_container_width=True)
    
    uploaded_file = st.file_uploader("Choose a palm image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Process image
            result, message = PalmistryAI().process_image(image_np)
            
            if result is None:
                st.error(message)
                return
                
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Palm Analysis")
                st.image(result, caption="Analyzed Palm", use_container_width=True)
            with col2:
                st.subheader("Palmistry Report")
                st.markdown(message)
                
            if message:
                report_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis": message
                }
                st.download_button(
                    label="Download Report",
                    data=json.dumps(report_data, indent=2),
                    file_name="palmistry_report.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()