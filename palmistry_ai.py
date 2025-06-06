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
import io
import base64
import tempfile
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from skimage.filters import frangi, sobel
from skimage import img_as_ubyte

# Try to import dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Successfully loaded environment variables from .env file")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("python-dotenv not available, environment variables from .env will not be loaded")

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

# Gemini API configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
GEMINI_MODEL = "gemini-pro"

# Model token limits (leaving room for response)
MODEL_TOKEN_LIMITS = {
    "llama3-70b-8192": 5500,
    "mistral-7b-8192": 5500,
    "llama2-70b-4096": 3500
}

# Try to load API keys from environment variables first
ENV_API_KEY = os.environ.get("GEMINI_API_KEY")

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
    # Using sample/dummy Groq-like keys that signal to the generate_report function
    # to use the local fallback
    SESSION_API_KEYS = [
        "gsk_local_fallback_key1",
        "gsk_local_fallback_key2",
        "gsk_local_fallback_key3"
    ]
    logger.warning("Using local fallback analysis. For AI-powered reports, set GEMINI_API_KEY in environment variables.")

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
        self.groq_api_url = GEMINI_API_URL
        
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

    def detect_palm_region(self, image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect the palm region using contour analysis. Fallback to whole image if detection fails.
        Args:
            image (np.ndarray): Input image
            debug (bool): If True, save debug images
        Returns:
            Tuple[np.ndarray, Tuple[int, int, int, int]]: (image with rectangle, (x, y, w, h))
        """
        try:
            image = self.ensure_bgr(image)
            processed = self.preprocess_image(image)

            # Fallback to contour-based detection
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if debug:
                cv2.imwrite("debug_palm_processed.png", processed)
                debug_img = image.copy()
                cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
                cv2.imwrite("debug_palm_contours.png", debug_img)

            if not contours:
                self.logger.warning("No contours found in the image, using whole image as palm region.")
                h, w = image.shape[:2]
                return image, (0, 0, w, h)

            # Find the largest contour (assumed to be the palm)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            result = image.copy()
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if debug:
                cv2.imwrite("debug_palm_bbox.png", result)
            return result, (x, y, w, h)
        except Exception as e:
            self.logger.error(f"Error in palm detection: {str(e)}. Using whole image as fallback.")
            h, w = image.shape[:2]
            return image, (0, 0, w, h)

    def extract_palm_lines(self, image: np.ndarray, palm_region: Tuple[int, int, int, int], debug: bool = False) -> List[Dict[str, Any]]:
        """
        Extract palm lines using ridge/valley (Frangi or Sobel) and contour-based approach.
        Always returns a set of palm lines, using fallback and synthetic lines if needed.
        """
        try:
            x, y, w, h = palm_region
            palm_roi = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
            try:
                ridge = frangi(blurred)
                ridge_img = img_as_ubyte(ridge)
            except Exception:
                ridge = sobel(blurred)
                ridge_img = img_as_ubyte(ridge)
            _, binary = cv2.threshold(ridge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_length = max(40, int(0.15 * min(w, h)))
            palm_lines = []
            for contour in contours:
                if cv2.arcLength(contour, False) < min_length:
                    continue
                x1, y1 = contour[0][0]
                x2, y2 = contour[-1][0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_type = self.classify_palm_line(x1, y1, x2, y2, angle, w, h)
                if line_type:
                    palm_lines.append({
                        'type': line_type,
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'length': float(length),
                        'angle': float(angle)
                    })
            # Fallback 1: Skeletonization if no lines
            if not palm_lines:
                from skimage.morphology import skeletonize
                skel = skeletonize((binary // 255).astype(bool))
                skel_img = (skel * 255).astype(np.uint8)
                contours, _ = cv2.findContours(skel_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    if len(contour) < min_length:
                        continue
                    x1, y1 = contour[0][0]
                    x2, y2 = contour[-1][0]
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    line_type = self.classify_palm_line(x1, y1, x2, y2, angle, w, h)
                    if line_type:
                        palm_lines.append({
                            'type': line_type,
                            'start': (int(x1), int(y1)),
                            'end': (int(x2), int(y2)),
                            'length': float(length),
                            'angle': float(angle)
                        })
            # Fallback 2: Synthetic lines if still empty
            if not palm_lines:
                palm_lines = [
                    {'type': 'life', 'start': (int(0.2*w), int(0.8*h)), 'end': (int(0.4*w), int(0.2*h)), 'length': float(h*0.7), 'angle': -45.0},
                    {'type': 'head', 'start': (int(0.2*w), int(0.5*h)), 'end': (int(0.8*w), int(0.5*h)), 'length': float(w*0.6), 'angle': 0.0},
                    {'type': 'heart', 'start': (int(0.2*w), int(0.3*h)), 'end': (int(0.8*w), int(0.3*h)), 'length': float(w*0.6), 'angle': 0.0},
                    {'type': 'fate', 'start': (int(0.5*w), int(0.8*h)), 'end': (int(0.5*w), int(0.2*h)), 'length': float(h*0.6), 'angle': 90.0},
                ]
            return palm_lines
        except Exception as e:
            self.logger.error(f"Error in extract_palm_lines (robust fallback): {str(e)}")
            # Always return synthetic lines as last resort
            h, w = image.shape[:2]
            return [
                {'type': 'life', 'start': (int(0.2*w), int(0.8*h)), 'end': (int(0.4*w), int(0.2*h)), 'length': float(h*0.7), 'angle': -45.0},
                {'type': 'head', 'start': (int(0.2*w), int(0.5*h)), 'end': (int(0.8*w), int(0.5*h)), 'length': float(w*0.6), 'angle': 0.0},
                {'type': 'heart', 'start': (int(0.2*w), int(0.3*h)), 'end': (int(0.8*w), int(0.3*h)), 'length': float(w*0.6), 'angle': 0.0},
                {'type': 'fate', 'start': (int(0.5*w), int(0.8*h)), 'end': (int(0.5*w), int(0.2*h)), 'length': float(h*0.6), 'angle': 90.0},
            ]

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

    def classify_palm_lines(self, lines: List[Dict], landmarks: List[Dict]) -> Dict:
        """
        Classify palm lines based on their properties and landmarks.
        
        Args:
            lines (List[Dict]): Detected palm lines
            landmarks (List[Dict]): Detected landmarks
            
        Returns:
            Dict: Classified palm lines
        """
        try:
            classified_lines = {}
            
            for line in lines:
                start = line["start"]
                end = line["end"]
                length = line["length"]
                angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
                
                # Classify based on position and angle
                line_type = self.classify_palm_line(
                    start[0], start[1], end[0], end[1], 
                    angle, line.get("width", 0), line.get("height", 0)
                )
                
                if line_type:
                    classified_lines[line_type] = {
                        "start": start,
                        "end": end,
                        "length": length,
                        "angle": angle,
                        "curvature": line.get("curvature", 0)
                    }
            
            return classified_lines
            
        except Exception as e:
            self.logger.error(f"Error classifying palm lines: {str(e)}")
            return {}

    def prepare_palm_data_for_ai(self, palm_data: dict) -> dict:
        """
        Prepare palm data for AI analysis.
        
        Args:
            palm_data (dict): Raw palm data containing lines, landmarks, and classified lines
            
        Returns:
            dict: Processed palm data ready for AI analysis
        """
        try:
            processed_data = {
                "timestamp": palm_data["timestamp"],
                "lines": {}
            }
            
            # Process classified lines
            for line_type, line_data in palm_data["classified_lines"].items():
                if line_data:
                    processed_data["lines"][line_type] = {
                        "length": line_data.get("length", 0),
                        "start": line_data.get("start", (0, 0)),
                        "end": line_data.get("end", (0, 0)),
                        "angle": line_data.get("angle", 0),
                        "curvature": line_data.get("curvature", 0)
                    }
            
            # Process landmarks
            if "landmarks" in palm_data:
                processed_data["landmarks"] = []
                for landmark in palm_data["landmarks"]:
                    processed_data["landmarks"].append({
                        "type": landmark.get("type", "unknown"),
                        "position": landmark.get("position", (0, 0))
                    })
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preparing palm data: {str(e)}")
            return {
                "timestamp": palm_data.get("timestamp", datetime.now().isoformat()),
                "error": str(e)
            }

    def draw_analysis(self, image: np.ndarray, lines: List[Dict], landmarks: List[Dict], 
                     classified_lines: Dict) -> np.ndarray:
        """
        Draw the palm analysis on the image.
        
        Args:
            image (np.ndarray): Input image
            lines (List[Dict]): Detected palm lines
            landmarks (List[Dict]): Detected landmarks
            classified_lines (Dict): Classified palm lines
            
        Returns:
            np.ndarray: Image with analysis drawn
        """
        try:
            # Create a copy of the image
            result_image = image.copy()
            
            # Draw lines
            for line in lines:
                start = tuple(map(int, line["start"]))
                end = tuple(map(int, line["end"]))
                color = self.line_colors.get(line.get("type", "unknown"), (255, 255, 255))
                cv2.line(result_image, start, end, color, 2)
                
                # Add line label
                mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                cv2.putText(result_image, line.get("type", "unknown").upper(), 
                           mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks
            for landmark in landmarks:
                pos = tuple(map(int, landmark["position"]))
                cv2.circle(result_image, pos, 5, (0, 255, 255), -1)
                cv2.putText(result_image, landmark.get("type", "unknown").upper(),
                           (pos[0] + 10, pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 2)
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error drawing analysis: {str(e)}")
            return image

    def assign_major_lines(self, lines, palm_region):
        x, y, w, h = palm_region
        center_x, center_y = w // 2, h // 2
        assigned = {'life': None, 'heart': None, 'head': None, 'fate': None}
        scores = []
        for line in lines:
            x1, y1 = line['start']
            x2, y2 = line['end']
            # Heuristics for each line
            life_score = -abs(x1 - int(0.2*w)) - abs(y1 - int(0.8*h)) - abs(x2 - int(0.4*w)) - abs(y2 - int(0.2*h))
            heart_score = -abs(y1 - int(0.3*h)) - abs(y2 - int(0.3*h)) - abs(x1 - int(0.2*w)) - abs(x2 - int(0.8*w))
            head_score = -abs(y1 - int(0.5*h)) - abs(y2 - int(0.5*h))
            fate_score = -abs(x1 - center_x) - abs(x2 - center_x) - abs(y1 - int(0.8*h)) - abs(y2 - int(0.2*h))
            scores.append((life_score, heart_score, head_score, fate_score, line))
        used = set()
        for idx, key in enumerate(['life', 'heart', 'head', 'fate']):
            best = max([s for s in scores if id(s[4]) not in used], key=lambda s: s[idx], default=None)
            if best:
                assigned[key] = best[4]
                used.add(id(best[4]))
        # Fallback: if any line is still None, assign a synthetic line
        for key in assigned:
            if assigned[key] is None:
                if key == 'life':
                    assigned[key] = {'type': 'life', 'start': (int(0.2*w), int(0.8*h)), 'end': (int(0.4*w), int(0.2*h)), 'length': float(h*0.7), 'angle': -45.0}
                elif key == 'head':
                    assigned[key] = {'type': 'head', 'start': (int(0.2*w), int(0.5*h)), 'end': (int(0.8*w), int(0.5*h)), 'length': float(w*0.6), 'angle': 0.0}
                elif key == 'heart':
                    assigned[key] = {'type': 'heart', 'start': (int(0.2*w), int(0.3*h)), 'end': (int(0.8*w), int(0.3*h)), 'length': float(w*0.6), 'angle': 0.0}
                elif key == 'fate':
                    assigned[key] = {'type': 'fate', 'start': (int(0.5*w), int(0.8*h)), 'end': (int(0.5*w), int(0.2*h)), 'length': float(h*0.6), 'angle': 90.0}
        return assigned

    def generate_ai_palmistry_report(self, palm_data: dict) -> str:
        """
        Generate a detailed, authentic palmistry report using chiromancy rules, gender, and age.
        """
        gender = palm_data.get("gender", "Unknown")
        age = palm_data.get("age", "Unknown")
        classified = palm_data.get("classified_lines", {})
        report = [f"# Palmistry Report\n\n**Gender:** {gender}  |  **Age:** {age}"]
        report.append("\n---\n")
        report.append("## Palm Line Analysis\n")
        # Life Line
        life = classified.get('life')
        if life['length'] > 0.7:
            report.append("- **Life Line:** Long and deep — vitality, strong health, zest for life. (Or maybe you just moisturize a lot?)")
        else:
            report.append("- **Life Line:** Short or faint — caution with health, or a life full of changes. Or maybe you just don't like commitment!")
        # Heart Line
        heart = classified.get('heart')
        if abs(heart['angle']) > 10:
            report.append("- **Heart Line:** Curved — warm, emotional, open-hearted. You probably cry at Pixar movies.")
        else:
            report.append("- **Heart Line:** Straight — rational in love, values stability. Or maybe you just ghost people efficiently.")
        # Head Line
        head = classified.get('head')
        if head['length'] > 0.7:
            report.append("- **Head Line:** Long — analytical, thoughtful, intelligent. You probably overthink texts.")
        else:
            report.append("- **Head Line:** Short — intuitive, creative, quick-thinking. Or you just make it up as you go!")
        # Fate Line
        fate = classified.get('fate')
        if fate['length'] > 0.7:
            report.append("- **Fate Line:** Strong — sense of destiny, career focus, life purpose. Or you just like making to-do lists.")
        else:
            report.append("- **Fate Line:** Weak or absent — self-made, values freedom, or changes career paths. Or maybe you just can't pick a Netflix show.")
        report.append("\n---\n")
        # Funny Age-based insights
        if age:
            try:
                age_val = int(age)
                if age_val < 10:
                    report.append("\n**You're so young, your palm lines are still in beta! Come back after you finish your homework.** 🍼")
                elif age_val < 18:
                    report.append("\n**Under 18? Your palm lines are still downloading. Try again after you survive puberty!** 🤓")
                elif age_val < 25:
                    report.append("\n**Ah, youth! Your palm lines are as fresh as your memes. Don't worry, life gets weirder.** 😎")
                elif age_val < 40:
                    report.append("\n**Prime of life! Your palm lines are bold, but your back probably already hurts.** 🏋️")
                elif age_val < 60:
                    report.append("\n**Seasoned! Your palm lines have seen things. Like dial-up internet and floppy disks.** 💾")
                else:
                    report.append("\n**Over 60? Your palm lines are vintage. You probably have stories that start with 'Back in my day...'.** 👴👵")
                    report.append("**Honestly, your palm lines are so old, they might be eligible for a senior discount.** 🦖")
            except Exception:
                report.append("\n**Your age is a mystery, just like your palm lines!** 🕵️")
        # Chiromancy-based summary
        report.append("\n## Chiromancy Synthesis\n")
        report.append("Your palm reveals a tapestry of strengths, opportunities, and questionable life choices. The interplay of your lines suggests a balance between heart, mind, and the urge to binge-watch TV at 2am.\n")
        report.append("\n---\n")
        report.append("### Recommendations\n- Embrace your strengths and nurture your growth.\n- Remember, palmistry is a tool for self-reflection, not prediction.\n- Use these insights to guide your journey with confidence. Or just to impress your friends at parties.\n")
        return '\n'.join(report)

    def process_image(self, image: np.ndarray, gender: str = None, age: int = None) -> Tuple[Optional[np.ndarray], str]:
        try:
            image = self.ensure_bgr(image)
            processed_image = self.preprocess_image(image)
            palm_region = self.detect_palm_region(processed_image)[1]
            lines = self.extract_palm_lines(processed_image, palm_region)
            landmarks = self.extract_hand_landmarks(processed_image, palm_region)
            assigned_lines = self.assign_major_lines(lines, palm_region)
            palm_data = {
                "timestamp": datetime.now().isoformat(),
                "lines": lines,
                "landmarks": landmarks,
                "classified_lines": assigned_lines,
                "gender": gender,
                "age": age
            }
            report = self.generate_ai_palmistry_report(palm_data)
            result_image = self.draw_analysis(processed_image, lines, landmarks, assigned_lines)
            return result_image, report
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None, f"Error processing image: {str(e)}"

# Palmistry AI - Streamlit Deployment Ready
# To deploy: Place this file and requirements.txt in your repo root. Add a 'photos' directory (optional) for gallery images.
# Set the entrypoint to 'palmistry_ai.py' in Streamlit Cloud.

# --- Apple-level CSS ---
st.markdown("""
    <style>
    html, body, .main, .stApp {
        background: #fff !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Arial, sans-serif !important;
        color: #222 !important;
    }
    .centered-container {
        max-width: 480px;
        margin: 48px auto 0 auto;
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 16px 0 rgba(60,60,60,0.08);
        padding: 2.5rem 2.2rem 2rem 2.2rem;
        border: 1.5px solid #e0e0e0;
    }
    .form-label {
        font-size: 1.08rem;
        color: #222;
        font-weight: 600;
        margin-bottom: 0.3em;
        margin-top: 1.1em;
        display: block;
        letter-spacing: 0.01em;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        background: #fff;
        font-size: 1.08rem;
        padding: 0.5rem 1rem;
        font-family: inherit !important;
        font-weight: 600;
        color: #222 !important;
        transition: border 0.2s, box-shadow 0.2s;
    }
    .stTextInput>div>input:focus, .stNumberInput>div>input:focus {
        border: 1.5px solid #888;
        background: #f5f5f5;
    }
    .stFileUploader>div>div>div>button {
        background: #fff;
        color: #222;
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        font-weight: 700;
        font-size: 1.08rem;
        padding: 0.7rem 2.2rem;
        box-shadow: 0 1px 4px #eaeaea;
        font-family: inherit !important;
        transition: background 0.2s, box-shadow 0.2s, border 0.2s;
    }
    .stFileUploader>div>div>div>button:hover {
        background: #f5f5f5;
        color: #222;
        border: 1.5px solid #eaeaea;
        box-shadow: 0 2px 8px #e0e0e0;
        transform: scale(1.03);
    }
    .stButton>button {
        background: #fff;
        color: #111;
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        font-weight: 700;
        font-size: 1.08rem;
        padding: 0.7rem 2.2rem;
        box-shadow: 0 1px 4px #eaeaea;
        font-family: inherit !important;
        transition: background 0.2s, box-shadow 0.2s, border 0.2s;
    }
    .stButton>button:hover {
        background: #f5f5f5;
        color: #111;
        border: 1.5px solid #eaeaea;
        box-shadow: 0 2px 8px #e0e0e0;
        transform: scale(1.03);
    }
    .report-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px 0 rgba(60,60,60,0.06);
        padding: 2rem 2.2rem;
        margin-bottom: 2rem;
        border: 1.5px solid #e0e0e0;
        font-family: inherit !important;
        font-weight: 600;
        color: #232323 !important;
        transition: box-shadow 0.2s, border 0.2s;
    }
    .report-card:hover {
        box-shadow: 0 4px 24px 0 rgba(60,60,60,0.10);
        border: 1.5px solid #eaeaea;
        transform: scale(1.01);
    }
    .stImage>img {
        background: #fff !important;
        border-radius: 12px;
        box-shadow: 0 1px 8px #eaeaea;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #111;
        font-family: inherit !important;
        letter-spacing: 0.5px;
        font-weight: 700;
        text-shadow: none;
        margin-bottom: 0.7em;
    }
    .stMarkdown p, .stMarkdown li {
        font-size: 1.08rem;
        color: #222 !important;
        font-family: inherit !important;
        font-weight: 500;
        margin-bottom: 0.7em;
    }
    @media (max-width: 700px) {
        .centered-container { padding: 1.2rem 0.5rem; }
        .report-card { padding: 1.2rem 0.5rem; }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_photo_paths():
    photo_dir = "photos"
    if os.path.exists(photo_dir) and os.path.isdir(photo_dir):
        return [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return []

def main():
    photo_paths = get_photo_paths()
    if photo_paths:
        banner_img_path = random.choice(photo_paths)
        banner_img = Image.open(banner_img_path)
        st.image(banner_img, caption="Palmistry Art", use_container_width=True)

    st.markdown(
        "<h1 style='text-align:center; font-size:2.5rem; color:#111; font-family:inherit; margin-bottom:0.2em;'>🖐️ Palmistry AI: Your Personalized Palm Reading</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:1.15rem; color:#444; margin-bottom:2.2em;'>Upload a clear palm image, select your gender and age, and receive a detailed, authentic palmistry report.</p>",
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown('<label class="form-label">Gender</label>', unsafe_allow_html=True)
        gender = st.selectbox("", ["Male", "Female", "Other"], key="gender_select")
        st.markdown('<label class="form-label">Age</label>', unsafe_allow_html=True)
        age = st.number_input("", min_value=5, max_value=120, value=25, key="age_input")
        st.markdown('<label class="form-label">Upload Palm Image</label>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="file_upload")
        st.markdown(
            "<div style='margin-top:1.2rem; text-align:center;'><span style='font-size:1.05rem; color:#888;'>✨ Your palm is unique. Let AI reveal its story! ✨</span></div>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        palmistry = PalmistryAI()
        result_image, report = palmistry.process_image(image_np, gender=gender, age=age)
        st.image(result_image, caption="Palm Analysis", use_container_width=True)
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(report, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.balloons()

    if photo_paths:
        st.markdown("### Palmistry Gallery")
        gallery_cols = st.columns(4)
        for idx, img_path in enumerate(photo_paths[:8]):
            with gallery_cols[idx % 4]:
                st.image(Image.open(img_path), use_container_width=True)

if __name__ == "__main__":
    main()