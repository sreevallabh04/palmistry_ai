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

    def generate_ai_palmistry_report(self, palm_data: dict) -> str:
        """
        Generate an AI-powered palmistry report using Gemini API with fallback options.
        
        Args:
            palm_data (dict): Processed palm data
            
        Returns:
            str: Generated palmistry report
        """
        # Prepare the palm data for AI analysis
        prepared_data = self.prepare_palm_data_for_ai(palm_data)
        
        # Try each API key in rotation
        for attempt in range(len(self.groq_api_keys)):
            api_key = self.groq_api_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.groq_api_keys)
            
            try:
                # Prepare the API request
                url = f"{GEMINI_API_URL}?key={api_key}"
                
                # Prepare the prompt
                prompt = f"""As an expert palm reader, analyze this palm data and provide a detailed palmistry reading:
                {json.dumps(prepared_data, indent=2)}
                
                Please provide a comprehensive analysis including:
                1. Overall personality traits
                2. Life path and destiny
                3. Career prospects
                4. Relationships and emotional life
                5. Health indicators
                6. Future opportunities and challenges
                
                Format the response in a clear, structured way with sections and bullet points where appropriate."""
                
                # Prepare the request payload
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2000
                    }
                }
                
                # Make the API request
                response = requests.post(
                    url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        content = result["candidates"][0]["content"]
                        if "parts" in content and len(content["parts"]) > 0:
                            return content["parts"][0]["text"]
                
                self.logger.warning(f"API request failed with status code {response.status_code}")
                
            except Exception as e:
                self.logger.error(f"Error generating AI report: {str(e)}")
                continue
        
        # If all API attempts failed, generate a local fallback report
        self.logger.info("Generating local fallback palmistry report")
        return self.generate_local_fallback_report(prepared_data)

    def generate_local_fallback_report(self, palm_data: dict) -> str:
        """
        Generate a basic palmistry report using local analysis when AI is unavailable.
        
        Args:
            palm_data (dict): Processed palm data
            
        Returns:
            str: Generated palmistry report
        """
        report = []
        report.append("# Palmistry Analysis Report")
        report.append("\n## Basic Analysis")
        
        # Analyze heart line
        if "heart_line" in palm_data:
            heart_line = palm_data["heart_line"]
            if heart_line["length"] > 0.7:
                report.append("- Strong emotional nature with deep capacity for love")
            else:
                report.append("- Practical approach to relationships and emotions")
        
        # Analyze head line
        if "head_line" in palm_data:
            head_line = palm_data["head_line"]
            if head_line["length"] > 0.7:
                report.append("- Strong intellectual capabilities and analytical mind")
            else:
                report.append("- Intuitive and creative thinking style")
        
        # Analyze life line
        if "life_line" in palm_data:
            life_line = palm_data["life_line"]
            if life_line["length"] > 0.7:
                report.append("- Strong vitality and physical energy")
            else:
                report.append("- Focus on quality over quantity in life experiences")
        
        report.append("\n## Recommendations")
        report.append("1. Consider consulting with a professional palm reader for a more detailed analysis")
        report.append("2. Keep in mind that palmistry is one of many tools for self-reflection")
        report.append("3. Use this reading as a starting point for personal growth")
        
        return "\n".join(report)

    def process_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Process the input image and generate palmistry analysis.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[Optional[np.ndarray], str]: Processed image and analysis report
        """
        try:
            # Ensure image is in BGR format
            image = self.ensure_bgr(image)
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Detect palm region
            palm_region = self.detect_palm_region(processed_image)
            if palm_region is None:
                return None, "No palm detected in the image. Please ensure the palm is clearly visible."
            
            # Extract palm lines
            lines = self.extract_palm_lines(processed_image, palm_region)
            if not lines:
                return None, "No palm lines detected. Please ensure the palm is clearly visible."
            
            # Extract landmarks
            landmarks = self.extract_hand_landmarks(processed_image, palm_region)
            
            # Classify palm lines
            classified_lines = self.classify_palm_lines(lines, landmarks)
            
            # Prepare palm data for AI analysis
            palm_data = {
                "timestamp": datetime.now().isoformat(),
                "lines": lines,
                "landmarks": landmarks,
                "classified_lines": classified_lines
            }
            
            # Generate AI report
            try:
                report = self.generate_ai_palmistry_report(palm_data)
            except Exception as e:
                self.logger.error(f"Error generating AI report: {str(e)}")
                report = self.generate_local_fallback_report(palm_data)
            
            # Draw analysis on image
            result_image = self.draw_analysis(processed_image, lines, landmarks, classified_lines)
            
            return result_image, report
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None, f"Error processing image: {str(e)}"

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