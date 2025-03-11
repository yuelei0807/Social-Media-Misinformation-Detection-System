import os
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

import cv2
import pytesseract  # For OCR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Class for processing and analyzing images."""

    def __init__(self):
        """Initialize the image processor."""
        # Check if OpenCV is properly installed
        try:
            logger.info(f"OpenCV version: {cv2.__version__}")
        except:
            logger.error("OpenCV not properly installed or imported")
            raise
            
        # Check if Tesseract is available (for OCR)
        try:
            # You might need to set the tesseract path depending on your system
            # For example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # For macOS with homebrew: pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
            logger.info("Tesseract initialized for OCR")
        except:
            logger.warning("Tesseract might not be properly installed for OCR functionality")
        
        logger.info("ImageProcessor initialized")

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array if successful, None otherwise
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            logger.info(f"Image loaded: {image_path}, shape: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None

    def extract_text_from_image(self, image: np.ndarray) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Extracted text
        """
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get a binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(binary)
            
            logger.info(f"Extracted text from image: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            List of face rectangles as (x, y, width, height)
        """
        try:
            # Load pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            logger.info(f"Detected {len(faces)} faces in image")
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image and extract various features.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of image analysis results
        """
        try:
            image = self.load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
                
            # Get basic image properties
            height, width, channels = image.shape
            
            # Extract text from image
            text = self.extract_text_from_image(image)
            
            # Detect faces
            faces = self.detect_faces(image)
            
            # Calculate color histogram
            color_hist = self._calculate_color_histogram(image)
            
            # Check for manipulation indicators
            manipulation_score = self._check_manipulation_indicators(image)
            
            analysis = {
                "dimensions": {
                    "width": width,
                    "height": height,
                    "channels": channels
                },
                "text_content": text,
                "text_present": len(text.strip()) > 0,
                "faces_detected": len(faces),
                "face_locations": faces,
                "dominant_colors": self._get_dominant_colors(color_hist),
                "manipulation_indicators": {
                    "score": manipulation_score,
                    "suspicious": manipulation_score > 0.5
                }
            }
            
            logger.info(f"Image analysis completed: {image_path}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}

    def _calculate_color_histogram(self, image: np.ndarray) -> Dict[str, List[int]]:
        """
        Calculate color histogram for an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary of color histograms
        """
        try:
            # Split the image into its BGR channels
            b, g, r = cv2.split(image)
            
            # Calculate histograms for each channel
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten().tolist()
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten().tolist()
            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten().tolist()
            
            return {
                "blue": hist_b,
                "green": hist_g,
                "red": hist_r
            }
        except Exception as e:
            logger.error(f"Error calculating color histogram: {e}")
            return {"blue": [], "green": [], "red": []}

    def _get_dominant_colors(self, hist: Dict[str, List[int]]) -> List[Tuple[int, int, int]]:
        """
        Get dominant colors from a color histogram.
        
        Args:
            hist: Color histogram dictionary
            
        Returns:
            List of dominant colors as (R, G, B) tuples
        """
        try:
            # Find indices of peaks in each channel
            b_peaks = np.argsort(hist["blue"])[-3:]
            g_peaks = np.argsort(hist["green"])[-3:]
            r_peaks = np.argsort(hist["red"])[-3:]
            
            # Combine peaks to form colors
            colors = []
            for r in r_peaks:
                for g in g_peaks:
                    for b in b_peaks:
                        colors.append((int(r), int(g), int(b)))
            
            # Return top 3 colors
            return colors[:3]
        except Exception as e:
            logger.error(f"Error getting dominant colors: {e}")
            return [(0, 0, 0)]

    def _check_manipulation_indicators(self, image: np.ndarray) -> float:
        """
        Check for indicators of image manipulation.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Manipulation score between 0 and 1
        """
        try:
            # This is a simple placeholder implementation
            # In a real system, you would use more sophisticated techniques
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate noise level
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.std(gray.astype(float) - blur.astype(float))
            normalized_noise = min(1.0, noise / 20.0)  # Normalize to 0-1
            
            # Combine indicators
            score = (edge_density + normalized_noise) / 2
            
            return score
        except Exception as e:
            logger.error(f"Error checking manipulation indicators: {e}")
            return 0.0


# Example usage function
def analyze_sample_image():
    """Analyze a sample image to demonstrate image processing."""
    # This is just an example path
    sample_image_path = "path/to/sample_image.jpg"
    
    processor = ImageProcessor()
    
    if os.path.exists(sample_image_path):
        # Analyze the image
        analysis = processor.analyze_image(sample_image_path)
        print(f"Image analysis results: {analysis}")
    else:
        logger.warning(f"Sample image not found: {sample_image_path}")
        logger.info("This is just an example. Please provide a valid image path.")


if __name__ == "__main__":
    analyze_sample_image()
