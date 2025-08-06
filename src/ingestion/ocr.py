import yaml
from typing import Dict, List, Any, Optional
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF
import io


class OCRProcessor:
    def __init__(self, config_path: str = "ingestion_config.yaml"):
        """Initialize OCR processor with configuration."""
        self.config = self._load_config(config_path)
        self.ocr_config = self.config.get("ingestion", {}).get("ocr", {})
        
        # Configure tesseract
        if self.ocr_config.get("engine") == "tesseract":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
        
        self.language = self.ocr_config.get("language", "eng")
        self.confidence_threshold = self.ocr_config.get("confidence_threshold", 0.7)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load ingestion configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Extract text with confidence scores
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                lang=self.language, 
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            text_blocks = []
            full_text = ""
            
            for i in range(len(ocr_data['text'])):
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if text and confidence > (self.confidence_threshold * 100):
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence / 100,
                        'bbox': (
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        )
                    })
                    full_text += text + " "
            
            return {
                'text': full_text.strip(),
                'text_blocks': text_blocks,
                'metadata': {
                    'image_path': image_path,
                    'language': self.language,
                    'confidence_threshold': self.confidence_threshold,
                    'num_blocks': len(text_blocks),
                    'avg_confidence': np.mean([block['confidence'] for block in text_blocks]) if text_blocks else 0
                }
            }
            
        except Exception as e:
            raise Exception(f"Error extracting text from image {image_path}: {e}")
    
    def extract_text_from_pdf_images(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from images embedded in PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing extracted text from all images
        """
        try:
            pdf_document = fitz.open(pdf_path)
            all_text = ""
            image_texts = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Get images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Process image
                    processed_image = self._preprocess_image(image)
                    
                    # Extract text
                    ocr_data = pytesseract.image_to_data(
                        processed_image, 
                        lang=self.language, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    page_text = ""
                    for i in range(len(ocr_data['text'])):
                        confidence = int(ocr_data['conf'][i])
                        text = ocr_data['text'][i].strip()
                        
                        if text and confidence > (self.confidence_threshold * 100):
                            page_text += text + " "
                    
                    if page_text.strip():
                        image_texts.append({
                            'page': page_num + 1,
                            'image_index': img_index,
                            'text': page_text.strip(),
                            'confidence': np.mean([int(ocr_data['conf'][i])/100 for i in range(len(ocr_data['text'])) if ocr_data['text'][i].strip()])
                        })
                        all_text += page_text + "\n"
            
            pdf_document.close()
            
            return {
                'text': all_text.strip(),
                'image_texts': image_texts,
                'metadata': {
                    'pdf_path': pdf_path,
                    'num_pages': len(pdf_document),
                    'num_images': len(image_texts),
                    'language': self.language
                }
            }
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF images {pdf_path}: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        # Convert to numpy array with proper data type
        img_array = np.array(image, dtype=np.uint8)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL Image
        return Image.fromarray(binary)
    
    def get_layout_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze document layout and structure.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing layout analysis results
        """
        try:
            image = Image.open(image_path)
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect lines and tables
            lines = self._detect_lines(gray)
            tables = self._detect_tables(gray)
            
            return {
                'lines': lines,
                'tables': tables,
                'metadata': {
                    'image_path': image_path,
                    'num_lines': len(lines),
                    'num_tables': len(tables)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error analyzing layout for {image_path}: {e}")
    
    def _detect_lines(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect horizontal and vertical lines in the image."""
        lines = []
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in h_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100:  # Minimum line length
                lines.append({'type': 'horizontal', 'bbox': (x, y, w, h)})
        
        for contour in v_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 100:  # Minimum line height
                lines.append({'type': 'vertical', 'bbox': (x, y, w, h)})
        
        return lines
    
    def _detect_tables(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect table structures in the image."""
        tables = []
        
        # Use line detection to find table boundaries
        lines = self._detect_lines(gray_image)
        
        # Group lines into potential table regions
        horizontal_lines = [line for line in lines if line['type'] == 'horizontal']
        vertical_lines = [line for line in lines if line['type'] == 'vertical']
        
        # Simple table detection based on line intersections
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Find bounding box of all lines
            all_bboxes = [line['bbox'] for line in lines]
            min_x = min([bbox[0] for bbox in all_bboxes])
            min_y = min([bbox[1] for bbox in all_bboxes])
            max_x = max([bbox[0] + bbox[2] for bbox in all_bboxes])
            max_y = max([bbox[1] + bbox[3] for bbox in all_bboxes])
            
            tables.append({
                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                'num_rows': len(horizontal_lines) - 1,
                'num_cols': len(vertical_lines) - 1
            })
        
        return tables 