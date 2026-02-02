import os
import pytesseract
from docx import Document
from pdf2image import convert_from_path
from pypdf import PdfReader
from PIL import Image

def extract_text_from_file(file_path):
    """
    Extract text from a file based on its extension.
    Supports: .png, .jpg, .jpeg, .pdf, .docx
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg']:
        return pytesseract.image_to_string(Image.open(file_path))
    
    elif ext == '.docx':
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    elif ext == '.pdf':
        try:
            # Try extracting text directly
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # If text is too short, assume it's a scanned PDF and use OCR
            if len(text.strip()) < 50:
                print("PDF seems to be scanned. Performing OCR on PDF pages...")
                images = convert_from_path(file_path)
                ocr_text = ""
                for i, image in enumerate(images):
                    print(f"Processing page {i+1}...")
                    ocr_text += pytesseract.image_to_string(image) + "\n"
                return ocr_text
            
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")
            
    elif ext == '.doc':
        # Legacy DOC support is tricky without external tools like 'antiword' or 'libreoffice'
        raise ValueError("legacy .doc format is not directly supported. Please convert it to .docx or .pdf")
    
    else:
        # Try as image for other extensions? Or just fail.
        try:
            return pytesseract.image_to_string(Image.open(file_path))
        except:
             raise ValueError(f"Unsupported file format: {ext}")