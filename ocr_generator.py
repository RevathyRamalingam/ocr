import os
import pytesseract
from docx import Document
from pdf2image import convert_from_path
from PIL import Image

def _perform_ocr_on_pdf(file_path):
    """Helper function to perform OCR on a PDF file."""
    images = convert_from_path(file_path)
    ocr_text = ""
    for i, image in enumerate(images):
        print(f"Processing page {i+1}...")
        ocr_text += pytesseract.image_to_string(image) + "\n"
    return ocr_text

def extract_text_from_file(file_path, force_ocr=False):
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
        print("Performing OCR on PDF pages...")
        return _perform_ocr_on_pdf(file_path)

    elif ext == '.doc':
        # Legacy DOC support is tricky without external tools like 'antiword' or 'libreoffice'
        raise ValueError("legacy .doc format is not directly supported. Please convert it to .docx or .pdf")
    
    else:
        # Try as image for other extensions? Or just fail.
        try:
            return pytesseract.image_to_string(Image.open(file_path))
        except:
             raise ValueError(f"Unsupported file format: {ext}")