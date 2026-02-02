from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Load environment variables from .env for groq api key
load_dotenv()

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Define the output structure
class OutputInfo(BaseModel):
    full_name: str | None = None
    phone_number: str | None = None
    email: str | None = None

# Load OCR text from scanned images
def convert_scanned_to_text(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

# Initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Create output parser
parser = JsonOutputParser(pydantic_object=OutputInfo)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting contact information from resumes.
Extract the full_name, phone_number, and email from the provided resume text.

{format_instructions}

Rules:
- Extract accurate information only
- If a field is not found, set it to null
- Phone number should include country code if present
- Email should be the primary/professional email if multiple are found
- Email should be in the format username@domain.com
- Return ONLY valid JSON, no explanations"""),
    ("user", "Extract contact information from this resume:\n\n{resume_text}")
])

# Create the chain
chain = prompt | llm | parser

def extract_fields(ocr_text):
    try:
        # Invoke the chain
        result = chain.invoke({
            "resume_text": ocr_text,
            "format_instructions": parser.get_format_instructions()
        })
        return OutputInfo(**result)
    except Exception as e:
        print(f"Error extracting information: {e}")
        return OutputInfo()

def main():
    # OCR the resume image
    print("Performing OCR on resume...")
    ocr_text = convert_scanned_to_text("resume_scan-1.png")
    print(f"OCR Text extracted ({len(ocr_text)} characters)\n")
    
    # Extract structured data
    print("Extracting contact information...")
    result = extract_fields(ocr_text)
    
    # Display results
    print("\n=== Extracted Contact Information ===")
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()