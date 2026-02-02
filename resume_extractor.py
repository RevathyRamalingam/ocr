from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
from pydantic_ai.models.groq import GroqModel
from pydantic_ai import Agent
from pydantic import BaseModel
import asyncio
import json

# Load environment variables from .env for grok api key
load_dotenv()

# ====== ADD THIS LINE ======
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# ===========================

# Define the output structure
class OutputInfo(BaseModel):
    full_name: str | None = None
    phone_number: str | None = None
    email: str | None = None

# Load OCR text from scanned images
def convert_scanned_to_text(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

groq_model = GroqModel(
    model_name="llama-3.3-70b-versatile"
)

# Create agent WITHOUT result_type
agent = Agent(
    model=groq_model,
    system_prompt="""You are an expert at extracting contact information from resumes.
Extract the full_name, phone_number, and email from the provided resume text.

Return the information in the following JSON format:
{
    "full_name": "extracted name or null",
    "phone_number": "extracted phone or null",
    "email": "extracted email or null"
}

Rules:
- Extract accurate information only
- If a field is not found, set it to null
- Phone number should include country code if present
- Email should be the primary/professional email if multiple are found
- Email should be in the format username@domain.com
- Return ONLY valid JSON, no explanations
"""
)

async def extract_fields(ocr_text):
    result = await agent.run(f"Extract contact information from this resume:\n\n{ocr_text}")
    print("result is \n",result)
    # Parse the JSON response
    try:
        # The result might be in result.data or result.output depending on version
        if hasattr(result, 'output'):
            json_str = result.output
        else:
            json_str = str(result)
        
        # Clean the response if it has markdown code blocks
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()
        
        extracted_json = json.loads(json_str)
        return OutputInfo(**extracted_json)
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {result}")
        # Return empty ContactInfo if parsing fails
        return OutputInfo()

def main():
    # OCR the resume image
    print("Performing OCR on resume...")
    ocr_text = convert_scanned_to_text("resume_scan-1.png")
    print(f"OCR Text extracted ({len(ocr_text)} characters)\n")
    
    # Extract structured data
    print("Extracting contact information...")
    result = asyncio.run(extract_fields(ocr_text))
    
    # Display results
    print("\n=== Extracted Contact Information ===")
    print(json.dumps(result.model_dump(),indent=2))

if __name__ == "__main__":
    main()