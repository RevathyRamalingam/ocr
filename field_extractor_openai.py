from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import asyncio

# Load environment variables from .env for OpenAI api key
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

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o",  # or "gpt-3.5-turbo" / "gpt-4-turbo"
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

# Create structured output LLM
structured_llm = llm.with_structured_output(OutputInfo)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting contact information from resumes.
Extract the full_name, phone_number, and email from the provided resume text.

Rules:
- Extract accurate information only
- If a field is not found, set it to null
- Phone number should include country code if present
- Email should be the primary/professional email if multiple are found
- Email should be in the format username@domain.com"""),
    ("human", "Extract contact information from this resume:\n\n{ocr_text}")
])

# Create the chain
chain = prompt | structured_llm

async def extract_fields(ocr_text):
    # This returns an actual OutputInfo object directly
    return await chain.ainvoke({"ocr_text": ocr_text})

async def main():  # Make main async
    # OCR the resume image
    print("Performing OCR on resume...")
    ocr_text = convert_scanned_to_text("resume_scan-1.png")
    print(f"OCR Text extracted ({len(ocr_text)} characters)\n")
    
    # Extract structured data
    print("Extracting contact information...")
    result = await extract_fields(ocr_text)  # Use await instead of asyncio.run()
    
    # Display results
    print("\n=== Extracted Contact Information ===")
    print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())  # Move asyncio.run() here