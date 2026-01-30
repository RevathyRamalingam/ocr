import pytesseract
from PIL import Image
from pydantic_ai.models.groq import GroqModel
from pydantic_ai import Agent
import asyncio

#load OCR text from scanned images
def ocr_resume(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

groq_model = GroqModel(
    model_name="llama-3.3-70b-versatile"
)

// langchain
// json schema input 
//json schema validation
// open ai GroqModel
//different resume formats, pdf,docs,tx




def build_prompt(ocr_text):
    return f"""
You are extracting contact information from a resume.

Extract the following fields:
- full_name
- phone_number
- email

Rules:
- Return ONLY valid JSON
- If a field is missing, set it to null
- Do NOT add explanations
- Phone number should include country code if present

Resume text:
\"\"\"
{ocr_text}
\"\"\"
"""




async def extract_resume_fields(ocr_text):
    prompt = build_prompt(ocr_text)
    agent = Agent(
        name="rag_agent",
        instructions=prompt,
        model=groq_model
    )
    response = await agent.run()
    return response


def main():
    ocr_text = ocr_resume("resume_scan-1.png")
    result = asyncio.run(extract_resume_fields(ocr_text))  # <- run async function
    print(result)


if __name__ == "__main__":
    main()

