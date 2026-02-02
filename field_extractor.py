import asyncio
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import ocr_generator
import dynamic_model_generator
import logging
import llm_instantiator

logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv(override=True)

async def main():
    print("=== Dynamic OCR Field Extractor ===")
    
    # 1. Get Model Selection
    while True:
        model_input = input("Select Model (openai/groq): ").strip().lower()
        if model_input in ["openai", "groq"]:
            break
        print("Invalid selection. Please choose 'openai' or 'groq'.")

    # 2. Get Fields to Extract
    fields_input = input("Enter fields to extract (comma-separated, e.g. 'field1,field2,field3'): ").strip()
    field_names = [f.strip() for f in fields_input.split(',') if f.strip()]
    print("fieldnames inputed are ",field_names)
    
    if not field_names:
        print("No fields specified. Exiting.")
        return

    # 3. Get Image/File Path
    default_file = "resume_scan-1.png"
    file_path = input(f"Enter file path (default: {default_file}): ").strip()
    if not file_path:
        file_path = default_file

    try:
        # Perform Extraction
        print(f"\nExtracting text from {file_path}...")
        extracted_text = ocr_generator.extract_text_from_file(file_path, force_ocr=True)
        print(f"Extraction complete. Extracted {len(extracted_text)} characters.")
        
        # Initialize Logic
        print(f"Initializing {model_input} model...")
        llm = llm_instantiator.get_llm(model_input)
        
        # Create Dynamic Model
        OutputModel = dynamic_model_generator.create_dynamic_output_model(field_names)
        
        # Build Chain
        if model_input == "openai":
            # OpenAI supports structured output natively
            structured_llm = llm.with_structured_output(OutputModel)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting information from documents.
                Extract the following fields from the provided text: {fields_list}.
                If a field is not found, set it to null."""),
                ("user", "Extract info from this text:\n\n{text}")
            ])
            
            chain = prompt | structured_llm
            print("Extracting data...")
            result = await chain.ainvoke({
                "fields_list": ", ".join(field_names),
                "text": extracted_text
            })
            
            # Result is already a pydantic object
            output_data = result.model_dump()

        else: # groq (using JSON parser)
            parser = JsonOutputParser(pydantic_object=OutputModel)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting information from documents.
                Extract the following fields from the provided text: {fields_list}.

                Model Instructions:
                {format_instructions}

                Rules:
                - If a field is not found, set it to null.
                - Return ONLY valid JSON."""),
                ("user", "Extract info from this text:\n\n{text}")
            ])
            
            chain = prompt | llm | parser
            print("Extracting data...")
            result = await chain.ainvoke({
                "fields_list": ", ".join(field_names),
                "text": extracted_text,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Result is usually a dict, but verify if we need to wrap it
            # parser usually returns a dict
            output_data = result

        # Display Result
        print("\n=== Extracted Information ===")
        print(json.dumps(output_data, indent=2))

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
