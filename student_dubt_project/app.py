import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import google.generativeai as genai
import os
import asyncio

# Set your WolframAlpha APP ID and configure it in the environment
WOLFRAM_ALPHA_APPID = "5RT7JE-AAL5L34LKR"  # Replace with your actual APP ID
os.environ["WOLFRAM_ALPHA_APPID"] = WOLFRAM_ALPHA_APPID

# Configure the Gemini API key
genai.configure(api_key="AIzaSyA3MzKibpGjCn3VCUvE3oo4-ZRtB9H9I4M")
from fastapi.middleware.cors import CORSMiddleware

# Add this code block right after you initialize the FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can restrict it to specific domains later.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Initialize WolframAlpha API Wrapper
wolfram_alpha_wrapper = WolframAlphaAPIWrapper()

# Pydantic model for input data validation
class StudentQueryRequest(BaseModel):
    query: str  # Student's question

# Function to query WolframAlpha for math-related questions
async def fetch_info_from_wolfram_alpha(query: str):
    try:
        # Run the synchronous WolframAlpha query in a separate thread to avoid blocking
        response = await asyncio.to_thread(wolfram_alpha_wrapper.run, query)
        if response:
            return response
        return "No result found."
    except Exception as e:
        return f"Error fetching from WolframAlpha: {str(e)}"

# Function to query Gemini LLM for general questions
def fetch_info_from_gemini(query: str):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(query)
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Error fetching from Gemini: {str(e)}"

# Function to process the student query
async def process_student_query(query: str):
    # Check if the query is math-related
    if "math" in query.lower() or any(op in query for op in ['+', '-', '*', '/', '=', 'square', 'cube']):
        # Fetch response from WolframAlpha
        return await fetch_info_from_wolfram_alpha(query)
    else:
        # Fetch response from Gemini LLM
        return fetch_info_from_gemini(query)

# FastAPI route for handling student queries
@app.post("/api/query")
async def handle_student_query(request: StudentQueryRequest):
    try:
        # Process the student query
        response = await process_student_query(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Main function to run the FastAPI app
def main():
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
