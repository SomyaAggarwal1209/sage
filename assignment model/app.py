import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load the API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please configure it as an environment variable.")

# Configure the generative AI API with the loaded key
genai.configure(api_key=GOOGLE_API_KEY)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set up the model with Gemini (Google Generative AI)
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
parser = StrOutputParser()

from fastapi.middleware.cors import CORSMiddleware

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can restrict it to specific domains later.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}

# Update the system prompt to generate questions
system_template = """
You are an intelligent assistant. Based on the subject {subject} and topic {topic}, generate an assignment consisting of:
- 5 easy questions,
- 3 medium-difficulty questions, and
- 2 hard questions.
Each question should be unique and clear.
"""

# Use ChatPromptTemplate to create the questions dynamically
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "Generate assignment questions for the student.")]
)

def generate_assignment(subject, topic):
    # Build the query dynamically
    query = {
        "subject": subject,
        "topic": topic
    }
    
    result = prompt_template | model | parser
    response = result.invoke(query)
    
    # Filter out the text and extract only the questions
    lines = response.splitlines()
    questions = [line.strip() for line in lines if line.strip() and (line.strip()[0].isdigit())]  
    formatted_questions = [f"{i+1}. {question.split('. ', 1)[-1]}" for i, question in enumerate(questions[:10])]
    
    return formatted_questions

# Define the request body schema
class AssignmentRequest(BaseModel):
    subject: str
    topic: str

# Define the response schema
class AssignmentResponse(BaseModel):
    subject: str
    topic: str
    questions: List[str]

# Define API endpoint for generating assignments
@app.post("/generate-assignment", response_model=AssignmentResponse)
async def generate_assignment_route(request: AssignmentRequest):
    subject = request.subject
    topic = request.topic
    
    # Validate input
    if not subject or not topic:
        raise HTTPException(status_code=400, detail="Both 'subject' and 'topic' fields are required.")
    
    # Generate the assignment questions
    questions = generate_assignment(subject, topic)
    
    # Return the questions in a JSON response
    return AssignmentResponse(subject=subject, topic=topic, questions=questions)
