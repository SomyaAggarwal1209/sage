from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging
import time
from selenium.webdriver.chrome.options import Options
import os
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please configure it as an environment variable.")

# Configure the generative AI API with the loaded key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize LangChain model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
parser = StrOutputParser()

# Define system message template for extracting search terms
system_template = "Extract the most relevant search terms for finding a flowchart or diagram or image as specified by user from the following text: Note: only give the search terms no extra text"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Create the LangChain pipeline
chain = prompt_template | model | parser

# Function to extract search terms using LangChain
def extract_search_terms(query: str):
    result = chain.invoke({"text": query})
    return result 

def search_images_with_metadata(search_terms):

    # Initialize Chrome driver
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    service = Service("./chromedriver.exe")  # Update with the correct path to your chromedriver
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Open Google Images
        driver.get("https://images.google.com")

        # Find the search box and enter search terms
        search_box = driver.find_element(By.CLASS_NAME, "gLFyf")
        search_box.send_keys(search_terms + Keys.ENTER)

        time.sleep(3)

        # Find image elements
        images = driver.find_elements(By.XPATH, "//div[@class='s6JM6d']//img[@class='YQ4gaf']")
        real_imgs = []
        
        # Extract metadata for the top 3 images
        for index, image in enumerate(images[:3]):
            try:
                driver.execute_script("arguments[0].scrollIntoView();", image)
                time.sleep(2)
                image.click()

                # Wait for the real image to load
                real_image = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//img[contains(@class, 'iPVvYb') and contains(@class, 'sFlh5c') and contains(@class, 'FyHeAf')]"))
                )
                if real_image:
                    img_url = real_image[0].get_attribute("src")
                    alt_text = real_image[0].get_attribute("alt")
                    metadata = {
                        "url": img_url,
                        "alt_text": alt_text,
                    }
                    print(metadata)
                    real_imgs.append(metadata)
                time.sleep(2)

                # Close image preview
                close_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "ioQ39e.wv9iH.MjJqGe.cd29Sd"))
                )
                close_button.click()
            except Exception:
                continue
        return real_imgs
    
    finally:
        driver.quit()

# Pydantic model for user query input
class UserQuery(BaseModel):
    user_query: str

# API endpoint to search for images based on user's query
@app.post("/search_image/")
async def search_flowchart(user_query: UserQuery):
    search_terms = extract_search_terms(user_query.user_query)
    
    # Retrieve the top 3 image metadata
    image_metadata = search_images_with_metadata(search_terms)
    
    return JSONResponse(content={"message": "Image search completed.", "search_terms": search_terms, "image_metadata": image_metadata})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
