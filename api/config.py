from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
