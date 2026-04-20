import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DEBUG: bool = os.getenv("DEBUG", False)
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "BackendOCR")

settings = Settings()
