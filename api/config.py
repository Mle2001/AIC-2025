import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    # OpenAI Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # File Upload Configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "524288000"))  # 500MB
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")
    
    # VideoRAG Configuration
    videorag_enabled: bool = os.getenv("VIDEORAG_ENABLED", "true").lower() == "true"
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

def get_openai_api_key() -> str:
    """Get OpenAI API key from settings or environment"""
    if settings.openai_api_key:
        return settings.openai_api_key
    
    # Fallback to environment variable
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or in .env file")
    
    return key

def validate_settings():
    """Validate required settings"""
    try:
        get_openai_api_key()
        return True
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return False
