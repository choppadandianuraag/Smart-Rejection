"""
Configuration settings for Smart Rejection system.
Loads environment variables and provides typed settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Get the project root directory (parent of config folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from .env file using absolute path
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    supabase_service_key: Optional[str] = Field(None, env="SUPABASE_SERVICE_KEY")
    
    # Hugging Face Configuration
    hf_token: Optional[str] = Field(None, env="HF_TOKEN")

    # Groq Configuration (for LLM-based ATS ranking)
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    
    # Model Configuration
    model_name: str = Field(
        default="NuMind/NuMarkdown-8B-Thinking",
        env="MODEL_NAME"
    )
    device: str = Field(default="cuda", env="DEVICE")
    
    # Application Settings
    debug: bool = Field(default=True, env="DEBUG")
    resume_upload_dir: Path = Field(
        default=Path("./uploads/resumes"),
        env="RESUME_UPLOAD_DIR"
    )
    
    # Processing Settings
    max_file_size_mb: int = 10
    supported_formats: list = [".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"]
    
    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars not defined in the model


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


# Create settings instance
settings = get_settings()
