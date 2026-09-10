"""
Central configuration. Loads settings from environment variables or .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gmail OAuth2 ---
    # Download credentials.json from Google Cloud Console -> APIs & Services -> Credentials
    gmail_client_secret_file: str = "credentials.json"
    
    # Auto-generated after first browser consent screen — do NOT edit manually
    gmail_token_file: str = "token.json"
    
    # Gmail API access scope (readonly)
    gmail_scopes: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]

    # --- MongoDB Connection ---
    # Change to your local MongoDB URI or MongoDB Atlas cloud connection string
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "mail_agent"

    # --- Local Storage Paths ---
    storage_root: str = "./storage"         # Root directory for all storage
    inbox_folder: str = "./storage/inbox"   # Raw incoming attachment staging directory
    reason_folder: str = "./storage/reason" # Consolidated single reasoning report directory

    # --- Gmail Polling Start Date ---
    # Floor date for first run (YYYY-MM-DD); ignored once cursor exists in Mongo
    gmail_start_date: str = "2026-09-09"

    # --- RAG Extraction Limits ---
    extracted_text_max_chars: int = 3000       # Legacy cap for short documents
    resume_extracted_text_max_chars: int = 15000 # Full context extraction cap (15,000 chars)

    # --- Canonical Resume Stacks ---
    resume_known_stacks: list[str] = [
        "ai", "devops", "full-stack", "java", "data-engineering",
        "mobile", "frontend", "backend", "cloud", "security",
        "consultant", "scrum-master", "project-manager", "hr", "other-tech",
    ]
    resume_min_stack_confidence: float = 0.4

    # --- On-Prem Qwen LLM Endpoint (OpenAI-Compatible) ---
    # Override via QWEN_BASE_URL in .env for production IP address
    qwen_base_url: str = "http://localhost:19298/v1"
    qwen_model: str = "Qwen/Qwen3.5-9B"
    qwen_api_key: str = "EMPTY"              # Use "EMPTY" for local vLLM/Ollama servers
    qwen_timeout_seconds: int = 300          # Timeout set to 300s (5 min) for large prompts
    qwen_resume_timeout_seconds: int = 300   # Legacy setting

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def storage_root_path(self) -> Path:
        p = Path(self.storage_root)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def inbox_path(self) -> Path:
        p = Path(self.inbox_folder)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reason_path(self) -> Path:
        p = Path(self.reason_folder)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
