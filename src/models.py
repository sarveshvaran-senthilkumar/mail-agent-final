"""
Data schemas. Kept as pydantic models so both Mongo I/O and LLM output
get validated at the boundary instead of trusting raw dicts everywhere.
"""
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class User(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    email: EmailStr
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


class LLMClassification(BaseModel):
    """Expected JSON shape returned by the Qwen classification call."""
    doc_type: str
    target_folders: list[str] = Field(default_factory=lambda: ["misc"])
    folder: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: Optional[str] = None

    @field_validator("target_folders", mode="before")
    @classmethod
    def coerce_target_folders(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        return ["misc"]

    def model_post_init(self, __context):
        if not self.target_folders and self.folder:
            self.target_folders = [self.folder]
        elif self.target_folders and not self.folder:
            self.folder = self.target_folders[0]


class Document(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: str
    message_id: str
    filename: str
    doc_type: str
    folder: str
    storage_path: str
    inbox_path: str                  # path of the raw copy in inbox/ staging folder
    llm_tag: str                     # e.g. "qwen3.5-9b:0.87"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}


class ProcessedMessage(BaseModel):
    """Idempotency record — one per Gmail message id we've already handled."""
    message_id: str
    internal_date_ms: int            # Gmail internalDate epoch milliseconds (millisecond precision)
    status: str                      # "success" | "failed"
    error: Optional[str] = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunCursor(BaseModel):
    """
    Singleton document in MongoDB. Holds the internalDate (ms) of the newest
    message processed in the last successful run. Next run queries Gmail for
    messages with internalDate strictly above this value.
    """
    key: str = "last_internal_date_ms"   # always this value — acts as document key
    value: int                            # epoch milliseconds
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StackExperience(BaseModel):
    """
    One detected technology stack from a resume, with experience measured
    specifically in that stack (not the candidate's total career years).
    """
    stack: str                           # canonical folder name e.g. "ai", "devops", "java"
    years: float                         # years of experience in THIS stack specifically
    experience_band: str                 # "0-2yrs" | "2-5yrs" | "5-10yrs" | "10+yrs"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: str = ""                   # one-sentence reason from the LLM


class ResumeAnalysis(BaseModel):
    """
    Full sub-classification result for one resume file.
    Stored in the resume_analyses MongoDB collection.
    A single resume produces one ResumeAnalysis containing all detected stacks.
    """
    message_id: str
    filename: str
    stacks: list[StackExperience]        # one entry per detected stack
    analysed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
