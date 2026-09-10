# Mail Agent — Single-Pass RAG & Prompt-Driven Document Processing Engine

A production-grade, RAG-assisted automated document processing agent that ingests email attachments from Gmail, extracts document text, and uses a single-pass LLM prompt engine (Qwen / OpenAI-compatible) to classify and file documents into structured disk directories and MongoDB collections.

---

## 🎯 1. Purpose and Functionality of the Agent

The **Mail Agent** solves the problem of manual email attachment sorting and candidate screening by providing an automated, intelligent document ingestion and classification pipeline:

* **Automated Gmail Ingestion**: Periodically polls Gmail for new emails containing file attachments, maintaining millisecond-precision timestamp cursors (`RunCursor`) to avoid double-processing.
* **Open-Ended LLM Semantic Domain Classification**: Replaces brittle `if/else` keyword lists with pure LLM semantic reasoning to evaluate candidate resumes, automatically filing them into **15 canonical technology stack directories** (`ai`, `devops`, `full-stack`, `java`, `data-engineering`, `mobile`, `frontend`, `backend`, `cloud`, `security`, `consultant`, `scrum-master`, `project-manager`, `hr`, `other-tech`).
* **Experience Band Sorting**: Calculates stack-specific experience tenure from position date ranges and sub-categorizes resumes into experience bands (`0-2yrs`, `2-5yrs`, `5-10yrs`, `10+yrs`).
* **Audit Trail & Reasoning Reports**: Generates strictly **one consolidated reasoning report (`*_reason.txt`)** per attachment under `storage/reason/` containing ISO timestamps, confidence scores, target folder assignments, and extracted context.
* **Idempotency & Database Metadata**: Stores structured document metadata (`user_id`, `message_id`, `storage_path`, `llm_tag`) in MongoDB and prevents duplicate processing via `processed_messages`.

---

## ⚙️ 2. How the Agent Works (Step-by-Step Pipeline)

```text
 ┌────────────────┐     1. Poll Gmail (after: cursor)      ┌─────────────────────────┐
 │   Gmail API    │ ─────────────────────────────────────> │  Download Attachment    │
 └────────────────┘                                        └────────────┬────────────┘
                                                                        │
                                                                        │ 2. Stage original file
                                                                        ▼
 ┌────────────────┐     4. Single-Pass Semantic Inference  ┌─────────────────────────┐
 │ On-Prem Qwen   │ <───────────────────────────────────── │     storage/inbox/      │
 │  (OpenAI SDK)  │ ─────────────────────────────────────> │  (alex_cv.pdf copy)     │
 └────────────────┘     Returns target_folders & reason    └────────────┬────────────┘
                                                                        │
                                                                        │ 3. Extract text (15k chars)
                                                                        ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                 File Storage & Audit                             │
 │  • Save copies to: storage/resumes/java/5-10yrs/ & storage/resumes/ai/2-5yrs/    │
 │  • Write report to: storage/reason/alex_cv_reason.txt                            │
 │  • Insert metadata to MongoDB & advance cursor                                   │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

1. **Authentication & Cursor Check**: Connects to Gmail via OAuth2 (`credentials.json` / `token.json`) and loads the last processed timestamp (`last_internal_date_ms`) from MongoDB.
2. **Attachment Staging**: Downloads incoming file attachments and stages an unaltered copy in `storage/inbox/` (e.g. `storage/inbox/Suresh_Veeraraghavan.pdf`).
3. **RAG Text Extraction**: Extracts text from PDF (`pdfplumber`), Word (`python-docx`), or plain text files up to 15,000 characters.
4. **Single-Pass LLM Classification**: Passes email metadata + extracted document text to Qwen via OpenAI SDK. The LLM infers candidate domains, calculates experience bands, and returns target folder paths in one call.
5. **Multi-Folder Filing & Reasoning Generation**: Copies the attachment into all LLM-selected target folders (`storage/resumes/java/10+yrs/`, `storage/resumes/ai/2-5yrs/`) with automatic filename collision de-confliction (`_1`, `_2`), writes a single `.txt` reasoning report to `storage/reason/`, inserts metadata to MongoDB, and updates the run cursor.

---

## 📥 3. Input Required for the Agent

### A. Input Data Format (Incoming Email & Attachment)
The agent processes emails containing attachments:
* **Email Metadata**: Sender Email, Subject Line, Body Text.
* **Document Attachment**: PDF (`.pdf`), Word Document (`.docx`), or Plain Text (`.txt`).

#### Sample Input Payload Example:
* **Sender**: `vendor@example.com`
* **Subject**: `Application for Senior Java & AI Architect Role`
* **Body**: `"Please find attached my resume for your consideration. I have 15 years Java architecture experience along with 2 years AI practice leadership."`
* **Attachment Filename**: `Suresh_Veeraraghavan.pdf`
* **Extracted RAG Text**:
  ```text
  SURESH VEERARAGHAVAN | Technology & AI Transformation Executive
  Profile: 26+ years experience in enterprise software, 15 years Java architecture, 2 years Azure architecture, and recent AI practice leadership.
  Career History:
  - 2000-2007: Software Engineering -> Java Development
  - 2008-2018: Enterprise Architecture & Java Solution Architecture
  - 2020-Present: VP Engineering & AI Practice Head (GenAI, RAG, Agentic AI)
  ```

### B. Required Service Inputs (Replacing Placeholders with Real Values)

To run the agent, copy `.env.example` to `.env` and `credentials.json.example` to `credentials.json` and replace dummy values:

| Configuration Key | Example / Default | Required Value & Setup Instructions |
| :--- | :--- | :--- |
| `QWEN_BASE_URL` | `http://YOUR_LLM_SERVER_IP:19298/v1` | **Set to your active LLM server URL** (e.g. `http://175.155.64.191:19298/v1` or local vLLM/Ollama port). |
| `QWEN_MODEL` | `Qwen/Qwen3.5-9B` | **Set to your model identifier** running on your LLM server endpoint. |
| `QWEN_API_KEY` | `EMPTY` | Set API Key if your endpoint requires auth (use `"EMPTY"` for local vLLM/Ollama). |
| `MONGO_URI` | `mongodb://localhost:27017` | Set your MongoDB connection string (local instance or cloud Atlas URI). |
| `GMAIL_CLIENT_SECRET_FILE` | `credentials.json` | Download OAuth Client credentials from **Google Cloud Console** (Gmail API -> Desktop App) and save as `credentials.json`. |

---

## 💡 4. Reasoning Behind the Design Choices & Inputs Provided

* **Why 15,000 Characters for Text Extraction?**
  Multi-page dense technical resumes often place career summaries on Page 1 and deep historical project experience on Pages 3–4. Extracting up to 15,000 characters ensures Qwen evaluates total candidate tenure and all domain experience accurately without truncating early work history.
* **Why Inbox-First Staging (`storage/inbox/`)?**
  Staging raw attachments before processing ensures an unaltered master backup exists on disk regardless of downstream LLM parsing, network timeouts, or multi-folder routing decisions.
* **Why Single-Pass Prompt Architecture over Hardcoded Keyword Lists?**
  Static keyword tables (`ai -> PyTorch`, `devops -> Docker`) break when candidates use new tools or express roles in non-standard phrasing. Single-pass prompt architecture lets the LLM perform holistic open-ended semantic domain reasoning in a single API call, reducing API round-trip latency while remaining adaptable to any candidate profile.
* **Why De-Conflict Filenames (`alex_cv_1.pdf`)?**
  Candidates frequently re-submit updated resumes with identical filenames. De-conflicting paths with matching numeric suffixes across `inbox/`, `reason/`, and `resumes/` ensures historical audit trails are preserved without accidentally overwriting previous submissions.

---

## 📁 5. Restructured File Architecture & File Descriptions

```text
mail-agent/
├── .env.example                # Configuration template file listing all environment variables and default placeholders.
├── .gitignore                  # Version control rules excluding credentials, tokens, local storage attachments, bytecode, and virtual environment.
├── README.md                   # Comprehensive documentation guide explaining purpose, setup, architecture, and execution procedures.
├── credentials.json.example    # Google Cloud OAuth 2.0 Client credentials template with dummy client ID and secret placeholders.
├── evals/                      # Test suite directory (unit tests and mock evaluation fixtures).
│   ├── conftest.py             # Shared pytest fixtures providing mock Gmail/Mongo services and isolated temporary storage.
│   ├── test_agent.py           # Unit tests verifying agent workflow execution, error isolation, idempotency, and multi-folder routing.
│   ├── test_database.py        # Unit tests verifying user creation, document metadata insertion, and message status tracking.
│   ├── test_llm.py             # Unit tests verifying LLM response parsing, fallback handling, and fence stripping.
│   └── test_tools.py           # Unit tests verifying email header parsing, attachment listing, and file collision de-confliction.
├── prompts/                    # System prompts and prompt builders module.
│   └── prompt.py               # Single-pass CLASSIFICATION_SYSTEM_PROMPT and prompt builder for open-ended LLM semantic domain inference.
├── src/                        # Main application source code module.
│   ├── agent.py                # Core pipeline orchestrator coordinating Gmail polling, text extraction, LLM classification, and storage.
│   ├── app.py                  # Application entry point supporting CLI batch execution (python app.py) and FastAPI server mode (python app.py serve).
│   ├── config.py               # Central settings loader parsing environment variables from .env using Pydantic Settings.
│   ├── database.py             # MongoDB database layer managing user records, document metadata, and timestamp polling cursors.
│   ├── llm.py                  # OpenAI client wrapper connecting to Qwen with JSON parsing resilience, anti-thinking options, and fallbacks.
│   ├── logger.py               # Structured JSON logger outputting timestamped execution events to stdout for observability.
│   ├── models.py               # Pydantic data schemas validating LLM outputs, MongoDB documents, users, and idempotency records.
│   ├── requirements.txt        # Python package requirements manifest listing FastAPI, Uvicorn, Pydantic, OpenAI, PyMongo, and PDF tools.
│   ├── tools.py                # Gmail OAuth2 client, attachment downloader, multi-format text extractor (PDF/DOCX), and de-conflicted file writer.
│   └── utils.py                # Helper utilities providing exponential backoff retry decorators and safe filename sanitization.
└── storage/                    # Root directory for local persistent file storage (excluded from Git).
    ├── inbox/                  # Staging directory storing raw original copies of all incoming email attachments before classification.
    ├── reason/                 # Consolidated reasoning report directory holding one single reasoning .txt file per attachment.
    ├── resumes/                # Sub-classified resume directory structured by domain and experience band (e.g. resumes/java/5-10yrs/).
    ├── invoices/               # Storage directory holding classified invoice documents.
    ├── contracts/              # Storage directory holding classified contract documents.
    └── misc/                   # Fallback directory for unclassified or zero-character extracted documents.
```

---

## 💻 6. Execution Procedure (CLI Commands)

### Step A: Setup Virtual Environment & Install Dependencies
```powershell
# Navigate to project directory
cd "D:\personal works\mannit\new\mail-agent"

# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1   # On Linux/macOS: source venv/bin/activate

# Install required Python packages
pip install -r src/requirements.txt
```

### Step B: Run Automated Unit Tests
```powershell
# Run the test suite in isolated temporary storage
.\venv\Scripts\python.exe -m pytest evals
```

### Step C: Execute Agent in CLI Batch Mode (Single Run)
Polls Gmail for new unprocessed emails, classifies attachments, writes files to `storage/`, updates MongoDB, and exits:
```powershell
.\venv\Scripts\python.exe src/app.py
```
*(On your first execution, a browser window will automatically open asking you to authorize Gmail access, which generates `token.json`.)*

### Step D: Execute Agent in Webhook Server Mode (FastAPI)
Starts a Uvicorn HTTP web server listening on port `8000`:
```powershell
.\venv\Scripts\python.exe src/app.py serve
```

### Step E: Trigger Agent via HTTP Call (Postman / cURL)
While the server is running in another terminal window:

* **Health Check**:
  ```powershell
  curl http://localhost:8000/health
  ```
* **Trigger Agent Run**:
  ```powershell
  curl -X POST http://localhost:8000/run -H "Content-Type: application/json"
  ```
