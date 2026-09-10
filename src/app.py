"""
Entry point.
  - `python app.py`        -> runs the agent once and exits (CLI/cron use).
  - `python app.py serve`  -> starts a FastAPI server with POST /run
                                (for push-trigger / webhook-style invocation).
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException

from src import agent
from src.logger import log_event

app = FastAPI(title="Mail Agent")


@app.get("/health")
def health():
    return {"status": "ok", "service": "Mail Agent API"}


@app.post("/run")
def trigger_run():
    try:
        summary = agent.run()
        return {
            "processed": summary.processed,
            "failed": summary.failed,
            "skipped": summary.skipped,
        }
    except Exception as e:
        log_event("api_run_failed", level="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
    else:
        agent.run()