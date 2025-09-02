from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

REPORT_AGENT_API_KEY = os.getenv("REPORT_AGENT_API_KEY", "change_me")
ANALYSIS_BASE_URL = os.getenv("ANALYSIS_BASE_URL", "http://127.0.0.1:8081")
COLLECTOR_BASE_URL = os.getenv("COLLECTOR_BASE_URL", "http://127.0.0.1:8082")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
USE_HF_SUMMARIZER = os.getenv("USE_HF_SUMMARIZER", "false").lower() == "true"

os.makedirs(OUTPUT_DIR, exist_ok=True)
