from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
JWT_SECRET = os.getenv('JWT_SECRET', 'changeme-please-set-in-env')
FERNET_KEY = os.getenv('FERNET_KEY', '')  # optional: base64 urlsafe
DATA_PATH = BASE_DIR / 'data' / 'store_data.csv'
