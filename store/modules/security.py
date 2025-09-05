import re
import time
import jwt
from cryptography.fernet import Fernet
from utils.config import JWT_SECRET, FERNET_KEY

# Simple input sanitizer
def sanitize_text(text: str) -> str:
    # remove control characters and excessive whitespace
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# JWT helpers
def create_token(identity: str, expires_in: int = 3600) -> str:
    payload = {"sub": identity, "iat": int(time.time()), "exp": int(time.time()) + expires_in}
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_token(token: str) -> dict:
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return {"ok": True, "sub": data.get('sub')}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Optional encryption helpers
def get_fernet():
    if not FERNET_KEY:
        # derive a key for demo (not for production!)
        return None
    return Fernet(FERNET_KEY.encode())

def encrypt_text(plain_text: str) -> bytes:
    f = get_fernet()
    if not f:
        raise RuntimeError("FERNET_KEY not configured")
    return f.encrypt(plain_text.encode())

def decrypt_text(token: bytes) -> str:
    f = get_fernet()
    if not f:
        raise RuntimeError("FERNET_KEY not configured")
    return f.decrypt(token).decode()
