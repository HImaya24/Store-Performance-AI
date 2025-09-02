from __future__ import annotations
import uvicorn
from .app import app

def main():
    uvicorn.run(app, host="127.0.0.1", port=8083)

if __name__ == "__main__":
    main()
