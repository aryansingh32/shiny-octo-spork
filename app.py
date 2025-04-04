"""
Root-level app.py to properly import the application for Render.
This is a common pattern for handling Python module imports in deployment.
"""

import os
import sys

# Add the current directory to the path so Python can find the packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app from the backend package
from backend.app import app

# This allows gunicorn to find the app when running with gunicorn app:app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 