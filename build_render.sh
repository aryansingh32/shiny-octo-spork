#!/bin/bash

# This script helps prepare the project for Render deployment

echo "Starting Render deployment preparation..."

# Print Python version information
echo "Python version:"
python --version
pip --version

# Install dependencies with verbose output
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt --verbose

# Create any necessary directories
mkdir -p backend/logs

# Handle pyttsx3 issues (common on cloud platforms)
echo "Checking TTS dependencies..."
if pip show pyttsx3 > /dev/null; then
  echo "Note: pyttsx3 is installed but might not work in cloud environments."
  echo "Text-to-speech functionality may be limited on Render."
else
  echo "pyttsx3 not installed, will use alternative TTS methods."
fi

# Verify critical dependencies
echo "Verifying critical dependencies..."
if pip show scikit-learn > /dev/null; then
  echo "scikit-learn installed successfully."
else
  echo "WARNING: scikit-learn installation failed. Check compatibility."
fi

if pip show numpy > /dev/null; then
  echo "numpy installed successfully."
else
  echo "WARNING: numpy installation failed. Check compatibility."
fi

echo "Render deployment preparation complete!" 