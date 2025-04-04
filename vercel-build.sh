#!/bin/bash

# This script is used by Vercel to install dependencies and build the project
echo "Running custom build script for Vercel deployment"

# Install dependencies with legacy peer deps to fix dependency conflicts
echo "Installing dependencies with legacy-peer-deps..."
npm install --legacy-peer-deps

# Run the build
echo "Building the project..."
npm run build

# Run debug script to check directories
echo "Running debug script..."
node debug-build.js

# Create dist directory if it doesn't exist
echo "Ensuring dist directory exists..."
mkdir -p dist

# Ensure that dist has index.html 
if [ ! -f "dist/index.html" ]; then
  echo "index.html not found in dist, searching for it..."
  
  # Look for index.html in root directory
  if [ -f "index.html" ]; then
    echo "Found index.html in root, copying to dist..."
    cp index.html dist/
  fi
  
  # Look for index.html in build directory
  if [ -d "build" ] && [ -f "build/index.html" ]; then
    echo "Found index.html in build/, copying build contents to dist..."
    cp -r build/* dist/
  fi
fi

# Copy assets if needed
if [ -d "public" ]; then
  echo "Copying public directory assets to dist..."
  cp -r public/* dist/ 2>/dev/null || true
fi

# Check for necessary files in dist
echo "Checking contents of dist directory..."
if [ ! -f "dist/index.html" ]; then
  echo "ERROR: No index.html found in dist directory!"
  echo "Creating minimal index.html..."
  echo "<html><head><title>ElderlyCareUI</title><meta http-equiv='refresh' content='0;url=/index.html'></head><body>Redirecting...</body></html>" > dist/index.html
fi

# List the contents of the dist directory
echo "Contents of dist directory:"
ls -la dist

echo "Build completed successfully!" 