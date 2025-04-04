#!/bin/bash

echo "=== ElderlyCareUI Deployment Helper ==="
echo "This script will help you deploy your ElderlyCareUI project to free hosting services"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Check if GitHub repository exists locally
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    
    # Create initial commit
    git add .
    git commit -m "Initial commit"
    
    echo "Please create a GitHub repository and enter the repository URL:"
    read repo_url
    
    git remote add origin $repo_url
    git branch -M main
    git push -u origin main
else
    echo "Git repository already initialized"
    
    # Push latest changes
    git add .
    git commit -m "Update before deployment"
    git push
fi

echo "=== Preparing for deployment ==="
echo "1. Backend - Render (free tier)"
echo "2. Frontend - Vercel (free tier)"

echo "=== Backend Deployment Steps (Render) ==="
echo "1. Go to https://render.com and sign up/login"
echo "2. Click 'New Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Use these settings:"
echo "   - Name: elderlycareui-backend"
echo "   - Environment: Python"
echo "   - Build Command: pip install -r backend/requirements.txt"
echo "   - Start Command: cd backend && gunicorn app:app"
echo "   - Plan: Free"

echo "=== Frontend Deployment Steps (Vercel) ==="
echo "1. Go to https://vercel.com and sign up/login"
echo "2. Click 'New Project' and import your GitHub repository"
echo "3. The vercel.json file should already be configured for you"
echo "4. Click 'Deploy'"

echo "=== IMPORTANT ==="
echo "After deploying the backend, get the URL and update it in vercel.json:"
echo "1. Update the API_BASE_URL in your vercel.json"
echo "2. Update the API destination in routes section of vercel.json"
echo "3. Redeploy the frontend to apply these changes"

echo "Deployment preparation completed!" 