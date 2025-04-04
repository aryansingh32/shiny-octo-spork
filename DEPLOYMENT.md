# Deployment Guide for ElderlyCareUI

This guide describes how to deploy the ElderlyCareUI project to free hosting platforms. The project consists of two main parts:
1. Backend (Flask API) - will be deployed to Render.com
2. Frontend (React + Vite) - will be deployed to Vercel.com

## Prerequisites

- A GitHub account
- Git installed on your local machine
- A Render.com account (free tier)
- A Vercel.com account (free tier)

## Step 1: Push Your Code to GitHub

1. Create a new repository on GitHub
2. Initialize Git in your project (if not already done)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/ElderlyCareUI.git
   git push -u origin main
   ```

## Step 2: Deploy the Backend to Render.com

1. Sign up/login to [Render.com](https://render.com)
2. Click "New Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `elderlycareui-backend` (or your preferred name)
   - **Environment**: Python
   - **Region**: Choose the closest to your target users
   - **Branch**: main
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && gunicorn app:app`
   - **Plan**: Free

5. Click "Create Web Service"
6. Wait for the deployment to complete (this may take a few minutes)
7. Once deployed, note the URL (typically `https://your-service-name.onrender.com`)

## Step 3: Update Frontend Configuration

After deploying the backend, you need to update the frontend configuration:

1. Edit `vercel.json` to update the backend URL:
   ```json
   {
     "routes": [
       { "handle": "filesystem" },
       { "src": "/api/(.*)", "dest": "https://your-backend-url.onrender.com/api/$1" },
       { "src": "/(.*)", "dest": "index.html" }
     ],
     "env": {
       "VITE_API_BASE_URL": "https://your-backend-url.onrender.com"
     }
   }
   ```

2. Commit and push these changes:
   ```bash
   git add vercel.json
   git commit -m "Update backend URL"
   git push
   ```

## Step 4: Deploy the Frontend to Vercel.com

1. Sign up/login to [Vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build` (should be detected automatically)
   - **Output Directory**: `dist` (should be detected automatically)

5. Click "Deploy"
6. Wait for the deployment to complete
7. Once deployed, Vercel will provide you with a URL for your application

## Step 5: Testing and Verification

1. Visit your frontend URL to ensure it's working correctly
2. Test API endpoints through the frontend
3. If there are issues:
   - Check Render logs for backend problems
   - Check Vercel logs for frontend problems

## Updating Your Application

When you make changes to your code:

1. Commit and push changes to GitHub
2. Render and Vercel will automatically redeploy your application

## Environment Variables

If your application requires environment variables:

- For backend: Add them in the Render dashboard under "Environment" section
- For frontend: Add them in the Vercel dashboard under "Environment Variables" section

## Custom Domains (Optional)

Both Render and Vercel allow you to set up custom domains:

- For Render: Go to your service > Settings > Custom Domain
- For Vercel: Go to your project > Settings > Domains

Note that SSL certificates are provided automatically by both services.

## Troubleshooting

- **Backend not responding**: Check Render logs for errors
- **Frontend not updating**: Make sure the Vercel build is completing successfully
- **CORS errors**: Ensure CORS is properly configured in backend/app.py
- **Environment variable issues**: Verify variables are set correctly in both platforms 