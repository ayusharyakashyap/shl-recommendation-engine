#!/bin/bash

# SHL Assessment Recommendation Engine - Cloud Deployment Helper
# This script helps you deploy your application to the cloud

echo "🌐 SHL Assessment Recommendation Engine - Cloud Deployment"
echo "=========================================================="
echo ""

echo "📋 Your submission is ready for cloud deployment!"
echo ""
echo "🎯 Recommended Platform: Streamlit Community Cloud (FREE)"
echo ""
echo "📝 Quick Steps to Deploy:"
echo ""
echo "1. 📂 Create GitHub Repository:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: 'shl-recommendation-engine'"
echo "   - Make it public"
echo "   - Don't initialize with README (we have our own)"
echo ""
echo "2. 🚀 Push Your Code:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/shl-recommendation-engine.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. 🌐 Deploy to Streamlit Cloud:"
echo "   - Visit: https://share.streamlit.io/"
echo "   - Click 'New app'"
echo "   - Connect your GitHub repository"
echo "   - Main file path: webapp/app.py"
echo "   - Click 'Deploy'!"
echo ""
echo "4. ✅ Get Your Public URL:"
echo "   - You'll get a URL like: https://YOUR-USERNAME-shl-recommendation-engine-webapp-app-xyz123.streamlit.app/"
echo "   - This URL will be accessible 24/7 for SHL recruiters!"
echo ""
echo "📊 Current Status:"
echo "✅ Code is ready for deployment"
echo "✅ Git repository initialized"
echo "✅ All files committed"
echo "✅ Docker configuration included"
echo "✅ Streamlit configuration optimized"
echo ""
echo "🔗 Alternative Platforms:"
echo "- Render.com: Professional deployment with custom domains"
echo "- Railway.app: One-click deployment"
echo "- Vercel: Serverless deployment"
echo ""
echo "📖 For detailed instructions, see: CLOUD_DEPLOYMENT_GUIDE.md"
echo ""
echo "🎉 Once deployed, update your SHL submission with the live URL!"
echo ""

# Check if user wants to open the deployment guide
read -p "📖 Would you like to open the detailed deployment guide? (y/n): " choice
case "$choice" in 
  y|Y ) 
    if command -v open >/dev/null 2>&1; then
      open CLOUD_DEPLOYMENT_GUIDE.md
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open CLOUD_DEPLOYMENT_GUIDE.md
    else
      echo "Please open CLOUD_DEPLOYMENT_GUIDE.md manually"
    fi
    ;;
  * ) 
    echo "📄 You can find the deployment guide in: CLOUD_DEPLOYMENT_GUIDE.md"
    ;;
esac

echo ""
echo "🌟 Good luck with your SHL submission!"