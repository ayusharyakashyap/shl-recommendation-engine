#!/bin/bash

# SHL Assessment Recommendation System Deployment Script
# This script sets up and runs both the API and web application

echo "🚀 SHL Assessment Recommendation System Deployment"
echo "=================================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "📦 Creating virtual environment..."
    cd ..
    python3 -m venv .venv
    cd SHL_Submission
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ../.venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if models exist
if [ ! -f "../src/models/hybrid_model.pkl" ]; then
    echo "⚠️  Trained models not found. The system will use mock data."
    echo "   To use real models, please run the training scripts first."
fi

# Function to start API server
start_api() {
    echo "🌐 Starting API server on port 8000..."
    cd api
    python api.py &
    API_PID=$!
    cd ..
    echo "✅ API server started (PID: $API_PID)"
    echo "📖 API Documentation: http://localhost:8000/docs"
}

# Function to start web app
start_webapp() {
    echo "🌐 Starting web application on port 8501..."
    cd webapp
    streamlit run app.py --server.port 8501 &
    WEBAPP_PID=$!
    cd ..
    echo "✅ Web application started (PID: $WEBAPP_PID)"
    echo "🔗 Web App URL: http://localhost:8501"
}

# Function to cleanup processes
cleanup() {
    echo "🧹 Cleaning up..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "🔴 API server stopped"
    fi
    if [ ! -z "$WEBAPP_PID" ]; then
        kill $WEBAPP_PID 2>/dev/null  
        echo "🔴 Web app stopped"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Parse command line arguments
case "$1" in
    "api")
        start_api
        echo "Press Ctrl+C to stop the API server"
        wait $API_PID
        ;;
    "webapp")
        start_webapp
        echo "Press Ctrl+C to stop the web application"
        wait $WEBAPP_PID
        ;;
    "both"|"")
        start_api
        sleep 2
        start_webapp
        
        echo ""
        echo "🎉 Both services are running!"
        echo "📊 Web App: http://localhost:8501" 
        echo "🔗 API Docs: http://localhost:8000/docs"
        echo "❤️  API Health: http://localhost:8000/health"
        echo ""
        echo "Press Ctrl+C to stop both services"
        
        # Wait for both processes
        wait $API_PID $WEBAPP_PID
        ;;
    "test")
        echo "🧪 Running test predictions..."
        python generate_predictions.py
        echo "✅ Test completed. Check ayush_arya_kashyap.csv for results."
        ;;
    *)
        echo "Usage: $0 [api|webapp|both|test]"
        echo "  api    - Start only the API server"
        echo "  webapp - Start only the web application"
        echo "  both   - Start both services (default)"
        echo "  test   - Generate test predictions CSV"
        exit 1
        ;;
esac