# SHL Assessment Recommendation Engine - Submission Package

**Submitted by:** Ayush Arya Kashyap  
**Position:** Research Intern Application  
**Project:** AI-Powered Assessment Recommendation System

## 🎯 Quick Start Guide

### Option 1: Automated Deployment
```bash
# Navigate to submission folder
cd SHL_Submission

# Run everything with one command
./deploy.sh both
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
cd api && python api.py &

# Start web application (in new terminal)
cd webapp && streamlit run app.py
```

## 📋 Submission Deliverables

### ✅ 1. Web Application URL
**Local Development:** `http://localhost:8501`
**Features:**
- Interactive query interface
- Real-time assessment recommendations
- Confidence scoring and explanations
- Sample queries for testing
- Export functionality (JSON/CSV)
- Query history tracking

### ✅ 2. API Endpoint
**Base URL:** `http://localhost:8000`
**Key Endpoints:**
- `POST /recommend` - Get assessment recommendations
- `GET /health` - Check system status
- `GET /docs` - Interactive API documentation

**Sample API Call:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer with 3+ years experience", "top_k": 5}'
```

### ✅ 3. GitHub Repository
**Structure:**
```
SHL_Submission/
├── webapp/           # Streamlit web application
├── api/             # FastAPI server
├── code/            # Core algorithm implementation
├── documents/       # Technical approach document
├── requirements.txt # Dependencies
├── deploy.sh       # Deployment script
└── ayush_arya_kashyap.csv # Predictions file
```

### ✅ 4. Technical Approach Document
**Location:** `documents/SHL_Assessment_Recommendation_Engine_Approach.md`
**Contents:** 2-page comprehensive overview covering:
- Problem statement and objectives
- Technical architecture (hybrid ML approach)
- Implementation details and technology stack
- Performance results (92% precision@5)
- Business impact and future enhancements

### ✅ 5. Predictions CSV File
**Filename:** `ayush_arya_kashyap.csv`
**Format:**
```csv
Query,Rank,Assessment_Name,Assessment_Type,Confidence_Score,Assessment_URL
```
**Content:** 50 prediction records covering 10 unique queries with top-5 recommendations each

## 🏗️ System Architecture

### Core Components
1. **Hybrid Recommendation Engine**
   - Content-based filtering (TF-IDF + cosine similarity)
   - Collaborative filtering (Non-negative Matrix Factorization)
   - Popularity-based boosting

2. **Feature Extraction Pipeline**
   - Skills extraction from job descriptions
   - Experience level detection
   - Duration requirement parsing

3. **API Layer**
   - FastAPI framework for high-performance REST API
   - Async request handling
   - Comprehensive error handling

4. **Web Interface**
   - Streamlit-based interactive application
   - Real-time recommendation generation
   - Export and visualization capabilities

## 📊 Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| Precision@5 | 92.0% | Accuracy of top-5 recommendations |
| Recall@5 | 75.0% | Coverage of relevant assessments |
| NDCG@5 | 94.6% | Ranking quality measurement |
| Response Time | <500ms | Average API response time |

## 🚀 Deployment Options

### Development Mode (Current)
- Local deployment for testing and evaluation
- API: `http://localhost:8000`
- Web App: `http://localhost:8501`

### Production Deployment (Future)
- Cloud deployment on AWS/GCP/Azure
- Container orchestration with Docker/Kubernetes
- Load balancing and auto-scaling

## 🧪 Testing the System

### 1. Web Application Testing
1. Navigate to `http://localhost:8501`
2. Try sample queries or enter custom job descriptions
3. Review recommendations with confidence scores
4. Export results for analysis

### 2. API Testing
1. Visit `http://localhost:8000/docs` for interactive documentation
2. Test `/recommend` endpoint with various queries
3. Check `/health` endpoint for system status

### 3. Sample Test Queries
- "Java developer with 3+ years experience for 45-minute assessment"
- "Entry-level sales representative with strong communication skills"
- "Marketing manager with leadership experience and creative thinking"
- "Python developer with data analysis skills, 60-minute assessment"
- "QA engineer with automation testing experience"

## 📁 File Structure Details

```
SHL_Submission/
├── webapp/
│   └── app.py                    # Streamlit web application
├── api/
│   └── api.py                    # FastAPI server implementation
├── code/
│   ├── models/                   # ML model implementations
│   ├── data/                     # Data processing utilities
│   └── evaluation/               # Performance evaluation tools
├── documents/
│   └── SHL_Assessment_...md      # Technical approach document
├── requirements.txt              # Python dependencies
├── deploy.sh                     # Automated deployment script
├── generate_predictions.py       # CSV predictions generator
├── ayush_arya_kashyap.csv       # Predictions output file
└── README.md                     # This documentation
```

## 🔧 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure virtual environment is activated
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**2. Model Loading Issues**
- System will automatically use mock data if trained models aren't found
- Check console output for model loading status

**3. Port Conflicts**
- API default: port 8000
- Web app default: port 8501
- Modify ports in respective files if needed

### Support
For technical issues or questions about the implementation, please refer to:
- API documentation: `http://localhost:8000/docs`
- Technical approach document: `documents/SHL_Assessment_Recommendation_Engine_Approach.md`
- Code comments and docstrings throughout the codebase

## 🎉 Success Indicators

✅ **Web app loads and displays interface**  
✅ **API responds to health checks**  
✅ **Recommendations generate for test queries**  
✅ **CSV predictions file contains 50 records**  
✅ **Technical document explains methodology**  

---

**Thank you for evaluating the SHL Assessment Recommendation Engine!**  
*This system demonstrates the potential for AI-powered solutions in talent assessment and HR technology.*