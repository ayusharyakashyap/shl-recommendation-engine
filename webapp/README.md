# SHL Assessment Recommendation Engine - Web App

## 🌐 Web-based RAG Tool

This is a Streamlit-based web interface for the SHL Assessment Recommendation Engine.

### 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Web App:**
   ```bash
   streamlit run app.py
   ```

3. **Access the App:**
   Open your browser to `http://localhost:8501`

### ✨ Features

- **Interactive Query Interface** - Enter job descriptions and get instant recommendations
- **Sample Queries** - Pre-built examples for quick testing
- **Real-time Results** - See recommendations with confidence scores
- **Export Options** - Download results as JSON or CSV
- **Query History** - Track recent searches
- **Explanations** - Understand why assessments were recommended

### 🔧 Configuration

The app can work in two modes:
- **API Mode**: Connects to the FastAPI server (if running)
- **Local Mode**: Uses the trained models directly

### 📊 Sample Queries

Try these example queries:
- "I need Java developers with 3+ years experience for a 45-minute assessment"
- "Looking for sales representatives, entry level, strong communication skills"
- "Need marketing manager with leadership experience and creative thinking"

### 🎯 Performance

The underlying model achieves:
- **92.0% Precision@5**
- **74.8% Recall@5** 
- **94.6% NDCG@5**

---

**Built for SHL Labs Research Intern Assignment**  
**Author:** Ayush Arya Kashyap