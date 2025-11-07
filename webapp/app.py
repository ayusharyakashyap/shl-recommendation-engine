"""
Web-based RAG (Retrieval-Augmented Generation) Tool for SHL Assessment Recommendations
A simple Streamlit web interface for the recommendation system
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to Python path to import our models
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

try:
    from models.hybrid import HybridRecommender
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def load_recommender():
    """Load the hybrid recommender model"""
    try:
        recommender = HybridRecommender()
        recommender.load_model(
            model_path="../../src/models/hybrid_model.pkl",
            content_model_path="../../src/models/content_based_model.pkl",
            cf_model_path="../../src/models/collaborative_filtering_model.pkl",
            processed_data_path="../../src/data/processed_data.json"
        )
        return recommender
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def call_api_endpoint(query, top_k=5, include_explanation=True):
    """Call the API endpoint if it's running"""
    try:
        response = requests.post(
            "http://localhost:8000/recommend",
            json={
                "query": query,
                "top_k": top_k,
                "include_explanation": include_explanation,
                "diverse": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def main():
    # Header
    st.title("🎯 SHL Assessment Recommendation Engine")
    st.markdown("### AI-Powered Assessment Selection for HR Professionals")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Configuration")
        top_k = st.slider("Number of Recommendations", 1, 10, 5)
        include_explanations = st.checkbox("Include Explanations", value=True)
        use_api = st.checkbox("Use API Endpoint", value=False)
        
        st.header("📊 System Info")
        st.info("**Model Performance:**\n- Precision@5: 92.0%\n- Recall@5: 74.8%\n- NDCG@5: 0.946")
        
        if st.button("🔄 Clear History"):
            st.session_state.query_history = []
            st.session_state.recommendations = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Job Description Input")
        
        # Sample queries for quick testing
        st.subheader("📝 Sample Queries")
        sample_queries = [
            "I need Java developers with 3+ years experience for a 45-minute assessment",
            "Looking for sales representatives, entry level, strong communication skills",
            "Need marketing manager with leadership experience and creative thinking",
            "Python developer with data analysis skills, 60-minute assessment",
            "QA engineer with automation testing experience, 1 hour maximum"
        ]
        
        selected_sample = st.selectbox("Choose a sample query:", [""] + sample_queries)
        
        # Query input
        query = st.text_area(
            "Enter your hiring requirements:",
            value=selected_sample if selected_sample else "",
            height=100,
            placeholder="Example: I need Java developers with 3+ years experience for a 45-minute assessment..."
        )
        
        # Recommendation button
        if st.button("🚀 Get Recommendations", type="primary"):
            if query.strip():
                with st.spinner("Generating recommendations..."):
                    recommendations = None
                    
                    # Try API first if selected
                    if use_api:
                        recommendations = call_api_endpoint(query, top_k, include_explanations)
                        if recommendations:
                            st.success("✅ Using API endpoint")
                        else:
                            st.warning("⚠️ API not available, using local model")
                    
                    # Use local model if API not available or not selected
                    if not recommendations and MODELS_AVAILABLE:
                        try:
                            if 'recommender' not in st.session_state:
                                with st.spinner("Loading recommendation model..."):
                                    st.session_state.recommender = load_recommender()
                            
                            if st.session_state.recommender:
                                recs = st.session_state.recommender.recommend(
                                    query, top_k=top_k, explain=include_explanations
                                )
                                recommendations = {
                                    "query": query,
                                    "recommendations": recs,
                                    "total_found": len(recs),
                                    "processing_time_ms": 0
                                }
                                st.success("✅ Using local model")
                        except Exception as e:
                            st.error(f"Error with local model: {e}")
                    
                    if recommendations:
                        st.session_state.recommendations = recommendations
                        st.session_state.query_history.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "query": query,
                            "results": len(recommendations["recommendations"])
                        })
                        st.rerun()
                    else:
                        st.error("❌ No recommendation service available. Please ensure models are trained.")
            else:
                st.warning("Please enter a job description or query.")
    
    with col2:
        st.header("📈 Recent Queries")
        if st.session_state.query_history:
            for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"{entry['timestamp']} - {entry['results']} results"):
                    st.write(entry['query'][:100] + "..." if len(entry['query']) > 100 else entry['query'])
        else:
            st.info("No queries yet. Try the samples above!")
    
    # Display recommendations
    if st.session_state.recommendations:
        st.header("🎯 Recommended Assessments")
        
        recs = st.session_state.recommendations["recommendations"]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Recommendations", len(recs))
        with col2:
            avg_confidence = sum(rec.get('confidence', 0) for rec in recs) / len(recs) if recs else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with col3:
            unique_types = len(set(rec.get('assessment_type', 'unknown') for rec in recs))
            st.metric("Assessment Types", unique_types)
        with col4:
            processing_time = st.session_state.recommendations.get('processing_time_ms', 0)
            st.metric("Processing Time", f"{processing_time:.1f}ms")
        
        # Individual recommendations
        for i, rec in enumerate(recs, 1):
            with st.expander(f"{i}. {rec.get('assessment_name', 'Unknown').title()}", expanded=i<=3):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Type:** {rec.get('assessment_type', 'Unknown').title()}")
                    st.write(f"**Confidence:** {rec.get('confidence', 0):.1f}%")
                    
                    if rec.get('related_skills'):
                        st.write(f"**Related Skills:** {', '.join(rec['related_skills'][:5])}")
                    
                    if include_explanations and rec.get('explanation', {}).get('reasons'):
                        st.write(f"**Why recommended:** {rec['explanation']['reasons'][0]}")
                    
                    if rec.get('assessment_url'):
                        st.write(f"**URL:** {rec['assessment_url']}")
                
                with col2:
                    # Confidence bar
                    confidence = rec.get('confidence', 0)
                    st.progress(confidence / 100)
                    
                    # Assessment type badge
                    type_colors = {
                        'technical': '🔧',
                        'communication': '💬',
                        'personality': '🧠',
                        'cognitive': '🎯',
                        'business': '💼',
                        'office_skills': '📊',
                        'general': '📝'
                    }
                    assessment_type = rec.get('assessment_type', 'general')
                    icon = type_colors.get(assessment_type, '📝')
                    st.write(f"{icon} {assessment_type.title()}")
        
        # Export options
        st.header("📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = json.dumps(st.session_state.recommendations, indent=2)
            st.download_button(
                "📄 Download as JSON",
                json_data,
                file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            if recs:
                df = pd.DataFrame([{
                    'Rank': i+1,
                    'Assessment Name': rec.get('assessment_name', ''),
                    'Type': rec.get('assessment_type', ''),
                    'Confidence': f"{rec.get('confidence', 0):.1f}%",
                    'URL': rec.get('assessment_url', '')
                } for i, rec in enumerate(recs)])
                
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📊 Download as CSV",
                    csv_data,
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    **SHL Assessment Recommendation Engine** | Built with ❤️ for SHL Labs  
    *Powered by Hybrid ML Models (Content-based + Collaborative Filtering)*
    """)

if __name__ == "__main__":
    main()