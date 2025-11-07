"""
Standalone API Endpoint for SHL Assessment Recommendations
Simplified FastAPI server for submission
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

try:
    from models.hybrid import HybridRecommender
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="AI-powered assessment recommendation endpoint for SHL",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender
recommender = None

class QueryRequest(BaseModel):
    """Request model for assessment recommendations"""
    query: str = Field(..., description="Job description or hiring requirements")
    top_k: int = Field(5, description="Number of recommendations to return")
    include_explanation: bool = Field(False, description="Include explanation")

class AssessmentRecommendation(BaseModel):
    """Individual assessment recommendation"""
    assessment_name: str
    assessment_type: str
    assessment_url: str
    confidence: float
    related_skills: List[str] = []
    explanation: Optional[str] = None

class RecommendationResponse(BaseModel):
    """API response model"""
    query: str
    recommendations: List[AssessmentRecommendation]
    total_found: int
    status: str

@app.on_event("startup")
async def startup_event():
    """Load the recommendation model on startup"""
    global recommender
    
    if MODELS_AVAILABLE:
        try:
            recommender = HybridRecommender()
            recommender.load_model(
                model_path="../../src/models/hybrid_model.pkl",
                content_model_path="../../src/models/content_based_model.pkl",
                cf_model_path="../../src/models/collaborative_filtering_model.pkl",
                processed_data_path="../../src/data/processed_data.json"
            )
            print("✅ Recommendation engine loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            recommender = None
    else:
        print("⚠️ Models not available - using mock responses")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if recommender else "models_not_loaded",
        "models_available": MODELS_AVAILABLE,
        "recommender_loaded": recommender is not None
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: QueryRequest):
    """
    Get assessment recommendations for a job query
    
    Returns a ranked list of SHL assessments suitable for the given job requirements.
    """
    try:
        if recommender:
            # Use actual model
            recommendations = recommender.recommend(
                query=request.query,
                top_k=request.top_k,
                explain=request.include_explanation
            )
            
            # Convert to response format
            assessment_recs = []
            for rec in recommendations:
                explanation = None
                if request.include_explanation and 'explanation' in rec:
                    if rec['explanation'].get('reasons'):
                        explanation = rec['explanation']['reasons'][0]
                
                assessment_rec = AssessmentRecommendation(
                    assessment_name=rec['assessment_name'],
                    assessment_type=rec['assessment_type'],
                    assessment_url=rec['assessment_url'],
                    confidence=rec['confidence'],
                    related_skills=rec.get('related_skills', []),
                    explanation=explanation
                )
                assessment_recs.append(assessment_rec)
            
            return RecommendationResponse(
                query=request.query,
                recommendations=assessment_recs,
                total_found=len(assessment_recs),
                status="success"
            )
        
        else:
            # Return mock data for demonstration
            mock_recommendations = [
                AssessmentRecommendation(
                    assessment_name="Java Programming Assessment",
                    assessment_type="technical",
                    assessment_url="https://www.shl.com/products/product-catalog/view/java-assessment/",
                    confidence=92.5,
                    related_skills=["java", "programming", "software development"],
                    explanation="Matches Java programming requirements" if request.include_explanation else None
                ),
                AssessmentRecommendation(
                    assessment_name="Problem Solving Assessment",
                    assessment_type="cognitive",
                    assessment_url="https://www.shl.com/products/product-catalog/view/problem-solving/",
                    confidence=87.3,
                    related_skills=["analytical thinking", "problem solving"],
                    explanation="Tests analytical and problem-solving abilities" if request.include_explanation else None
                ),
                AssessmentRecommendation(
                    assessment_name="Team Collaboration Assessment",
                    assessment_type="personality",
                    assessment_url="https://www.shl.com/products/product-catalog/view/team-collaboration/",
                    confidence=82.1,
                    related_skills=["teamwork", "collaboration", "communication"],
                    explanation="Evaluates team collaboration skills" if request.include_explanation else None
                )
            ]
            
            return RecommendationResponse(
                query=request.query,
                recommendations=mock_recommendations[:request.top_k],
                total_found=len(mock_recommendations[:request.top_k]),
                status="mock_data"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing"""
    return {
        "sample_queries": [
            "I need Java developers with 3+ years experience for a 45-minute assessment",
            "Looking for sales representatives, entry level, strong communication skills",
            "Need marketing manager with leadership experience and creative thinking",
            "Python developer with data analysis skills, 60-minute assessment",
            "QA engineer with automation testing experience, 1 hour maximum"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SHL Assessment Recommendation API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )