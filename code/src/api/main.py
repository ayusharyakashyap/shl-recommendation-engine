"""
FastAPI REST API for SHL Assessment Recommendation Engine
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
import os
import uvicorn

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid import HybridRecommender

app = FastAPI(
    title="SHL Assessment Recommendation Engine",
    description="AI-powered system to recommend relevant assessments from SHL's product catalog",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = None

class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    query: str = Field(..., description="Job description or hiring query", min_length=10)
    top_k: int = Field(5, description="Number of recommendations to return", ge=1, le=20)
    include_explanation: bool = Field(False, description="Include explanation for recommendations")
    diverse: bool = Field(False, description="Ensure diverse recommendation types")

class AssessmentRecommendation(BaseModel):
    """Response model for individual assessment recommendation"""
    assessment_name: str
    assessment_type: str
    assessment_url: str
    hybrid_score: float
    confidence: float
    related_skills: List[str] = []
    explanation: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    query: str
    recommendations: List[AssessmentRecommendation]
    total_found: int
    processing_time_ms: float

class ExplanationRequest(BaseModel):
    """Request model for detailed explanation"""
    query: str = Field(..., description="Original hiring query")
    assessment_url: str = Field(..., description="Assessment URL to explain")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup"""
    global recommender
    
    try:
        print("Loading SHL Assessment Recommendation Engine...")
        
        # Initialize hybrid recommender
        recommender = HybridRecommender(
            content_weight=0.6,
            collaborative_weight=0.4,
            popularity_weight=0.1
        )
        
        # Load pre-trained models
        recommender.load_model(
            model_path="src/models/hybrid_model.pkl",
            content_model_path="src/models/content_based_model.pkl",
            cf_model_path="src/models/collaborative_filtering_model.pkl",
            processed_data_path="src/data/processed_data.json"
        )
        
        print("✅ Recommendation engine loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading recommendation engine: {e}")
        recommender = None

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SHL Assessment Recommendation Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if recommender and recommender.is_trained else "unhealthy",
        model_loaded=recommender is not None and recommender.is_trained,
        version="1.0.0"
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations for a hiring query
    
    Returns ranked list of recommended assessments with confidence scores.
    """
    if not recommender or not recommender.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation engine not available. Please try again later."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Get recommendations
        if request.diverse:
            recommendations = recommender.recommend_with_diversity(
                query=request.query,
                top_k=request.top_k
            )
        else:
            recommendations = recommender.recommend(
                query=request.query,
                top_k=request.top_k,
                explain=request.include_explanation
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to response format
        assessment_recs = []
        for rec in recommendations:
            assessment_rec = AssessmentRecommendation(
                assessment_name=rec['assessment_name'],
                assessment_type=rec['assessment_type'],
                assessment_url=rec['assessment_url'],
                hybrid_score=rec['hybrid_score'],
                confidence=rec['confidence'],
                related_skills=rec.get('related_skills', [])
            )
            
            if request.include_explanation and 'explanation' in rec:
                assessment_rec.explanation = rec['explanation']
            
            assessment_recs.append(assessment_rec)
        
        return RecommendationResponse(
            query=request.query,
            recommendations=assessment_recs,
            total_found=len(assessment_recs),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.post("/explain")
async def explain_recommendation(request: ExplanationRequest):
    """
    Get detailed explanation for why a specific assessment was recommended
    """
    if not recommender or not recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not available"
        )
    
    try:
        explanation = recommender.get_recommendation_explanation(
            query=request.query,
            assessment_url=request.assessment_url
        )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )

@app.get("/assessments/popular")
async def get_popular_assessments(top_k: int = Query(10, ge=1, le=50)):
    """Get most popular assessments across all queries"""
    if not recommender or not recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not available"
        )
    
    try:
        # Get popularity scores
        popularity_scores = recommender.popularity_scores
        
        # Sort by popularity
        sorted_assessments = sorted(
            popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Get assessment details
        popular_assessments = []
        data = recommender.collaborative_recommender.data
        
        for assessment_url, score in sorted_assessments:
            assessment_data = data[data['assessment_url'] == assessment_url].iloc[0]
            
            popular_assessments.append({
                'assessment_name': assessment_data['assessment_name'],
                'assessment_type': assessment_data['assessment_type'],
                'assessment_url': assessment_url,
                'popularity_score': score,
                'usage_count': int(score * data['assessment_url'].value_counts().max())
            })
        
        return {
            'popular_assessments': popular_assessments,
            'total_assessments': len(popularity_scores)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving popular assessments: {str(e)}"
        )

@app.get("/assessments/types")
async def get_assessment_types():
    """Get available assessment types and their counts"""
    if not recommender or not recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not available"
        )
    
    try:
        data = recommender.collaborative_recommender.data
        type_counts = data['assessment_type'].value_counts().to_dict()
        
        return {
            'assessment_types': type_counts,
            'total_types': len(type_counts)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving assessment types: {str(e)}"
        )

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and performance metrics"""
    if not recommender or not recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not available"
        )
    
    try:
        data = recommender.collaborative_recommender.data
        
        stats = {
            'total_records': len(data),
            'unique_queries': data['query_id'].nunique(),
            'unique_assessments': data['assessment_url'].nunique(),
            'assessment_types': data['assessment_type'].value_counts().to_dict(),
            'job_roles': data['job_role'].value_counts().to_dict(),
            'avg_assessments_per_query': len(data) / data['query_id'].nunique(),
            'model_weights': {
                'content_weight': recommender.content_weight,
                'collaborative_weight': recommender.collaborative_weight,
                'popularity_weight': recommender.popularity_weight
            }
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving system stats: {str(e)}"
        )

if __name__ == "__main__":
    # For development only
    print("Starting SHL Assessment Recommendation Engine API...")
    print("API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )