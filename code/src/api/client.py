"""
Test client for SHL Assessment Recommendation Engine API
"""
import requests
import json
import time

class SHLAPIClient:
    """Client for interacting with SHL Recommendation API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_recommendations(self, query: str, top_k: int = 5, 
                          include_explanation: bool = False, 
                          diverse: bool = False):
        """Get assessment recommendations"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "include_explanation": include_explanation,
                "diverse": diverse
            }
            
            response = requests.post(
                f"{self.base_url}/recommend",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def explain_recommendation(self, query: str, assessment_url: str):
        """Get explanation for a specific recommendation"""
        try:
            payload = {
                "query": query,
                "assessment_url": assessment_url
            }
            
            response = requests.post(
                f"{self.base_url}/explain",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_popular_assessments(self, top_k: int = 10):
        """Get most popular assessments"""
        try:
            response = requests.get(f"{self.base_url}/assessments/popular?top_k={top_k}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def demo_api():
    """Demo the API functionality"""
    client = SHLAPIClient()
    
    print("🚀 SHL Assessment Recommendation Engine API Demo")
    print("=" * 60)
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if "error" in health or not health.get("model_loaded", False):
        print("❌ API not ready. Please start the API server first.")
        return
    
    # Test queries
    test_queries = [
        "I need Java developers with 3+ years experience for a 45-minute assessment",
        "Looking for sales representatives, entry level, 1-hour test",
        "Need to assess marketing manager with strong communication skills"
    ]
    
    print("\n2. Recommendation Tests:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 50)
        
        # Get recommendations
        result = client.get_recommendations(
            query=query,
            top_k=3,
            include_explanation=True
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"Processing time: {result['processing_time_ms']:.1f}ms")
        print(f"Total found: {result['total_found']}")
        print("\nRecommendations:")
        
        for j, rec in enumerate(result['recommendations'], 1):
            print(f"{j}. {rec['assessment_name'].title()}")
            print(f"   Type: {rec['assessment_type']}")
            print(f"   Confidence: {rec['confidence']:.1f}%")
            if rec.get('explanation') and rec['explanation'].get('reasons'):
                print(f"   Reason: {rec['explanation']['reasons'][0]}")
            print()
    
    # Popular assessments
    print("\n3. Popular Assessments:")
    print("-" * 30)
    popular = client.get_popular_assessments(top_k=5)
    
    if "error" not in popular:
        for i, assessment in enumerate(popular['popular_assessments'], 1):
            print(f"{i}. {assessment['assessment_name'].title()}")
            print(f"   Type: {assessment['assessment_type']}")
            print(f"   Usage: {assessment['usage_count']} times")
            print()
    
    # System stats
    print("\n4. System Statistics:")
    print("-" * 30)
    stats = client.get_system_stats()
    
    if "error" not in stats:
        print(f"Total Records: {stats['total_records']}")
        print(f"Unique Queries: {stats['unique_queries']}")
        print(f"Unique Assessments: {stats['unique_assessments']}")
        print(f"Avg Assessments/Query: {stats['avg_assessments_per_query']:.1f}")
        print("\nModel Weights:")
        for weight, value in stats['model_weights'].items():
            print(f"  {weight}: {value:.2f}")
    
    print("\n✅ API Demo completed!")

if __name__ == "__main__":
    demo_api()