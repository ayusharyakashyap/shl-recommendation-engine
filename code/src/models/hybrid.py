"""
Hybrid recommendation system combining content-based and collaborative filtering
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, 
                 content_weight: float = 0.6, 
                 collaborative_weight: float = 0.4,
                 popularity_weight: float = 0.1):
        """
        Initialize hybrid recommender
        
        Args:
            content_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering recommendations
            popularity_weight: Weight for popularity-based recommendations
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight
        
        # Normalize weights
        total_weight = content_weight + collaborative_weight + popularity_weight
        self.content_weight /= total_weight
        self.collaborative_weight /= total_weight
        self.popularity_weight /= total_weight
        
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.popularity_scores = {}
        self.is_trained = False
        
    def train(self, processed_data_path: str, 
              content_model_path: Optional[str] = None,
              cf_model_path: Optional[str] = None):
        """Train all components of the hybrid system"""
        
        print("Training Hybrid Recommendation System...")
        print("=" * 50)
        
        # Train or load content-based model
        if content_model_path and os.path.exists(content_model_path):
            print("Loading pre-trained content-based model...")
            self.content_recommender.load_model(content_model_path)
        else:
            print("Training content-based model...")
            self.content_recommender.train(processed_data_path)
            if content_model_path:
                self.content_recommender.save_model(content_model_path)
        
        # Train or load collaborative filtering model
        if cf_model_path and os.path.exists(cf_model_path):
            print("Loading pre-trained collaborative filtering model...")
            self.collaborative_recommender.load_model(cf_model_path)
            # Still need to load data for popularity scores
            self.collaborative_recommender.load_data(processed_data_path)
        else:
            print("Training collaborative filtering model...")
            self.collaborative_recommender.train(processed_data_path)
            if cf_model_path:
                self.collaborative_recommender.save_model(cf_model_path)
        
        # Calculate popularity scores
        self._calculate_popularity_scores()
        
        self.is_trained = True
        print("Hybrid system training completed!")
        
    def _calculate_popularity_scores(self):
        """Calculate popularity scores for assessments"""
        data = self.collaborative_recommender.data
        
        # Count how many times each assessment appears
        assessment_counts = data['assessment_url'].value_counts()
        
        # Calculate popularity scores (normalized)
        max_count = assessment_counts.max()
        self.popularity_scores = {}
        
        for assessment_url, count in assessment_counts.items():
            self.popularity_scores[assessment_url] = count / max_count
        
        print(f"Calculated popularity scores for {len(self.popularity_scores)} assessments")
    
    def recommend(self, query: str, top_k: int = 5, 
                  explain: bool = False) -> List[Dict]:
        """Generate hybrid recommendations"""
        if not self.is_trained:
            raise ValueError("Hybrid system must be trained before making recommendations")
        
        # Get recommendations from each component
        content_recs = self.content_recommender.recommend(query, top_k=20)
        cf_recs = self.collaborative_recommender.recommend(query, top_k=20)
        
        # Create a unified assessment dictionary
        all_assessments = {}
        
        # Add content-based recommendations
        for rec in content_recs:
            url = rec['assessment_url']
            all_assessments[url] = {
                'assessment_name': rec['assessment_name'],
                'assessment_type': rec['assessment_type'],
                'assessment_url': url,
                'content_score': rec['similarity_score'],
                'cf_score': 0.0,
                'popularity_score': self.popularity_scores.get(url, 0.0),
                'related_skills': rec.get('related_skills', [])
            }
        
        # Add/update with collaborative filtering recommendations
        for rec in cf_recs:
            url = rec['assessment_url']
            if url in all_assessments:
                all_assessments[url]['cf_score'] = rec['cf_score']
            else:
                all_assessments[url] = {
                    'assessment_name': rec['assessment_name'],
                    'assessment_type': rec['assessment_type'],
                    'assessment_url': url,
                    'content_score': 0.0,
                    'cf_score': rec['cf_score'],
                    'popularity_score': self.popularity_scores.get(url, 0.0),
                    'related_skills': []
                }
        
        # Calculate hybrid scores
        for assessment in all_assessments.values():
            # Normalize scores to 0-1 range
            content_score_norm = min(assessment['content_score'], 1.0)
            cf_score_norm = min(assessment['cf_score'] / 2.0, 1.0)  # CF scores tend to be higher
            popularity_score_norm = assessment['popularity_score']
            
            # Calculate weighted hybrid score
            hybrid_score = (
                self.content_weight * content_score_norm +
                self.collaborative_weight * cf_score_norm +
                self.popularity_weight * popularity_score_norm
            )
            
            assessment['hybrid_score'] = hybrid_score
            assessment['confidence'] = min(hybrid_score * 100, 100)
        
        # Sort by hybrid score and return top_k
        sorted_assessments = sorted(
            all_assessments.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:top_k]
        
        # Add explanation if requested
        if explain:
            for assessment in sorted_assessments:
                assessment['explanation'] = self._generate_explanation(
                    query, assessment
                )
        
        return sorted_assessments
    
    def _generate_explanation(self, query: str, assessment: Dict) -> Dict:
        """Generate explanation for why assessment was recommended"""
        explanation = {
            'reasons': [],
            'score_breakdown': {
                'content_score': assessment['content_score'],
                'collaborative_score': assessment['cf_score'],
                'popularity_score': assessment['popularity_score'],
                'hybrid_score': assessment['hybrid_score']
            }
        }
        
        # Content-based reasons
        if assessment['content_score'] > 0.5:
            if assessment['related_skills']:
                explanation['reasons'].append(
                    f"Matches required skills: {', '.join(assessment['related_skills'][:3])}"
                )
            explanation['reasons'].append(
                f"High content similarity ({assessment['content_score']:.2f})"
            )
        
        # Collaborative filtering reasons
        if assessment['cf_score'] > 1.0:
            explanation['reasons'].append(
                "Recommended based on similar hiring scenarios"
            )
        
        # Popularity reasons
        if assessment['popularity_score'] > 0.5:
            explanation['reasons'].append(
                "Popular choice among similar queries"
            )
        
        # Assessment type reasons
        type_relevance = {
            'technical': ['developer', 'engineer', 'programmer'],
            'communication': ['manager', 'sales', 'marketing'],
            'personality': ['cultural fit', 'team', 'leadership'],
            'cognitive': ['analyst', 'complex', 'problem solving']
        }
        
        query_lower = query.lower()
        assessment_type = assessment['assessment_type']
        
        if assessment_type in type_relevance:
            relevant_terms = type_relevance[assessment_type]
            if any(term in query_lower for term in relevant_terms):
                explanation['reasons'].append(
                    f"Assessment type ({assessment_type}) matches job requirements"
                )
        
        return explanation
    
    def recommend_with_diversity(self, query: str, top_k: int = 5) -> List[Dict]:
        """Generate diverse recommendations by ensuring type variety"""
        # Get more recommendations than needed
        initial_recs = self.recommend(query, top_k=top_k * 2)
        
        # Select diverse recommendations
        diverse_recs = []
        used_types = set()
        
        # First pass: ensure type diversity
        for rec in initial_recs:
            assessment_type = rec['assessment_type']
            if assessment_type not in used_types and len(diverse_recs) < top_k:
                diverse_recs.append(rec)
                used_types.add(assessment_type)
        
        # Second pass: fill remaining slots with highest scores
        for rec in initial_recs:
            if len(diverse_recs) >= top_k:
                break
            if rec not in diverse_recs:
                diverse_recs.append(rec)
        
        return diverse_recs[:top_k]
    
    def get_recommendation_explanation(self, query: str, assessment_url: str) -> Dict:
        """Get detailed explanation for a specific recommendation"""
        # Get individual explanations
        content_explanation = self.content_recommender.explain_recommendation(
            query, assessment_url
        )
        
        # Get collaborative filtering info
        cf_neighbors = self.collaborative_recommender.get_query_neighbors(query, top_k=3)
        
        explanation = {
            'query': query,
            'assessment_url': assessment_url,
            'content_based_explanation': content_explanation,
            'similar_queries': cf_neighbors,
            'popularity_info': {
                'popularity_score': self.popularity_scores.get(assessment_url, 0.0),
                'rank_by_popularity': self._get_popularity_rank(assessment_url)
            }
        }
        
        return explanation
    
    def _get_popularity_rank(self, assessment_url: str) -> int:
        """Get popularity rank of an assessment"""
        sorted_by_popularity = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (url, score) in enumerate(sorted_by_popularity, 1):
            if url == assessment_url:
                return rank
        
        return len(sorted_by_popularity)
    
    def evaluate_weights(self, test_queries: List[str], ground_truth: List[List[str]]) -> Dict:
        """Evaluate different weight combinations"""
        weight_combinations = [
            (0.7, 0.3, 0.0),
            (0.6, 0.4, 0.0),
            (0.5, 0.5, 0.0),
            (0.6, 0.3, 0.1),
            (0.5, 0.4, 0.1),
            (0.4, 0.4, 0.2)
        ]
        
        results = {}
        
        for content_w, cf_w, pop_w in weight_combinations:
            # Temporarily change weights
            old_weights = (self.content_weight, self.collaborative_weight, self.popularity_weight)
            self.content_weight = content_w
            self.collaborative_weight = cf_w
            self.popularity_weight = pop_w
            
            # Evaluate
            precision_scores = []
            for query, truth in zip(test_queries, ground_truth):
                recs = self.recommend(query, top_k=5)
                rec_urls = [rec['assessment_url'] for rec in recs]
                
                # Calculate precision@5
                hits = len(set(rec_urls) & set(truth))
                precision = hits / min(len(rec_urls), len(truth))
                precision_scores.append(precision)
            
            avg_precision = np.mean(precision_scores)
            results[f"C{content_w}_CF{cf_w}_P{pop_w}"] = avg_precision
            
            # Restore weights
            self.content_weight, self.collaborative_weight, self.popularity_weight = old_weights
        
        return results
    
    def save_model(self, model_path: str):
        """Save the hybrid model configuration"""
        import pickle
        
        model_data = {
            'content_weight': self.content_weight,
            'collaborative_weight': self.collaborative_weight,
            'popularity_weight': self.popularity_weight,
            'popularity_scores': self.popularity_scores
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Hybrid model saved to {model_path}")
    
    def load_model(self, model_path: str, 
                   content_model_path: str, 
                   cf_model_path: str,
                   processed_data_path: str = "src/data/processed_data.json"):
        """Load the hybrid model"""
        import pickle
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.content_weight = model_data['content_weight']
        self.collaborative_weight = model_data['collaborative_weight']
        self.popularity_weight = model_data['popularity_weight']
        self.popularity_scores = model_data['popularity_scores']
        
        # Load component models
        self.content_recommender.load_model(content_model_path)
        self.collaborative_recommender.load_model(cf_model_path)
        
        # Load data for collaborative recommender
        self.collaborative_recommender.load_data(processed_data_path)
        
        self.is_trained = True
        print(f"Hybrid model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    hybrid_recommender = HybridRecommender(
        content_weight=0.6,
        collaborative_weight=0.4,
        popularity_weight=0.1
    )
    
    # Train the hybrid system
    hybrid_recommender.train(
        processed_data_path="src/data/processed_data.json",
        content_model_path="src/models/content_based_model.pkl",
        cf_model_path="src/models/collaborative_filtering_model.pkl"
    )
    
    # Test recommendations
    test_queries = [
        "I need a Python developer with 3 years experience for a 60-minute assessment",
        "Looking for sales representatives, new graduates, 1-hour assessment",
        "Need to assess marketing manager candidates with strong communication skills"
    ]
    
    print("\n" + "="*80)
    print("HYBRID RECOMMENDATION SYSTEM RESULTS")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 60)
        
        # Get regular recommendations
        recommendations = hybrid_recommender.recommend(query, top_k=5, explain=True)
        
        for j, rec in enumerate(recommendations, 1):
            print(f"{j}. {rec['assessment_name'].title()}")
            print(f"   Type: {rec['assessment_type']}")
            print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
            print(f"   Confidence: {rec['confidence']:.1f}%")
            if 'explanation' in rec and rec['explanation']['reasons']:
                print(f"   Reasons: {rec['explanation']['reasons'][0]}")
            print()
    
    # Save the hybrid model
    hybrid_recommender.save_model("src/models/hybrid_model.pkl")
    
    print("Hybrid recommendation system demo completed!")