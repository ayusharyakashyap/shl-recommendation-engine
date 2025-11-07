"""
Collaborative filtering recommendation model for SHL Assessment Recommendation Engine
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle

class CollaborativeFilteringRecommender:
    """Collaborative filtering based on query-assessment interaction patterns"""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.nmf_model = None
        self.query_features = None
        self.assessment_features = None
        self.query_to_idx = {}
        self.assessment_to_idx = {}
        self.idx_to_query = {}
        self.idx_to_assessment = {}
        self.interaction_matrix = None
        self.is_trained = False
        
    def load_data(self, processed_data_path: str):
        """Load processed data"""
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
        
        self.data = pd.DataFrame(data)
        print(f"Loaded {len(self.data)} records for collaborative filtering")
        return self.data
    
    def create_interaction_matrix(self):
        """Create query-assessment interaction matrix"""
        # Get unique queries and assessments
        unique_queries = self.data['query_id'].unique()
        unique_assessments = self.data['assessment_url'].unique()
        
        # Create mappings
        self.query_to_idx = {query: idx for idx, query in enumerate(unique_queries)}
        self.assessment_to_idx = {assessment: idx for idx, assessment in enumerate(unique_assessments)}
        self.idx_to_query = {idx: query for query, idx in self.query_to_idx.items()}
        self.idx_to_assessment = {idx: assessment for assessment, idx in self.assessment_to_idx.items()}
        
        # Create interaction matrix (queries x assessments)
        n_queries = len(unique_queries)
        n_assessments = len(unique_assessments)
        
        interaction_matrix = np.zeros((n_queries, n_assessments))
        
        # Fill interaction matrix
        for _, row in self.data.iterrows():
            query_idx = self.query_to_idx[row['query_id']]
            assessment_idx = self.assessment_to_idx[row['assessment_url']]
            interaction_matrix[query_idx, assessment_idx] = 1.0
        
        self.interaction_matrix = interaction_matrix
        print(f"Created interaction matrix: {interaction_matrix.shape}")
        print(f"Sparsity: {(1 - np.count_nonzero(interaction_matrix) / interaction_matrix.size) * 100:.1f}%")
        
        return interaction_matrix
    
    def create_weighted_interaction_matrix(self):
        """Create weighted interaction matrix based on assessment characteristics"""
        # Start with basic interaction matrix
        weighted_matrix = self.interaction_matrix.copy()
        
        # Add weights based on assessment type popularity and query complexity
        for query_id, query_idx in self.query_to_idx.items():
            # Get query data
            query_data = self.data[self.data['query_id'] == query_id].iloc[0]
            
            # Parse query features
            try:
                skills = json.loads(query_data['query_skills'])
                duration_info = json.loads(query_data['duration_info'])
                experience_info = json.loads(query_data['experience_info'])
            except:
                skills = {}
                duration_info = {}
                experience_info = {}
            
            # Calculate query complexity weight
            complexity_weight = 1.0
            complexity_weight += len(skills) * 0.1  # More skills = higher weight
            complexity_weight += (1.0 if duration_info else 0.0) * 0.1  # Duration specified
            complexity_weight += (1.0 if experience_info else 0.0) * 0.1  # Experience specified
            
            # Apply complexity weight to all interactions for this query
            weighted_matrix[query_idx, :] *= complexity_weight
        
        # Add assessment type weights
        for assessment_url, assessment_idx in self.assessment_to_idx.items():
            # Get assessment data
            assessment_data = self.data[self.data['assessment_url'] == assessment_url].iloc[0]
            assessment_type = assessment_data['assessment_type']
            
            # Weight based on assessment type
            type_weights = {
                'technical': 1.2,      # Higher weight for technical assessments
                'communication': 1.1,   # Slightly higher for communication
                'personality': 1.0,     # Baseline
                'cognitive': 1.1,       # Slightly higher for cognitive
                'business': 1.0,        # Baseline
                'office_skills': 0.9,   # Slightly lower
                'general': 0.8          # Lower weight for general assessments
            }
            
            type_weight = type_weights.get(assessment_type, 1.0)
            weighted_matrix[:, assessment_idx] *= type_weight
        
        self.weighted_interaction_matrix = weighted_matrix
        print("Created weighted interaction matrix")
        
        return weighted_matrix
    
    def train(self, processed_data_path: str):
        """Train the collaborative filtering model"""
        # Load data
        self.load_data(processed_data_path)
        
        # Create interaction matrices
        self.create_interaction_matrix()
        self.create_weighted_interaction_matrix()
        
        # Train NMF model
        self.nmf_model = NMF(
            n_components=self.n_components,
            init='random',
            random_state=42,
            max_iter=200,
            alpha_W=0.02,
            alpha_H=0.02,
            l1_ratio=0.01
        )
        
        # Fit the model
        self.query_features = self.nmf_model.fit_transform(self.weighted_interaction_matrix)
        self.assessment_features = self.nmf_model.components_.T
        
        # Calculate reconstruction for evaluation
        reconstruction = np.dot(self.query_features, self.assessment_features.T)
        reconstruction_error = np.mean((self.weighted_interaction_matrix - reconstruction) ** 2)
        
        self.is_trained = True
        print(f"NMF training completed. Reconstruction error: {reconstruction_error:.4f}")
        print(f"Query features shape: {self.query_features.shape}")
        print(f"Assessment features shape: {self.assessment_features.shape}")
    
    def find_similar_queries(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar queries based on NMF features"""
        # Calculate similarity with all existing queries
        similarities = cosine_similarity([query_vector], self.query_features)[0]
        
        # Get top similar queries
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_queries = []
        for idx in top_indices:
            query_id = self.idx_to_query[idx]
            similarity = similarities[idx]
            similar_queries.append((query_id, similarity))
        
        return similar_queries
    
    def get_query_vector(self, query_text: str) -> np.ndarray:
        """Get query vector for a new query by finding similar existing queries"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.processor import DataProcessor
        
        # Process the new query
        processor = DataProcessor("")
        query_info = {
            'skills': processor.extract_skills(query_text),
            'experience': processor.extract_experience_level(query_text),
            'duration': processor.extract_duration(query_text),
            'job_role': processor.extract_job_role(query_text)
        }
        
        # Find most similar existing query based on features
        best_similarity = 0
        best_query_vector = None
        
        for query_id, query_idx in self.query_to_idx.items():
            # Get existing query data
            existing_query_data = self.data[self.data['query_id'] == query_id].iloc[0]
            
            # Calculate feature similarity
            similarity = self._calculate_query_similarity(query_info, existing_query_data)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_query_vector = self.query_features[query_idx]
        
        # If no good match found, use average query vector
        if best_query_vector is None or best_similarity < 0.3:
            best_query_vector = np.mean(self.query_features, axis=0)
        
        return best_query_vector
    
    def _calculate_query_similarity(self, query_info: Dict, existing_query_data: pd.Series) -> float:
        """Calculate similarity between new query and existing query"""
        try:
            existing_skills = json.loads(existing_query_data['query_skills'])
            existing_experience = json.loads(existing_query_data['experience_info'])
            existing_duration = json.loads(existing_query_data['duration_info'])
        except:
            existing_skills = {}
            existing_experience = {}
            existing_duration = {}
        
        similarity = 0.0
        
        # Job role similarity
        if query_info['job_role'] == existing_query_data['job_role']:
            similarity += 0.4
        
        # Skills similarity
        new_skills_set = set([skill for category in query_info['skills'].values() for skill in category])
        existing_skills_set = set([skill for category in existing_skills.values() for skill in category])
        
        if new_skills_set and existing_skills_set:
            skill_overlap = len(new_skills_set & existing_skills_set) / len(new_skills_set | existing_skills_set)
            similarity += skill_overlap * 0.4
        
        # Duration similarity
        has_duration_new = len(query_info['duration']) > 0
        has_duration_existing = len(existing_duration) > 0
        if has_duration_new == has_duration_existing:
            similarity += 0.1
        
        # Experience similarity
        has_exp_new = len(query_info['experience']) > 0
        has_exp_existing = len(existing_experience) > 0
        if has_exp_new == has_exp_existing:
            similarity += 0.1
        
        return similarity
    
    def recommend(self, query: str, top_k: int = 5) -> List[Dict]:
        """Generate recommendations using collaborative filtering"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Get query vector
        query_vector = self.get_query_vector(query)
        
        # Calculate scores for all assessments
        assessment_scores = np.dot(query_vector, self.assessment_features.T)
        
        # Get top recommendations
        top_indices = np.argsort(assessment_scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            assessment_url = self.idx_to_assessment[idx]
            
            # Get assessment details
            assessment_data = self.data[self.data['assessment_url'] == assessment_url].iloc[0]
            
            rec = {
                'assessment_name': assessment_data['assessment_name'],
                'assessment_type': assessment_data['assessment_type'],
                'assessment_url': assessment_url,
                'cf_score': float(assessment_scores[idx]),
                'confidence': min(assessment_scores[idx] * 50, 100)  # Scale to 0-100
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_query_neighbors(self, query: str, top_k: int = 3) -> List[Dict]:
        """Get similar queries and their assessment preferences"""
        query_vector = self.get_query_vector(query)
        similar_queries = self.find_similar_queries(query_vector, top_k)
        
        neighbors = []
        for query_id, similarity in similar_queries:
            # Get original query text and its assessments
            query_data = self.data[self.data['query_id'] == query_id]
            
            neighbor = {
                'query_id': query_id,
                'similarity': float(similarity),
                'original_query': query_data.iloc[0]['original_query'],
                'assessments': query_data['assessment_name'].tolist()
            }
            neighbors.append(neighbor)
        
        return neighbors
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'nmf_model': self.nmf_model,
            'query_features': self.query_features,
            'assessment_features': self.assessment_features,
            'query_to_idx': self.query_to_idx,
            'assessment_to_idx': self.assessment_to_idx,
            'idx_to_query': self.idx_to_query,
            'idx_to_assessment': self.idx_to_assessment,
            'interaction_matrix': self.interaction_matrix,
            'weighted_interaction_matrix': self.weighted_interaction_matrix,
            'n_components': self.n_components
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Collaborative filtering model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.nmf_model = model_data['nmf_model']
        self.query_features = model_data['query_features']
        self.assessment_features = model_data['assessment_features']
        self.query_to_idx = model_data['query_to_idx']
        self.assessment_to_idx = model_data['assessment_to_idx']
        self.idx_to_query = model_data['idx_to_query']
        self.idx_to_assessment = model_data['idx_to_assessment']
        self.interaction_matrix = model_data['interaction_matrix']
        self.weighted_interaction_matrix = model_data['weighted_interaction_matrix']
        self.n_components = model_data['n_components']
        self.is_trained = True
        
        print(f"Collaborative filtering model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    cf_recommender = CollaborativeFilteringRecommender(n_components=8)
    
    # Train the model
    cf_recommender.train("src/data/processed_data.json")
    
    # Test recommendations
    test_query = "I need a Java developer with 2 years experience for a 45-minute assessment"
    recommendations = cf_recommender.recommend(test_query, top_k=5)
    
    print(f"\nCollaborative Filtering Recommendations for: '{test_query}'")
    print("-" * 70)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['assessment_name'].title()}")
        print(f"   Type: {rec['assessment_type']}")
        print(f"   CF Score: {rec['cf_score']:.3f}")
        print(f"   Confidence: {rec['confidence']:.1f}%")
        print()
    
    # Show similar queries
    neighbors = cf_recommender.get_query_neighbors(test_query, top_k=3)
    print("Similar queries:")
    for neighbor in neighbors:
        print(f"- Similarity: {neighbor['similarity']:.3f}")
        print(f"  Query: {neighbor['original_query'][:100]}...")
        print()
    
    # Save the model
    cf_recommender.save_model("src/models/collaborative_filtering_model.pkl")