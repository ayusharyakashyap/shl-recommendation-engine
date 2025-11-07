"""
Content-based recommendation model for SHL Assessment Recommendation Engine
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle

class ContentBasedRecommender:
    """Content-based recommendation system using TF-IDF and feature matching"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.assessment_features = None
        self.query_features = None
        self.assessment_matrix = None
        self.scaler = StandardScaler()
        self.assessments_df = None
        self.is_trained = False
        
    def load_data(self, processed_data_path: str):
        """Load processed data"""
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
        
        # Convert back to DataFrame
        df = pd.DataFrame(data)
        
        # Parse JSON strings back to dictionaries
        df['query_skills'] = df['query_skills'].apply(json.loads)
        df['experience_info'] = df['experience_info'].apply(json.loads)
        df['duration_info'] = df['duration_info'].apply(json.loads)
        
        self.data = df
        print(f"Loaded {len(df)} processed records")
        return df
    
    def create_text_features(self):
        """Create text-based features for TF-IDF"""
        # Create comprehensive text representation for each assessment
        assessment_texts = []
        assessment_info = []
        
        # Group by assessment to get unique assessments
        unique_assessments = self.data.groupby('assessment_url').first().reset_index()
        
        for _, row in unique_assessments.iterrows():
            # Combine assessment name and type for text features
            text_features = f"{row['assessment_name']} {row['assessment_type']}"
            
            # Add related skills if this assessment appears with certain skills
            related_queries = self.data[self.data['assessment_url'] == row['assessment_url']]
            all_skills = []
            for _, query_row in related_queries.iterrows():
                skills = query_row['query_skills']
                for category, skill_list in skills.items():
                    all_skills.extend(skill_list)
            
            # Add unique skills to text
            unique_skills = list(set(all_skills))
            text_features += " " + " ".join(unique_skills)
            
            assessment_texts.append(text_features)
            assessment_info.append({
                'assessment_name': row['assessment_name'],
                'assessment_type': row['assessment_type'],
                'assessment_url': row['assessment_url'],
                'related_skills': unique_skills
            })
        
        self.assessment_texts = assessment_texts
        self.assessments_df = pd.DataFrame(assessment_info)
        print(f"Created text features for {len(assessment_texts)} unique assessments")
        
    def create_numerical_features(self):
        """Create numerical features for assessments"""
        numerical_features = []
        
        for _, assessment in self.assessments_df.iterrows():
            # Get all queries that use this assessment
            related_queries = self.data[self.data['assessment_url'] == assessment['assessment_url']]
            
            # Extract numerical features based on associated queries
            features = {
                'avg_query_length': related_queries['query_length'].mean(),
                'num_related_queries': len(related_queries),
                'has_duration_constraint': any(related_queries['duration_info'].apply(lambda x: len(x) > 0)),
                'requires_experience': any(related_queries['experience_info'].apply(lambda x: len(x) > 0)),
                'num_skill_categories': len(set([cat for _, query in related_queries.iterrows() 
                                               for cat in query['query_skills'].keys()]))
            }
            
            # Assessment type encoding
            type_encoding = {
                'technical': 1.0, 'communication': 0.8, 'personality': 0.6,
                'cognitive': 0.7, 'business': 0.5, 'office_skills': 0.4, 'general': 0.2
            }
            features['type_score'] = type_encoding.get(assessment['assessment_type'], 0.0)
            
            numerical_features.append(list(features.values()))
        
        self.numerical_features = np.array(numerical_features)
        self.feature_names = list(features.keys())
        print(f"Created {len(self.feature_names)} numerical features")
        
    def train(self, processed_data_path: str):
        """Train the content-based recommender"""
        # Load data
        self.load_data(processed_data_path)
        
        # Create features
        self.create_text_features()
        self.create_numerical_features()
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.assessment_texts)
        
        # Scale numerical features
        self.scaled_numerical_features = self.scaler.fit_transform(self.numerical_features)
        
        # Combine TF-IDF and numerical features
        self.combined_features = np.hstack([
            self.tfidf_matrix.toarray(),
            self.scaled_numerical_features
        ])
        
        self.is_trained = True
        print(f"Training completed. Feature matrix shape: {self.combined_features.shape}")
        
    def process_query(self, query: str) -> Dict:
        """Process a new query to extract features"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.processor import DataProcessor
        
        # Create temporary processor
        processor = DataProcessor("")
        
        # Extract features
        skills = processor.extract_skills(query)
        experience = processor.extract_experience_level(query)
        duration = processor.extract_duration(query)
        job_role = processor.extract_job_role(query)
        
        return {
            'query': query,
            'skills': skills,
            'experience': experience,
            'duration': duration,
            'job_role': job_role,
            'skills_text': ' '.join([skill for category in skills.values() for skill in category]),
            'query_length': len(query)
        }
    
    def create_query_features(self, query_info: Dict) -> np.ndarray:
        """Create feature vector for a new query"""
        # Create text for TF-IDF
        query_text = f"{query_info['skills_text']} {query_info['job_role']}"
        
        # Transform using trained TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query_text]).toarray()
        
        # Create numerical features
        numerical_features = [
            query_info['query_length'],
            1,  # num_related_queries (default for new query)
            len(query_info['duration']) > 0,
            len(query_info['experience']) > 0,
            len(query_info['skills']),
            0.5  # type_score (default for query)
        ]
        
        # Scale numerical features
        scaled_numerical = self.scaler.transform([numerical_features])
        
        # Combine features
        combined_query_features = np.hstack([query_tfidf, scaled_numerical])
        
        return combined_query_features
    
    def recommend(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Dict]:
        """Generate recommendations for a query"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Process query
        query_info = self.process_query(query)
        
        # Create query features
        query_features = self.create_query_features(query_info)
        
        # Calculate similarities
        similarities = cosine_similarity(query_features, self.combined_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                rec = {
                    'assessment_name': self.assessments_df.iloc[idx]['assessment_name'],
                    'assessment_type': self.assessments_df.iloc[idx]['assessment_type'],
                    'assessment_url': self.assessments_df.iloc[idx]['assessment_url'],
                    'similarity_score': float(similarities[idx]),
                    'related_skills': self.assessments_df.iloc[idx]['related_skills'],
                    'confidence': min(similarities[idx] * 100, 100)
                }
                recommendations.append(rec)
        
        return recommendations
    
    def explain_recommendation(self, query: str, assessment_url: str) -> Dict:
        """Explain why a specific assessment was recommended"""
        query_info = self.process_query(query)
        
        # Find the assessment
        assessment_idx = self.assessments_df[
            self.assessments_df['assessment_url'] == assessment_url
        ].index
        
        if len(assessment_idx) == 0:
            return {"error": "Assessment not found"}
        
        assessment_idx = assessment_idx[0]
        assessment = self.assessments_df.iloc[assessment_idx]
        
        # Calculate feature contributions
        query_features = self.create_query_features(query_info)
        assessment_features = self.combined_features[assessment_idx:assessment_idx+1]
        
        similarity = cosine_similarity(query_features, assessment_features)[0][0]
        
        explanation = {
            'assessment_name': assessment['assessment_name'],
            'assessment_type': assessment['assessment_type'],
            'similarity_score': float(similarity),
            'query_skills': query_info['skills'],
            'assessment_skills': assessment['related_skills'],
            'skill_overlap': list(set(query_info['skills_text'].split()) & 
                                set(assessment['related_skills'])),
            'job_role_match': query_info['job_role'],
            'assessment_category': assessment['assessment_type']
        }
        
        return explanation
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'assessments_df': self.assessments_df,
            'combined_features': self.combined_features,
            'feature_names': self.feature_names,
            'assessment_texts': self.assessment_texts
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.scaler = model_data['scaler']
        self.assessments_df = model_data['assessments_df']
        self.combined_features = model_data['combined_features']
        self.feature_names = model_data['feature_names']
        self.assessment_texts = model_data['assessment_texts']
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    recommender = ContentBasedRecommender()
    
    # Train the model
    recommender.train("src/data/processed_data.json")
    
    # Test recommendations
    test_query = "I need a Python developer with 3 years experience for a 1-hour assessment"
    recommendations = recommender.recommend(test_query, top_k=5)
    
    print(f"\nRecommendations for: '{test_query}'")
    print("-" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['assessment_name'].title()}")
        print(f"   Type: {rec['assessment_type']}")
        print(f"   Confidence: {rec['confidence']:.1f}%")
        print(f"   Skills: {rec['related_skills'][:3]}")
        print()
    
    # Save the model
    recommender.save_model("src/models/content_based_model.pkl")