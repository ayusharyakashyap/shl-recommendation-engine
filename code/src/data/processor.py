"""
Data preprocessing and feature extraction for SHL Assessment Recommendation Engine
"""
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from textblob import TextBlob
import json

class DataProcessor:
    """Handles data loading, cleaning, and feature extraction"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.processed_data = None
        self.skills_dict = self._create_skills_dictionary()
        self.experience_patterns = self._create_experience_patterns()
        self.duration_patterns = self._create_duration_patterns()
        
    def load_data(self) -> pd.DataFrame:
        """Load the Excel dataset"""
        self.df = pd.read_excel(self.data_path)
        print(f"Loaded {len(self.df)} records from {self.data_path}")
        return self.df
    
    def _create_skills_dictionary(self) -> Dict[str, List[str]]:
        """Create a dictionary of skill categories and related terms"""
        return {
            'programming': ['java', 'python', 'javascript', 'js', 'sql', 'html', 'css', 'selenium', 'automation'],
            'database': ['sql', 'mysql', 'postgresql', 'database', 'data', 'analytics'],
            'web_development': ['html', 'css', 'javascript', 'drupal', 'web', 'frontend', 'backend'],
            'testing': ['testing', 'qa', 'quality', 'selenium', 'automation', 'manual testing'],
            'office_tools': ['excel', 'microsoft', 'office', 'word', 'powerpoint'],
            'communication': ['english', 'communication', 'writing', 'verbal', 'presentation'],
            'leadership': ['leadership', 'management', 'team', 'manager', 'lead', 'senior'],
            'sales': ['sales', 'selling', 'business development', 'revenue'],
            'marketing': ['marketing', 'brand', 'advertising', 'seo', 'content', 'social media'],
            'analytics': ['data', 'analytics', 'analysis', 'statistics', 'reporting', 'tableau'],
            'personality': ['personality', 'cultural fit', 'behavior', 'attitude', 'soft skills']
        }
    
    def _create_experience_patterns(self) -> List[Tuple[str, str]]:
        """Create patterns to extract experience levels"""
        return [
            (r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', 'years_experience'),
            (r'(\d+)-(\d+)\s*years', 'years_range'),
            (r'new\s*graduate|fresh|entry\s*level|0-\d+\s*years', 'entry_level'),
            (r'senior|experienced|expert|\d+\+\s*years', 'senior_level'),
            (r'junior|intermediate|mid\s*level', 'mid_level')
        ]
    
    def _create_duration_patterns(self) -> List[Tuple[str, str]]:
        """Create patterns to extract assessment duration"""
        return [
            (r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:hours?|hrs?|minutes?|mins?)', 'duration_range'),
            (r'(\d+)\s*(?:hours?|hrs?)', 'hours'),
            (r'(\d+)\s*(?:minutes?|mins?)', 'minutes'),
            (r'about\s*(?:an?\s*)?hour|1\s*hour', '60_minutes'),
            (r'30-40\s*min|30\s*to\s*40\s*min', '35_minutes'),
            (r'90\s*min|1\.5\s*hour', '90_minutes')
        ]
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text using predefined categories"""
        text_lower = text.lower()
        extracted_skills = {}
        
        for category, skills in self.skills_dict.items():
            found_skills = []
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
            if found_skills:
                extracted_skills[category] = found_skills
        
        return extracted_skills
    
    def extract_experience_level(self, text: str) -> Dict[str, str]:
        """Extract experience level information"""
        text_lower = text.lower()
        experience_info = {}
        
        for pattern, exp_type in self.experience_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if exp_type == 'years_experience':
                    experience_info['years'] = int(match.group(1))
                    experience_info['level'] = 'experienced' if int(match.group(1)) >= 3 else 'junior'
                elif exp_type == 'years_range':
                    experience_info['min_years'] = int(match.group(1))
                    experience_info['max_years'] = int(match.group(2))
                    experience_info['level'] = 'experienced' if int(match.group(1)) >= 3 else 'junior'
                else:
                    experience_info['level'] = exp_type
                break
        
        return experience_info
    
    def extract_duration(self, text: str) -> Dict[str, int]:
        """Extract assessment duration in minutes"""
        text_lower = text.lower()
        duration_info = {}
        
        for pattern, duration_type in self.duration_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if duration_type == 'duration_range':
                    # Convert to minutes if needed
                    val1, val2 = int(match.group(1)), int(match.group(2))
                    if 'hour' in text_lower:
                        val1, val2 = val1 * 60, val2 * 60
                    duration_info['min_duration'] = val1
                    duration_info['max_duration'] = val2
                    duration_info['avg_duration'] = (val1 + val2) // 2
                elif duration_type == 'hours':
                    minutes = int(match.group(1)) * 60
                    duration_info['duration'] = minutes
                elif duration_type == 'minutes':
                    duration_info['duration'] = int(match.group(1))
                else:
                    # Predefined durations
                    duration_info['duration'] = int(duration_type.split('_')[0])
                break
        
        return duration_info
    
    def extract_job_role(self, text: str) -> str:
        """Extract job role/position from text"""
        text_lower = text.lower()
        
        role_patterns = {
            'developer': ['developer', 'programmer', 'engineer', 'java', 'python'],
            'analyst': ['analyst', 'data analyst', 'business analyst'],
            'sales': ['sales', 'sales representative', 'business development'],
            'marketing': ['marketing', 'brand manager', 'content writer'],
            'qa': ['qa', 'quality assurance', 'tester'],
            'manager': ['manager', 'team lead', 'supervisor', 'coo'],
            'admin': ['admin', 'assistant', 'administrative'],
            'consultant': ['consultant', 'advisor']
        }
        
        for role, keywords in role_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return role
        
        return 'general'
    
    def extract_assessment_features(self, url: str) -> Dict[str, str]:
        """Extract features from assessment URL"""
        # Extract assessment name from URL
        parts = url.split('/')
        if 'view' in parts:
            idx = parts.index('view')
            if idx + 1 < len(parts):
                assessment_name = parts[idx + 1]
            else:
                assessment_name = 'unknown'
        else:
            assessment_name = 'unknown'
        
        # Clean assessment name
        assessment_name = assessment_name.replace('-', ' ').replace('%28', '(').replace('%29', ')')
        
        # Categorize assessment type
        assessment_type = self._categorize_assessment(assessment_name)
        
        return {
            'assessment_name': assessment_name,
            'assessment_type': assessment_type,
            'assessment_url': url
        }
    
    def _categorize_assessment(self, name: str) -> str:
        """Categorize assessment based on name"""
        name_lower = name.lower()
        
        if any(skill in name_lower for skill in ['java', 'python', 'sql', 'javascript', 'css', 'html']):
            return 'technical'
        elif any(skill in name_lower for skill in ['communication', 'english', 'verbal', 'writing']):
            return 'communication'
        elif any(skill in name_lower for skill in ['personality', 'opq', 'leadership', 'behavior']):
            return 'personality'
        elif any(skill in name_lower for skill in ['numerical', 'verbal', 'reasoning', 'cognitive']):
            return 'cognitive'
        elif any(skill in name_lower for skill in ['sales', 'marketing', 'business']):
            return 'business'
        elif any(skill in name_lower for skill in ['excel', 'office', 'administrative']):
            return 'office_skills'
        else:
            return 'general'
    
    def process_dataset(self) -> pd.DataFrame:
        """Process the entire dataset and extract features"""
        if self.df is None:
            self.load_data()
        
        processed_rows = []
        
        for idx, row in self.df.iterrows():
            query = row['Query']
            url = row['Assessment_url']
            
            # Extract query features
            skills = self.extract_skills(query)
            experience = self.extract_experience_level(query)
            duration = self.extract_duration(query)
            job_role = self.extract_job_role(query)
            
            # Extract assessment features
            assessment_features = self.extract_assessment_features(url)
            
            # Combine all features
            processed_row = {
                'original_query': query,
                'query_skills': skills,
                'experience_info': experience,
                'duration_info': duration,
                'job_role': job_role,
                'assessment_name': assessment_features['assessment_name'],
                'assessment_type': assessment_features['assessment_type'],
                'assessment_url': url,
                'skills_text': ' '.join([skill for category in skills.values() for skill in category]),
                'query_length': len(query),
                'query_id': f"query_{len(set(self.df['Query'].iloc[:idx+1]))}"
            }
            
            processed_rows.append(processed_row)
        
        self.processed_data = pd.DataFrame(processed_rows)
        print(f"Processed {len(self.processed_data)} records with extracted features")
        
        return self.processed_data
    
    def save_processed_data(self, output_path: str):
        """Save processed data to JSON for easy loading"""
        if self.processed_data is not None:
            # Convert complex objects to strings for JSON serialization
            export_data = self.processed_data.copy()
            export_data['query_skills'] = export_data['query_skills'].apply(json.dumps)
            export_data['experience_info'] = export_data['experience_info'].apply(json.dumps)
            export_data['duration_info'] = export_data['duration_info'].apply(json.dumps)
            
            export_data.to_json(output_path, orient='records', indent=2)
            print(f"Saved processed data to {output_path}")
        else:
            print("No processed data to save. Run process_dataset() first.")
    
    def get_feature_summary(self) -> Dict:
        """Get summary statistics of extracted features"""
        if self.processed_data is None:
            return {}
        
        summary = {
            'total_records': len(self.processed_data),
            'unique_queries': self.processed_data['query_id'].nunique(),
            'unique_assessments': self.processed_data['assessment_name'].nunique(),
            'job_roles': self.processed_data['job_role'].value_counts().to_dict(),
            'assessment_types': self.processed_data['assessment_type'].value_counts().to_dict()
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("Gen_AI Dataset.xlsx")
    df = processor.load_data()
    processed_df = processor.process_dataset()
    
    print("\nFeature Summary:")
    summary = processor.get_feature_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save processed data
    processor.save_processed_data("src/data/processed_data.json")