"""
Evaluation metrics and validation for SHL Assessment Recommendation Engine
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid import HybridRecommender
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender

class RecommendationEvaluator:
    """Comprehensive evaluation system for recommendation models"""
    
    def __init__(self):
        self.data = None
        self.query_groups = None
        
    def load_data(self, processed_data_path: str):
        """Load processed data for evaluation"""
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
        
        self.data = pd.DataFrame(data)
        
        # Group by query for evaluation
        self.query_groups = self.data.groupby('query_id')
        print(f"Loaded {len(self.data)} records for evaluation")
        print(f"Found {len(self.query_groups)} unique queries")
        
    def precision_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        if len(recommended_k) == 0:
            return 0.0
        
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(recommended_k)
    
    def recall_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if len(relevant) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = len([item for item in recommended_k if item in relevant_set])
        return hits / len(relevant)
    
    def f1_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate F1@K"""
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended: List[str], relevant: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
            """Calculate DCG@K"""
            relevance_scores = relevance_scores[:k]
            dcg = 0.0
            for i, score in enumerate(relevance_scores):
                dcg += (2**score - 1) / np.log2(i + 2)
            return dcg
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        relevance_scores = [1.0 if item in relevant_set else 0.0 for item in recommended_k]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(relevant), k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_average_precision(self, recommended: List[str], relevant: List[str]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        if len(relevant) == 0:
            return 0.0
        
        relevant_set = set(relevant)
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended):
            if item in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        if relevant_count == 0:
            return 0.0
        
        return precision_sum / len(relevant)
    
    def coverage(self, all_recommendations: List[List[str]], 
                 all_assessments: List[str]) -> float:
        """Calculate catalog coverage"""
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / len(all_assessments)
    
    def diversity(self, recommendations: List[str], 
                  assessment_types: Dict[str, str]) -> float:
        """Calculate type diversity of recommendations"""
        if len(recommendations) == 0:
            return 0.0
        
        types = [assessment_types.get(url, 'unknown') for url in recommendations]
        unique_types = set(types)
        
        return len(unique_types) / len(recommendations)
    
    def evaluate_model(self, model, test_queries: List[str], 
                      ground_truth: List[List[str]], 
                      k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate a single model"""
        results = {f'precision@{k}': [] for k in k_values}
        results.update({f'recall@{k}': [] for k in k_values})
        results.update({f'f1@{k}': [] for k in k_values})
        results.update({f'ndcg@{k}': [] for k in k_values})
        results['map'] = []
        results['diversity'] = []
        
        # Get assessment types for diversity calculation
        assessment_types = {}
        for _, row in self.data.iterrows():
            assessment_types[row['assessment_url']] = row['assessment_type']
        
        all_recommendations = []
        
        for query, truth in zip(test_queries, ground_truth):
            try:
                # Get recommendations
                if hasattr(model, 'recommend'):
                    recs = model.recommend(query, top_k=max(k_values))
                    rec_urls = [rec['assessment_url'] for rec in recs]
                else:
                    # Fallback for different model interfaces
                    rec_urls = []
                
                all_recommendations.append(rec_urls)
                
                # Calculate metrics for each k
                for k in k_values:
                    results[f'precision@{k}'].append(
                        self.precision_at_k(rec_urls, truth, k)
                    )
                    results[f'recall@{k}'].append(
                        self.recall_at_k(rec_urls, truth, k)
                    )
                    results[f'f1@{k}'].append(
                        self.f1_at_k(rec_urls, truth, k)
                    )
                    results[f'ndcg@{k}'].append(
                        self.ndcg_at_k(rec_urls, truth, k)
                    )
                
                # Calculate MAP
                results['map'].append(
                    self.mean_average_precision(rec_urls, truth)
                )
                
                # Calculate diversity
                results['diversity'].append(
                    self.diversity(rec_urls[:5], assessment_types)
                )
                
            except Exception as e:
                print(f"Error evaluating query '{query[:50]}...': {e}")
                # Add zeros for failed queries
                for k in k_values:
                    results[f'precision@{k}'].append(0.0)
                    results[f'recall@{k}'].append(0.0)
                    results[f'f1@{k}'].append(0.0)
                    results[f'ndcg@{k}'].append(0.0)
                results['map'].append(0.0)
                results['diversity'].append(0.0)
        
        # Calculate averages
        avg_results = {}
        for metric, values in results.items():
            avg_results[f'avg_{metric}'] = np.mean(values)
            avg_results[f'std_{metric}'] = np.std(values)
        
        # Calculate coverage
        all_assessment_urls = self.data['assessment_url'].unique().tolist()
        avg_results['coverage'] = self.coverage(all_recommendations, all_assessment_urls)
        
        return avg_results
    
    def cross_validate_model(self, model_class, model_params: Dict, 
                           n_splits: int = 5) -> Dict:
        """Perform cross-validation on a model"""
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        # Get unique queries
        unique_queries = self.data['query_id'].unique()
        
        # Prepare query texts and ground truth
        query_texts = []
        ground_truths = []
        
        for query_id in unique_queries:
            query_data = self.data[self.data['query_id'] == query_id]
            query_text = query_data.iloc[0]['original_query']
            truth_urls = query_data['assessment_url'].tolist()
            
            query_texts.append(query_text)
            ground_truths.append(truth_urls)
        
        # Perform k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(unique_queries)):
            print(f"Evaluating fold {fold + 1}/{n_splits}...")
            
            # Split data
            train_queries = [unique_queries[i] for i in train_idx]
            test_queries = [query_texts[i] for i in test_idx]
            test_truths = [ground_truths[i] for i in test_idx]
            
            # Create training data for this fold
            train_data = self.data[self.data['query_id'].isin(train_queries)]
            
            # Save temporary training data
            train_data_path = f"temp_train_fold_{fold}.json"
            train_data.to_json(train_data_path, orient='records')
            
            try:
                # Train model
                model = model_class(**model_params)
                model.train(train_data_path)
                
                # Evaluate on test set
                fold_result = self.evaluate_model(model, test_queries, test_truths)
                fold_results.append(fold_result)
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {e}")
                fold_results.append({})
            
            finally:
                # Clean up temporary file
                if os.path.exists(train_data_path):
                    os.remove(train_data_path)
        
        # Aggregate results across folds
        if not fold_results or all(not result for result in fold_results):
            return {}
        
        aggregated_results = {}
        
        # Get all metrics from successful folds
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())
        
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in fold_results if result]
            aggregated_results[f'cv_{metric}'] = np.mean(values)
            aggregated_results[f'cv_std_{metric}'] = np.std(values)
        
        return aggregated_results
    
    def compare_models(self, models: Dict[str, object], 
                      test_queries: List[str], 
                      ground_truth: List[List[str]]) -> pd.DataFrame:
        """Compare multiple models"""
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            model_results = self.evaluate_model(model, test_queries, ground_truth)
            results[model_name] = model_results
        
        # Convert to DataFrame for easy comparison
        comparison_df = pd.DataFrame(results).T
        return comparison_df
    
    def generate_evaluation_report(self, results: Dict, 
                                 output_path: str = "evaluation_report.md"):
        """Generate a comprehensive evaluation report"""
        report = "# SHL Assessment Recommendation Engine - Evaluation Report\n\n"
        
        report += "## Model Performance Summary\n\n"
        
        # Key metrics table
        key_metrics = ['avg_precision@5', 'avg_recall@5', 'avg_f1@5', 
                      'avg_ndcg@5', 'avg_map', 'coverage', 'avg_diversity']
        
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        for metric in key_metrics:
            if metric in results:
                report += f"| {metric.replace('avg_', '').replace('@', '@')} | {results[metric]:.4f} |\n"
        
        report += "\n## Detailed Metrics\n\n"
        
        # Precision@K
        report += "### Precision@K\n"
        for k in [1, 3, 5, 10]:
            metric = f'avg_precision@{k}'
            if metric in results:
                report += f"- Precision@{k}: {results[metric]:.4f} ± {results.get(f'std_precision@{k}', 0):.4f}\n"
        
        # Recall@K
        report += "\n### Recall@K\n"
        for k in [1, 3, 5, 10]:
            metric = f'avg_recall@{k}'
            if metric in results:
                report += f"- Recall@{k}: {results[metric]:.4f} ± {results.get(f'std_recall@{k}', 0):.4f}\n"
        
        # NDCG@K
        report += "\n### NDCG@K\n"
        for k in [1, 3, 5, 10]:
            metric = f'avg_ndcg@{k}'
            if metric in results:
                report += f"- NDCG@{k}: {results[metric]:.4f} ± {results.get(f'std_ndcg@{k}', 0):.4f}\n"
        
        # Other metrics
        report += "\n### Other Metrics\n"
        report += f"- Mean Average Precision: {results.get('avg_map', 0):.4f}\n"
        report += f"- Coverage: {results.get('coverage', 0):.4f}\n"
        report += f"- Diversity: {results.get('avg_diversity', 0):.4f}\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Evaluation report saved to {output_path}")
        return report

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of all models"""
    evaluator = RecommendationEvaluator()
    evaluator.load_data("src/data/processed_data.json")
    
    # Prepare test data
    unique_queries = evaluator.data['query_id'].unique()
    test_queries = []
    ground_truths = []
    
    for query_id in unique_queries:
        query_data = evaluator.data[evaluator.data['query_id'] == query_id]
        query_text = query_data.iloc[0]['original_query']
        truth_urls = query_data['assessment_url'].tolist()
        
        test_queries.append(query_text)
        ground_truths.append(truth_urls)
    
    print("=== SHL Assessment Recommendation Engine Evaluation ===")
    print(f"Test queries: {len(test_queries)}")
    print(f"Average ground truth size: {np.mean([len(truth) for truth in ground_truths]):.1f}")
    
    # Load and evaluate hybrid model
    try:
        hybrid_model = HybridRecommender()
        hybrid_model.load_model(
            model_path="src/models/hybrid_model.pkl",
            content_model_path="src/models/content_based_model.pkl",
            cf_model_path="src/models/collaborative_filtering_model.pkl"
        )
        
        print("\nEvaluating Hybrid Model...")
        hybrid_results = evaluator.evaluate_model(hybrid_model, test_queries, ground_truths)
        
        print("\nHybrid Model Results:")
        print("-" * 40)
        for metric, value in hybrid_results.items():
            if 'avg_' in metric:
                print(f"{metric}: {value:.4f}")
        
        # Generate report
        evaluator.generate_evaluation_report(
            hybrid_results, 
            "hybrid_model_evaluation_report.md"
        )
        
    except Exception as e:
        print(f"Error evaluating hybrid model: {e}")

if __name__ == "__main__":
    run_comprehensive_evaluation()