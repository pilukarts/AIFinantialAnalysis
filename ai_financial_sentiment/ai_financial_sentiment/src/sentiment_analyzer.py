"""
AI Financial Sentiment Analysis Module
Using BERT-based transformers for financial news sentiment
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class FinancialSentimentAnalyzer:
    """
    Advanced sentiment analysis for financial news using BERT
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize the sentiment analyzer with pre-trained model"""
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # TF-IDF for financial keyword extraction
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.sentiment_labels = {
            0: 'negative',
            1: 'neutral', 
            2: 'positive'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess financial text"""
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Financial specific preprocessing
        financial_abbreviations = {
            'q1': 'quarter 1',
            'q2': 'quarter 2', 
            'q3': 'quarter 3',
            'q4': 'quarter 4',
            'yoy': 'year over year',
            'qoq': 'quarter over quarter'
        }
        
        for abbr, full_form in financial_abbreviations.items():
            text = text.replace(abbr, full_form)
            
        return text
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries with sentiment scores and labels
        """
        results = []
        
        for text in texts:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                clean_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            # Calculate confidence
            confidence = probabilities[0][predicted_class].item()
            
            # Get all probabilities for detailed analysis
            probs = probabilities[0].cpu().numpy()
            
            result = {
                'text': text,
                'clean_text': clean_text,
                'sentiment': self.sentiment_labels[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'negative': probs[0],
                    'neutral': probs[1], 
                    'positive': probs[2]
                }
            }
            
            results.append(result)
            
        return results
    
    def analyze_financial_correlation(self, sentiment_results: List[Dict], 
                                    stock_data: pd.DataFrame) -> Dict:
        """
        Analyze correlation between sentiment and stock price movements
        
        Args:
            sentiment_results: Results from sentiment analysis
            stock_data: DataFrame with stock price data
            
        Returns:
            Dictionary with correlation analysis
        """
        # Create sentiment time series
        sentiments = [r['confidence'] if r['sentiment'] == 'positive' 
                     else -r['confidence'] if r['sentiment'] == 'negative' 
                     else 0 for r in sentiment_results]
        
        if len(stock_data) < 2:
            return {'correlation': 0, 'insights': 'Insufficient data for correlation'}
        
        # Calculate stock price changes
        stock_changes = stock_data['close'].pct_change().dropna()
        
        # Align data
        min_len = min(len(sentiments), len(stock_changes))
        if min_len == 0:
            return {'correlation': 0, 'insights': 'No aligned data available'}
        
        sentiments_aligned = sentiments[:min_len]
        stock_changes_aligned = stock_changes[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(sentiments_aligned, stock_changes_aligned)[0, 1]
        
        # Generate insights
        insights = []
        if correlation > 0.3:
            insights.append("Strong positive correlation: Positive sentiment often precedes price increases")
        elif correlation < -0.3:
            insights.append("Strong negative correlation: Positive sentiment may indicate selling opportunity")
        elif abs(correlation) < 0.1:
            insights.append("Low correlation: Sentiment shows weak relationship with price movements")
        else:
            insights.append("Moderate correlation: Some relationship between sentiment and price movements")
            
        return {
            'correlation': correlation,
            'insights': insights,
            'sentiment_mean': np.mean(sentiments_aligned),
            'price_change_mean': np.mean(stock_changes_aligned),
            'sample_size': min_len
        }
    
    def extract_financial_keywords(self, texts: List[str]) -> List[Dict]:
        """
        Extract and rank financial keywords from text corpus
        """
        if not texts:
            return []
            
        # Fit TF-IDF on all texts
        processed_texts = [self.preprocess_text(text) for text in texts if text]
        tfidf_matrix = self.tfidf.fit_transform(processed_texts)
        
        # Get feature names and scores
        feature_names = self.tfidf.get_feature_names_out()
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Create keyword ranking
        keyword_scores = list(zip(feature_names, mean_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {'keyword': keyword, 'score': float(score)} 
            for keyword, score in keyword_scores[:20]
        ]
    
    def generate_sentiment_summary(self, results: List[Dict]) -> Dict:
        """Generate comprehensive sentiment summary"""
        if not results:
            return {}
            
        total = len(results)
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'), 
            'neutral': sentiments.count('neutral')
        }
        
        sentiment_percentages = {
            k: round(v / total * 100, 2) 
            for k, v in sentiment_counts.items()
        }
        
        return {
            'total_articles': total,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'average_confidence': round(np.mean(confidences), 3),
            'confidence_std': round(np.std(confidences), 3),
            'overall_sentiment': max(sentiment_counts, key=sentiment_counts.get),
            'confidence_level': 'high' if np.mean(confidences) > 0.8 else 'medium' if np.mean(confidences) > 0.6 else 'low'
        }


if __name__ == "__main__":
    # Example usage
    analyzer = FinancialSentimentAnalyzer()
    
    # Test with sample financial news
    sample_news = [
        "Apple reports strong Q4 earnings with revenue up 15% year over year",
        "Tesla stock falls 8% amid concerns over production delays", 
        "Microsoft announces new AI partnership with OpenAI",
        "Amazon reports mixed results with cloud revenue growth slowing"
    ]
    
    results = analyzer.analyze_sentiment(sample_news)
    
    print("\\nSentiment Analysis Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
        print(f"   Text: {result['text'][:100]}...")
        print()
    
    # Generate summary
    summary = analyzer.generate_sentiment_summary(results)
    print("Sentiment Summary:", summary)