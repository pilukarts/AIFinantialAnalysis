"""
Demo script for AI Financial Sentiment Analysis
Simulates financial analysis without requiring external APIs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.sentiment_analyzer import FinancialSentimentAnalyzer
from src.data_collector import FinancialDataCollector
from src.market_predictor import MarketPredictor

def create_sample_data():
    """Create sample financial data for demonstration"""
    
    # Sample stock price data
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Simulate realistic stock price movements
    base_price = 150
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for return_rate in returns[1:]:
        new_price = prices[-1] * (1 + return_rate)
        prices.append(new_price)
    
    stock_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Add technical indicators
    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['RSI'] = 50 + np.random.normal(0, 15, len(dates))  # Simplified RSI
    stock_data['MACD'] = np.random.normal(0, 2, len(dates))  # Simplified MACD
    stock_data['BB_Upper'] = stock_data['SMA_20'] * 1.02
    stock_data['BB_Lower'] = stock_data['SMA_20'] * 0.98
    
    # Sample financial news
    sample_news = [
        {
            'title': 'Apple Reports Strong Q4 Earnings, Beats Revenue Expectations',
            'summary': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, with revenue reaching $119 billion, up 8% year-over-year.',
            'published_date': '2024-11-15 14:30:00',
            'publisher': 'Reuters',
            'symbols': ['AAPL'],
            'source': 'Demo Data'
        },
        {
            'title': 'Tesla Stock Rises on Optimistic Production Forecasts',
            'summary': 'Tesla shares gained 5% after the company announced plans to increase production capacity at its Gigafactory facilities.',
            'published_date': '2024-11-14 16:45:00',
            'publisher': 'Demo News',
            'symbols': ['TSLA'],
            'source': 'Demo Data'
        },
        {
            'title': 'Microsoft Azure Growth Slows, Cloud Competition Intensifies',
            'summary': 'Microsoft reported slower than expected growth in its Azure cloud division, prompting concerns about increased competition.',
            'published_date': '2024-11-13 11:20:00',
            'publisher': 'Financial Times',
            'symbols': ['MSFT'],
            'source': 'Demo Data'
        },
        {
            'title': 'Amazon Announces Major Logistics Expansion',
            'summary': 'Amazon unveiled plans to expand its logistics network with 50 new fulfillment centers across North America.',
            'published_date': '2024-11-12 09:15:00',
            'publisher': 'Wall Street Journal',
            'symbols': ['AMZN'],
            'source': 'Demo Data'
        },
        {
            'title': 'NVIDIA Stock Volatile After AI Chip Supply Concerns',
            'summary': 'NVIDIA experienced increased volatility following reports of potential supply chain constraints for its latest AI chips.',
            'published_date': '2024-11-11 15:00:00',
            'publisher': 'MarketWatch',
            'symbols': ['NVDA'],
            'source': 'Demo Data'
        }
    ]
    
    return stock_data, sample_news

def create_demo_collector():
    """Create a demo version of data collector"""
    
    class DemoDataCollector:
        def __init__(self):
            self.stock_data, self.news_data = create_sample_data()
        
        def get_stock_data(self, symbol, period="1mo", interval="1d"):
            """Return sample stock data"""
            return self.stock_data.copy()
        
        def get_financial_news(self, symbols, days_back=7):
            """Return sample news data"""
            return self.news_data
        
        def get_company_info(self, symbol):
            """Return sample company info"""
            company_info = {
                'AAPL': {
                    'company_name': 'Apple Inc.',
                    'sector': 'Technology',
                    'industry': 'Consumer Electronics',
                    'market_cap': 3000000000000,
                    'pe_ratio': 28.5
                },
                'MSFT': {
                    'company_name': 'Microsoft Corporation',
                    'sector': 'Technology',
                    'industry': 'Software',
                    'market_cap': 2800000000000,
                    'pe_ratio': 32.1
                },
                'GOOGL': {
                    'company_name': 'Alphabet Inc.',
                    'sector': 'Technology',
                    'industry': 'Internet Services',
                    'market_cap': 1700000000000,
                    'pe_ratio': 24.8
                },
                'AMZN': {
                    'company_name': 'Amazon.com Inc.',
                    'sector': 'Consumer Discretionary',
                    'industry': 'E-commerce',
                    'market_cap': 1500000000000,
                    'pe_ratio': 45.2
                },
                'TSLA': {
                    'company_name': 'Tesla Inc.',
                    'sector': 'Consumer Discretionary',
                    'industry': 'Automotive',
                    'market_cap': 800000000000,
                    'pe_ratio': 65.4
                },
                'NVDA': {
                    'company_name': 'NVIDIA Corporation',
                    'sector': 'Technology',
                    'industry': 'Semiconductors',
                    'market_cap': 1200000000000,
                    'pe_ratio': 42.7
                }
            }
            return company_info.get(symbol, {})
    
    return DemoDataCollector()

def run_demo_analysis():
    """Run a complete demonstration of the AI analysis system"""
    
    print("🎯 AI Financial Sentiment Analysis - Demo Mode")
    print("=" * 60)
    print("This demo uses simulated data to showcase the system's capabilities.")
    print("No external APIs are required for this demonstration.")
    print("=" * 60)
    
    # Initialize components with demo data
    sentiment_analyzer = FinancialSentimentAnalyzer()
    demo_collector = create_demo_collector()
    
    # Create a modified predictor that uses demo data
    class DemoMarketPredictor:
        def __init__(self):
            self.demo_collector = demo_collector
        
        def train_price_prediction_model(self, symbol):
            """Simulate model training"""
            print(f"Training price prediction model for {symbol}...")
            
            # Simulate training process
            import time
            time.sleep(1)  # Simulate processing time
            
            # Return simulated training results
            return {
                'symbol': symbol,
                'price_mse': 0.0234,
                'direction_accuracy': 0.734,
                'feature_importance': {
                    'Close': 0.25,
                    'Volume': 0.15,
                    'RSI': 0.20,
                    'Sentiment_Score': 0.18,
                    'MACD': 0.12,
                    'BB_Upper': 0.10
                },
                'training_samples': 180,
                'test_samples': 45,
                'features_used': 15
            }
        
        def predict_price_movement(self, symbol, days_ahead=1):
            """Generate simulated predictions"""
            print(f"Generating price predictions for {symbol}...")
            
            # Simulate prediction process
            import time
            time.sleep(0.5)
            
            # Get sample data
            stock_data = self.demo_collector.get_stock_data(symbol)
            current_price = stock_data['Close'].iloc[-1]
            
            # Generate realistic predictions
            np.random.seed(hash(symbol) % 1000)  # Consistent but varied results
            
            # Simulate ML prediction logic
            price_change = np.random.normal(0.02, 0.05)  # 2% average change with 5% std
            confidence = np.random.uniform(0.65, 0.85)
            direction = 'bullish' if price_change > 0 else 'bearish'
            
            # Generate trading signals based on confidence
            signals = []
            if direction == 'bullish' and confidence > 0.7:
                signals.append("BUY signal: Model predicts positive movement with high confidence")
            elif direction == 'bearish' and confidence > 0.7:
                signals.append("SELL signal: Model predicts negative movement with high confidence")
            else:
                signals.append("HOLD signal: Model predictions are uncertain")
            
            # Add sentiment context
            sentiment_context = "Positive sentiment detected (high confidence) supports bullish outlook"
            if direction == 'bearish':
                sentiment_context = "Negative sentiment detected (high confidence) may indicate selling pressure"
            signals.append(sentiment_context)
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(current_price * (1 + price_change)),
                'predicted_change': float(price_change),
                'predicted_direction': direction,
                'confidence': float(confidence),
                'days_ahead': days_ahead,
                'trading_signals': signals,
                'recommendation': 'STRONG_BUY' if direction == 'bullish' and confidence > 0.8 else
                               'BUY' if direction == 'bullish' and confidence > 0.7 else
                               'STRONG_SELL' if direction == 'bearish' and confidence > 0.8 else
                               'SELL' if direction == 'bearish' and confidence > 0.7 else 'HOLD'
            }
        
        def generate_market_report(self, symbols):
            """Generate simulated market report"""
            print("Generating comprehensive market report...")
            
            report = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbols_analyzed': symbols,
                'individual_analysis': {},
                'market_summary': {
                    'symbols_with_predictions': len(symbols),
                    'bullish_predictions': len(symbols) // 2,
                    'bearish_predictions': len(symbols) // 2,
                    'high_confidence_predictions': len(symbols) * 3 // 4,
                    'average_sentiment_score': 0.15,
                    'most_volatile_symbol': symbols[0],
                    'strongest_performer': 3.2
                },
                'recommendations': []
            }
            
            # Analyze each symbol
            for symbol in symbols:
                training_results = self.train_price_prediction_model(symbol)
                prediction_results = self.predict_price_movement(symbol)
                company_info = self.demo_collector.get_company_info(symbol)
                current_data = self.demo_collector.get_stock_data(symbol)
                
                # Simulate sentiment analysis
                news_data = self.demo_collector.get_financial_news([symbol], days_back=3)
                news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                             for article in news_data if article.get('title')]
                sentiment_results = []
                if news_texts:
                    sentiment_results = sentiment_analyzer.analyze_sentiment(news_texts)
                
                sentiment_summary = sentiment_analyzer.generate_sentiment_summary(sentiment_results) if sentiment_results else {}
                
                report['individual_analysis'][symbol] = {
                    'company_info': company_info,
                    'current_data': {
                        'price': float(current_data['Close'].iloc[-1]),
                        'volume': int(current_data['Volume'].iloc[-1]),
                        'rsi': float(current_data['RSI'].iloc[-1])
                    },
                    'training_results': training_results,
                    'prediction': prediction_results,
                    'sentiment_analysis': sentiment_summary,
                    'recent_news_count': len(news_data),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Add to recommendations if not HOLD
                if prediction_results.get('recommendation') != 'HOLD':
                    recommendation = {
                        'symbol': symbol,
                        'action': prediction_results['recommendation'],
                        'confidence': prediction_results['confidence'],
                        'reasoning': prediction_results.get('trading_signals', []),
                        'target_price': prediction_results.get('predicted_price'),
                        'current_price': prediction_results.get('current_price'),
                        'expected_return': prediction_results.get('predicted_change')
                    }
                    report['recommendations'].append(recommendation)
            
            return report
    
    # Run the demo
    print("\\n1. Initializing AI components...")
    demo_predictor = DemoMarketPredictor()
    
    print("\\n2. Analyzing sample financial news...")
    sample_news = demo_collector.news_data
    news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                 for article in sample_news if article.get('title')]
    
    sentiment_results = sentiment_analyzer.analyze_sentiment(news_texts[:3])  # Limit for demo
    print(f"   Analyzed {len(sentiment_results)} news articles")
    
    print("\\n3. Training AI models...")
    training_results = demo_predictor.train_price_prediction_model('AAPL')
    print(f"   Model accuracy: {training_results['direction_accuracy']:.1%}")
    
    print("\\n4. Generating predictions...")
    prediction = demo_predictor.predict_price_movement('AAPL')
    print(f"   Predicted direction: {prediction['predicted_direction'].title()}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    
    print("\\n5. Creating market report...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    market_report = demo_predictor.generate_market_report(symbols)
    
    # Display results
    print("\\n" + "=" * 60)
    print("📊 DEMO ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\\n🤖 SENTIMENT ANALYSIS:")
    sentiment_summary = sentiment_analyzer.generate_sentiment_summary(sentiment_results)
    for key, value in sentiment_summary.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\\n📈 SAMPLE PREDICTION (AAPL):")
    for key, value in prediction.items():
        if isinstance(value, float):
            if 'price' in key.lower():
                print(f"   {key.replace('_', ' ').title()}: ${value:.2f}")
            elif 'change' in key.lower() or 'confidence' in key.lower():
                print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\\n🎯 TRADING SIGNALS:")
    for signal in prediction['trading_signals']:
        print(f"   • {signal}")
    
    print("\\n📊 MARKET SUMMARY:")
    summary = market_report['market_summary']
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\\n🔝 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(market_report['recommendations'][:3], 1):
        print(f"   {i}. {rec['symbol']}: {rec['action']} (Confidence: {rec['confidence']:.1%})")
    
    print("\\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\\n📋 Next Steps:")
    print("• Install real dependencies: pip install -r requirements.txt")
    print("• Get API keys for real data (optional)")
    print("• Run full analysis: python main.py --symbol AAPL")
    print("• Start dashboard: python main.py --dashboard")
    print("\\n💡 This demo showcases the AI capabilities without external dependencies.")
    print("   Real analysis will provide more accurate and timely results.")
    print("=" * 60)

if __name__ == "__main__":
    run_demo_analysis()