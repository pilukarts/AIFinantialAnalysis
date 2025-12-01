"""
AI Financial Sentiment Analysis & Market Prediction
Main entry point for the complete analysis system
"""

import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import our modules
from sentiment_analyzer import FinancialSentimentAnalyzer
from data_collector import FinancialDataCollector
from market_predictor import MarketPredictor

def save_results(data: Dict, filename: str):
    """Save analysis results to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {filename}")

def run_single_analysis(symbol: str, output_dir: str = "results") -> Dict:
    """Run complete analysis for a single stock symbol"""
    print(f"\\n=== Running AI Analysis for {symbol} ===")
    
    # Initialize components
    sentiment_analyzer = FinancialSentimentAnalyzer()
    data_collector = FinancialDataCollector()
    predictor = MarketPredictor()
    
    print("\\n1. Collecting financial data...")
    stock_data = data_collector.get_stock_data(symbol, period="3mo")
    company_info = data_collector.get_company_info(symbol)
    
    print("2. Fetching financial news...")
    news_data = data_collector.get_financial_news([symbol], days_back=14)
    news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                 for article in news_data if article.get('title')]
    
    print("3. Analyzing sentiment...")
    sentiment_results = []
    if news_texts:
        sentiment_results = sentiment_analyzer.analyze_sentiment(news_texts)
        sentiment_summary = sentiment_analyzer.generate_sentiment_summary(sentiment_results)
    else:
        sentiment_summary = {}
    
    print("4. Training prediction models...")
    training_results = predictor.train_price_prediction_model(symbol)
    
    print("5. Generating predictions...")
    prediction_results = predictor.predict_price_movement(symbol)
    
    print("6. Creating comprehensive report...")
    
    # Compile results
    analysis_results = {
        'analysis_metadata': {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': '3 months',
            'news_analyzed': len(news_data),
            'analysis_version': '1.0'
        },
        'company_info': company_info,
        'current_market_data': {
            'stock_data': stock_data.to_dict('records') if not stock_data.empty else [],
            'latest_price': float(stock_data['Close'].iloc[-1]) if not stock_data.empty else 0,
            'price_change_pct': float(stock_data['Close'].pct_change().iloc[-1] * 100) if not stock_data.empty else 0,
            'volume': int(stock_data['Volume'].iloc[-1]) if not stock_data.empty else 0
        },
        'sentiment_analysis': {
            'raw_results': sentiment_results,
            'summary': sentiment_summary,
            'keywords': sentiment_analyzer.extract_financial_keywords(news_texts)
        },
        'prediction_analysis': {
            'model_training': training_results,
            'price_prediction': prediction_results,
            'confidence_metrics': {
                'prediction_confidence': prediction_results.get('confidence', 0),
                'model_accuracy': training_results.get('direction_accuracy', 0),
                'sample_size': training_results.get('training_samples', 0)
            }
        },
        'news_coverage': {
            'total_articles': len(news_data),
            'recent_articles': news_data[:5],  # Latest 5 articles
            'sentiment_distribution': sentiment_summary.get('sentiment_distribution', {})
        },
        'trading_recommendation': {
            'action': prediction_results.get('recommendation', 'HOLD'),
            'target_price': prediction_results.get('predicted_price'),
            'expected_return': prediction_results.get('predicted_change'),
            'risk_level': 'HIGH' if prediction_results.get('confidence', 0) < 0.6 else 'MEDIUM',
            'reasoning': prediction_results.get('trading_signals', [])
        }
    }
    
    return analysis_results

def run_multi_analysis(symbols: List[str], output_dir: str = "results") -> Dict:
    """Run analysis for multiple stock symbols"""
    print(f"\\n=== Running Multi-Stock AI Analysis ===")
    
    predictor = MarketPredictor()
    
    # Generate comprehensive market report
    market_report = predictor.generate_market_report(symbols)
    
    # Add metadata
    market_report['analysis_metadata'] = {
        'analysis_type': 'multi_stock',
        'symbols_analyzed': symbols,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_symbols': len(symbols),
        'version': '1.0'
    }
    
    return market_report

def print_summary_report(results: Dict, analysis_type: str = 'single'):
    """Print a formatted summary report"""
    print("\\n" + "="*60)
    print("📊 AI FINANCIAL ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    if analysis_type == 'single':
        symbol = results['analysis_metadata']['symbol']
        print(f"\\n🏢 Company: {results['company_info'].get('company_name', 'N/A')}")
        print(f"📈 Symbol: {symbol}")
        print(f"💰 Current Price: ${results['current_market_data']['latest_price']:.2f}")
        print(f"📊 24h Change: {results['current_market_data']['price_change_pct']:.2f}%")
        
        # Sentiment summary
        sentiment = results['sentiment_analysis']['summary']
        if sentiment:
            print(f"\\n💭 SENTIMENT ANALYSIS:")
            print(f"   Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A').title()}")
            print(f"   Confidence Level: {sentiment.get('confidence_level', 'N/A').title()}")
            print(f"   Articles Analyzed: {sentiment.get('total_articles', 0)}")
        
        # Prediction summary
        prediction = results['prediction_analysis']['price_prediction']
        if prediction:
            print(f"\\n🤖 AI PREDICTION:")
            print(f"   Predicted Direction: {prediction.get('predicted_direction', 'N/A').title()}")
            print(f"   Confidence: {prediction.get('confidence', 0)*100:.1f}%")
            print(f"   Expected Change: {prediction.get('predicted_change', 0)*100:.2f}%")
        
        # Trading recommendation
        recommendation = results['trading_recommendation']
        print(f"\\n🎯 TRADING RECOMMENDATION:")
        print(f"   Action: {recommendation.get('action', 'HOLD')}")
        print(f"   Risk Level: {recommendation.get('risk_level', 'UNKNOWN')}")
        
        if recommendation.get('reasoning'):
            print(f"   Reasoning:")
            for reason in recommendation['reasoning'][:3]:
                print(f"   • {reason}")
    
    else:  # multi-analysis
        print(f"\\n🏢 MULTI-STOCK MARKET ANALYSIS")
        print(f"   Symbols Analyzed: {len(results['symbols_analyzed'])}")
        print(f"   Analysis Date: {results['analysis_metadata']['analysis_date']}")
        
        summary = results['market_summary']
        print(f"\\n📊 MARKET OVERVIEW:")
        print(f"   Bullish Predictions: {summary.get('bullish_predictions', 0)}")
        print(f"   Bearish Predictions: {summary.get('bearish_predictions', 0)}")
        print(f"   High Confidence: {summary.get('high_confidence_predictions', 0)}")
        print(f"   Average Sentiment: {summary.get('average_sentiment_score', 0):.2f}")
        
        if results.get('recommendations'):
            print(f"\\n🎯 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'][:3], 1):
                print(f"   {i}. {rec['symbol']}: {rec['action']} (Confidence: {rec['confidence']*100:.1f}%)")
    
    print("\\n" + "="*60)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='AI Financial Sentiment Analysis & Market Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol AAPL                    # Analyze single stock
  python main.py --symbols AAPL MSFT GOOGL       # Analyze multiple stocks  
  python main.py --symbol AAPL --dashboard       # Start interactive dashboard
  python main.py --symbols AAPL MSFT --output custom_results/
        """
    )
    
    parser.add_argument('--symbol', type=str, help='Single stock symbol to analyze (e.g., AAPL)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple stock symbols to analyze')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true', help='Start interactive dashboard')
    parser.add_argument('--save', action='store_true', help='Save results to JSON files')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.symbol and not args.symbols:
        parser.error("Must specify either --symbol or --symbols")
    
    symbols = [args.symbol] if args.symbol else args.symbols
    
    # Create output directory
    import os
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if args.dashboard:
            print("🚀 Starting AI Financial Dashboard...")
            print("Dashboard will be available at: http://localhost:8050")
            from dashboards.dashboard import app
            app.run_server(debug=False, host='0.0.0.0', port=8050)
        
        elif len(symbols) == 1:
            # Single symbol analysis
            results = run_single_analysis(symbols[0], args.output)
            
            # Print summary
            print_summary_report(results, 'single')
            
            # Save results if requested
            if args.save:
                filename = f"{args.output}/{symbols[0]}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_results(results, filename)
        
        else:
            # Multi-symbol analysis
            results = run_multi_analysis(symbols, args.output)
            
            # Print summary
            print_summary_report(results, 'multi')
            
            # Save results if requested
            if args.save:
                filename = f"{args.output}/market_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_results(results, filename)
    
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Error during analysis: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()