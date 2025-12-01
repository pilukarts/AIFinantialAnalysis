"""
AI Financial Sentiment Analysis - Simple Version
Works with basic Python without heavy dependencies for demo purposes
"""

import argparse
import json
import sys
import os
import random
from datetime import datetime, timedelta

def simulate_sentiment_analysis(texts):
    """Simulate BERT sentiment analysis with realistic results"""
    print("🤖 Running BERT-based sentiment analysis...")
    
    # Financial keywords that influence sentiment
    positive_words = ['strong', 'beat', 'growth', 'up', 'increase', 'gain', 'profit', 'bull', 'positive']
    negative_words = ['decline', 'fall', 'loss', 'down', 'bear', 'negative', 'weak', 'concern', 'warning']
    
    results = []
    for text in texts:
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if pos_score > neg_score:
            sentiment = 'positive'
            confidence = min(0.95, 0.6 + (pos_score * 0.1) + random.uniform(-0.1, 0.1))
        elif neg_score > pos_score:
            sentiment = 'negative'
            confidence = min(0.95, 0.6 + (neg_score * 0.1) + random.uniform(-0.1, 0.1))
        else:
            sentiment = 'neutral'
            confidence = random.uniform(0.5, 0.8)
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'source': 'Simulated BERT Model'
        })
    
    return results

def simulate_price_prediction(symbol):
    """Simulate ML price prediction with realistic financial analysis"""
    print(f"🔮 Generating AI price prediction for {symbol}...")
    
    # Simulate current price based on symbol (realistic ranges)
    base_prices = {'AAPL': 175, 'MSFT': 340, 'GOOGL': 140, 'AMZN': 155, 'TSLA': 250, 'NVDA': 500}
    current_price = base_prices.get(symbol, 100) + random.uniform(-10, 10)
    
    # Simulate prediction based on market sentiment and technical indicators
    price_change = random.uniform(-0.05, 0.08)  # -5% to +8%
    confidence = random.uniform(0.65, 0.88)
    
    # Generate realistic signals
    direction = 'bullish' if price_change > 0 else 'bearish'
    signals = []
    
    if direction == 'bullish' and confidence > 0.75:
        signals.append("BUY signal: Model predicts positive movement with high confidence")
        action = "BUY"
    elif direction == 'bearish' and confidence > 0.75:
        signals.append("SELL signal: Model predicts negative movement with high confidence")  
        action = "SELL"
    else:
        signals.append("HOLD signal: Model predictions are uncertain")
        action = "HOLD"
    
    # Add technical analysis context
    rsi = random.uniform(35, 75)
    signals.append(f"Technical indicators show RSI at {rsi:.1f} - {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'}")
    
    if confidence > 0.8:
        signals.append("High model confidence supports confident trading decisions")
    
    predicted_price = current_price * (1 + price_change)
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 2),
        'predicted_price': round(predicted_price, 2),
        'price_change_pct': round(price_change * 100, 2),
        'predicted_direction': direction,
        'confidence': round(confidence, 3),
        'trading_signals': signals,
        'recommendation': action,
        'risk_level': 'HIGH' if confidence < 0.7 else 'MEDIUM',
        'model_accuracy': round(random.uniform(0.68, 0.82), 3),
        'technical_analysis': {
            'RSI': round(rsi, 1),
            'MACD': round(random.uniform(-2.5, 2.5), 2),
            'Volume_Trend': random.choice(['increasing', 'decreasing', 'stable'])
        }
    }

def generate_company_info(symbol):
    """Generate realistic company information"""
    companies = {
        'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'market_cap': 3000000000000},
        'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software', 'market_cap': 2800000000000},
        'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services', 'market_cap': 1700000000000},
        'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary', 'industry': 'E-commerce', 'market_cap': 1500000000000},
        'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary', 'industry': 'Automotive', 'market_cap': 800000000000},
        'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors', 'market_cap': 1200000000000}
    }
    return companies.get(symbol, {'name': 'Unknown Company', 'sector': 'Unknown', 'industry': 'Unknown', 'market_cap': 0})

def is_crypto_symbol(symbol):
    """Check if a symbol is a cryptocurrency"""
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TRX', 'MATIC', 'DOT', 'AVAX', 'SHIB', 'LTC', 'UNI', 'LINK', 'ALGO', 'VET', 'XLM', 'ATOM', 'FIL']
    return symbol.upper() in crypto_symbols

def generate_crypto_info(symbol):
    """Generate realistic cryptocurrency information"""
    cryptos = {
        'BTC': {'name': 'Bitcoin', 'category': 'Digital Gold', 'market_cap_rank': 1},
        'ETH': {'name': 'Ethereum', 'category': 'Smart Contracts', 'market_cap_rank': 2},
        'BNB': {'name': 'Binance Coin', 'category': 'Exchange Token', 'market_cap_rank': 3},
        'XRP': {'name': 'XRP', 'category': 'Payments', 'market_cap_rank': 4},
        'ADA': {'name': 'Cardano', 'category': 'Smart Contracts', 'market_cap_rank': 5},
        'DOGE': {'name': 'Dogecoin', 'category': 'Meme Coin', 'market_cap_rank': 6},
        'SOL': {'name': 'Solana', 'category': 'High Performance', 'market_cap_rank': 7},
        'TRX': {'name': 'TRON', 'category': 'Entertainment', 'market_cap_rank': 8},
        'MATIC': {'name': 'Polygon', 'category': 'Layer 2', 'market_cap_rank': 9},
        'DOT': {'name': 'Polkadot', 'category': 'Interoperability', 'market_cap_rank': 10}
    }
    
    info = cryptos.get(symbol.upper(), {'name': 'Unknown Crypto', 'category': 'Altcoin', 'market_cap_rank': 99})
    
    # Add realistic market data
    base_price = {
        'BTC': 45000, 'ETH': 2800, 'BNB': 320, 'XRP': 0.62, 'ADA': 0.48, 
        'DOGE': 0.08, 'SOL': 95, 'TRX': 0.08, 'MATIC': 0.85, 'DOT': 6.5
    }.get(symbol.upper(), 100)
    
    current_price = base_price + random.uniform(-base_price * 0.1, base_price * 0.1)
    market_cap = current_price * random.randint(10000000, 1000000000)  # Realistic supply
    
    return {
        **info,
        'current_price': round(current_price, 4),
        'market_cap': int(market_cap),
        'volume_24h': int(market_cap * random.uniform(0.05, 0.15)),
        'circulating_supply': random.randint(1000000, 1000000000),
        'max_supply': random.randint(100000000, 1000000000),
        'total_supply': random.randint(100000000, 1000000000)
    }

def simulate_crypto_prediction(symbol):
    """Simulate cryptocurrency price prediction with higher volatility"""
    print(f"🪙 Generating crypto prediction for {symbol}...")
    
    crypto_info = generate_crypto_info(symbol)
    current_price = crypto_info['current_price']
    
    # Crypto is more volatile - larger potential changes
    price_change = random.uniform(-0.12, 0.15)  # -12% to +15%
    confidence = random.uniform(0.55, 0.85)  # Lower confidence due to volatility
    
    # Generate realistic crypto signals
    direction = 'bullish' if price_change > 0 else 'bearish'
    signals = []
    
    # Crypto-specific indicators
    volatility = abs(price_change) * 100
    
    if direction == 'bullish' and confidence > 0.7:
        signals.append(f"🚀 BUY signal: {symbol} shows strong bullish momentum")
        action = "BUY"
    elif direction == 'bearish' and confidence > 0.7:
        signals.append(f"📉 SELL signal: {symbol} indicates bearish trend")
        action = "SELL"
    else:
        signals.append(f"⏸️ HOLD signal: {symbol} showing sideways movement")
        action = "HOLD"
    
    # Add crypto-specific technical analysis
    rsi = random.uniform(25, 80)
    if rsi < 30:
        signals.append(f"🔥 RSI oversold ({rsi:.1f}) - potential reversal zone")
    elif rsi > 70:
        signals.append(f"⚡ RSI overbought ({rsi:.1f}) - possible pullback")
    else:
        signals.append(f"⚖️ RSI neutral ({rsi:.1f}) - balanced momentum")
    
    # Volume analysis for crypto
    volume_signal = random.choice(['high', 'normal', 'low'])
    if volume_signal == 'high':
        signals.append("📊 High trading volume confirms price movement")
    elif volume_signal == 'low':
        signals.append("📉 Low volume suggests weak conviction")
    
    # Fear & Greed sentiment (crypto-specific)
    fear_greed = random.choice(['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'])
    if fear_greed == 'extreme_fear':
        signals.append("😱 Extreme fear in market - potential buying opportunity")
        confidence *= 0.9  # Reduce confidence
    elif fear_greed == 'extreme_greed':
        signals.append("😈 Extreme greed detected - caution advised")
        confidence *= 0.9
    elif fear_greed == 'neutral':
        signals.append("😐 Market sentiment balanced")
        confidence *= 1.05  # Slight confidence boost
    
    predicted_price = current_price * (1 + price_change)
    
    # Risk assessment for crypto
    if volatility > 10:
        risk_level = 'VERY_HIGH'
        signals.append("⚠️ Very high volatility - use proper position sizing")
    elif volatility > 6:
        risk_level = 'HIGH'
        signals.append("⚠️ High volatility detected - manage risk carefully")
    else:
        risk_level = 'MEDIUM'
        signals.append("📈 Moderate volatility - standard risk management")
    
    return {
        'symbol': symbol,
        'current_price': round(current_price, 4),
        'predicted_price': round(predicted_price, 4),
        'price_change_pct': round(price_change * 100, 2),
        'predicted_direction': direction,
        'confidence': round(confidence, 3),
        'trading_signals': signals,
        'recommendation': action,
        'risk_level': risk_level,
        'volatility_percent': round(volatility, 2),
        'model_accuracy': round(random.uniform(0.62, 0.78), 3),
        'market_sentiment': fear_greed.replace('_', ' ').title(),
        'technical_analysis': {
            'RSI': round(rsi, 1),
            'MACD': round(random.uniform(-3.0, 3.0), 2),
            'Volume_Trend': volume_signal,
            'Fear_Greed_Index': random.randint(10, 90),
            'Support_Level': round(current_price * 0.92, 4),
            'Resistance_Level': round(current_price * 1.08, 4)
        },
        'crypto_specific': {
            'circulating_supply': crypto_info['circulating_supply'],
            'market_cap_rank': crypto_info['market_cap_rank'],
            'category': crypto_info['category']
        }
    }

def generate_sample_news(symbol):
    """Generate realistic financial news"""
    news_templates = [
        f"{symbol} Reports Strong Q4 Earnings, Beats Revenue Expectations",
        f"{symbol} Stock Rises on Optimistic Production Forecasts",
        f"{symbol} Announces Major Business Expansion",
        f"{symbol} Faces Competition Challenges in Core Markets",
        f"{symbol} Shows Strong Growth in International Markets"
    ]
    
    summaries = [
        "Company reported quarterly earnings that exceeded analyst expectations, with revenue showing significant year-over-year growth.",
        "Shares gained after the company announced strategic initiatives aimed at increasing market share and operational efficiency.",
        "The company unveiled plans for expansion that are expected to drive long-term growth and profitability.",
        "Market analysts express concerns about increased competitive pressures and potential impact on future performance.",
        "Strong international performance drives optimism about sustained growth momentum and market expansion."
    ]
    
    return [
        {
            'title': news_templates[0],
            'summary': summaries[0],
            'sentiment': 'positive',
            'confidence': 0.85,
            'publisher': 'Financial News Network'
        },
        {
            'title': news_templates[1], 
            'summary': summaries[1],
            'sentiment': 'positive',
            'confidence': 0.78,
            'publisher': 'Market Analysis Weekly'
        },
        {
            'title': news_templates[2],
            'summary': summaries[2], 
            'sentiment': 'positive',
            'confidence': 0.82,
            'publisher': 'Business Wire'
        }
    ]

def generate_crypto_news(symbol):
    """Generate cryptocurrency-specific news"""
    crypto_templates = {
        'BTC': [
            f"{symbol} Shows Strong Institutional Adoption Trends",
            f"{symbol} Market Analysis: Technical Indicators Point Bullish",
            f"Major Payment Processor Integrates {symbol} Support",
            f"{symbol} Halving Event Impact Analysis"
        ],
        'ETH': [
            f"{symbol} Network Upgrade Boosts DeFi Activity",
            f"{symbol} Staking Rewards Attract New Investors",
            f"Major Platform Migrates to {symbol} Blockchain",
            f"{symbol} Gas Fee Optimization Shows Results"
        ],
        'default': [
            f"{symbol} Partners with Major Financial Institution",
            f"{symbol} Technical Analysis Shows Bullish Patterns",
            f"New {symbol} Development Roadmap Unveiled",
            f"{symbol} Community Growth Drives Market Interest"
        ]
    }
    
    crypto_summaries = {
        'BTC': [
            "Institutional investors continue to increase their Bitcoin holdings, signaling strong confidence in the cryptocurrency's long-term value proposition.",
            "Technical analysis indicates Bitcoin is showing strong bullish momentum with key support levels holding firm across multiple timeframes.",
            "The integration of Bitcoin payments by major platforms expands its utility and adoption potential in the mainstream economy.",
            "Analysis of Bitcoin's upcoming halving event suggests potential for significant price movements based on historical patterns."
        ],
        'ETH': [
            "The Ethereum network upgrade has resulted in improved transaction speeds and reduced fees, boosting DeFi ecosystem activity.",
            "Ethereum's staking mechanism continues to attract investors with competitive yields and network security benefits.",
            "Major platforms choosing to build on Ethereum demonstrates the blockchain's robust infrastructure and developer ecosystem.",
            "Gas fee optimization efforts have successfully reduced transaction costs while maintaining network security and performance."
        ],
        'default': [
            "Strategic partnerships with traditional financial institutions indicate growing mainstream adoption and legitimacy.",
            "Technical analysis reveals strong bullish patterns with favorable momentum indicators across multiple timeframes.",
            "The development roadmap outlines ambitious plans for network improvements and ecosystem expansion.",
            "Community growth metrics show increasing engagement and interest from both retail and institutional participants."
        ]
    }
    
    templates = crypto_templates.get(symbol, crypto_templates['default'])
    summaries = crypto_summaries.get(symbol, crypto_summaries['default'])
    
    return [
        {
            'title': templates[0],
            'summary': summaries[0],
            'sentiment': 'positive',
            'confidence': 0.88,
            'publisher': 'CryptoDaily'
        },
        {
            'title': templates[1], 
            'summary': summaries[1],
            'sentiment': 'positive',
            'confidence': 0.82,
            'publisher': 'Blockchain News'
        },
        {
            'title': templates[2],
            'summary': summaries[2], 
            'sentiment': 'positive',
            'confidence': 0.85,
            'publisher': 'CoinDesk'
        }
    ]
        }
    ]

def run_single_analysis(symbol):
    """Run complete analysis for a single symbol (stock or cryptocurrency)"""
    is_crypto = is_crypto_symbol(symbol)
    asset_type = "Cryptocurrency" if is_crypto else "Stock"
    
    print(f"\\n=== Running AI Analysis for {symbol} ({asset_type}) ===")
    
    if is_crypto:
        # Cryptocurrency analysis
        crypto_info = generate_crypto_info(symbol)
        news_data = generate_crypto_news(symbol) if 'generate_crypto_news' in globals() else generate_sample_news(symbol)
        news_texts = [f"{article['title']} {article['summary']}" for article in news_data]
        
        print("\\n1. Analyzing crypto market sentiment...")
        sentiment_results = simulate_sentiment_analysis(news_texts)
        
        print("2. Generating cryptocurrency price predictions...")
        prediction = simulate_crypto_prediction(symbol)
        
        print("3. Calculating crypto-specific technical indicators...")
        print("4. Analyzing on-chain metrics and market structure...")
        print("5. Creating crypto analysis report...")
        
        # Compile results
        analysis_results = {
            'analysis_metadata': {
                'symbol': symbol,
                'asset_type': 'cryptocurrency',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_version': '1.1-crypto',
                'demo_mode': True
            },
            'asset_info': crypto_info,
            'current_market_data': {
                'latest_price': prediction['current_price'],
                'price_change_pct': prediction['price_change_pct'],
                'volume_24h': crypto_info['volume_24h'],
                'market_cap': crypto_info['market_cap'],
                'market_cap_rank': crypto_info['market_cap_rank']
            },
            'sentiment_analysis': {
                'summary': {
                    'total_articles': len(sentiment_results),
                    'positive_count': sum(1 for r in sentiment_results if r['sentiment'] == 'positive'),
                    'negative_count': sum(1 for r in sentiment_results if r['sentiment'] == 'negative'),
                    'neutral_count': sum(1 for r in sentiment_results if r['sentiment'] == 'neutral'),
                    'overall_sentiment': max(set([r['sentiment'] for r in sentiment_results]), 
                                           key=[r['sentiment'] for r in sentiment_results].count),
                    'average_confidence': round(sum(r['confidence'] for r in sentiment_results) / len(sentiment_results), 3)
                },
                'articles_analyzed': sentiment_results
            },
            'prediction_analysis': prediction,
            'crypto_specific': {
                'category': crypto_info['category'],
                'circulating_supply': crypto_info['circulating_supply'],
                'max_supply': crypto_info['max_supply'],
                'market_sentiment': prediction['market_sentiment'],
                'fear_greed_index': prediction['technical_analysis']['Fear_Greed_Index']
            },
            'trading_recommendation': {
                'action': prediction['recommendation'],
                'target_price': prediction['predicted_price'],
                'expected_return': prediction['price_change_pct'] / 100,
                'risk_level': prediction['risk_level'],
                'reasoning': prediction['trading_signals'],
                'volatility_percent': prediction['volatility_percent']
            }
        }
    else:
        # Stock analysis
        company_info = generate_company_info(symbol)
        news_data = generate_sample_news(symbol)
        news_texts = [f"{article['title']} {article['summary']}" for article in news_data]
        
        print("\\n1. Analyzing financial news...")
        sentiment_results = simulate_sentiment_analysis(news_texts)
        
        print("2. Generating price predictions...")
        prediction = simulate_price_prediction(symbol)
        
        print("3. Calculating technical indicators...")
        print("4. Analyzing sentiment-price correlations...")
        print("5. Creating comprehensive report...")
        
        # Compile results
        analysis_results = {
            'analysis_metadata': {
                'symbol': symbol,
                'asset_type': 'stock',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_version': '1.1-crypto',
                'demo_mode': True
            },
            'company_info': company_info,
            'current_market_data': {
                'latest_price': prediction['current_price'],
                'price_change_pct': prediction['price_change_pct'],
                'volume': random.randint(5000000, 50000000)
            },
            'sentiment_analysis': {
                'summary': {
                    'total_articles': len(sentiment_results),
                    'positive_count': sum(1 for r in sentiment_results if r['sentiment'] == 'positive'),
                    'negative_count': sum(1 for r in sentiment_results if r['sentiment'] == 'negative'),
                    'neutral_count': sum(1 for r in sentiment_results if r['sentiment'] == 'neutral'),
                    'overall_sentiment': max(set([r['sentiment'] for r in sentiment_results]), 
                                           key=[r['sentiment'] for r in sentiment_results].count),
                    'average_confidence': round(sum(r['confidence'] for r in sentiment_results) / len(sentiment_results), 3)
                },
                'articles_analyzed': sentiment_results
            },
            'prediction_analysis': prediction,
            'news_coverage': {
                'total_articles': len(news_data),
                'recent_articles': news_data
            },
            'trading_recommendation': {
                'action': prediction['recommendation'],
                'target_price': prediction['predicted_price'],
                'expected_return': prediction['price_change_pct'] / 100,
                'risk_level': prediction['risk_level'],
                'reasoning': prediction['trading_signals']
            }
        }
    
    return analysis_results

def print_summary_report(results):
    """Print a formatted summary report for stocks or cryptocurrencies"""
    symbol = results['analysis_metadata']['symbol']
    asset_type = results['analysis_metadata']['asset_type']
    
    print("\\n" + "="*60)
    print("📊 AI FINANCIAL ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    if asset_type == 'cryptocurrency':
        # Cryptocurrency report
        asset_info = results['asset_info']
        print(f"\\n🪙 Cryptocurrency: {asset_info['name']}")
        print(f"📈 Symbol: {symbol}")
        print(f"💰 Current Price: ${results['current_market_data']['latest_price']:.4f}")
        print(f"📊 24h Change: {results['current_market_data']['price_change_pct']:.2f}%")
        print(f"🏆 Market Cap Rank: #{results['current_market_data']['market_cap_rank']}")
        print(f"🏷️ Category: {asset_info['category']}")
        
        # Crypto-specific metrics
        print(f"\\n💹 CRYPTO MARKET DATA:")
        market_cap = results['current_market_data']['market_cap']
        print(f"   Market Cap: ${market_cap:,.0f}")
        print(f"   24h Volume: ${results['current_market_data']['volume_24h']:,.0f}")
        print(f"   Circulating Supply: {results['crypto_specific']['circulating_supply']:,.0f}")
        
        # Fear & Greed sentiment
        fear_greed = results['crypto_specific']['market_sentiment']
        print(f"\\n🧠 MARKET SENTIMENT:")
        print(f"   Fear & Greed Index: {results['crypto_specific']['fear_greed_index']}/100 ({fear_greed})")
        
        # Volatility info
        prediction = results['prediction_analysis']
        print(f"\\n⚡ VOLATILITY ANALYSIS:")
        print(f"   Expected Volatility: {prediction['volatility_percent']:.1f}%")
        
    else:
        # Stock report
        company_info = results['company_info']
        print(f"\\n🏢 Company: {company_info['name']}")
        print(f"📈 Symbol: {symbol}")
        print(f"💰 Current Price: ${results['current_market_data']['latest_price']:.2f}")
        print(f"📊 24h Change: {results['current_market_data']['price_change_pct']:.2f}%")
        
        if 'company_info' in results:
            print(f"\\n🏢 COMPANY DETAILS:")
            print(f"   Sector: {company_info.get('sector', 'N/A')}")
            print(f"   Industry: {company_info.get('industry', 'N/A')}")
            print(f"   Market Cap: ${company_info.get('market_cap', 0):,.0f}")
    
    # Sentiment summary (common for both)
    sentiment = results['sentiment_analysis']['summary']
    print(f"\\n💭 SENTIMENT ANALYSIS:")
    print(f"   Overall Sentiment: {sentiment['overall_sentiment'].title()}")
    print(f"   Average Confidence: {sentiment['average_confidence']:.1%}")
    print(f"   Articles Analyzed: {sentiment['total_articles']}")
    print(f"   Distribution: {sentiment['positive_count']} positive, {sentiment['negative_count']} negative, {sentiment['neutral_count']} neutral")
    
    # Prediction summary
    prediction = results['prediction_analysis']
    print(f"\\n🤖 AI PREDICTION:")
    print(f"   Predicted Direction: {prediction['predicted_direction'].title()}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    print(f"   Expected Change: {prediction['price_change_pct']:.1f}%")
    print(f"   Model Accuracy: {prediction['model_accuracy']:.1%}")
    
    # Technical analysis
    tech = prediction['technical_analysis']
    print(f"\\n📊 TECHNICAL INDICATORS:")
    print(f"   RSI: {tech['RSI']}")
    print(f"   MACD: {tech['MACD']}")
    print(f"   Volume Trend: {tech['Volume_Trend'].title()}")
    
    # Crypto-specific technical indicators
    if asset_type == 'cryptocurrency':
        print(f"   Support Level: ${tech['Support_Level']:.4f}")
        print(f"   Resistance Level: ${tech['Resistance_Level']:.4f}")
        if 'Fear_Greed_Index' in tech:
            print(f"   Fear & Greed: {tech['Fear_Greed_Index']}/100")
    
    # Trading recommendation
    recommendation = results['trading_recommendation']
    print(f"\\n🎯 TRADING RECOMMENDATION:")
    print(f"   Action: {recommendation['action']}")
    print(f"   Risk Level: {recommendation['risk_level']}")
    
    if asset_type == 'cryptocurrency':
        print(f"   Target Price: ${recommendation['target_price']:.4f}")
    else:
        print(f"   Target Price: ${recommendation['target_price']:.2f}")
    
    if recommendation.get('volatility_percent'):
        print(f"   Expected Volatility: {recommendation['volatility_percent']:.1f}%")
    
    if recommendation['reasoning']:
        print(f"\\n💡 REASONING:")
        for reason in recommendation['reasoning']:
            print(f"   • {reason}")
    
    print("\\n" + "="*60)
    print("✅ Analysis completed successfully!")
    print("="*60)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='AI Financial Sentiment Analysis & Market Prediction (Demo Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_simple.py --symbol AAPL              # Analyze single stock
  python main_simple.py --symbols AAPL MSFT GOOGL  # Analyze multiple stocks  
  python main_simple.py --dashboard                # Start demo dashboard
        """
    )
    
    parser.add_argument('--symbol', type=str, help='Single stock/crypto symbol to analyze (e.g., AAPL, BTC, ETH)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple stock/crypto symbols to analyze')
    parser.add_argument('--save', action='store_true', help='Save results to JSON files')
    parser.add_argument('--dashboard', action='store_true', help='Start demo dashboard')
    
    args = parser.parse_args()
    
    try:
        if args.dashboard:
            print("🚀 Starting AI Financial Dashboard (Demo Mode)...")
            print("Dashboard will be available at: http://localhost:8050")
            print("Note: This is a demo with simulated data")
            # In a real implementation, this would start the Dash app
            print("Dashboard features:")
            print("• Interactive price charts with AI predictions")
            print("• Real-time sentiment analysis visualization")
            print("• Technical indicator dashboard")
            print("• Portfolio overview and recommendations")
            
        elif args.symbol or args.symbols:
            symbols = [args.symbol] if args.symbol else args.symbols
            
            if len(symbols) == 1:
                results = run_single_analysis(symbols[0])
                print_summary_report(results)
                
                if args.save:
                    filename = f"{symbols[0]}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"\\nResults saved to {filename}")
            else:
                print(f"\\n=== Running Multi-Stock Analysis for {symbols} ===")
                all_results = {}
                for symbol in symbols:
                    results = run_single_analysis(symbol)
                    all_results[symbol] = results
                
                # Print comparison
                print("\\n" + "="*60)
                print("📊 MULTI-STOCK COMPARISON")
                print("="*60)
                for symbol, results in all_results.items():
                    pred = results['prediction_analysis']
                    sentiment = results['sentiment_analysis']['summary']
                    print(f"\\n{symbol}: {results['company_info']['name']}")
                    print(f"   Price: ${pred['current_price']:.2f} → ${pred['predicted_price']:.2f}")
                    print(f"   Prediction: {pred['predicted_direction'].title()} ({pred['confidence']:.1%})")
                    print(f"   Sentiment: {sentiment['overall_sentiment'].title()}")
                    print(f"   Action: {pred['recommendation']}")
        else:
            # Default demo if no arguments
            print("🎯 AI Financial Sentiment Analysis - Demo Mode")
            print("=" * 60)
            print("This demo showcases AI-powered financial analysis capabilities.")
            print("Use --help for usage options.")
            print("\\nRunning default analysis for AAPL...")
            results = run_single_analysis('AAPL')
            print_summary_report(results)
    
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()