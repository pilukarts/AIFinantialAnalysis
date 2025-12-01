"""
Financial Data Integration Module
Handles data collection from multiple sources (Yahoo Finance, News APIs, Crypto APIs, etc.)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import time
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Cryptocurrency API endpoints
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

class FinancialDataCollector:
    """
    Comprehensive financial data collector for market analysis
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, symbol: str, period: str = "1mo", 
                      interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock price data using Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            print(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to price data"""
        try:
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volatility
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            
            return df
            
        except Exception as e:
            print(f"Error adding technical indicators: {str(e)}")
            return df
    
    def get_financial_news(self, symbols: List[str], 
                          days_back: int = 7) -> List[Dict]:
        """
        Fetch financial news for given symbols
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
            
        Returns:
            List of news articles with metadata
        """
        all_news = []
        
        # Try multiple news sources
        news_sources = [
            self._get_yahoo_finance_news,
            self._get_market_watch_news,
            self._get_cnbc_news
        ]
        
        for source in news_sources:
            try:
                source_news = source(symbols, days_back)
                all_news.extend(source_news)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"News source failed: {str(e)}")
                continue
        
        # Remove duplicates based on title similarity
        unique_news = self._remove_duplicates(all_news)
        
        print(f"Collected {len(unique_news)} unique news articles")
        return unique_news
    
    def _get_yahoo_finance_news(self, symbols: List[str], 
                               days_back: int) -> List[Dict]:
        """Get news from Yahoo Finance"""
        news_articles = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                news = ticker.news
                
                if news:
                    for article in news:
                        # Filter by date
                        pub_date = datetime.fromtimestamp(article['providerPublishTime'])
                        if (datetime.now() - pub_date).days <= days_back:
                            news_articles.append({
                                'title': article['title'],
                                'summary': article.get('summary', ''),
                                'link': article['link'],
                                'publisher': article['publisher'],
                                'published_date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                                'symbols': [symbol],
                                'source': 'Yahoo Finance'
                            })
                            
            except Exception as e:
                print(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
                continue
                
        return news_articles
    
    def _get_market_watch_news(self, symbols: List[str], 
                              days_back: int) -> List[Dict]:
        """Get news from MarketWatch (placeholder - would need proper API)"""
        # This is a placeholder - in real implementation you'd use MarketWatch API
        return []
    
    def _get_cnbc_news(self, symbols: List[str], 
                      days_back: int) -> List[Dict]:
        """Get news from CNBC (placeholder - would need proper API)"""
        # This is a placeholder - in real implementation you'd use CNBC API  
        return []
    
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []
            
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_words = set(article['title'].lower().split())
            
            # Check if similar title exists
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.lower().split())
                similarity = len(title_words.intersection(seen_words)) / len(title_words.union(seen_words))
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(article['title'])
        
        return unique_articles
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get detailed company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            key_metrics = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'profit_margin': info.get('profitMargins', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
            return key_metrics
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
    
    def get_market_summary(self, symbols: List[str]) -> Dict:
        """Get comprehensive market summary for multiple symbols"""
        market_data = {}
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            # Get price data
            stock_data = self.get_stock_data(symbol, period="1mo")
            company_info = self.get_company_info(symbol)
            
            if not stock_data.empty and company_info:
                # Calculate current metrics
                latest_price = stock_data['Close'].iloc[-1]
                price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
                
                market_data[symbol] = {
                    'company_info': company_info,
                    'current_price': float(latest_price),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'volume': int(stock_data['Volume'].iloc[-1]),
                    'avg_volume': int(stock_data['Volume'].rolling(20).mean().iloc[-1]),
                    'technical_indicators': {
                        'rsi': float(stock_data['RSI'].iloc[-1]),
                        'macd': float(stock_data['MACD'].iloc[-1]),
                        'bb_position': 'upper' if latest_price > stock_data['BB_Upper'].iloc[-1] else 
                                      'lower' if latest_price < stock_data['BB_Lower'].iloc[-1] else 'middle'
                    }
                }
            
            time.sleep(0.1)  # Rate limiting
        
        return market_data


class CryptoDataCollector:
    """
    Cryptocurrency data collector using CoinGecko API
    Handles Bitcoin, Ethereum and other major cryptocurrencies
    """
    
    def __init__(self):
        self.base_url = COINGECKO_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Popular cryptocurrency symbols mapping
        self.crypto_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'SOL': 'solana',
            'TRX': 'tron',
            'MATIC': 'matic-network',
            'DOT': 'polkadot',
            'AVAX': 'avalanche-2',
            'SHIB': 'shiba-inu',
            'LTC': 'litecoin',
            'UNI': 'uniswap',
            'LINK': 'chainlink',
            'ALGO': 'algorand',
            'VET': 'vechain',
            'XLM': 'stellar',
            'ATOM': 'cosmos',
            'FIL': 'filecoin'
        }
    
    def get_crypto_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch cryptocurrency price data from CoinGecko
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days of data to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol to CoinGecko ID
            crypto_id = self._get_crypto_id(symbol)
            if not crypto_id:
                print(f"Cryptocurrency {symbol} not found")
                return pd.DataFrame()
            
            url = f"{self.base_url}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days <= 30 else 'hourly'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' not in data:
                print(f"No price data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data.get('market_caps', [])
            
            df_data = []
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                df_data.append({
                    'Date': datetime.fromtimestamp(timestamp / 1000),
                    'Close': price,
                    'Volume': volume,
                    'Market_Cap': volume  # Using market cap as volume proxy
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('Date')
            df['Open'] = df['Close']  # CoinGecko doesn't provide OHLC, using close for all
            df['High'] = df['Close']
            df['Low'] = df['Close']
            
            # Add technical indicators
            df = self._add_crypto_technical_indicators(df)
            
            print(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_crypto_id(self, symbol: str) -> Optional[str]:
        """Convert symbol to CoinGecko ID"""
        return self.crypto_map.get(symbol.upper())
    
    def _add_crypto_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators specifically for cryptocurrency"""
        try:
            # Moving averages
            df['SMA_7'] = df['Close'].rolling(window=7).mean()
            df['SMA_30'] = df['Close'].rolling(window=30).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI (important for crypto)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Crypto-specific indicators
            # Fear & Greed Index proxy (using volatility)
            df['Volatility'] = df['Close'].rolling(window=7).std()
            
            # Price momentum
            df['Momentum_7'] = df['Close'].pct_change(periods=7)
            df['Momentum_30'] = df['Close'].pct_change(periods=30)
            
            # Volume analysis
            df['Volume_SMA'] = df['Volume'].rolling(window=7).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_1d'] = df['Close'].pct_change(periods=1)
            df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
            
            # Support and resistance levels
            df['Support'] = df['Low'].rolling(window=20).min()
            df['Resistance'] = df['High'].rolling(window=20).max()
            
            return df
            
        except Exception as e:
            print(f"Error adding crypto technical indicators: {str(e)}")
            return df
    
    def get_crypto_info(self, symbol: str) -> Dict:
        """Get detailed cryptocurrency information"""
        try:
            crypto_id = self._get_crypto_id(symbol)
            if not crypto_id:
                return {}
            
            url = f"{self.base_url}/coins/{crypto_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            market_data = data.get('market_data', {})
            
            info = {
                'symbol': symbol.upper(),
                'name': data.get('name', 'N/A'),
                'description': data.get('description', {}).get('en', '')[:200] + '...' if data.get('description') else 'N/A',
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'market_cap_rank': market_data.get('market_cap_rank', 0),
                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'total_supply': market_data.get('total_supply', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),  # All time high
                'atl': market_data.get('atl', {}).get('usd', 0),  # All time low
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                'volume_change_24h': market_data.get('volume_change_percentage_24h', 0),
                'last_updated': market_data.get('last_updated', 'N/A')
            }
            
            return info
            
        except Exception as e:
            print(f"Error fetching crypto info for {symbol}: {str(e)}")
            return {}
    
    def get_crypto_news(self, symbols: List[str], days_back: int = 7) -> List[Dict]:
        """
        Get cryptocurrency news (using general financial news sources)
        """
        news_articles = []
        
        # For crypto, we'll search for news using the same sources
        # but focus on crypto-related keywords
        crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'cryptocurrency', 'blockchain', 'defi']
        
        # This is a simplified implementation - in a real system you'd use
        # crypto-specific news APIs like CryptoNews, NewsAPI with crypto filters, etc.
        try:
            # Placeholder for crypto news sources
            # You could integrate with APIs like:
            # - CryptoPanic API
            # - CoinTelegraph API  
            # - Cointelegraph API
            # - Messari API
            
            # For now, return empty list as this would require external APIs
            pass
            
        except Exception as e:
            print(f"Error fetching crypto news: {str(e)}")
        
        return news_articles
    
    def get_crypto_market_summary(self, symbols: List[str]) -> Dict:
        """Get comprehensive market summary for multiple cryptocurrencies"""
        market_data = {}
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            # Get price data and info
            crypto_data = self.get_crypto_data(symbol, days=30)
            crypto_info = self.get_crypto_info(symbol)
            
            if not crypto_data.empty and crypto_info:
                # Calculate current metrics
                latest_price = crypto_data['Close'].iloc[-1]
                price_change = crypto_data['Close'].iloc[-1] - crypto_data['Close'].iloc[-2]
                price_change_pct = (price_change / crypto_data['Close'].iloc[-2]) * 100
                
                # Crypto-specific metrics
                volume_24h = crypto_data['Volume'].iloc[-1]
                avg_volume = crypto_data['Volume'].rolling(7).mean().iloc[-1]
                
                market_data[symbol] = {
                    'crypto_info': crypto_info,
                    'current_price': float(latest_price),
                    'price_change_24h': float(price_change),
                    'price_change_24h_pct': float(price_change_pct),
                    'volume_24h': float(volume_24h),
                    'avg_volume_7d': float(avg_volume),
                    'market_cap': float(crypto_info.get('market_cap', 0)),
                    'market_cap_rank': int(crypto_info.get('market_cap_rank', 0)),
                    'technical_indicators': {
                        'rsi': float(crypto_data['RSI'].iloc[-1]) if not pd.isna(crypto_data['RSI'].iloc[-1]) else 50,
                        'macd': float(crypto_data['MACD'].iloc[-1]) if not pd.isna(crypto_data['MACD'].iloc[-1]) else 0,
                        'momentum_7d': float(crypto_data['Momentum_7'].iloc[-1]) if not pd.isna(crypto_data['Momentum_7'].iloc[-1]) else 0,
                        'volume_ratio': float(crypto_data['Volume_Ratio'].iloc[-1]) if not pd.isna(crypto_data['Volume_Ratio'].iloc[-1]) else 1.0,
                        'volatility': float(crypto_data['Volatility'].iloc[-1]) if not pd.isna(crypto_data['Volatility'].iloc[-1]) else 0
                    },
                    'support_resistance': {
                        'support': float(crypto_data['Support'].iloc[-1]) if not pd.isna(crypto_data['Support'].iloc[-1]) else latest_price,
                        'resistance': float(crypto_data['Resistance'].iloc[-1]) if not pd.isna(crypto_data['Resistance'].iloc[-1]) else latest_price
                    }
                }
            
            time.sleep(0.1)  # Rate limiting to be respectful to the API
        
        return market_data
    
    def get_top_cryptocurrencies(self, limit: int = 10) -> List[Dict]:
        """Get top cryptocurrencies by market cap"""
        try:
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            top_cryptos = []
            for coin in data:
                top_cryptos.append({
                    'symbol': coin['symbol'].upper(),
                    'name': coin['name'],
                    'current_price': coin['current_price'],
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'price_change_24h': coin['price_change_percentage_24h'],
                    'volume_24h': coin['total_volume']
                })
            
            return top_cryptos
            
        except Exception as e:
            print(f"Error fetching top cryptocurrencies: {str(e)}")
            return []
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a cryptocurrency"""
        return symbol.upper() in self.crypto_map


if __name__ == "__main__":
    # Example usage
    collector = FinancialDataCollector()
    
    # Test with popular stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Get market summary
    market_summary = collector.get_market_summary(test_symbols)
    
    print("Market Summary:")
    print(json.dumps(market_summary, indent=2))
    
    # Get financial news
    news = collector.get_financial_news(test_symbols, days_back=3)
    
    print(f"\\nFound {len(news)} news articles")
    for article in news[:3]:
        print(f"- {article['title']} ({article['publisher']})")

# Test cryptocurrency functionality
def test_crypto_collector():
    print("\\n" + "="*50)
    print("CRYPTOCURRENCY COLLECTOR TEST")
    print("="*50)
    
    crypto_collector = CryptoDataCollector()
    
    # Test popular cryptocurrencies
    crypto_symbols = ['BTC', 'ETH', 'ADA', 'SOL']
    
    # Get crypto market summary
    crypto_summary = crypto_collector.get_crypto_market_summary(crypto_symbols)
    
    print("\\nCrypto Market Summary:")
    print(json.dumps(crypto_summary, indent=2))
    
    # Get top cryptocurrencies
    top_cryptos = crypto_collector.get_top_cryptocurrencies(limit=5)
    
    print("\\nTop 5 Cryptocurrencies:")
    for crypto in top_cryptos:
        print(f"- {crypto['name']} ({crypto['symbol']}): ${crypto['current_price']:,.2f} "
              f"({crypto['price_change_24h']:+.2f}%)")
    
    # Test individual crypto data
    print("\\nTesting individual crypto data for BTC:")
    btc_data = crypto_collector.get_crypto_data('BTC', days=7)
    if not btc_data.empty:
        print(f"Latest BTC price: ${btc_data['Close'].iloc[-1]:,.2f}")
        print(f"7-day change: {((btc_data['Close'].iloc[-1] / btc_data['Close'].iloc[0]) - 1) * 100:+.2f}%")
    
    return crypto_summary, top_cryptos

if __name__ == "__main__":
    # Run both stock and crypto tests
    print("Running comprehensive financial data collector tests...")
    test_crypto_collector()