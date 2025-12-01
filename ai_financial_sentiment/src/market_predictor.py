"""
Financial Analysis & Prediction Module
Combines sentiment analysis with financial data for market insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
try:
    from .sentiment_analyzer import FinancialSentimentAnalyzer
    from .data_collector import FinancialDataCollector, CryptoDataCollector
except ImportError:
    from sentiment_analyzer import FinancialSentimentAnalyzer
    from data_collector import FinancialDataCollector, CryptoDataCollector

class MarketPredictor:
    """
    Advanced market prediction using sentiment and technical analysis
    """
    
    def __init__(self):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.data_collector = FinancialDataCollector()
        self.crypto_collector = CryptoDataCollector()
        self.price_models = {}  # Price prediction models per symbol
        self.sentiment_models = {}  # Sentiment prediction models per symbol
        self.crypto_models = {}  # Crypto prediction models per symbol
        self.scalers = {}  # Scalers for feature normalization
        self.crypto_scalers = {}  # Scalers for crypto features
        
    def prepare_features(self, price_data: pd.DataFrame, 
                        sentiment_data: List[Dict]) -> pd.DataFrame:
        """
        Prepare combined features from price data and sentiment analysis
        
        Args:
            price_data: Stock price data with technical indicators
            sentiment_data: List of sentiment analysis results
            
        Returns:
            DataFrame with combined features
        """
        # Start with price data features
        features = price_data[['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']].copy()
        
        # Add price-based features
        features['Price_MA_Ratio'] = features['Close'] / features['Close'].rolling(20).mean()
        features['Volume_MA_Ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
        features['Volatility'] = features['Close'].rolling(20).std()
        
        # Add sentiment features
        sentiment_features = self._create_sentiment_features(sentiment_data, len(price_data))
        
        # Merge sentiment features with price features
        for feature_name, values in sentiment_features.items():
            features[feature_name] = values
        
        # Create target variables
        features['Price_Change_Next'] = features['Close'].shift(-1) / features['Close'] - 1
        features['Price_Direction'] = (features['Price_Change_Next'] > 0).astype(int)
        
        # Remove rows with NaN values
        features = features.dropna()
        
        return features
    
    def _create_sentiment_features(self, sentiment_data: List[Dict], 
                                  data_length: int) -> Dict[str, List]:
        """
        Create sentiment-based features aligned with price data timestamps
        """
        features = {
            'Sentiment_Score': [0.0] * data_length,
            'Sentiment_Confidence': [0.5] * data_length,
            'Positive_Ratio': [0.33] * data_length,
            'Negative_Ratio': [0.33] * data_length,
            'News_Volume': [0] * data_length,
            'Sentiment_MA_3': [0.0] * data_length,
            'Sentiment_MA_7': [0.0] * data_length
        }
        
        if not sentiment_data:
            return features
        
        # Group sentiment by date and calculate metrics
        sentiment_by_date = {}
        for item in sentiment_data:
            try:
                if 'published_date' in item:
                    date_key = pd.to_datetime(item['published_date']).date()
                    if date_key not in sentiment_by_date:
                        sentiment_by_date[date_key] = []
                    sentiment_by_date[date_key].append(item)
            except:
                continue
        
        # Calculate daily sentiment metrics
        daily_sentiments = []
        for date, items in sentiment_by_date.items():
            if items:
                positive_count = sum(1 for item in items if item.get('sentiment') == 'positive')
                negative_count = sum(1 for item in items if item.get('sentiment') == 'negative')
                neutral_count = sum(1 for item in items if item.get('sentiment') == 'neutral')
                
                total_items = len(items)
                avg_confidence = np.mean([item.get('confidence', 0.5) for item in items])
                sentiment_score = (positive_count - negative_count) / total_items if total_items > 0 else 0
                
                daily_sentiments.append({
                    'date': date,
                    'sentiment_score': sentiment_score,
                    'confidence': avg_confidence,
                    'positive_ratio': positive_count / total_items,
                    'negative_ratio': negative_count / total_items,
                    'news_volume': total_items
                })
        
        # Sort by date
        daily_sentiments.sort(key=lambda x: x['date'])
        
        # Create time series features (simplified assignment for demo)
        # In real implementation, you'd align this with actual price data timestamps
        for i, feature_name in enumerate(features.keys()):
            if daily_sentiments:
                if feature_name == 'Sentiment_Score':
                    values = [s['sentiment_score'] for s in daily_sentiments]
                elif feature_name == 'Sentiment_Confidence':
                    values = [s['confidence'] for s in daily_sentiments]
                elif feature_name == 'Positive_Ratio':
                    values = [s['positive_ratio'] for s in daily_sentiments]
                elif feature_name == 'Negative_Ratio':
                    values = [s['negative_ratio'] for s in daily_sentiments]
                elif feature_name == 'News_Volume':
                    values = [s['news_volume'] for s in daily_sentiments]
                else:
                    values = [0.0] * min(len(daily_sentiments), data_length)
                
                # Pad or truncate to match data length
                if len(values) < data_length:
                    values.extend([values[-1]] * (data_length - len(values)))
                else:
                    values = values[:data_length]
                
                features[feature_name] = values
        
        return features
    
    def train_price_prediction_model(self, symbol: str) -> Dict:
        """
        Train price prediction model for a specific symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with training results and model metrics
        """
        print(f"Training price prediction model for {symbol}...")
        
        # Collect data
        price_data = self.data_collector.get_stock_data(symbol, period="6mo")
        news_data = self.data_collector.get_financial_news([symbol], days_back=30)
        
        if price_data.empty:
            return {'error': 'No price data available'}
        
        # Analyze sentiment
        news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                     for article in news_data if article.get('title')]
        sentiment_results = []
        
        if news_texts:
            sentiment_results = self.sentiment_analyzer.analyze_sentiment(news_texts)
        
        # Prepare features
        features = self.prepare_features(price_data, sentiment_results)
        
        if len(features) < 20:
            return {'error': 'Insufficient data for training'}
        
        # Prepare training data
        feature_columns = [col for col in features.columns 
                          if col not in ['Price_Change_Next', 'Price_Direction']]
        X = features[feature_columns].values
        y_price = features['Price_Change_Next'].values
        y_direction = features['Price_Direction'].values
        
        # Split data
        X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = \
            train_test_split(X, y_price, y_direction, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train price prediction model (regression)
        price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        price_model.fit(X_train_scaled, y_price_train)
        
        # Train direction prediction model (classification)
        direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        direction_model.fit(X_train_scaled, y_dir_train)
        
        # Evaluate models
        price_pred = price_model.predict(X_test_scaled)
        direction_pred = direction_model.predict(X_test_scaled)
        
        price_mse = mean_squared_error(y_price_test, price_pred)
        direction_accuracy = accuracy_score(y_dir_test, direction_pred)
        
        # Store models and scaler
        self.price_models[symbol] = price_model
        self.sentiment_models[symbol] = direction_model
        self.scalers[symbol] = scaler
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, price_model.feature_importances_))
        
        return {
            'symbol': symbol,
            'price_mse': float(price_mse),
            'direction_accuracy': float(direction_accuracy),
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(feature_columns)
        }
    
    def predict_price_movement(self, symbol: str, days_ahead: int = 1) -> Dict:
        """
        Predict future price movement for a symbol
        
        Args:
            symbol: Stock symbol
            days_ahead: Number of days ahead to predict
            
        Returns:
            Dictionary with prediction results
        """
        if symbol not in self.price_models:
            return {'error': f'No trained model for {symbol}. Train the model first.'}
        
        # Get latest data
        price_data = self.data_collector.get_stock_data(symbol, period="1mo")
        news_data = self.data_collector.get_financial_news([symbol], days_back=7)
        
        if price_data.empty:
            return {'error': 'No current data available'}
        
        # Analyze latest sentiment
        news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                     for article in news_data[-10:] if article.get('title')]  # Latest 10 articles
        sentiment_results = []
        
        if news_texts:
            sentiment_results = self.sentiment_analyzer.analyze_sentiment(news_texts)
        
        # Prepare features
        features = self.prepare_features(price_data, sentiment_results)
        
        if features.empty:
            return {'error': 'Unable to prepare features for prediction'}
        
        # Get latest features
        latest_features = features.iloc[-1:][features.columns.difference(['Price_Change_Next', 'Price_Direction'])].values
        
        # Scale features
        latest_features_scaled = self.scalers[symbol].transform(latest_features)
        
        # Make predictions
        price_change_pred = self.price_models[symbol].predict(latest_features_scaled)[0]
        direction_pred = self.sentiment_models[symbol].predict(latest_features_scaled)[0]
        direction_proba = self.sentiment_models[symbol].predict_proba(latest_features_scaled)[0]
        
        # Calculate prediction confidence
        prediction_confidence = max(direction_proba)
        
        # Generate trading signals
        current_price = price_data['Close'].iloc[-1]
        predicted_price = current_price * (1 + price_change_pred)
        
        # Generate insights
        signals = []
        if direction_pred == 1 and prediction_confidence > 0.6:
            signals.append("BUY signal: Model predicts positive movement with high confidence")
        elif direction_pred == 0 and prediction_confidence > 0.6:
            signals.append("SELL signal: Model predicts negative movement with high confidence")
        else:
            signals.append("HOLD signal: Model predictions are uncertain")
        
        # Add sentiment context
        if sentiment_results:
            sentiment_summary = self.sentiment_analyzer.generate_sentiment_summary(sentiment_results)
            overall_sentiment = sentiment_summary.get('overall_sentiment', 'neutral')
            confidence = sentiment_summary.get('confidence_level', 'medium')
            
            if overall_sentiment == 'positive':
                signals.append(f"Positive sentiment detected ({confidence} confidence) supports bullish outlook")
            elif overall_sentiment == 'negative':
                signals.append(f"Negative sentiment detected ({confidence} confidence) may indicate selling pressure")
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_change': float(price_change_pred),
            'predicted_direction': 'bullish' if direction_pred == 1 else 'bearish',
            'confidence': float(prediction_confidence),
            'days_ahead': days_ahead,
            'trading_signals': signals,
            'recommendation': 'STRONG_BUY' if direction_pred == 1 and prediction_confidence > 0.8 else
                           'BUY' if direction_pred == 1 and prediction_confidence > 0.6 else
                           'STRONG_SELL' if direction_pred == 0 and prediction_confidence > 0.8 else
                           'SELL' if direction_pred == 0 and prediction_confidence > 0.6 else 'HOLD'
        }
    
    def generate_market_report(self, symbols: List[str]) -> Dict:
        """
        Generate comprehensive market analysis report
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            Dictionary with complete market analysis
        """
        print("Generating comprehensive market report...")
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_analyzed': symbols,
            'individual_analysis': {},
            'market_summary': {},
            'recommendations': []
        }
        
        # Analyze each symbol
        for symbol in symbols:
            print(f"Analyzing {symbol}...")
            
            # Train model and get predictions
            training_results = self.train_price_prediction_model(symbol)
            prediction_results = self.predict_price_movement(symbol)
            
            # Get company info and market data
            company_info = self.data_collector.get_company_info(symbol)
            current_data = self.data_collector.get_stock_data(symbol, period="1mo")
            
            # Collect recent news
            news_data = self.data_collector.get_financial_news([symbol], days_back=3)
            news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                         for article in news_data if article.get('title')]
            
            sentiment_results = []
            if news_texts:
                sentiment_results = self.sentiment_analyzer.analyze_sentiment(news_texts)
            
            report['individual_analysis'][symbol] = {
                'company_info': company_info,
                'current_data': {
                    'price': float(current_data['Close'].iloc[-1]) if not current_data.empty else 0,
                    'volume': int(current_data['Volume'].iloc[-1]) if not current_data.empty else 0,
                    'rsi': float(current_data['RSI'].iloc[-1]) if not current_data.empty and 'RSI' in current_data.columns else 0
                },
                'training_results': training_results,
                'prediction': prediction_results,
                'sentiment_analysis': self.sentiment_analyzer.generate_sentiment_summary(sentiment_results) if sentiment_results else {},
                'recent_news_count': len(news_data),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Generate market summary and recommendations
        report['market_summary'] = self._generate_market_summary(report['individual_analysis'])
        report['recommendations'] = self._generate_recommendations(report['individual_analysis'])
        
        return report
    
    def _generate_market_summary(self, analysis_results: Dict) -> Dict:
        """Generate overall market summary from individual analyses"""
        if not analysis_results:
            return {}
        
        summary = {
            'symbols_with_predictions': 0,
            'bullish_predictions': 0,
            'bearish_predictions': 0,
            'high_confidence_predictions': 0,
            'average_sentiment_score': 0,
            'most_volatile_symbol': None,
            'strongest_performer': None
        }
        
        sentiment_scores = []
        price_changes = []
        volatilities = {}
        
        for symbol, data in analysis_results.items():
            if 'prediction' in data and data['prediction'].get('predicted_direction'):
                summary['symbols_with_predictions'] += 1
                
                direction = data['prediction']['predicted_direction']
                confidence = data['prediction']['confidence']
                
                if direction == 'bullish':
                    summary['bullish_predictions'] += 1
                else:
                    summary['bearish_predictions'] += 1
                
                if confidence > 0.7:
                    summary['high_confidence_predictions'] += 1
            
            # Collect sentiment scores
            sentiment_data = data.get('sentiment_analysis', {})
            if sentiment_data:
                pos_pct = sentiment_data.get('sentiment_percentages', {}).get('positive', 0)
                neg_pct = sentiment_data.get('sentiment_percentages', {}).get('negative', 0)
                sentiment_score = (pos_pct - neg_pct) / 100
                sentiment_scores.append(sentiment_score)
            
            # Track price changes
            if 'current_data' in data:
                price_changes.append(data['current_data'].get('price_change_pct', 0))
                volatilities[symbol] = abs(data['current_data'].get('price_change_pct', 0))
        
        # Calculate aggregates
        summary['average_sentiment_score'] = np.mean(sentiment_scores) if sentiment_scores else 0
        
        if volatilities:
            summary['most_volatile_symbol'] = max(volatilities, key=volatilities.get)
            summary['strongest_performer'] = max(price_changes) if price_changes else 0
        
        return summary
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate trading recommendations based on analysis"""
        recommendations = []
        
        for symbol, data in analysis_results.items():
            prediction = data.get('prediction', {})
            sentiment = data.get('sentiment_analysis', {})
            
            if prediction.get('recommendation') != 'HOLD':
                recommendation = {
                    'symbol': symbol,
                    'action': prediction['recommendation'],
                    'confidence': prediction['confidence'],
                    'reasoning': prediction.get('trading_signals', []),
                    'target_price': prediction.get('predicted_price'),
                    'current_price': prediction.get('current_price'),
                    'expected_return': prediction.get('predicted_change')
                }
                recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a cryptocurrency"""
        return self.crypto_collector.is_crypto_symbol(symbol)
    
    def predict_crypto(self, symbol: str, days_ahead: int = 7) -> Dict:
        """
        Generate cryptocurrency-specific prediction
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Get crypto data
            crypto_data = self.crypto_collector.get_crypto_data(symbol, days=30)
            if crypto_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Prepare features for crypto
            features = self._prepare_crypto_features(crypto_data)
            
            # Generate prediction using crypto-specific model
            prediction_result = self._generate_crypto_prediction(features, crypto_data, days_ahead)
            
            # Add crypto-specific indicators
            prediction_result.update(self._analyze_crypto_sentiment(symbol))
            prediction_result.update(self._generate_crypto_trading_signals(crypto_data, prediction_result))
            
            return prediction_result
            
        except Exception as e:
            return {'error': f'Crypto prediction failed for {symbol}: {str(e)}'}
    
    def _prepare_crypto_features(self, crypto_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare crypto-specific features for ML model"""
        features = crypto_data.copy()
        
        # Add crypto-specific ratios and indicators
        features['Price_to_MA7_Ratio'] = features['Close'] / features['SMA_7']
        features['Price_to_MA30_Ratio'] = features['Close'] / features['SMA_30']
        features['Volume_Trend'] = features['Volume'].rolling(window=3).mean()
        features['Volatility_Ratio'] = features['Volatility'] / features['Volatility'].rolling(window=20).mean()
        
        # RSI-based signals
        features['RSI_Signal'] = np.where(features['RSI'] > 70, -1, 
                                         np.where(features['RSI'] < 30, 1, 0))
        
        # MACD signals
        features['MACD_Signal_Cross'] = np.where(features['MACD'] > features['MACD_Signal'], 1, -1)
        
        # Support/Resistance analysis
        features['Distance_from_Resistance'] = (features['Resistance'] - features['Close']) / features['Close']
        features['Distance_from_Support'] = (features['Close'] - features['Support']) / features['Close']
        
        # Clean up NaN values
        features = features.fillna(method='forward').fillna(method='backward').fillna(0)
        
        # Select relevant features for prediction
        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Histogram',
            'Price_to_MA7_Ratio', 'Price_to_MA30_Ratio', 'Volatility_Ratio',
            'RSI_Signal', 'MACD_Signal_Cross', 'Distance_from_Resistance',
            'Distance_from_Support', 'Momentum_7', 'Momentum_30'
        ]
        
        return features[feature_columns]
    
    def _generate_crypto_prediction(self, features: pd.DataFrame, 
                                   crypto_data: pd.DataFrame, days_ahead: int) -> Dict:
        """Generate crypto-specific price prediction"""
        if len(features) < 50:  # Need sufficient data for ML
            return self._generate_simple_crypto_prediction(crypto_data, days_ahead)
        
        try:
            # Create target variable (future price)
            crypto_data['Future_Price'] = crypto_data['Close'].shift(-days_ahead)
            crypto_data['Price_Change_Future'] = crypto_data['Future_Price'] / crypto_data['Close'] - 1
            
            # Remove rows with NaN targets
            valid_data = crypto_data.dropna(subset=['Future_Price'])
            
            if len(valid_data) < 20:
                return self._generate_simple_crypto_prediction(crypto_data, days_ahead)
            
            # Prepare training data
            X = features.loc[valid_data.index]
            y_direction = (valid_data['Price_Change_Future'] > 0).astype(int)
            y_magnitude = valid_data['Price_Change_Future']
            
            # Train models
            if len(X) > 10:
                X_train, X_test, y_dir_train, y_dir_test = train_test_split(
                    X, y_direction, test_size=0.2, random_state=42)
                mag_train, mag_test, _, _ = train_test_split(
                    X, y_magnitude, test_size=0.2, random_state=42)
                
                # Direction prediction
                direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
                direction_model.fit(X_train, y_dir_train)
                
                # Magnitude prediction
                magnitude_model = RandomForestRegressor(n_estimators=100, random_state=42)
                magnitude_model.fit(X_train, mag_train)
                
                # Generate prediction
                latest_features = X.iloc[-1:].values
                direction_prob = direction_model.predict_proba(latest_features)[0]
                direction = 'bullish' if direction_prob[1] > 0.5 else 'bearish'
                confidence = max(direction_prob)
                
                # Magnitude prediction
                predicted_change = magnitude_model.predict(latest_features)[0]
                current_price = crypto_data['Close'].iloc[-1]
                predicted_price = current_price * (1 + predicted_change)
                
                return {
                    'symbol': crypto_data.index.name or 'Unknown',
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'predicted_change': float(predicted_change),
                    'predicted_change_pct': float(predicted_change * 100),
                    'predicted_direction': direction,
                    'confidence': float(confidence),
                    'prediction_method': 'ML_Based',
                    'days_ahead': days_ahead,
                    'model_accuracy': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.6 else 'Low'
                }
            else:
                return self._generate_simple_crypto_prediction(crypto_data, days_ahead)
                
        except Exception as e:
            print(f"ML prediction failed, falling back to simple prediction: {str(e)}")
            return self._generate_simple_crypto_prediction(crypto_data, days_ahead)
    
    def _generate_simple_crypto_prediction(self, crypto_data: pd.DataFrame, days_ahead: int) -> Dict:
        """Generate simple crypto prediction based on technical indicators"""
        latest = crypto_data.iloc[-1]
        
        # Simple trend analysis
        price_change_7d = latest.get('Momentum_7', 0)
        price_change_30d = latest.get('Momentum_30', 0)
        rsi = latest.get('RSI', 50)
        
        # Scoring system
        bullish_signals = 0
        bearish_signals = 0
        trading_signals = []
        
        # RSI signals
        if rsi < 30:
            bullish_signals += 1
            trading_signals.append("RSI oversold - potential buying opportunity")
        elif rsi > 70:
            bearish_signals += 1
            trading_signals.append("RSI overbought - potential selling opportunity")
        
        # Moving average signals
        if latest.get('Close', 0) > latest.get('SMA_7', 0):
            bullish_signals += 1
            trading_signals.append("Price above 7-day MA - bullish trend")
        else:
            bearish_signals += 1
            trading_signals.append("Price below 7-day MA - bearish trend")
        
        # MACD signals
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        if macd > macd_signal:
            bullish_signals += 1
            trading_signals.append("MACD above signal - bullish momentum")
        else:
            bearish_signals += 1
            trading_signals.append("MACD below signal - bearish momentum")
        
        # Volume analysis
        volume_ratio = latest.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:
            trading_signals.append(f"High volume activity (2x average) - strong interest")
        
        # Momentum analysis
        if price_change_7d > 0.05:  # 5% gain in 7 days
            bullish_signals += 1
            trading_signals.append("Strong positive momentum over 7 days")
        elif price_change_7d < -0.05:  # 5% loss in 7 days
            bearish_signals += 1
            trading_signals.append("Strong negative momentum over 7 days")
        
        # Determine overall direction
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            direction = 'neutral'
            confidence = 0.5
        else:
            bullish_ratio = bullish_signals / total_signals
            direction = 'bullish' if bullish_ratio > 0.6 else 'bearish' if bullish_ratio < 0.4 else 'neutral'
            confidence = max(bullish_ratio, 1 - bullish_ratio)
        
        # Simple price prediction (trend continuation)
        current_price = latest.get('Close', 0)
        
        # More conservative prediction for crypto due to volatility
        if direction == 'bullish':
            predicted_change = 0.02 + (confidence - 0.5) * 0.04  # 2-4% potential gain
            recommendation = 'BUY'
        elif direction == 'bearish':
            predicted_change = -0.02 - (confidence - 0.5) * 0.04  # 2-4% potential loss
            recommendation = 'SELL'
        else:
            predicted_change = 0
            recommendation = 'HOLD'
        
        predicted_price = current_price * (1 + predicted_change)
        
        return {
            'symbol': 'Crypto',
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_change': float(predicted_change),
            'predicted_change_pct': float(predicted_change * 100),
            'predicted_direction': direction,
            'confidence': float(confidence),
            'prediction_method': 'Technical_Analysis',
            'trading_signals': trading_signals,
            'recommendation': recommendation,
            'days_ahead': days_ahead,
            'model_accuracy': 'Technical',
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    
    def _analyze_crypto_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment for cryptocurrency (placeholder for future implementation)"""
        # For now, return neutral sentiment for crypto
        # In a full implementation, this would analyze crypto news, social media, etc.
        return {
            'sentiment_analysis': {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_percentages': {'positive': 33, 'neutral': 34, 'negative': 33},
                'key_themes': ['technical_analysis', 'price_momentum'],
                'news_count': 0,
                'sentiment_sources': ['technical_indicators']
            }
        }
    
    def _generate_crypto_trading_signals(self, crypto_data: pd.DataFrame, prediction: Dict) -> Dict:
        """Generate crypto-specific trading signals"""
        signals = []
        
        latest = crypto_data.iloc[-1]
        current_price = latest.get('Close', 0)
        
        # Risk management for crypto
        volatility = latest.get('Volatility', 0)
        if volatility > current_price * 0.1:  # High volatility (10% of price)
            signals.append("⚠️ High volatility detected - consider position sizing")
        
        # Support/Resistance levels
        support = latest.get('Support', current_price * 0.95)
        resistance = latest.get('Resistance', current_price * 1.05)
        
        if current_price < support * 1.02:  # Close to support
            signals.append(f"📈 Near support level (${support:.2f}) - potential bounce")
        elif current_price > resistance * 0.98:  # Close to resistance
            signals.append(f"📉 Near resistance level (${resistance:.2f}) - potential breakout")
        
        # Volume confirmation
        volume_ratio = latest.get('Volume_Ratio', 1)
        if volume_ratio > 2:
            signals.append("📊 High volume confirms price movement")
        elif volume_ratio < 0.5:
            signals.append("📉 Low volume - weak price confirmation")
        
        return {
            'trading_signals': prediction.get('trading_signals', []) + signals,
            'risk_level': 'High' if volatility > current_price * 0.15 else 'Medium' if volatility > current_price * 0.08 else 'Low',
            'support_level': support,
            'resistance_level': resistance
        }
    
    def analyze_crypto_portfolio(self, crypto_symbols: List[str], days_ahead: int = 7) -> Dict:
        """Analyze multiple cryptocurrencies"""
        analysis_results = {}
        
        for symbol in crypto_symbols:
            print(f"Analyzing {symbol}...")
            prediction = self.predict_crypto(symbol, days_ahead)
            
            # Get additional crypto data
            crypto_info = self.crypto_collector.get_crypto_info(symbol)
            crypto_data = self.crypto_collector.get_crypto_data(symbol, days=7)
            
            analysis_results[symbol] = {
                'prediction': prediction,
                'crypto_info': crypto_info,
                'current_data': {
                    'price_change_pct': crypto_info.get('price_change_24h', 0),
                    'volume_24h': crypto_info.get('total_volume', 0),
                    'market_cap': crypto_info.get('market_cap', 0)
                } if crypto_info else {}
            }
            
        return analysis_results


if __name__ == "__main__":
    # Example usage
    predictor = MarketPredictor()
    
    # Test with a single symbol
    test_symbol = 'AAPL'
    
    print(f"Training model and generating prediction for {test_symbol}...")
    prediction = predictor.predict_price_movement(test_symbol)
    
    print("Prediction Results:")
    print(json.dumps(prediction, indent=2))