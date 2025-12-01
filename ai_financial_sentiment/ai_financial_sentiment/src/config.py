"""
AI Financial Sentiment Analysis & Market Prediction
Configuration and Constants
"""

import os
from typing import Dict, List

# API Keys (set these in your environment)
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'demo')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'demo')

# Stock Symbols to analyze
DEFAULT_STOCKS = [
    'AAPL',    # Apple
    'MSFT',    # Microsoft  
    'GOOGL',   # Google
    'AMZN',    # Amazon
    'TSLA',    # Tesla
    'META',    # Meta
    'NVDA',    # NVIDIA
    'NFLX'     # Netflix
]

# News keywords for financial sentiment
FINANCIAL_KEYWORDS = [
    'earnings', 'revenue', 'profit', 'loss', 'stock price', 'market',
    'quarterly', 'annual', 'dividend', 'merger', 'acquisition', 'IPO',
    'growth', 'decline', 'bullish', 'bearish', 'volatility', 'trading'
]

# Model Configuration
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Dashboard Configuration
DASHBOARD_PORT = 8050
DASHBOARD_HOST = 'localhost'

# Data Paths
DATA_PATH = 'data'
MODELS_PATH = 'models'
LOGS_PATH = 'logs'

# Visualization Colors
SENTIMENT_COLORS = {
    'positive': '#2E8B57',    # Sea Green
    'negative': '#DC143C',    # Crimson  
    'neutral': '#708090'      # Slate Gray
}

MARKET_COLORS = {
    'bullish': '#32CD32',     # Lime Green
    'bearish': '#FF4500',     # Orange Red
    'sideways': '#FFD700'     # Gold
}

# Analysis Parameters
SENTIMENT_WINDOW_DAYS = 7
CORRELATION_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.6