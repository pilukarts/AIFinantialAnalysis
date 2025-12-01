"""
Interactive Financial Dashboard
Real-time visualization of AI-powered market analysis
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from market_predictor import MarketPredictor
from sentiment_analyzer import FinancialSentimentAnalyzer
from data_collector import FinancialDataCollector

# Initialize the predictor
predictor = MarketPredictor()
sentiment_analyzer = FinancialSentimentAnalyzer()
data_collector = FinancialDataCollector()

# Dash app setup
app = dash.Dash(__name__)
app.title = "AI Financial Sentiment Dashboard"

# Color schemes
COLORS = {
    'background': '#1e1e1e',
    'surface': '#2d2d2d',
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'danger': '#f44336',
    'warning': '#ff9800',
    'success': '#4CAF50',
    'text': '#ffffff',
    'text_secondary': '#cccccc'
}

# Define layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 AI Financial Sentiment Analysis Dashboard", 
                style={'color': COLORS['text'], 'textAlign': 'center', 'marginBottom': '10px'}),
        html.P("Powered by Advanced Machine Learning & Sentiment Analysis", 
               style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'marginBottom': '20px'})
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Select Stock Symbol:", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[
                    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                    {'label': 'Meta (META)', 'value': 'META'},
                    {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
                    {'label': 'Netflix (NFLX)', 'value': 'NFLX'}
                ],
                value='AAPL',
                style={'marginBottom': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Analysis Type:", style={'color': COLORS['text'], 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='analysis-dropdown',
                options=[
                    {'label': 'Price Prediction', 'value': 'price'},
                    {'label': 'Sentiment Analysis', 'value': 'sentiment'},
                    {'label': 'Technical Analysis', 'value': 'technical'},
                    {'label': 'Market Overview', 'value': 'overview'}
                ],
                value='overview',
                style={'marginBottom': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Button('🔄 Refresh Data', id='refresh-button', 
                       style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                              'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                              'cursor': 'pointer', 'fontSize': '16px'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'backgroundColor': COLORS['surface'], 'margin': '20px'}),
    
    # Main content area
    html.Div(id='main-content', style={'padding': '20px'}),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'})
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# Callback to update data
@app.callback(
    Output('stored-data', 'children'),
    [Input('refresh-button', 'n_clicks')],
    [Input('symbol-dropdown', 'value')]
)
def update_data(n_clicks, selected_symbol):
    """Update stored data when refresh button is clicked or symbol changes"""
    if not selected_symbol:
        return json.dumps({})
    
    try:
        # Collect comprehensive data
        print(f"Updating data for {selected_symbol}...")
        
        # Get stock data
        stock_data = data_collector.get_stock_data(selected_symbol, period="3mo")
        
        # Get company info
        company_info = data_collector.get_company_info(selected_symbol)
        
        # Get news and sentiment
        news_data = data_collector.get_financial_news([selected_symbol], days_back=7)
        news_texts = [article.get('title', '') + ' ' + article.get('summary', '') 
                     for article in news_data if article.get('title')]
        
        sentiment_results = []
        if news_texts:
            sentiment_results = sentiment_analyzer.analyze_sentiment(news_texts)
        
        # Train model and get prediction
        training_results = predictor.train_price_prediction_model(selected_symbol)
        prediction_results = predictor.predict_price_movement(selected_symbol)
        
        # Prepare data for storage
        stored_data = {
            'symbol': selected_symbol,
            'stock_data': stock_data.to_dict('records') if not stock_data.empty else [],
            'company_info': company_info,
            'news_data': news_data,
            'sentiment_results': sentiment_results,
            'training_results': training_results,
            'prediction_results': prediction_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return json.dumps(stored_data)
        
    except Exception as e:
        print(f"Error updating data: {str(e)}")
        return json.dumps({'error': str(e)})

# Main content callback
@app.callback(
    Output('main-content', 'children'),
    [Input('stored-data', 'children')],
    [Input('analysis-dropdown', 'value')]
)
def update_main_content(stored_data_json, analysis_type):
    """Update main content based on selected analysis type"""
    
    if not stored_data_json or stored_data_json == '{}':
        return html.Div([
            html.H3("No data available. Please select a symbol and click refresh.", 
                   style={'color': COLORS['text_secondary'], 'textAlign': 'center'})
        ])
    
    try:
        data = json.loads(stored_data_json)
        symbol = data.get('symbol', '')
        
        if analysis_type == 'overview':
            return create_overview_tab(data)
        elif analysis_type == 'price':
            return create_price_tab(data)
        elif analysis_type == 'sentiment':
            return create_sentiment_tab(data)
        elif analysis_type == 'technical':
            return create_technical_tab(data)
        else:
            return html.Div("Unknown analysis type", style={'color': COLORS['text']})
            
    except Exception as e:
        return html.Div([
            html.H3(f"Error loading data: {str(e)}", 
                   style={'color': COLORS['danger'], 'textAlign': 'center'})
        ])

def create_overview_tab(data):
    """Create overview tab with key metrics and summary"""
    symbol = data.get('symbol', '')
    prediction = data.get('prediction_results', {})
    company_info = data.get('company_info', {})
    sentiment_summary = data.get('sentiment_results', [])
    
    if not sentiment_summary:
        sentiment_summary = {}
    else:
        sentiment_summary = sentiment_analyzer.generate_sentiment_summary(sentiment_summary)
    
    # Key metrics cards
    metrics_cards = html.Div([
        html.Div([
            html.H3(f"${prediction.get('current_price', 0):.2f}", 
                   style={'margin': '0', 'color': COLORS['primary']}),
            html.P("Current Price", style={'margin': '0', 'color': COLORS['text_secondary']})
        ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
                 'textAlign': 'center', 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H3(f"{prediction.get('predicted_direction', 'N/A').title()}", 
                   style={'margin': '0', 'color': COLORS['success'] if prediction.get('predicted_direction') == 'bullish' else COLORS['danger']}),
            html.P("Predicted Direction", style={'margin': '0', 'color': COLORS['text_secondary']})
        ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
                 'textAlign': 'center', 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H3(f"{prediction.get('confidence', 0)*100:.1f}%", 
                   style={'margin': '0', 'color': COLORS['secondary']}),
            html.P("Prediction Confidence", style={'margin': '0', 'color': COLORS['text_secondary']})
        ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
                 'textAlign': 'center', 'width': '22%', 'display': 'inline-block', 'margin': '1%'}),
        
        html.Div([
            html.H3(f"{sentiment_summary.get('overall_sentiment', 'N/A').title()}", 
                   style={'margin': '0', 'color': COLORS['warning']}),
            html.P("News Sentiment", style={'margin': '0', 'color': COLORS['text_secondary']})
        ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
                 'textAlign': 'center', 'width': '22%', 'display': 'inline-block', 'margin': '1%'})
    ])
    
    # Trading signals
    signals = prediction.get('trading_signals', [])
    signals_list = html.Div([
        html.H4("🤖 AI Trading Signals", style={'color': COLORS['text']}),
        html.Ul([html.Li(signal, style={'color': COLORS['text_secondary'], 'marginBottom': '5px'}) 
                for signal in signals])
    ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
             'marginTop': '20px'})
    
    # Company info
    company_section = html.Div([
        html.H4("🏢 Company Information", style={'color': COLORS['text']}),
        html.P(f"Company: {company_info.get('company_name', 'N/A')}", style={'color': COLORS['text_secondary']}),
        html.P(f"Sector: {company_info.get('sector', 'N/A')}", style={'color': COLORS['text_secondary']}),
        html.P(f"Industry: {company_info.get('industry', 'N/A')}", style={'color': COLORS['text_secondary']}),
        html.P(f"Market Cap: ${company_info.get('market_cap', 0):,}", style={'color': COLORS['text_secondary']})
    ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px',
             'marginTop': '20px'})
    
    return html.Div([
        html.H2(f"📊 {symbol} Market Overview", style={'color': COLORS['text'], 'marginBottom': '30px'}),
        metrics_cards,
        signals_list,
        company_section
    ])

def create_price_tab(data):
    """Create price analysis tab"""
    symbol = data.get('symbol', '')
    stock_data = data.get('stock_data', [])
    prediction = data.get('prediction_results', {})
    
    if not stock_data:
        return html.Div("No price data available", style={'color': COLORS['text_secondary']})
    
    # Convert to DataFrame
    df = pd.DataFrame(stock_data)
    
    # Price chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Movement & Predictions', 'Volume Analysis'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(x=df['Date'] if 'Date' in df.columns else df.index, 
                  y=df['Close'], name='Actual Price', 
                  line=dict(color=COLORS['primary'])),
        row=1, col=1
    )
    
    # Add prediction if available
    if prediction.get('predicted_price'):
        fig.add_trace(
            go.Scatter(x=[df.index[-1], df.index[-1] + 1], 
                      y=[prediction['current_price'], prediction['predicted_price']], 
                      name='Predicted Price', 
                      line=dict(color=COLORS['danger'], dash='dash')),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', 
               marker_color=COLORS['secondary'], opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{symbol} Price Analysis with AI Predictions",
        template='plotly_dark',
        height=600,
        showlegend=True
    )
    
    return html.Div([
        html.H2(f"📈 {symbol} Price Analysis", style={'color': COLORS['text'], 'marginBottom': '20px'}),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Price Prediction Summary", style={'color': COLORS['text']}),
            html.P(f"Current Price: ${prediction.get('current_price', 0):.2f}", style={'color': COLORS['text_secondary']}),
            html.P(f"Predicted Price: ${prediction.get('predicted_price', 0):.2f}", style={'color': COLORS['text_secondary']}),
            html.P(f"Expected Change: {prediction.get('predicted_change', 0)*100:.2f}%", style={'color': COLORS['text_secondary']}),
            html.P(f"Recommendation: {prediction.get('recommendation', 'N/A')}", style={'color': COLORS['success']})
        ], style={'backgroundColor': COLORS['surface'], 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
    ])

def create_sentiment_tab(data):
    """Create sentiment analysis tab"""
    symbol = data.get('symbol', '')
    sentiment_results = data.get('sentiment_results', [])
    news_data = data.get('news_data', [])
    
    if not sentiment_results:
        return html.Div("No sentiment data available", style={'color': COLORS['text_secondary']})
    
    # Sentiment distribution
    sentiment_summary = sentiment_analyzer.generate_sentiment_summary(sentiment_results)
    sentiment_counts = sentiment_summary.get('sentiment_distribution', {})
    
    fig_sentiment = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title=f"{symbol} News Sentiment Distribution",
        color_discrete_map={
            'positive': COLORS['success'],
            'negative': COLORS['danger'], 
            'neutral': COLORS['warning']
        }
    )
    fig_sentiment.update_layout(template='plotly_dark', height=400)
    
    # Recent news with sentiment
    news_items = []
    for i, (news, sentiment) in enumerate(zip(news_data[:10], sentiment_results[:10])):
        news_items.append({
            'title': news.get('title', '')[:50] + '...',
            'sentiment': sentiment.get('sentiment', 'neutral'),
            'confidence': sentiment.get('confidence', 0),
            'publisher': news.get('publisher', 'Unknown')
        })
    
    # News sentiment table
    fig_news = go.Figure(data=[go.Table(
        header=dict(values=['Title', 'Sentiment', 'Confidence', 'Publisher'],
                   fill_color=COLORS['surface'],
                   align='left',
                   font=dict(color=COLORS['text'])),
        cells=dict(values=[[item['title'] for item in news_items],
                          [item['sentiment'] for item in news_items],
                          [f"{item['confidence']:.2f}" for item in news_items],
                          [item['publisher'] for item in news_items]],
                  fill_color=COLORS['background'],
                  align='left',
                  font=dict(color=COLORS['text_secondary']))
    )])
    
    fig_news.update_layout(title="Recent News Sentiment Analysis", template='plotly_dark', height=400)
    
    return html.Div([
        html.H2(f"💭 {symbol} Sentiment Analysis", style={'color': COLORS['text'], 'marginBottom': '20px'}),
        dcc.Graph(figure=fig_sentiment),
        html.Br(),
        dcc.Graph(figure=fig_news)
    ])

def create_technical_tab(data):
    """Create technical analysis tab"""
    symbol = data.get('symbol', '')
    stock_data = data.get('stock_data', [])
    
    if not stock_data:
        return html.Div("No technical data available", style={'color': COLORS['text_secondary']})
    
    df = pd.DataFrame(stock_data)
    
    # Technical indicators chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price', 
                  line=dict(color=COLORS['primary'])),
        row=1, col=1
    )
    
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(color=COLORS['danger'], dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(color=COLORS['success'], dash='dot')),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                      line=dict(color=COLORS['warning'])),
            row=2, col=1
        )
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                      line=dict(color=COLORS['secondary'])),
            row=3, col=1
        )
        if 'MACD_Signal' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                          line=dict(color=COLORS['danger'])),
                row=3, col=1
            )
    
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return html.Div([
        html.H2(f"📊 {symbol} Technical Analysis", style={'color': COLORS['text'], 'marginBottom': '20px'}),
        dcc.Graph(figure=fig)
    ])

if __name__ == '__main__':
    print("Starting AI Financial Sentiment Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    app.run_server(debug=True, host='0.0.0.0', port=8050)