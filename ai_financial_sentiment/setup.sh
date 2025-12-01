#!/bin/bash

# AI Financial Sentiment Analysis - Quick Setup Script
# This script sets up the development environment and runs initial tests

echo "🚀 Setting up AI Financial Sentiment Analysis Project..."
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version detected - Version OK"
else 
    echo "❌ Python $python_version detected - Requires Python 3.8+"
    exit 1
fi

# Create virtual environment
echo "\\n📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed successfully"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data models logs results
echo "✅ Directories created"

# Copy environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Environment file created (.env)"
    echo "📝 Note: Update .env file with your API keys for full functionality"
else
    echo "✅ Environment file already exists"
fi

# Run a quick test
echo "\\n🧪 Running quick functionality test..."
python3 -c "
try:
    from src.sentiment_analyzer import FinancialSentimentAnalyzer
    from src.data_collector import FinancialDataCollector
    from src.market_predictor import MarketPredictor
    print('✅ All modules imported successfully')
    
    print('🎉 All tests passed! System is ready.')
    
except Exception as e:
    print(f'❌ Test failed: {str(e)}')
    exit(1)
"

# Create quick start script
echo "\\n📝 Creating quick start scripts..."

# Create run_analysis.sh
cat > run_analysis.sh << 'EOF'
#!/bin/bash
# Quick analysis script
source venv/bin/activate
echo "Running AI Financial Analysis for AAPL..."
python main.py --symbol AAPL --save
EOF

# Create run_dashboard.sh
cat > run_dashboard.sh << 'EOF'
#!/bin/bash
# Quick dashboard script
source venv/bin/activate
echo "Starting AI Financial Dashboard..."
echo "Dashboard will open at: http://localhost:8050"
python main.py --dashboard
EOF

chmod +x run_analysis.sh run_dashboard.sh
echo "✅ Quick start scripts created"

# Final setup summary
echo "\\n🎉 Setup completed successfully!"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo "1. Update .env file with your API keys (optional)"
echo "2. Run analysis: ./run_analysis.sh"
echo "3. Start dashboard: ./run_dashboard.sh"
echo "4. Or use: python main.py --help"
echo ""
echo "📚 Quick Commands:"
echo "• Single stock: python main.py --symbol AAPL"
echo "• Multiple stocks: python main.py --symbols AAPL MSFT GOOGL"
echo "• Interactive dashboard: python main.py --dashboard"
echo "• Save results: python main.py --symbol AAPL --save"
echo ""
echo "🔗 Important URLs:"
echo "• Dashboard: http://localhost:8050"
echo "• Documentation: README.md"
echo ""
echo "⚠️  Disclaimer: This is for educational purposes only."
echo "   Not financial advice. Use at your own risk."
echo ""
echo "Happy analyzing! 📈"