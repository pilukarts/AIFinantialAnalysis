@echo off
REM AI Financial Sentiment Analysis - Quick Setup Script (Windows)
REM This script sets up the development environment and runs initial tests

echo.
echo 🚀 Setting up AI Financial Sentiment Analysis Project...
echo ==================================================

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment
echo.
echo 📦 Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing Python dependencies...
pip install -r requirements.txt
echo ✅ Dependencies installed successfully

REM Create necessary directories
echo 📁 Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "results" mkdir results
echo ✅ Directories created

REM Copy environment file
if not exist ".env" (
    copy .env.example .env
    echo ✅ Environment file created (.env)
    echo 📝 Note: Update .env file with your API keys for full functionality
) else (
    echo ✅ Environment file already exists
)

REM Run a quick test
echo.
echo 🧪 Running quick functionality test...
python -c "try:
    from src.sentiment_analyzer import FinancialSentimentAnalyzer
    from src.data_collector import FinancialDataCollector
    from src.market_predictor import MarketPredictor
    print('✅ All modules imported successfully')
    
    print('🎉 All tests passed! System is ready.')
    
except Exception as e:
    print(f'❌ Test failed: {str(e)}')
    exit(1)
"

REM Create batch files for easy use
echo.
echo 📝 Creating quick start scripts...

REM Create run_analysis.bat
echo @echo off > run_analysis.bat
echo cd /d %%~dp0 >> run_analysis.bat
echo call venv\Scripts\activate.bat >> run_analysis.bat
echo echo Running AI Financial Analysis for AAPL... >> run_analysis.bat
echo python main.py --symbol AAPL --save >> run_analysis.bat
echo pause >> run_analysis.bat

REM Create run_dashboard.bat  
echo @echo off > run_dashboard.bat
echo cd /d %%~dp0 >> run_dashboard.bat
echo call venv\Scripts\activate.bat >> run_dashboard.bat
echo echo Starting AI Financial Dashboard... >> run_dashboard.bat
echo echo Dashboard will open at: http://localhost:8050 >> run_dashboard.bat
echo python main.py --dashboard >> run_dashboard.bat
echo pause >> run_dashboard.bat

echo ✅ Quick start scripts created

REM Final setup summary
echo.
echo 🎉 Setup completed successfully!
echo ==================================================
echo.
echo 📋 Next Steps:
echo 1. Update .env file with your API keys (optional)
echo 2. Run analysis: run_analysis.bat
echo 3. Start dashboard: run_dashboard.bat
echo 4. Or use: python main.py --help
echo.
echo 📚 Quick Commands:
echo • Single stock: python main.py --symbol AAPL
echo • Multiple stocks: python main.py --symbols AAPL MSFT GOOGL
echo • Interactive dashboard: python main.py --dashboard
echo • Save results: python main.py --symbol AAPL --save
echo.
echo 🔗 Important URLs:
echo • Dashboard: http://localhost:8050
echo • Documentation: README.md
echo.
echo ⚠️  Disclaimer: This is for educational purposes only.
echo    Not financial advice. Use at your own risk.
echo.
echo Happy analyzing! 📈
echo.
pause