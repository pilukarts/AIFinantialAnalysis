# 🔧 SOLUCIÓN DE ERRORES - AI Financial Sentiment Analysis

## ❌ Problemas Identificados

Los errores que experimentaste se deben a problemas de importación de módulos:

1. **ModuleNotFoundError: No module named 'sentiment_analyzer'**
2. **Error en scripts de Windows (setup.bat)**

## ✅ SOLUCIONES INMEDIATAS

### **Opción 1: Usar Versión Simplificada (RECOMENDADO)**

```bash
# La versión simplificada NO requiere dependencias pesadas
python main_simple.py --symbol AAPL

# Con múltiples stocks
python main_simple.py --symbols AAPL MSFT GOOGL --save

# Con dashboard demo
python main_simple.py --dashboard
```

### **Opción 2: Corregir Importaciones en Versión Completa**

Si quieres usar la versión completa, edita estos archivos:

**En `/src/market_predictor.py`:**
```python
# Línea 20-21, cambiar a:
try:
    from .sentiment_analyzer import FinancialSentimentAnalyzer
    from .data_collector import FinancialDataCollector
except ImportError:
    from sentiment_analyzer import FinancialSentimentAnalyzer
    from data_collector import FinancialDataCollector
```

**En `/src/data_collector.py`:**
```python
# Al inicio del archivo, añadir:
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### **Opción 3: Setup Manual para Windows**

1. **Crear entorno virtual:**
```cmd
python -m venv venv
venv\Scripts\activate
```

2. **Instalar solo dependencias básicas:**
```cmd
pip install pandas numpy matplotlib seaborn plotly dash requests beautifulsoup4
```

3. **Probar imports:**
```cmd
python -c "from src.sentiment_analyzer import FinancialSentimentAnalyzer; print('OK')"
```

4. **Ejecutar análisis simple:**
```cmd
python main_simple.py --symbol AAPL
```

## 🚀 COMANDOS QUE FUNCIONAN AHORA

### **Demo Inmediato (sin dependencias):**
```bash
python main_simple.py
python main_simple.py --symbol AAPL
python main_simple.py --symbols AAPL MSFT GOOGL --save
python main_simple.py --dashboard
```

### **Análisis Completo (con setup correcto):**
```bash
# Opción A: Usar versión simple
python main_simple.py --symbol AAPL --save

# Opción B: Setup completo con dependencias
bash setup.sh  # En Linux/Mac
setup.bat      # En Windows

# Luego usar scripts generados
run_analysis.bat   # En Windows
./run_analysis.sh  # En Linux/Mac
```

## 📱 QUICK START - 3 Comandos

### **Para Probar Inmediatamente:**
```bash
# 1. Ver ayuda
python main_simple.py --help

# 2. Análisis básico
python main_simple.py --symbol AAPL

# 3. Análisis completo con guardado
python main_simple.py --symbols AAPL MSFT GOOGL --save
```

### **Para Usar Dashboard:**
```bash
python main_simple.py --dashboard
```
Luego abrir: http://localhost:8050

## 🔧 Debugging Paso a Paso

### **1. Verificar Python:**
```cmd
python --version
```

### **2. Verificar estructura de archivos:**
```cmd
dir src
# Debe mostrar: sentiment_analyzer.py, data_collector.py, market_predictor.py
```

### **3. Probar imports uno por uno:**
```cmd
python -c "import sys; print('Python OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import numpy; print('Numpy OK')"
python -c "from src.sentiment_analyzer import FinancialSentimentAnalyzer; print('Analyzer OK')"
```

### **4. Si fallan los imports:**
- Usar `main_simple.py` que no requiere dependencias pesadas
- O instalar: `pip install pandas numpy matplotlib seaborn plotly dash`

## 📊 Salida Esperada

### **Demo Simple:**
```
🎯 AI Financial Sentiment Analysis - Demo Mode
============================================================
This demo showcases AI-powered financial analysis capabilities.

=== Running AI Analysis for AAPL ===

1. Analyzing financial news...
2. Generating price predictions...
3. Calculating technical indicators...
4. Analyzing sentiment-price correlations...
5. Creating comprehensive report...

============================================================
📊 AI FINANCIAL ANALYSIS SUMMARY REPORT
============================================================

🏢 Company: Apple Inc.
📈 Symbol: AAPL
💰 Current Price: $173.45
📊 24h Change: +2.34%

💭 SENTIMENT ANALYSIS:
   Overall Sentiment: Positive
   Average Confidence: 81.5%
   Articles Analyzed: 3
   Distribution: 2 positive, 0 negative, 1 neutral

🤖 AI PREDICTION:
   Predicted Direction: Bullish
   Confidence: 78.5%
   Expected Change: +3.2%
   Model Accuracy: 74.6%

📊 TECHNICAL INDICATORS:
   RSI: 45.2
   MACD: 1.45
   Volume Trend: Increasing

🎯 TRADING RECOMMENDATION:
   Action: BUY
   Risk Level: MEDIUM
   Target Price: $178.98
   Reasoning:
   • BUY signal: Model predicts positive movement with high confidence
   • Technical indicators show RSI at 45.2 - neutral

============================================================
✅ Analysis completed successfully!
============================================================
```

## 🎯 RECOMENDACIÓN FINAL

**Para uso inmediato y sin problemas:**

1. **Usa `main_simple.py`** - Funciona sin dependencias pesadas
2. **Los comandos básicos funcionan:**
   ```bash
   python main_simple.py --symbol AAPL
   python main_simple.py --symbols AAPL MSFT GOOGL --save
   python main_simple.py --dashboard
   ```

3. **Para versión completa** - Sigue las instrucciones de instalación de dependencias

## 🆘 Si Aún Tienes Problemas

1. **Usa siempre la versión simple:** `main_simple.py`
2. **Ejecuta el demo standalone:** `python demo_standalone.py`
3. **Verifica que Python esté instalado:** `python --version`
4. **Revisa la estructura de archivos** según el README

¡El proyecto está diseñado para funcionar inmediatamente con la versión simple!