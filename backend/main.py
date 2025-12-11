from flask import Flask, jsonify, request
from flask_cors import CORS
from predictions import get_predictions, get_live_insights
from quantum_predictor import (
    get_quantum_prediction,
    get_batch_predictions,
    get_available_models,
    get_selected_features,
    QUANTUM_AVAILABLE
)
from quantum_regime_detector import (
    detect_market_regime,
    get_regime_history,
    get_accuracy_comparison,
    REGIME_DEFINITIONS
)
from datetime import datetime

# Create a Flask application instance
app = Flask(__name__)
# Enable CORS to allow requests from your frontend
CORS(app)

@app.route("/")
def home():
    """A simple route to check if the backend is running."""
    return "Quantum ML Backend is running!"

@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'quantum_available': QUANTUM_AVAILABLE,
        'version': '2.0.0'
    })

@app.route("/api/predictions")
def api_predictions():
    """
    The main API endpoint that the frontend will call.
    It fetches the prediction data and returns it as JSON.
    """
    # Call the function that runs the models on historical data
    prediction_data = get_predictions()
    
    # Return the data in a JSON format
    # The frontend expects a plain array, so we return that directly.
    return jsonify(prediction_data)

@app.route("/api/live-trading/<symbol>")
def api_live_trading(symbol):
    """
    Fetches live trading insights for a specific symbol.
    """
    data = get_live_insights(symbol)
    return jsonify(data)

@app.route("/api/quantum-predict", methods=['POST'])
def quantum_predict():
    """
    Multi-timeframe quantum prediction endpoint.
    
    Request body:
    {
        "ticker": "^FTSE",
        "model": "quantum_vqc",  // or "random_forest", etc.
        "timeframe": "daily"      // "hourly", "daily", or "weekly"
    }
    """
    try:
        data = request.get_json()
        ticker = data.get('ticker', '^FTSE')
        model_name = data.get('model', 'quantum_vqc')
        timeframe = data.get('timeframe', 'daily')
        
        result = get_quantum_prediction(ticker, model_name, timeframe)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route("/api/batch-predict", methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple models.
    """
    try:
        data = request.get_json()
        ticker = data.get('ticker', '^FTSE')
        models = data.get('models', ['quantum_vqc', 'random_forest'])
        timeframe = data.get('timeframe', 'daily')
        
        result = get_batch_predictions(ticker, models, timeframe)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/models")
def list_models():
    """List all available models."""
    try:
        models = get_available_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/features")
def get_features():
    """Get selected features."""
    try:
        features = get_selected_features()
        return jsonify(features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# VOLATILITY REGIME DETECTION ENDPOINTS
# =============================================================================

@app.route("/api/regime/detect", methods=['GET', 'POST'])
def detect_regime():
    """
    Detect current market regime using quantum clustering.
    
    GET: Uses default ^FTSE ticker
    POST: Accepts {"ticker": "^FTSE"} in body
    """
    try:
        ticker = "^FTSE"
        if request.method == 'POST':
            data = request.get_json() or {}
            ticker = data.get('ticker', '^FTSE')
        
        result = detect_market_regime(ticker)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/regime/history", methods=['GET', 'POST'])
def regime_history():
    """
    Get regime history for visualization.
    
    Query params or POST body:
    - ticker: Stock ticker (default: ^FTSE)
    - days: Number of days of history (default: 252)
    """
    try:
        ticker = "^FTSE"
        days = 252
        
        if request.method == 'POST':
            data = request.get_json() or {}
            ticker = data.get('ticker', '^FTSE')
            days = data.get('days', 252)
        else:
            ticker = request.args.get('ticker', '^FTSE')
            days = int(request.args.get('days', 252))
        
        result = get_regime_history(ticker, days)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/regime/accuracy")
def regime_accuracy():
    """
    Get accuracy comparison between base and regime-aware models.
    """
    try:
        ticker = request.args.get('ticker', '^FTSE')
        result = get_accuracy_comparison(ticker)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/regime/definitions")
def regime_definitions():
    """
    Get regime definitions and characteristics.
    """
    return jsonify(REGIME_DEFINITIONS)


# This block ensures the server runs only when the script is executed directly
if __name__ == "__main__":
    print("=" * 60)
    print("   QUANTUM ML BACKEND SERVER")
    print("=" * 60)
    print(f"Quantum Available: {QUANTUM_AVAILABLE}")
    print("\nEndpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/predictions - Historical predictions")
    print("  POST /api/quantum-predict - Quantum prediction")
    print("  POST /api/batch-predict - Batch predictions")
    print("  GET  /api/models - List models")
    print("  GET  /api/features - Get features")
    print("\n🔮 Regime Detection Endpoints:")
    print("  GET/POST /api/regime/detect - Detect current regime")
    print("  GET/POST /api/regime/history - Get regime history")
    print("  GET      /api/regime/accuracy - Model accuracy comparison")
    print("  GET      /api/regime/definitions - Regime definitions")
    print("=" * 60)
    
    # app.run() starts the development server
    # debug=True enables auto-reloading
    # host='0.0.0.0' makes it accessible from your local network
    app.run(host='0.0.0.0', port=5000, debug=True)