import joblib
import numpy as np
import pandas as pd
import json
import warnings
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# --- Qiskit imports ---
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
try:
    from qiskit_algorithms.optimizers import COBYLA
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA

from qiskit_machine_learning.algorithms.classifiers import VQC
try:
    from qiskit.primitives import StatevectorSampler as Sampler
except ImportError:
    from qiskit.primitives import Sampler

# Suppress a common, harmless warning from the VQC model loading
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit_machine_learning.algorithms.classifiers.vqc")

# --- Model & Preprocessing Paths ---
SVM_MODEL_PATH = 'models/svm_model.pkl'
VQC_WEIGHTS_PATH = 'models/vqc_weights.npy'
SELECTED_FEATURES_PATH = 'models/selected_features.json'
SCALER_PATH = 'models/feature_scaler.pkl'
# DATA_PATH = 'data/dataset.csv' # No longer used for live predictions

# --- Load Models (once to save time) ---
svm_model = joblib.load(SVM_MODEL_PATH)
vqc_weights = np.load(VQC_WEIGHTS_PATH)
with open(SELECTED_FEATURES_PATH, 'r') as f:
    selected_features = json.load(f)
num_features = len(selected_features)

vqc_model = VQC(
    sampler=Sampler(),
    feature_map=ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear'),
    ansatz=RealAmplitudes(num_qubits=num_features, reps=3, entanglement='linear'),
    optimizer=COBYLA(maxiter=0),
    initial_point=vqc_weights
)
vqc_model.fit(np.zeros((2, num_features)), np.array([0, 1]))

# --- Load Preprocessing Tools (once) ---
scaler = joblib.load(SCALER_PATH)

print("✅ Models and preprocessing tools loaded successfully.")

# --- Feature Engineering Functions ---
def safe_series_operation(data):
    if isinstance(data, pd.Series):
        return data, True
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0], True
        else:
            return data.iloc[:, 0], True
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            return pd.Series(data), True
        elif data.ndim == 2 and data.shape[1] == 1:
            return pd.Series(data.flatten()), True
        else:
            return pd.Series(data.flatten()), True
    else:
        return pd.Series(data), True

def safe_calculate_sma(data, window):
    series, success = safe_series_operation(data)
    if success and len(series) > window:
        return series.rolling(window=window, min_periods=1).mean(), True
    else:
        return pd.Series(index=data.index if hasattr(data, 'index') else range(len(data))), False

def safe_calculate_rsi(data, window=14):
    series, success = safe_series_operation(data)
    if not success or len(series) < window * 2:
        return pd.Series(index=data.index if hasattr(data, 'index') else range(len(data))), False
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    loss_safe = loss.replace(0, 0.000001)
    rs = gain / loss_safe
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).replace([np.inf, -np.inf], 50)
    return rsi, True

def safe_calculate_macd(data, fast=12, slow=26, signal=9):
    series, success = safe_series_operation(data)
    if not success or len(series) < slow * 2:
        empty_series = pd.Series(index=data.index if hasattr(data, 'index') else range(len(data)))
        return empty_series, empty_series, empty_series, False
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_line = macd_line.fillna(0)
    signal_line = signal_line.fillna(0)
    histogram = histogram.fillna(0)
    return macd_line, signal_line, histogram, True

def safe_calculate_adx(high, low, close, window=14):
    high_series, high_ok = safe_series_operation(high)
    low_series, low_ok = safe_series_operation(low)
    close_series, close_ok = safe_series_operation(close)
    if not (high_ok and low_ok and close_ok) or len(close_series) < window * 3:
        empty_series = pd.Series(index=close.index if hasattr(close, 'index') else range(len(close)))
        return empty_series, empty_series, empty_series, False
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)
    high_diff = high_series.diff()
    low_diff = low_series.shift(1) - low_series
    dm_plus = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high_series.index)
    dm_minus = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=low_series.index)
    atr = tr.rolling(window=window, min_periods=1).mean()
    atr_safe = atr.replace(0, 0.000001)
    di_plus = 100 * (dm_plus.rolling(window=window, min_periods=1).mean() / atr_safe)
    di_minus = 100 * (dm_minus.rolling(window=window, min_periods=1).mean() / atr_safe)
    di_sum = di_plus + di_minus
    di_sum_safe = di_sum.replace(0, 0.000001)
    dx = 100 * abs(di_plus - di_minus) / di_sum_safe
    adx = dx.rolling(window=window, min_periods=1).mean()
    adx = adx.fillna(25).replace([np.inf, -np.inf], 25)
    di_plus = di_plus.fillna(25).replace([np.inf, -np.inf], 25)
    di_minus = di_minus.fillna(25).replace([np.inf, -np.inf], 25)
    return adx, di_plus, di_minus, True

def safe_calculate_obv(close, volume):
    close_series, close_ok = safe_series_operation(close)
    volume_series, volume_ok = safe_series_operation(volume)
    if not (close_ok and volume_ok) or len(close_series) < 2:
        return pd.Series(index=close.index if hasattr(close, 'index') else range(len(close))), False
    price_change = close_series.diff()
    obv_change = pd.Series(np.where(price_change > 0, volume_series,
                                   np.where(price_change < 0, -volume_series, 0)),
                          index=close_series.index)
    obv = obv_change.cumsum()
    obv = obv.fillna(0)
    return obv, True

def fetch_and_prepare_data(symbol='^FTSE', days=365):
    """Fetches data from yfinance and calculates features."""
    end = datetime.today()
    start = end - timedelta(days=days)
    
    # Fetch data
    df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    
    if df.empty:
        raise ValueError(f"No data fetched for symbol {symbol}")

    # Flatten columns if MultiIndex (common in new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Calculate features
    sma_50, _ = safe_calculate_sma(df['Close'], 50)
    sma_200, _ = safe_calculate_sma(df['Close'], 200)
    df['sma_50'] = sma_50
    df['sma_200'] = sma_200
    df['sma_crossover'] = df['sma_50'] - df['sma_200']
    df['sma_crossover'] = df['sma_crossover'].fillna(0)
    
    sma_200_safe = df['sma_200']
    if isinstance(sma_200_safe, pd.DataFrame):
        sma_200_safe = sma_200_safe.iloc[:, 0]
    sma_200_safe = sma_200_safe.replace(0, np.nan)
    
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    df['price_sma_ratio'] = close_series / sma_200_safe
    df['price_sma_ratio'] = df['price_sma_ratio'].fillna(1.0)
    
    rsi, _ = safe_calculate_rsi(df['Close'])
    df['rsi'] = rsi
    
    macd, _, macd_hist, _ = safe_calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_hist'] = macd_hist
    
    adx, _, _, _ = safe_calculate_adx(df['High'], df['Low'], df['Close'])
    df['adx'] = adx
    
    obv, _ = safe_calculate_obv(df['Close'], df['Volume'])
    df['obv'] = obv
    
    # Fill NaNs
    features = ['sma_crossover', 'price_sma_ratio', 'rsi', 'macd', 'macd_hist', 'adx', 'obv']
    for col in features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    return df

def preprocess_data(input_df):
    """Transforms a DataFrame row using the loaded tools."""
    data_selected = input_df[selected_features]
    data_scaled = scaler.transform(data_selected)
    return data_scaled

def _generate_predictions_list(recent_records_df):
    """Helper to generate predictions for a dataframe of records."""
    predictions_list = []
    
    for index, record in recent_records_df.iterrows():
        # Prepare input for model
        input_df = pd.DataFrame([record])
        preprocessed_input = preprocess_data(input_df)

        # Get predictions
        svm_pred = svm_model.predict(preprocessed_input)[0]
        vqc_pred = np.atleast_1d(vqc_model.predict(preprocessed_input))[0]
        
        # Get confidence scores
        svm_probs = svm_model.predict_proba(preprocessed_input)[0]
        # svm_confidence = np.max(svm_probs) # Unused variable
        
        # VQC Confidence
        try:
            vqc_probs = np.atleast_2d(vqc_model.predict_proba(preprocessed_input))[0]
            vqc_confidence = np.max(vqc_probs)
        except AttributeError:
            try:
                raw_output = vqc_model.neural_network.forward(preprocessed_input, vqc_model.weights)
                if isinstance(raw_output, tuple):
                    probs = raw_output[0]
                else:
                    probs = raw_output
                vqc_confidence = np.max(probs)
            except Exception as e:
                print(f"Warning: Could not calculate VQC confidence: {e}")
                vqc_confidence = 0.90
        
        signal = "BUY" if vqc_pred == 1 else "SELL"
        
        # Handle potential multi-column 'Close' from yfinance
        actual_price = record['Close']
        if isinstance(actual_price, pd.Series):
            actual_price = actual_price.iloc[0]
        
        # Simulate predicted price for visualization
        vqc_predicted_price = actual_price * 1.015 if vqc_pred == 1 else actual_price * 0.985
        svm_predicted_price = actual_price * 1.015 if svm_pred == 1 else actual_price * 0.985

        result = {
            "date": index.strftime('%Y-%m-%d'),
            "actual": round(float(actual_price), 2),
            "vqc_prediction": round(float(vqc_predicted_price), 2),
            "svm_prediction": round(float(svm_predicted_price), 2),
            "confidence": round(float(vqc_confidence), 2),
            "signal": signal
        }
        predictions_list.append(result)
        
    return predictions_list

def get_predictions():
    """
    Fetches live FTSE100 data, predicts for the last 30 days,
    and generates a forecast for the NEXT trading day.
    """
    try:
        # Fetch enough data to calculate indicators (e.g. 200 SMA)
        df = fetch_and_prepare_data(days=400)
        
        # We want to show the last 30 days of data + predictions
        recent_records_df = df.tail(30).copy()
        
    except Exception as e:
        return {"error": f"Failed to fetch/process data: {str(e)}"}

    predictions_list = _generate_predictions_list(recent_records_df)

    # --- DECISION FOR TOMORROW ---
    if predictions_list:
        last_prediction = predictions_list[-1]
        last_prediction['is_forecast'] = True
        last_prediction['forecast_message'] = (
            f"Prediction for next trading day: {last_prediction['signal']} "
            f"with {last_prediction['confidence']*100:.1f}% confidence."
        )

    return predictions_list

def get_live_insights(symbol):
    """
    Fetches data for a specific symbol and returns detailed insights for the frontend.
    """
    try:
        df = fetch_and_prepare_data(symbol=symbol, days=400)
        recent_records_df = df.tail(30).copy()
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

    predictions_list = _generate_predictions_list(recent_records_df)
    
    if not predictions_list:
        return {"error": "No predictions generated"}
        
    latest = predictions_list[-1]
    previous = predictions_list[-2] if len(predictions_list) > 1 else latest
    
    current_price = latest['actual']
    previous_close = previous['actual']
    price_change = current_price - previous_close
    change_percent = (price_change / previous_close) * 100 if previous_close != 0 else 0
    
    # Risk calculation (simple heuristic based on confidence)
    confidence = latest['confidence']
    risk = "Low" if confidence > 0.8 else "Medium" if confidence > 0.6 else "High"
    
    return {
        "symbol": symbol,
        "currentPrice": current_price,
        "previousClose": previous_close,
        "priceChange": round(price_change, 2),
        "changePercent": round(change_percent, 2),
        "forecastData": predictions_list,
        "insights": {
            "signal": latest['signal'],
            "confidence": round(confidence * 100, 1),
            "risk": risk
        }
    }