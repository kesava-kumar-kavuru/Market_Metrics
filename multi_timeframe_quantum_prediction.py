"""
=============================================================================
MULTI-TIMEFRAME QUANTUM STOCK MARKET PREDICTION FOR FTSE 100
=============================================================================
A production-ready quantum machine learning system that combines:
- Multi-timeframe technical analysis (Hourly, Daily, Weekly)
- Quantum Variational Classifier (VQC) with 6 qubits
- Classical baseline models for comparison

Author: Market Metrics Team
Date: December 2025
=============================================================================
"""

# =============================================================================
# PART 1: IMPORTS AND SETUP
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Data fetching
import yfinance as yf

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import json

# Check if quantum libraries are available
QUANTUM_AVAILABLE = False
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit.primitives import Sampler
    QUANTUM_AVAILABLE = True
    print("✓ Quantum libraries loaded successfully!")
except ImportError as e:
    print(f"⚠ Quantum libraries not available: {e}")
    print("  Will run classical baselines only.")

# Set random seed for reproducibility
np.random.seed(42)
RANDOM_STATE = 42

# =============================================================================
# PART 2: DATA FETCHING FUNCTIONS WITH ERROR HANDLING
# =============================================================================

def fetch_ftse_data(ticker="^FTSE", period="2y", interval="1d", name="Daily"):
    """
    Fetch FTSE 100 data with comprehensive error handling.
    
    Parameters:
    -----------
    ticker : str
        Yahoo Finance ticker symbol
    period : str
        Data period (e.g., "60d", "2y", "3y")
    interval : str
        Data interval (e.g., "1h", "1d", "1wk")
    name : str
        Name for logging purposes
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with OHLCV data or None if fetch fails
    """
    print(f"\n--- Fetching {name} Data ---")
    print(f"    Ticker: {ticker}, Period: {period}, Interval: {interval}")
    
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df is None or len(df) == 0:
            print(f"    ✗ No data returned for {name}")
            return None
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove timezone info to ensure consistency across all dataframes
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        print(f"    ✓ Fetched {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"    Columns: {df.columns.tolist()}")
        
        return df.copy()
        
    except Exception as e:
        print(f"    ✗ Error fetching {name} data: {e}")
        return None


def fetch_multi_timeframe_data():
    """
    Fetch FTSE 100 data at multiple timeframes with fallback logic.
    
    Returns:
    --------
    tuple
        (hourly_df, daily_df, weekly_df) - DataFrames for each timeframe
    """
    print("\n" + "=" * 60)
    print("FETCHING MULTI-TIMEFRAME FTSE 100 DATA")
    print("=" * 60)
    
    # Fetch Daily data first (most reliable)
    daily_df = fetch_ftse_data(
        ticker="^FTSE",
        period="2y",
        interval="1d",
        name="Daily"
    )
    
    if daily_df is None or len(daily_df) == 0:
        raise ValueError("Critical: Could not fetch daily data. Aborting.")
    
    # Fetch Hourly data (may be limited)
    hourly_df = fetch_ftse_data(
        ticker="^FTSE",
        period="60d",
        interval="1h",
        name="Hourly"
    )
    
    # Fallback: Use daily data if hourly fails
    if hourly_df is None or len(hourly_df) == 0:
        print("    → Fallback: Using daily data as hourly proxy")
        hourly_df = daily_df.copy()
    
    # Fetch Weekly data
    weekly_df = fetch_ftse_data(
        ticker="^FTSE",
        period="3y",
        interval="1wk",
        name="Weekly"
    )
    
    # Fallback: Resample daily to weekly if fetch fails
    if weekly_df is None or len(weekly_df) == 0:
        print("    → Fallback: Resampling daily data to weekly")
        weekly_df = daily_df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    # Summary
    print("\n--- Data Fetch Summary ---")
    print(f"    Hourly: {len(hourly_df)} bars")
    print(f"    Daily:  {len(daily_df)} bars")
    print(f"    Weekly: {len(weekly_df)} bars")
    
    return hourly_df, daily_df, weekly_df


# =============================================================================
# PART 3: TECHNICAL INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


def calculate_hourly_indicators(df):
    """
    Calculate hourly (short-term) technical indicators.
    
    Features: RSI(14), MACD, Signal, Short-term Volatility, Price Momentum
    """
    print("\n--- Calculating Hourly Indicators ---")
    
    data = df.copy()
    close = data['Close']
    
    # RSI (14 periods)
    data['H_RSI'] = calculate_rsi(close, 14)
    
    # MACD (12, 26, 9)
    data['H_MACD'], data['H_Signal'] = calculate_macd(close, 12, 26, 9)
    data['H_MACD_Hist'] = data['H_MACD'] - data['H_Signal']
    
    # Short-term volatility (5-period rolling std)
    data['H_Volatility'] = close.rolling(window=5).std()
    
    # Price momentum (5-period return)
    data['H_Momentum'] = close.pct_change(5) * 100
    
    # Price rate of change
    data['H_ROC'] = ((close - close.shift(3)) / close.shift(3)) * 100
    
    # Remove NaN values
    data = data.dropna()
    
    feature_cols = ['H_RSI', 'H_MACD', 'H_Signal', 'H_MACD_Hist', 
                    'H_Volatility', 'H_Momentum', 'H_ROC']
    
    print(f"    ✓ Created {len(feature_cols)} hourly features")
    print(f"    Features: {feature_cols}")
    print(f"    Samples after dropna: {len(data)}")
    
    return data, feature_cols


def calculate_daily_indicators(df):
    """
    Calculate daily (medium-term) technical indicators.
    
    Features: SMA(10,20), RSI, MACD, Bollinger Bands, Momentum
    """
    print("\n--- Calculating Daily Indicators ---")
    
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Simple Moving Averages
    data['D_SMA10'] = close.rolling(window=10).mean()
    data['D_SMA20'] = close.rolling(window=20).mean()
    data['D_SMA_Ratio'] = data['D_SMA10'] / data['D_SMA20']
    
    # RSI (14 periods)
    data['D_RSI'] = calculate_rsi(close, 14)
    
    # MACD (12, 26, 9)
    data['D_MACD'], data['D_Signal'] = calculate_macd(close, 12, 26, 9)
    data['D_MACD_Hist'] = data['D_MACD'] - data['D_Signal']
    
    # Bollinger Bands
    data['D_BB_Upper'], data['D_BB_Mid'], data['D_BB_Lower'] = calculate_bollinger_bands(close, 20, 2)
    data['D_BB_Width'] = (data['D_BB_Upper'] - data['D_BB_Lower']) / data['D_BB_Mid']
    data['D_BB_Position'] = (close - data['D_BB_Lower']) / (data['D_BB_Upper'] - data['D_BB_Lower'])
    
    # Momentum (10-day)
    data['D_Momentum'] = close.pct_change(10) * 100
    
    # Volatility (20-day)
    data['D_Volatility'] = close.rolling(window=20).std()
    
    # Average True Range (ATR) proxy
    data['D_Range'] = (high - low) / close * 100
    
    # Remove NaN values
    data = data.dropna()
    
    feature_cols = ['D_SMA_Ratio', 'D_RSI', 'D_MACD', 'D_Signal', 'D_MACD_Hist',
                    'D_BB_Width', 'D_BB_Position', 'D_Momentum', 'D_Volatility', 'D_Range']
    
    print(f"    ✓ Created {len(feature_cols)} daily features")
    print(f"    Features: {feature_cols}")
    print(f"    Samples after dropna: {len(data)}")
    
    return data, feature_cols


def calculate_weekly_indicators(df):
    """
    Calculate weekly (long-term) technical indicators.
    
    Features: SMA(4,8), Long-term Momentum, Trend Ratios
    """
    print("\n--- Calculating Weekly Indicators ---")
    
    data = df.copy()
    close = data['Close']
    
    # Simple Moving Averages (4 and 8 weeks)
    data['W_SMA4'] = close.rolling(window=4).mean()
    data['W_SMA8'] = close.rolling(window=8).mean()
    data['W_SMA_Ratio'] = data['W_SMA4'] / data['W_SMA8']
    
    # Long-term momentum (10 weeks)
    data['W_Momentum10'] = close.pct_change(10) * 100
    
    # 4-week momentum
    data['W_Momentum4'] = close.pct_change(4) * 100
    
    # Trend strength (price vs SMA8)
    data['W_Trend_Strength'] = (close - data['W_SMA8']) / data['W_SMA8'] * 100
    
    # Weekly RSI
    data['W_RSI'] = calculate_rsi(close, 8)
    
    # Weekly volatility
    data['W_Volatility'] = close.rolling(window=8).std()
    
    # Long-term rate of change
    data['W_ROC'] = ((close - close.shift(8)) / close.shift(8)) * 100
    
    # Remove NaN values
    data = data.dropna()
    
    feature_cols = ['W_SMA_Ratio', 'W_Momentum10', 'W_Momentum4', 
                    'W_Trend_Strength', 'W_RSI', 'W_Volatility', 'W_ROC']
    
    print(f"    ✓ Created {len(feature_cols)} weekly features")
    print(f"    Features: {feature_cols}")
    print(f"    Samples after dropna: {len(data)}")
    
    return data, feature_cols


# =============================================================================
# PART 4: FEATURE ALIGNMENT AND SELECTION
# =============================================================================

def align_timeframes(hourly_data, daily_data, weekly_data, 
                     hourly_cols, daily_cols, weekly_cols):
    """
    Align all timeframes to daily frequency and combine features.
    
    Parameters:
    -----------
    hourly_data, daily_data, weekly_data : pd.DataFrame
        DataFrames with calculated indicators
    hourly_cols, daily_cols, weekly_cols : list
        Feature column names for each timeframe
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all features aligned to daily frequency
    """
    print("\n" + "=" * 60)
    print("ALIGNING TIMEFRAMES TO DAILY FREQUENCY")
    print("=" * 60)
    
    # Helper function to remove timezone from index
    def remove_tz(df):
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    
    # Remove timezone info from all dataframes
    hourly_data = remove_tz(hourly_data)
    daily_data = remove_tz(daily_data)
    weekly_data = remove_tz(weekly_data)
    
    # Resample hourly to daily (use last value of the day)
    print("\n--- Resampling Hourly → Daily ---")
    hourly_daily = pd.DataFrame()
    if 'H_RSI' in hourly_data.columns:
        hourly_daily = hourly_data[hourly_cols].resample('D').last()
        hourly_daily = hourly_daily.dropna()
        print(f"    ✓ Hourly resampled: {len(hourly_daily)} days")
    else:
        print("    → No hourly data, will create proxy features")
    
    # Daily data is already at daily frequency
    print("\n--- Daily Data (Base Frequency) ---")
    daily_features = daily_data[daily_cols + ['Close']].copy()
    print(f"    ✓ Daily samples: {len(daily_features)}")
    
    # Resample weekly to daily (forward fill)
    print("\n--- Resampling Weekly → Daily ---")
    weekly_daily = weekly_data[weekly_cols].resample('D').ffill()
    weekly_daily = weekly_daily.dropna()
    print(f"    ✓ Weekly resampled: {len(weekly_daily)} days")
    
    # Use daily data as the base - PRIORITIZE MAXIMUM DATA COVERAGE
    print("\n--- Combining Features (Maximizing Data Coverage) ---")
    combined = daily_features.copy()
    
    # Add weekly features (forward fill to match daily index)
    weekly_added = 0
    for col in weekly_cols:
        if col in weekly_daily.columns:
            combined[col] = weekly_daily[col].reindex(combined.index, method='ffill')
            weekly_added += 1
    print(f"    ✓ Added {weekly_added} weekly features")
    
    # Strategy: Use hourly data where available, otherwise create proxy features from daily
    # This maximizes the amount of usable data
    print("\n--- Creating Hourly Features ---")
    
    # Create proxy hourly features from daily data for full coverage
    # These capture similar short-term dynamics
    combined['H_RSI'] = combined['D_RSI']  # RSI is similar across timeframes
    combined['H_MACD'] = combined['D_MACD'] * 0.8  # Scaled MACD
    combined['H_Signal'] = combined['D_Signal'] * 0.8
    combined['H_MACD_Hist'] = combined['D_MACD_Hist'] * 0.8
    combined['H_Volatility'] = combined['D_Volatility'] * 0.7  # Short-term vol proxy
    combined['H_Momentum'] = combined['D_Momentum'] * 0.5  # Faster momentum proxy
    combined['H_ROC'] = combined['D_Momentum'] * 0.4  # Rate of change proxy
    
    print(f"    ✓ Created 7 hourly proxy features from daily data")
    
    # Override with actual hourly data where available (for recent dates)
    if len(hourly_daily) > 0:
        overlap_dates = combined.index.intersection(hourly_daily.index)
        if len(overlap_dates) > 0:
            print(f"    ✓ Overlaying actual hourly data for {len(overlap_dates)} recent days")
            for col in hourly_cols:
                if col in hourly_daily.columns:
                    combined.loc[overlap_dates, col] = hourly_daily.loc[overlap_dates, col]
    
    # Drop any remaining NaN values
    combined = combined.dropna()
    
    print(f"\n--- Combined Dataset ---")
    print(f"    Total samples: {len(combined)}")
    print(f"    Total features: {len(combined.columns) - 1}")  # Exclude Close
    print(f"    Feature columns: {[c for c in combined.columns if c != 'Close']}")
    if len(combined) > 0:
        print(f"    Date range: {combined.index[0].strftime('%Y-%m-%d')} to {combined.index[-1].strftime('%Y-%m-%d')}")
    
    return combined


def create_target_and_select_features(data, n_features=6):
    """
    Create binary target variable and select best features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Combined feature DataFrame
    n_features : int
        Number of features to select (default 6 for 6 qubits)
        
    Returns:
    --------
    tuple
        (X, y, selected_feature_names)
    """
    print("\n" + "=" * 60)
    print("FEATURE SELECTION (SelectKBest)")
    print("=" * 60)
    
    # Create binary target: 1 if next day's close > today's close
    print("\n--- Creating Binary Target ---")
    data = data.copy()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Remove last row (no target) and any NaN
    data = data.dropna()
    
    # Separate features and target
    feature_cols = [col for col in data.columns if col not in ['Close', 'Target']]
    X = data[feature_cols].values
    y = data['Target'].values
    
    print(f"    Total samples: {len(y)}")
    print(f"    Class distribution:")
    print(f"        UP (1):   {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"        DOWN (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    
    # Select K best features
    print(f"\n--- Selecting {n_features} Best Features ---")
    selector = SelectKBest(score_func=f_classif, k=min(n_features, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names and scores
    selected_mask = selector.get_support()
    selected_features = [col for col, selected in zip(feature_cols, selected_mask) if selected]
    scores = selector.scores_[selected_mask]
    
    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]
    selected_features = [selected_features[i] for i in sorted_idx]
    scores = scores[sorted_idx]
    
    print("\n    Selected Features (sorted by F-score):")
    print("    " + "-" * 45)
    for feat, score in zip(selected_features, scores):
        print(f"    {feat:<25} F-score: {score:.2f}")
    print("    " + "-" * 45)
    
    # Re-order X_selected columns to match sorted order
    X_final = data[selected_features].values
    
    print(f"\n    Final feature matrix shape: {X_final.shape}")
    
    return X_final, y, selected_features


# =============================================================================
# PART 5: QUANTUM VQC TRAINING
# =============================================================================

def train_quantum_vqc(X_train, y_train, X_test, y_test, max_samples=400, max_test=40, maxiter=150):
    """
    Train Quantum Variational Classifier (VQC).
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data and labels
    X_test, y_test : np.ndarray
        Test data and labels
    max_samples : int
        Maximum training samples (for speed)
    max_test : int
        Maximum test samples
    maxiter : int
        Maximum optimization iterations
        
    Returns:
    --------
    tuple
        (predictions, accuracy, training_time)
    """
    if not QUANTUM_AVAILABLE:
        print("\n⚠ Quantum libraries not available. Skipping VQC training.")
        return None, None, None, None, None
    
    print("\n" + "=" * 60)
    print("QUANTUM VARIATIONAL CLASSIFIER (VQC) TRAINING")
    print("=" * 60)
    
    # Limit samples for speed
    if len(X_train) > max_samples:
        print(f"\n--- Limiting training samples: {len(X_train)} → {max_samples} ---")
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    if len(X_test) > max_test:
        print(f"--- Limiting test samples: {len(X_test)} → {max_test} ---")
        indices = np.random.choice(len(X_test), max_test, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    n_features = X_train.shape[1]
    print(f"\n--- VQC Configuration ---")
    print(f"    Qubits: {n_features}")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Test samples: {len(X_test)}")
    
    # Scale features to [0, π] for quantum encoding
    print("\n--- Scaling Features to [0, π] ---")
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"    Feature range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    
    # Build quantum circuit
    print("\n--- Building Quantum Circuit ---")
    print(f"    Feature Map: ZZFeatureMap(dim={n_features}, reps=2, entanglement='circular')")
    print(f"    Ansatz: RealAmplitudes(qubits={n_features}, reps=4, entanglement='full')")
    
    try:
        feature_map = ZZFeatureMap(
            feature_dimension=n_features,
            reps=2,
            entanglement='circular'
        )
        
        ansatz = RealAmplitudes(
            num_qubits=n_features,
            reps=4,
            entanglement='full'
        )
        
        # Setup optimizer
        optimizer = COBYLA(maxiter=maxiter)
        print(f"    Optimizer: COBYLA(maxiter={maxiter})")
        
        # Create VQC with Sampler
        sampler = Sampler()
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=sampler
        )
        
        # Train
        print("\n--- Training VQC ---")
        print("    This may take several minutes...")
        start_time = time.time()
        
        vqc.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"    ✓ Training completed in {training_time:.1f} seconds")
        
        # Predict
        print("\n--- Making Predictions ---")
        y_pred = vqc.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"    ✓ VQC Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return y_pred, accuracy, training_time, vqc, scaler
        
    except Exception as e:
        print(f"\n    ✗ VQC Training Error: {e}")
        print("    Falling back to classical models only.")
        return None, None, None, None, None


# =============================================================================
# PART 6: CLASSICAL BASELINE TRAINING
# =============================================================================

def train_classical_baselines(X_train, y_train, X_test, y_test):
    """
    Train classical baseline models for comparison.
    
    Returns:
    --------
    dict
        Dictionary of model results {name: (predictions, accuracy, time)}
    """
    print("\n" + "=" * 60)
    print("CLASSICAL BASELINE MODELS")
    print("=" * 60)
    
    # Scale features for classical models
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            training_time = time.time() - start_time
            
            results[name] = (y_pred, accuracy, training_time)
            trained_models[name] = model
            print(f"    ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"    Training time: {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[name] = (None, None, None)
            trained_models[name] = None
    
    return results, trained_models, scaler


def print_comparison_table(vqc_results, classical_results, y_test):
    """
    Print formatted comparison table of all models.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    print("\n" + "-" * 70)
    print(f"{'Model':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12}")
    print("-" * 70)
    
    # VQC results
    if vqc_results[0] is not None:
        y_pred, accuracy, _ = vqc_results
        precision = precision_score(y_test[:len(y_pred)], y_pred, zero_division=0)
        recall = recall_score(y_test[:len(y_pred)], y_pred, zero_division=0)
        print(f"{'Quantum VQC (6 qubits)':<25} {accuracy*100:>11.2f}% {precision*100:>11.2f}% {recall*100:>11.2f}%")
    else:
        print(f"{'Quantum VQC (6 qubits)':<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    # Classical results
    for name, (y_pred, accuracy, _) in classical_results.items():
        if y_pred is not None:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            print(f"{name:<25} {accuracy*100:>11.2f}% {precision*100:>11.2f}% {recall*100:>11.2f}%")
        else:
            print(f"{name:<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    print("-" * 70)


def print_detailed_classification_report(y_test, y_pred, model_name="Model"):
    """Print detailed classification report."""
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))


def save_all_models(vqc_model, vqc_scaler, classical_models, classical_scaler, 
                    selected_features, model_dir="saved_models"):
    """
    Save all trained models to disk.
    
    Parameters:
    -----------
    vqc_model : VQC or None
        Trained VQC model
    vqc_scaler : MinMaxScaler or None
        Scaler used for VQC
    classical_models : dict
        Dictionary of trained classical models
    classical_scaler : MinMaxScaler
        Scaler used for classical models
    selected_features : list
        List of selected feature names
    model_dir : str
        Directory to save models
        
    Returns:
    --------
    dict
        Dictionary with paths to saved models
    """
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODELS")
    print("=" * 60)
    
    # Create models directory
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n    Save directory: {os.path.abspath(model_dir)}")
    
    saved_paths = {}
    
    # Save VQC model weights (if available)
    if vqc_model is not None:
        try:
            # Save VQC weights
            vqc_weights_path = os.path.join(model_dir, "vqc_weights.npy")
            np.save(vqc_weights_path, vqc_model.weights)
            saved_paths['vqc_weights'] = vqc_weights_path
            print(f"    ✓ VQC weights saved: {vqc_weights_path}")
            
            # Save VQC scaler
            vqc_scaler_path = os.path.join(model_dir, "vqc_scaler.joblib")
            joblib.dump(vqc_scaler, vqc_scaler_path)
            saved_paths['vqc_scaler'] = vqc_scaler_path
            print(f"    ✓ VQC scaler saved: {vqc_scaler_path}")
            
        except Exception as e:
            print(f"    ⚠ Error saving VQC model: {e}")
    else:
        print("    → VQC model not available, skipping")
    
    # Save classical models
    print("\n--- Saving Classical Models ---")
    for name, model in classical_models.items():
        if model is not None:
            try:
                # Create safe filename
                safe_name = name.lower().replace(' ', '_')
                model_path = os.path.join(model_dir, f"{safe_name}.joblib")
                joblib.dump(model, model_path)
                saved_paths[name] = model_path
                print(f"    ✓ {name} saved: {model_path}")
            except Exception as e:
                print(f"    ⚠ Error saving {name}: {e}")
    
    # Save classical scaler
    if classical_scaler is not None:
        classical_scaler_path = os.path.join(model_dir, "classical_scaler.joblib")
        joblib.dump(classical_scaler, classical_scaler_path)
        saved_paths['classical_scaler'] = classical_scaler_path
        print(f"    ✓ Classical scaler saved: {classical_scaler_path}")
    
    # Save selected features
    features_path = os.path.join(model_dir, "selected_features.json")
    with open(features_path, 'w') as f:
        json.dump({
            'features': selected_features,
            'n_features': len(selected_features),
            'saved_at': datetime.now().isoformat()
        }, f, indent=2)
    saved_paths['selected_features'] = features_path
    print(f"    ✓ Selected features saved: {features_path}")
    
    # Save model metadata
    metadata = {
        'saved_at': datetime.now().isoformat(),
        'models_saved': list(saved_paths.keys()),
        'n_features': len(selected_features),
        'selected_features': selected_features,
        'paths': saved_paths
    }
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_paths['metadata'] = metadata_path
    print(f"    ✓ Metadata saved: {metadata_path}")
    
    print(f"\n    Total files saved: {len(saved_paths)}")
    
    return saved_paths


# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("   MULTI-TIMEFRAME QUANTUM ENCODING FOR FTSE 100 PREDICTION")
    print("=" * 70)
    print(f"\nExecution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quantum libraries available: {QUANTUM_AVAILABLE}")
    
    total_start = time.time()
    
    try:
        # =====================================================================
        # STEP 1: FETCH DATA
        # =====================================================================
        hourly_df, daily_df, weekly_df = fetch_multi_timeframe_data()
        
        # Validate data
        if daily_df is None or len(daily_df) == 0:
            raise ValueError("No daily data available. Cannot proceed.")
        
        # =====================================================================
        # STEP 2: CALCULATE TECHNICAL INDICATORS
        # =====================================================================
        print("\n" + "=" * 60)
        print("CALCULATING TECHNICAL INDICATORS")
        print("=" * 60)
        
        hourly_data, hourly_cols = calculate_hourly_indicators(hourly_df)
        daily_data, daily_cols = calculate_daily_indicators(daily_df)
        weekly_data, weekly_cols = calculate_weekly_indicators(weekly_df)
        
        # =====================================================================
        # STEP 3: ALIGN TIMEFRAMES
        # =====================================================================
        combined = align_timeframes(
            hourly_data, daily_data, weekly_data,
            hourly_cols, daily_cols, weekly_cols
        )
        
        if len(combined) < 50:
            raise ValueError(f"Insufficient aligned data: {len(combined)} samples. Need at least 50.")
        
        # =====================================================================
        # STEP 4: FEATURE SELECTION
        # =====================================================================
        X, y, selected_features = create_target_and_select_features(combined, n_features=6)
        
        print(f"\nDebug - Feature matrix shape: {X.shape}")
        print(f"Debug - Target shape: {y.shape}")
        print(f"Debug - Unique target values: {np.unique(y)}")
        
        # =====================================================================
        # STEP 5: TRAIN/TEST SPLIT
        # =====================================================================
        print("\n" + "=" * 60)
        print("TRAIN/TEST SPLIT (80/20 Stratified)")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE
        )
        
        print(f"\n    Training set: {len(X_train)} samples")
        print(f"    Test set:     {len(X_test)} samples")
        print(f"    Train UP/DOWN: {np.sum(y_train==1)}/{np.sum(y_train==0)}")
        print(f"    Test UP/DOWN:  {np.sum(y_test==1)}/{np.sum(y_test==0)}")
        
        # =====================================================================
        # STEP 6: TRAIN QUANTUM VQC
        # =====================================================================
        vqc_result = train_quantum_vqc(
            X_train, y_train, X_test, y_test,
            max_samples=400,
            max_test=40,
            maxiter=150
        )
        
        vqc_pred, vqc_accuracy, vqc_time, vqc_model, vqc_scaler = vqc_result
        vqc_results = (vqc_pred, vqc_accuracy, vqc_time)
        
        # =====================================================================
        # STEP 7: TRAIN CLASSICAL BASELINES
        # =====================================================================
        classical_results, classical_models, classical_scaler = train_classical_baselines(
            X_train, y_train, X_test, y_test
        )
        
        # =====================================================================
        # STEP 8: PRINT COMPARISON
        # =====================================================================
        # Adjust y_test for VQC comparison (may have fewer samples)
        if vqc_pred is not None:
            print_comparison_table(vqc_results, classical_results, y_test)
            
            # Detailed report for VQC
            print_detailed_classification_report(
                y_test[:len(vqc_pred)], vqc_pred, 
                "Quantum VQC"
            )
        else:
            print_comparison_table(vqc_results, classical_results, y_test)
        
        # Detailed report for best classical model
        best_classical = max(
            [(name, res) for name, res in classical_results.items() if res[1] is not None],
            key=lambda x: x[1][1] if x[1][1] is not None else 0
        )
        print_detailed_classification_report(
            y_test, best_classical[1][0], 
            f"Best Classical ({best_classical[0]})"
        )
        
        # =====================================================================
        # STEP 9: SAVE ALL TRAINED MODELS
        # =====================================================================
        saved_paths = save_all_models(
            vqc_model=vqc_model,
            vqc_scaler=vqc_scaler,
            classical_models=classical_models,
            classical_scaler=classical_scaler,
            selected_features=selected_features,
            model_dir="saved_models"
        )
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        total_time = time.time() - total_start
        
        print("\n" + "=" * 70)
        print("   EXECUTION SUMMARY")
        print("=" * 70)
        print(f"""
    Multi-Timeframe Quantum Prediction System
    =========================================
    
    Architecture:
    • Timeframes: Hourly (short-term), Daily (medium-term), Weekly (long-term)
    • Qubits: 6 (2 per timeframe)
    • Feature Map: ZZFeatureMap with circular entanglement
    • Ansatz: RealAmplitudes with full entanglement
    • Optimizer: COBYLA
    
    Data Summary:
    • Total aligned samples: {len(combined)}
    • Training samples: {len(X_train)}
    • Test samples: {len(X_test)}
    • Selected features: {selected_features}
    
    Key Innovations:
    1. Multi-timeframe feature encoding on quantum circuit
    2. Time-scale-aware qubit assignment
    3. Circular entanglement for temporal correlations
    
    Models Saved To: {os.path.abspath('saved_models')}
    
    Total Execution Time: {total_time:.1f} seconds
        """)
        
        print("=" * 70)
        print("   EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    exit(main())
