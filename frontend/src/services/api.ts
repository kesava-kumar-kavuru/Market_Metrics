// Dummy API service for quantum ML trading data
export interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  prediction?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

export interface ModelAccuracy {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface PredictionData {
  date: string;
  actual: number;
  vqc_prediction: number;
  svm_prediction: number;
  confidence: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
}

export interface QuantumMetrics {
  circuit_depth: number;
  quantum_volume: number;
  gate_fidelity: number;
  coherence_time: number;
  error_rate: number;
}

// Quantum Predictor Types
export interface QuantumPredictRequest {
  features: number[];
  model?: string;
  return_probabilities?: boolean;
}

export interface QuantumPredictResponse {
  prediction: number;
  signal: 'BUY' | 'SELL';
  confidence: number;
  model_used: string;
  probabilities?: number[];
  features_used: number;
}

export interface AvailableModel {
  name: string;
  type: string;
  accuracy: number;
}

export interface ModelsResponse {
  available_models: AvailableModel[];
  default_model: string;
}

export interface FeaturesResponse {
  features: string[];
  count: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
  timestamp: string;
}

// Dummy data generators
const generateMarketData = (days: number = 30): MarketData[] => {
  const data: MarketData[] = [];
  let currentPrice = 7800; // FTSE 100 base price
  
  for (let i = days; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    
    const volatility = 0.02;
    const change = (Math.random() - 0.5) * volatility * currentPrice;
    const open = currentPrice;
    const close = currentPrice + change;
    const high = Math.max(open, close) + Math.random() * 20;
    const low = Math.min(open, close) - Math.random() * 20;
    
    data.push({
      timestamp: date.toISOString(),
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 1000000) + 500000,
      prediction: Math.random() > 0.6 ? 'BUY' : Math.random() > 0.3 ? 'HOLD' : 'SELL',
      confidence: 0.7 + Math.random() * 0.25
    });
    
    currentPrice = close;
  }
  
  return data;
};

// API functions
export const fetchMarketData = async (): Promise<MarketData[]> => {
  await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
  return generateMarketData(30);
};

export const fetchModelAccuracies = async (): Promise<ModelAccuracy[]> => {
  await new Promise(resolve => setTimeout(resolve, 300));
  return [
    {
      model: 'VQC (Quantum)',
      accuracy: 87.3,
      precision: 85.1,
      recall: 89.2,
      f1Score: 87.1
    },
    {
      model: 'SVM (Classical)',
      accuracy: 82.1,
      precision: 80.4,
      recall: 83.8,
      f1Score: 82.0
    },
    {
      model: 'IBM Quantum',
      accuracy: 89.1,
      precision: 87.5,
      recall: 90.3,
      f1Score: 88.9
    }
  ];
};

export const fetchPredictions = async (): Promise<PredictionData[]> => {
  const res = await fetch("http://127.0.0.1:5000/api/predictions");
  if (!res.ok) throw new Error("Failed to fetch predictions");
  const data = await res.json();
  // Support both array and {predictions: array} response
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.predictions)) return data.predictions;
  return [];
};

export const fetchQuantumMetrics = async (): Promise<QuantumMetrics> => {
  await new Promise(resolve => setTimeout(resolve, 200));
  return {
    circuit_depth: 12,
    quantum_volume: 64,
    gate_fidelity: 0.995,
    coherence_time: 150, // microseconds
    error_rate: 0.005
  };
};

export const fetchLiveMetrics = async () => {
  await new Promise(resolve => setTimeout(resolve, 100));
  return {
    currentPrice: 7845.32 + (Math.random() - 0.5) * 20,
    dailyChange: (Math.random() - 0.5) * 100,
    volume: Math.floor(Math.random() * 500000) + 1000000,
    volatility: 0.15 + Math.random() * 0.1,
    nextPrediction: Math.random() > 0.5 ? 'BUY' : 'SELL',
    confidence: 0.8 + Math.random() * 0.15
  };
};

// ============= QUANTUM PREDICTOR API =============
const API_BASE = "http://127.0.0.1:5000";

export const fetchQuantumPrediction = async (
  features: number[],
  model: string = 'vqc',
  returnProbabilities: boolean = true
): Promise<QuantumPredictResponse> => {
  const res = await fetch(`${API_BASE}/api/quantum-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      features,
      model,
      return_probabilities: returnProbabilities
    })
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.error || 'Failed to fetch quantum prediction');
  }
  return res.json();
};

export const fetchBatchPredictions = async (
  featuresList: number[][],
  models: string[] = ['vqc']
): Promise<{results: any[], total_predictions: number}> => {
  const res = await fetch(`${API_BASE}/api/batch-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      features_list: featuresList,
      models
    })
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.error || 'Failed to fetch batch predictions');
  }
  return res.json();
};

export const fetchAvailableModels = async (): Promise<ModelsResponse> => {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error('Failed to fetch available models');
  return res.json();
};

export const fetchSelectedFeatures = async (): Promise<FeaturesResponse> => {
  const res = await fetch(`${API_BASE}/api/features`);
  if (!res.ok) throw new Error('Failed to fetch selected features');
  return res.json();
};

export const fetchAPIHealth = async (): Promise<HealthResponse> => {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('API health check failed');
  return res.json();
};