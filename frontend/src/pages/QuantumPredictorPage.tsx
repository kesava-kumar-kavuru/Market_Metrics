import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  BarChart3,
  Brain,
  Layers,
  Target
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';

// Types
interface PredictionResponse {
  prediction: number;
  prediction_label: string;
  confidence: number;
  current_price: number;
  model_name: string;
  model_accuracy: number;
  timestamp: string;
  features: { name: string; value: number }[];
  ticker: string;
  timeframe: string;
}

interface HistoricalPrediction {
  timestamp: string;
  prediction: string;
  confidence: number;
  price: number;
}

// API endpoint
const API_BASE_URL = 'http://localhost:5000/api';

const QuantumPredictorPage = () => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalPrediction[]>([]);
  const [selectedModel, setSelectedModel] = useState('quantum_vqc');
  const [timeframe, setTimeframe] = useState('daily');

  // Model comparison data - actual trained model accuracies
  // IBM Brisbane VQC: Precision 0.695, Recall 0.700, F1 0.690
  const modelComparison = [
    { model: 'VQC (IBM Brisbane)', accuracy: 0.70, color: '#8b5cf6' },
    { model: 'VQC (Simulator)', accuracy: 0.633, color: '#06b6d4' },
    { model: 'Random Forest', accuracy: 0.467, color: '#3b82f6' },
  ];

  // Fetch prediction from backend
  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/quantum-predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: '^FTSE',
          model: selectedModel,
          timeframe: timeframe
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setPrediction(data);

      // Add to historical data
      setHistoricalData(prev => [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        prediction: data.prediction_label,
        confidence: data.confidence,
        price: data.current_price
      }].slice(-10));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh every 5 minutes
  useEffect(() => {
    fetchPrediction();
    const interval = setInterval(fetchPrediction, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [selectedModel, timeframe]);

  // Confidence Gauge Component
  const ConfidenceGauge = ({ confidence }: { confidence: number }) => {
    const percentage = confidence * 100;
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference * (1 - confidence);

    return (
      <div className="relative w-32 h-32">
        <svg className="transform -rotate-90 w-32 h-32" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            className="text-muted/20"
          />
          <circle
            cx="50"
            cy="50"
            r="45"
            stroke="url(#gradient)"
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center flex-col">
          <div className="text-2xl font-bold text-foreground">{percentage.toFixed(0)}%</div>
          <div className="text-xs text-muted-foreground">Confidence</div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-primary/10">
              <Brain className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-3xl font-bold text-foreground">
              Quantum Stock Predictor
            </h1>
          </div>
          <p className="text-muted-foreground">
            Multi-Timeframe FTSE 100 Prediction using 6-Qubit VQC
          </p>
        </div>
        <Button
          onClick={fetchPrediction}
          disabled={loading}
          variant="quantum"
          size="default"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          {loading ? 'Fetching...' : 'Get Prediction'}
        </Button>
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Selection */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg text-foreground">
              <Layers className="w-5 h-5 text-primary" />
              Model Selection
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {[
              { value: 'quantum_vqc', label: 'Quantum VQC (6 Qubits)', icon: Zap, accuracy: '65%' },
              { value: 'random_forest', label: 'Random Forest', icon: BarChart3, accuracy: '58%' },
              { value: 'gradient_boosting', label: 'Gradient Boosting', icon: TrendingUp, accuracy: '57%' },
              { value: 'logistic_regression', label: 'Logistic Regression', icon: Activity, accuracy: '62%' }
            ].map(({ value, label, icon: Icon, accuracy }) => (
              <button
                key={value}
                onClick={() => setSelectedModel(value)}
                className={`w-full px-4 py-3 rounded-lg transition-all duration-200 flex items-center gap-3 border-2 ${
                  selectedModel === value
                    ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border-primary text-foreground shadow-md shadow-primary/20'
                    : 'bg-card/50 border-border/50 text-muted-foreground hover:bg-muted/50 hover:border-primary/50 hover:text-foreground'
                }`}
              >
                <Icon className={`w-5 h-5 ${selectedModel === value ? 'text-primary' : ''}`} />
                <span className="font-medium flex-1 text-left">{label}</span>
                <Badge variant={selectedModel === value ? "default" : "outline"} className={selectedModel === value ? 'bg-primary text-white' : ''}>
                  {accuracy}
                </Badge>
                {selectedModel === value && <CheckCircle className="w-5 h-5 text-emerald-400" />}
              </button>
            ))}
          </CardContent>
        </Card>

        {/* Timeframe Selection */}
        <Card className="bg-card/50 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg text-foreground">
              <Clock className="w-5 h-5 text-primary" />
              Timeframe Analysis
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: 'hourly', label: 'Hourly', desc: '60 days data', qubits: '0-1' },
                { value: 'daily', label: 'Daily', desc: '2 years data', qubits: '2-3' },
                { value: 'weekly', label: 'Weekly', desc: '3 years data', qubits: '4-5' }
              ].map(({ value, label, desc, qubits }) => (
                <button
                  key={value}
                  onClick={() => setTimeframe(value)}
                  className={`px-4 py-4 rounded-lg transition-all duration-200 text-center border-2 ${
                    timeframe === value
                      ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border-primary text-foreground shadow-md shadow-primary/20'
                      : 'bg-card/50 border-border/50 text-muted-foreground hover:bg-muted/50 hover:border-primary/50 hover:text-foreground'
                  }`}
                >
                  <div className={`font-bold mb-1 ${timeframe === value ? 'text-primary' : ''}`}>{label}</div>
                  <div className="text-xs opacity-80">{desc}</div>
                  <div className="text-xs mt-2 opacity-60">Qubits: {qubits}</div>
                </button>
              ))}
            </div>

            {/* Qubit Assignment Info */}
            <div className="mt-4 p-3 rounded-lg bg-muted/20 border border-border/30">
              <div className="text-sm font-medium mb-2 text-foreground">6-Qubit Assignment:</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-purple-500" />
                  <span className="text-muted-foreground">Q0-1: Hourly</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="text-muted-foreground">Q2-3: Daily</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-cyan-500" />
                  <span className="text-muted-foreground">Q4-5: Weekly</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="bg-destructive/10 border-destructive/30">
          <CardContent className="p-4 flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-destructive flex-shrink-0" />
            <div>
              <div className="font-bold text-destructive mb-1">Connection Error</div>
              <div className="text-sm text-muted-foreground">
                {error}. Ensure the backend API is running at {API_BASE_URL}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Prediction Display */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Prediction Card */}
          <Card className="lg:col-span-2 bg-gradient-to-br from-primary/10 via-background to-cyan-500/10 border-primary/30">
            <CardContent className="p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <div className="text-muted-foreground text-sm mb-1">Latest Prediction</div>
                  <div className="text-4xl font-bold text-foreground">
                    FTSE 100
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-muted-foreground text-sm mb-1">Current Price</div>
                  <div className="text-3xl font-bold text-foreground">
                    £{prediction.current_price?.toFixed(2) || '8,250.00'}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6 mb-6">
                <div>
                  <div className="text-muted-foreground text-sm mb-2">Prediction</div>
                  <div className={`text-5xl font-bold flex items-center gap-3 ${
                    prediction.prediction === 1 ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {prediction.prediction === 1 ? (
                      <>
                        <TrendingUp className="w-12 h-12" />
                        UP
                      </>
                    ) : (
                      <>
                        <TrendingDown className="w-12 h-12" />
                        DOWN
                      </>
                    )}
                  </div>
                  <div className="text-muted-foreground text-sm mt-2">
                    Next day direction forecast
                  </div>
                </div>

                <div className="flex items-center justify-center">
                  <ConfidenceGauge confidence={prediction.confidence || 0.65} />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 pt-6 border-t border-border/30">
                <div>
                  <div className="text-muted-foreground text-xs mb-1">Model</div>
                  <div className="text-foreground font-semibold text-sm flex items-center gap-1">
                    <Zap className="w-3 h-3 text-primary" />
                    {prediction.model_name || 'Quantum VQC'}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs mb-1">Accuracy</div>
                  <div className="text-foreground font-semibold text-sm">
                    {((prediction.model_accuracy || 0.65) * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs mb-1">Timestamp</div>
                  <div className="text-foreground font-semibold text-sm">
                    {new Date().toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Feature Importance */}
          <Card className="bg-card/50 border-border/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg text-foreground">
                <Target className="w-5 h-5 text-primary" />
                Top Features
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {(prediction.features || [
                { name: 'W_Trend_Strength', value: 0.89 },
                { name: 'D_BB_Width', value: 0.76 },
                { name: 'D_Volatility', value: 0.72 },
                { name: 'H_Volatility', value: 0.68 },
                { name: 'W_Momentum4', value: 0.61 },
                { name: 'W_Volatility', value: 0.55 }
              ]).map((feature, idx) => (
                <div key={idx}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">{feature.name}</span>
                    <span className="text-foreground font-semibold">
                      {(feature.value * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Progress value={feature.value * 100} className="h-2" />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Historical Predictions Chart */}
      {historicalData.length > 0 && (
        <Card className="bg-card/50 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Clock className="w-5 h-5 text-primary" />
              Recent Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="hsl(var(--muted-foreground))" 
                  tick={{ fontSize: 12 }} 
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))" 
                  domain={[0, 1]} 
                  ticks={[0, 0.5, 1]} 
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="hsl(var(--primary))"
                  strokeWidth={3}
                  dot={{ fill: 'hsl(var(--primary))', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Model Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-card/50 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <BarChart3 className="w-5 h-5 text-primary" />
              Model Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="model"
                  stroke="hsl(var(--muted-foreground))"
                  tick={{ fontSize: 10 }}
                  angle={-15}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))" 
                  domain={[0, 1]} 
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="accuracy" radius={[8, 8, 0, 0]}>
                  {modelComparison.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-foreground">
              <Brain className="w-5 h-5 text-primary" />
              System Architecture
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-purple-500 rounded-full mt-2 flex-shrink-0" />
              <div>
                <div className="text-foreground font-semibold mb-1">Multi-Timeframe Analysis</div>
                <div className="text-muted-foreground">
                  Processes hourly, daily, and weekly data simultaneously
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
              <div>
                <div className="text-foreground font-semibold mb-1">6-Qubit Quantum Circuit</div>
                <div className="text-muted-foreground">
                  ZZFeatureMap + RealAmplitudes ansatz with entanglement
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0" />
              <div>
                <div className="text-foreground font-semibold mb-1">Real-time yfinance Data</div>
                <div className="text-muted-foreground">
                  Live FTSE 100 price and technical indicators
                </div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 flex-shrink-0" />
              <div>
                <div className="text-foreground font-semibold mb-1">SelectKBest Features</div>
                <div className="text-muted-foreground">
                  6 optimal features selected via F-score analysis
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer Info */}
      <div className="text-center text-muted-foreground text-sm py-4">
        <p className="flex items-center justify-center gap-2">
          <Zap className="w-4 h-4 text-primary" />
          Powered by Qiskit Quantum Computing | 📊 Data from Yahoo Finance
        </p>
        <p className="mt-2 text-xs">
          Research prototype. Not financial advice. Always consult professionals before trading.
        </p>
      </div>
    </div>
  );
};

export default QuantumPredictorPage;
