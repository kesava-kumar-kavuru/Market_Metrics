import { Card } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { 
  ArrowRight, 
  ArrowLeft,
  Brain, 
  Cpu, 
  Database, 
  GitBranch, 
  Layers, 
  Zap,
  TrendingUp,
  BarChart3,
  Activity,
  Target,
  Workflow,
  CircuitBoard,
  Gauge,
  CheckCircle2
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const ArchitecturePage = () => {
  const navigate = useNavigate();

  const pipelineSteps = [
    {
      step: 1,
      title: "Data Acquisition",
      description: "Fetch FTSE 100 market data using yfinance API",
      details: ["Historical price data (2010-present)", "OHLCV data extraction", "Auto-adjusted prices"],
      icon: Database,
      color: "from-blue-500 to-cyan-500"
    },
    {
      step: 2,
      title: "Feature Engineering",
      description: "Calculate technical indicators for market analysis",
      details: ["SMA (50, 200 periods)", "RSI, MACD, ADX", "OBV, Price ratios"],
      icon: Layers,
      color: "from-cyan-500 to-teal-500"
    },
    {
      step: 3,
      title: "Data Preprocessing",
      description: "Prepare data for quantum circuit encoding",
      details: ["MinMaxScaler normalization", "Feature selection (SelectKBest)", "Train/test split (80/20)"],
      icon: GitBranch,
      color: "from-teal-500 to-green-500"
    },
    {
      step: 4,
      title: "Quantum Encoding",
      description: "Encode classical data into quantum states",
      details: ["ZZFeatureMap encoding", "3-qubit representation", "Linear entanglement"],
      icon: CircuitBoard,
      color: "from-green-500 to-emerald-500"
    },
    {
      step: 5,
      title: "VQC Training",
      description: "Train variational quantum classifier",
      details: ["RealAmplitudes ansatz", "COBYLA optimizer", "IBM Brisbane backend"],
      icon: Brain,
      color: "from-emerald-500 to-purple-500"
    },
    {
      step: 6,
      title: "Prediction & Evaluation",
      description: "Generate predictions and evaluate performance",
      details: ["Binary classification (Up/Down)", "Accuracy: 70%", "Precision: 69.5%"],
      icon: Target,
      color: "from-purple-500 to-pink-500"
    }
  ];

  const quantumCircuitLayers = [
    {
      name: "ZZFeatureMap",
      description: "Data encoding layer that maps classical features to quantum states using rotation gates and ZZ entanglement",
      specs: {
        "Feature Dimension": "3 qubits",
        "Repetitions": "1-2 reps",
        "Entanglement": "Linear",
        "Gate Types": "RY, RZ, CNOT"
      },
      color: "primary"
    },
    {
      name: "RealAmplitudes Ansatz",
      description: "Variational layer with trainable parameters that learns optimal quantum state transformations",
      specs: {
        "Qubits": "3",
        "Repetitions": "1-3 reps",
        "Parameters": "6-12",
        "Gate Types": "RY, CNOT"
      },
      color: "secondary"
    },
    {
      name: "Measurement Layer",
      description: "Collapses quantum states to classical bits for classification output",
      specs: {
        "Classical Bits": "3",
        "Output": "Binary class",
        "Method": "Z-basis measurement",
        "Shots": "1024"
      },
      color: "accent"
    }
  ];

  const modelComparison = [
    {
      model: "VQC (IBM Brisbane)",
      accuracy: "70.0%",
      precision: "69.5%",
      recall: "70.0%",
      f1: "69.0%",
      time: "445.7s",
      type: "quantum"
    },
    {
      model: "VQC (Simulator)",
      accuracy: "87.3%",
      precision: "86.8%",
      recall: "87.3%",
      f1: "86.5%",
      time: "12.3s",
      type: "quantum"
    },
    {
      model: "Random Forest",
      accuracy: "52.1%",
      precision: "51.8%",
      recall: "52.1%",
      f1: "51.2%",
      time: "0.8s",
      type: "classical"
    }
  ];

  const features = [
    {
      title: "Real-time Market Data",
      description: "Live FTSE 100 data integration via yfinance with automatic updates",
      icon: Activity
    },
    {
      title: "Quantum Computing",
      description: "IBM Quantum hardware execution on Brisbane QPU with 127 qubits",
      icon: Cpu
    },
    {
      title: "Technical Analysis",
      description: "7 technical indicators: SMA, RSI, MACD, ADX, OBV, and derived features",
      icon: BarChart3
    },
    {
      title: "Volatility Detection",
      description: "Quantum clustering for market regime identification (Bull/Bear/Sideways/Crisis)",
      icon: Gauge
    },
    {
      title: "Model Comparison",
      description: "Side-by-side comparison of quantum vs classical ML approaches",
      icon: GitBranch
    },
    {
      title: "Interactive Dashboard",
      description: "Real-time visualization with React, Recharts, and Tailwind CSS",
      icon: TrendingUp
    }
  ];

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-border/40 bg-background/95 backdrop-blur">
        <div className="w-full mx-auto px-4 sm:px-6 md:px-8 lg:px-12 py-4">
          <div className="flex items-center justify-between">
            <Button 
              variant="ghost" 
              onClick={() => navigate('/')}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
            <Badge className="bg-primary/20 text-primary border-primary/30">
              System Architecture
            </Badge>
          </div>
        </div>
      </header>

      <main className="relative z-10 w-full mx-auto px-4 sm:px-6 md:px-8 lg:px-12 xl:px-16 py-8 md:py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4">
            <span className="text-primary">Quantum ML</span>
            {" "}<span className="text-foreground">Architecture</span>
          </h1>
          <p className="text-muted-foreground max-w-3xl mx-auto text-base md:text-lg">
            A hybrid classical-quantum approach to equity market direction classification 
            using Variational Quantum Classifiers on IBM Quantum hardware
          </p>
        </div>

        {/* ML Pipeline Flow */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <Workflow className="h-6 w-6 text-primary" />
            <h2 className="text-2xl font-bold text-foreground">ML Pipeline Flow</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {pipelineSteps.map((step, index) => (
              <Card key={index} className="p-5 bg-card/50 border-border/50 relative overflow-hidden group hover:bg-card/70 transition-all">
                <div className={`absolute top-0 left-0 w-1 h-full bg-gradient-to-b ${step.color}`}></div>
                <div className="flex items-start gap-4">
                  <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${step.color} flex items-center justify-center flex-shrink-0`}>
                    <span className="text-white font-bold text-sm">{step.step}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-base mb-1 text-foreground">{step.title}</h3>
                    <p className="text-xs text-muted-foreground mb-2">{step.description}</p>
                    <ul className="space-y-1">
                      {step.details.map((detail, idx) => (
                        <li key={idx} className="text-xs text-muted-foreground flex items-center gap-1.5">
                          <CheckCircle2 className="h-3 w-3 text-emerald-400 flex-shrink-0" />
                          <span>{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
                {index < pipelineSteps.length - 1 && (
                  <ArrowRight className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/50" />
                )}
              </Card>
            ))}
          </div>
        </section>

        {/* Quantum Circuit Architecture */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <CircuitBoard className="h-6 w-6 text-purple-400" />
            <h2 className="text-2xl font-bold text-foreground">Quantum Circuit Architecture</h2>
          </div>

          {/* Circuit Diagram Visualization */}
          <Card className="p-6 bg-card/30 border-border/50 mb-6">
            <h3 className="font-semibold mb-4 text-center text-foreground">Complete VQC Circuit</h3>
            <div className="bg-white rounded-lg p-4 flex justify-center">
              <img 
                src="/circuit_image.png" 
                alt="Variational Quantum Classifier Circuit showing ZZFeatureMap and RealAmplitudes layers"
                className="max-w-full h-auto"
                style={{ maxHeight: '300px' }}
              />
            </div>
            <p className="text-xs text-muted-foreground text-center mt-4">
              3-qubit VQC with ZZFeatureMap (x[0], x[1], x[2]) for data encoding and RealAmplitudes (θ[0]-θ[11]) for variational optimization
            </p>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {quantumCircuitLayers.map((layer, index) => (
              <Card key={index} className="p-5 bg-card/50 border-border/50">
                <div className="flex items-center gap-2 mb-3">
                  <div className={`w-3 h-3 rounded-full bg-${layer.color}`}></div>
                  <h3 className="font-semibold text-foreground">{layer.name}</h3>
                </div>
                <p className="text-xs text-muted-foreground mb-4">{layer.description}</p>
                <div className="space-y-2">
                  {Object.entries(layer.specs).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-muted-foreground">{key}</span>
                      <span className="font-medium text-foreground">{value}</span>
                    </div>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Model Performance Comparison */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="h-6 w-6 text-emerald-400" />
            <h2 className="text-2xl font-bold text-foreground">Model Performance Comparison</h2>
          </div>

          <Card className="p-6 bg-card/30 border-border/50 overflow-x-auto">
            <table className="w-full min-w-[600px]">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="text-left py-3 px-4 font-semibold text-muted-foreground">Model</th>
                  <th className="text-center py-3 px-4 font-semibold text-muted-foreground">Accuracy</th>
                  <th className="text-center py-3 px-4 font-semibold text-muted-foreground">Precision</th>
                  <th className="text-center py-3 px-4 font-semibold text-muted-foreground">Recall</th>
                  <th className="text-center py-3 px-4 font-semibold text-muted-foreground">F1-Score</th>
                  <th className="text-center py-3 px-4 font-semibold text-muted-foreground">Time</th>
                </tr>
              </thead>
              <tbody>
                {modelComparison.map((model, index) => (
                  <tr key={index} className="border-b border-border/30 hover:bg-card/50">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {model.type === 'quantum' ? (
                          <Cpu className="h-4 w-4 text-primary" />
                        ) : (
                          <Brain className="h-4 w-4 text-muted-foreground" />
                        )}
                        <span className="font-medium text-foreground">{model.model}</span>
                        {model.type === 'quantum' && (
                          <Badge variant="outline" className="text-xs">Quantum</Badge>
                        )}
                      </div>
                    </td>
                    <td className="text-center py-3 px-4">
                      <span className={model.type === 'quantum' ? 'text-cyan-400 font-semibold' : 'text-foreground'}>
                        {model.accuracy}
                      </span>
                    </td>
                    <td className="text-center py-3 px-4 text-foreground">{model.precision}</td>
                    <td className="text-center py-3 px-4 text-foreground">{model.recall}</td>
                    <td className="text-center py-3 px-4 text-foreground">{model.f1}</td>
                    <td className="text-center py-3 px-4 text-muted-foreground">{model.time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </section>

        {/* Technical Stack */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <Zap className="h-6 w-6 text-primary" />
            <h2 className="text-2xl font-bold text-foreground">Technical Stack</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <Database className="h-4 w-4 text-blue-400" />
                Data Layer
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• yfinance - Market data API</li>
                <li>• pandas - Data manipulation</li>
                <li>• numpy - Numerical computing</li>
              </ul>
            </Card>

            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <Cpu className="h-4 w-4 text-purple-400" />
                Quantum Layer
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Qiskit 1.4.4 - Quantum SDK</li>
                <li>• Qiskit ML 0.8.3 - VQC</li>
                <li>• IBM Brisbane - QPU Backend</li>
              </ul>
            </Card>

            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <Brain className="h-4 w-4 text-green-400" />
                Classical ML
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• scikit-learn - ML algorithms</li>
                <li>• Random Forest - Baseline</li>
                <li>• MinMaxScaler - Normalization</li>
              </ul>
            </Card>

            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <Layers className="h-4 w-4 text-cyan-400" />
                Backend
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Flask - REST API</li>
                <li>• Flask-CORS - Cross-origin</li>
                <li>• joblib - Model persistence</li>
              </ul>
            </Card>

            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <Activity className="h-4 w-4 text-pink-400" />
                Frontend
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• React 18 + TypeScript</li>
                <li>• Vite - Build tool</li>
                <li>• Tailwind CSS - Styling</li>
              </ul>
            </Card>

            <Card className="p-5 bg-card/50 border-border/50">
              <h3 className="font-semibold mb-3 flex items-center gap-2 text-foreground">
                <BarChart3 className="h-4 w-4 text-orange-400" />
                Visualization
              </h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Recharts - React charts</li>
                <li>• shadcn/ui - Components</li>
                <li>• Lucide Icons - Icons</li>
              </ul>
            </Card>
          </div>
        </section>

        {/* Key Features */}
        <section className="mb-12">
          <div className="flex items-center gap-3 mb-6">
            <CheckCircle2 className="h-6 w-6 text-emerald-400" />
            <h2 className="text-2xl font-bold text-foreground">Key Features</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, index) => (
              <Card key={index} className="p-5 bg-card/50 border-border/50 hover:bg-card/70 transition-all">
                <feature.icon className="h-8 w-8 text-primary mb-3" />
                <h3 className="font-semibold mb-2 text-foreground">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </section>

        {/* CTA */}
        <div className="text-center">
          <Button 
            onClick={() => navigate('/dashboard')}
            size="lg"
            className="text-lg px-8 py-6"
          >
            Explore Dashboard
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </main>
    </div>
  );
};

export default ArchitecturePage;
