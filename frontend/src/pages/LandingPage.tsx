import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { ArrowRight, Brain, TrendingUp, Zap, BarChart3 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const LandingPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "Variational Quantum Classifier",
      description: "Advanced VQC model for market prediction with 70% accuracy on IBM Brisbane"
    },
    {
      icon: TrendingUp,
      title: "FTSE 100 Analysis",
      description: "Real-time analysis of FTSE 100 market data using yfinance"
    },
    {
      icon: Zap,
      title: "IBM Quantum Integration",
      description: "Leveraging IBM Quantum computers for enhanced predictions"
    },
    {
      icon: BarChart3,
      title: "SVM Comparison",
      description: "Side-by-side comparison with classical SVM algorithms"
    }
  ];

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Quantum Background Effects */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-secondary/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-3/4 left-1/2 w-64 h-64 bg-accent/10 rounded-full blur-2xl animate-pulse" style={{ animationDelay: '4s' }}></div>
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-border/40 bg-background/95 backdrop-blur w-full">
        <div className="w-full mx-auto px-4 sm:px-6 md:px-8 lg:px-12 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 gradient-quantum rounded-xl flex items-center justify-center">
                <span className="text-white font-bold">Q</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-foreground">
                  QuantumTrader ML
                </h1>
                <p className="text-sm text-muted-foreground">Quantum Market Prediction</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge className="animate-quantum-pulse hidden md:flex">
                <div className="w-2 h-2 bg-accent rounded-full mr-2"></div>
                Live Trading
              </Badge>

              <a
                href="https://q-folio-mu.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-3 py-2 rounded-md bg-white text-black text-sm font-medium border border-gray-300 hover:bg-gray-50"
              >
                qfolio.ai
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 w-full mx-auto px-4 sm:px-6 md:px-8 lg:px-12 xl:px-16 py-12 md:py-16">
        <div className="text-center w-full max-w-6xl mx-auto mb-12 md:mb-16">
          <Badge className="mb-6 border-primary/20 text-primary">
            Hackathon Project • FTSE 100 Classification
          </Badge>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6">
            <span className="text-primary">
              Quantum ML
            </span>
            <br />
            <span className="text-foreground">Market Predictor</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto px-2">
            Revolutionary quantum machine learning approach to predict FTSE 100 market direction 
            using Variational Quantum Classifiers with 70% accuracy on IBM Quantum hardware.
          </p>

          <div className="flex justify-center mb-12">
            <Button 
              onClick={() => navigate('/dashboard')}
              size="lg"
              variant="quantum"
              className="text-lg px-12 py-7 text-white"
            >
              Go to Dashboard
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 md:gap-5 lg:gap-6 mb-12 md:mb-16 w-full">
            <Card className="p-4 md:p-5 lg:p-6 bg-card/50 border-border/50">
              <div className="text-2xl md:text-3xl font-bold text-cyan-400">70.0%</div>
              <div className="text-xs md:text-sm text-muted-foreground">VQC Accuracy</div>
            </Card>
            <Card className="p-4 md:p-5 lg:p-6 bg-card/50 border-border/50">
              <div className="text-2xl md:text-3xl font-bold text-emerald-400">FTSE 100</div>
              <div className="text-xs md:text-sm text-muted-foreground">Market Data</div>
            </Card>
            <Card className="p-4 md:p-5 lg:p-6 bg-card/50 border-border/50">
              <div className="text-2xl md:text-3xl font-bold text-purple-400">Real-time</div>
              <div className="text-xs md:text-sm text-muted-foreground">Predictions</div>
            </Card>
            <Card className="p-4 md:p-5 lg:p-6 bg-card/50 border-border/50">
              <div className="text-2xl md:text-3xl font-bold text-blue-400">IBM Q</div>
              <div className="text-xs md:text-sm text-muted-foreground">Quantum Backend</div>
            </Card>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-5 lg:gap-6 mb-12 md:mb-16 w-full">
          {features.map((feature, index) => (
            <Card key={index} className="p-5 md:p-6 bg-card/50 border-border/50 hover:bg-card/70 transition-all duration-300">
              <feature.icon className="h-10 w-10 md:h-12 md:w-12 text-primary mb-3 md:mb-4" />
              <h3 className="text-base md:text-lg font-semibold mb-2 text-foreground">{feature.title}</h3>
              <p className="text-xs md:text-sm text-muted-foreground">{feature.description}</p>
            </Card>
          ))}
        </div>

        {/* Technical Overview */}
        <Card className="p-6 md:p-8 bg-card/30 border-border/50 w-full">
          <div className="text-center">
            <h2 className="text-2xl md:text-3xl font-bold mb-3 md:mb-4 text-foreground">Technical Architecture</h2>
            <p className="text-sm md:text-base text-muted-foreground mb-5 md:mb-6 max-w-3xl mx-auto px-2">
              Our quantum machine learning system combines classical preprocessing with quantum 
              variational circuits to achieve superior market prediction accuracy on FTSE 100 data.
            </p>
            <div className="flex justify-center">
              <Button
                onClick={() => navigate('/architecture')}
                variant="outline"
                size="lg"
              >
                View Architecture
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
};

export default LandingPage;