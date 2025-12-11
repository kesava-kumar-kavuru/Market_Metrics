import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";

// Mock FTSE 100 data
const mockData = [
  { time: "09:00", price: 7850, prediction: "up" },
  { time: "10:00", price: 7865, prediction: "up" },
  { time: "11:00", price: 7840, prediction: "down" },
  { time: "12:00", price: 7825, prediction: "down" },
  { time: "13:00", price: 7845, prediction: "up" },
  { time: "14:00", price: 7880, prediction: "up" },
  { time: "15:00", price: 7895, prediction: "up" },
  { time: "16:00", price: 7910, prediction: "up" },
];

export function MarketChart() {
  const currentPrice = 7910;
  const dailyChange = "+60 (+0.76%)";
  const quantumPrediction = "UP";
  const confidence = "87.3%";

  return (
    <Card className="border-border/50">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-foreground">FTSE 100 - Quantum Analysis</CardTitle>
          <Badge className={quantumPrediction === "UP" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}>
            {quantumPrediction} {confidence}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Current Price */}
          <div className="flex items-baseline space-x-2">
            <span className="text-3xl font-bold text-foreground">
              {currentPrice.toLocaleString()}
            </span>
            <span className="text-emerald-400 font-medium">{dailyChange}</span>
          </div>

          {/* Mini Chart Visualization */}
          <div className="h-32 relative bg-muted/10 rounded-lg p-4 border border-primary/20">
            <div className="flex items-end justify-between h-full">
              {mockData.map((point, index) => (
                <div key={index} className="flex flex-col items-center space-y-1 flex-1">
                  <div
                    className={`w-2 bg-gradient-to-t rounded-full ${
                      point.prediction === "up" 
                        ? "from-emerald-500/60 to-emerald-400" 
                        : "from-red-500/60 to-red-400"
                    }`}
                    style={{
                      height: `${((point.price - 7800) / 150) * 100}%`,
                      minHeight: "4px"
                    }}
                  />
                  <span className="text-xs text-muted-foreground rotate-45 mt-2">
                    {point.time}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Quantum Indicators */}
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="text-center p-2 bg-cyan-500/10 rounded border border-cyan-500/20">
              <div className="text-xs text-muted-foreground">Volatility</div>
              <div className="font-semibold text-cyan-400">Low</div>
            </div>
            <div className="text-center p-2 bg-purple-500/10 rounded border border-purple-500/20">
              <div className="text-xs text-muted-foreground">Momentum</div>
              <div className="font-semibold text-purple-400">Strong</div>
            </div>
            <div className="text-center p-2 bg-emerald-500/10 rounded border border-emerald-500/20">
              <div className="text-xs text-muted-foreground">Signal</div>
              <div className="font-semibold text-emerald-400">Buy</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}