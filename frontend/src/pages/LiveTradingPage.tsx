import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Badge } from "../components/ui/badge";
import { TrendingUp, TrendingDown, Activity, BarChart3, Loader2 } from "lucide-react";
import ForecastChart from "../components/ForecastChart";
import CompanyMetrics from "../components/CompanyMetrics";
import { companies } from "../lib/mockData";

const LiveTradingPage = () => {
  const [selectedCompany, setSelectedCompany] = useState("VOD.L");
  const [companyData, setCompanyData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:5000/api/live-trading/${selectedCompany}`);
        if (!response.ok) {
          throw new Error("Failed to fetch data");
        }
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        setCompanyData(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedCompany]);

  const staticInfo = companies.find(c => c.symbol === selectedCompany);

  const chartData = companyData?.forecastData?.map((item: any) => ({
    date: item.date,
    historical: item.actual,
    predicted: item.vqc_prediction,
    confidenceUpper: item.vqc_prediction * (1 + (1 - item.confidence) * 0.5),
    confidenceLower: item.vqc_prediction * (1 - (1 - item.confidence) * 0.5),
  })) || [];

  if (loading && !companyData) {
      return (
          <div className="min-h-screen bg-background flex items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
      );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Quantum Live Trading
                </h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20">
                Real-Time Quantum Trading
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-3xl font-bold text-foreground mb-2">Live Trading Dashboard</h2>
              <p className="text-muted-foreground">Select a company to view and trade in real time</p>
            </div>
            <Select value={selectedCompany} onValueChange={setSelectedCompany}>
              <SelectTrigger className="w-64 bg-card border-border">
                <SelectValue placeholder="Select a company" />
              </SelectTrigger>
              <SelectContent>
                {companies.map((company: { symbol: string; name: string }) => (
                  <SelectItem key={company.symbol} value={company.symbol}>
                    <div className="flex items-center space-x-3">
                      <span className="font-medium">{company.symbol}</span>
                      <span className="text-muted-foreground">{company.name}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {error && (
            <div className="p-4 mb-6 bg-destructive/10 text-destructive rounded-lg border border-destructive/20">
                Error: {error}
            </div>
        )}

        {companyData && staticInfo && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              <Card className="lg:col-span-2 bg-gradient-to-br from-card to-card/80 border-border/50">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-2xl font-bold text-foreground">
                        {staticInfo.name}
                      </CardTitle>
                      <p className="text-muted-foreground">{staticInfo.symbol} • {staticInfo.sector}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-foreground">
                        ${companyData.currentPrice?.toFixed(2)}
                      </div>
                      <div className={`flex items-center space-x-1 ${
                        companyData.priceChange >= 0 ? 'text-success' : 'text-destructive'
                      }`}>
                        {companyData.priceChange >= 0 ? 
                          <TrendingUp className="h-4 w-4" /> : 
                          <TrendingDown className="h-4 w-4" />
                        }
                        <span className="font-medium">
                          {companyData.priceChange >= 0 ? '+' : ''}${companyData.priceChange?.toFixed(2)} ({companyData.changePercent?.toFixed(2)}%)
                        </span>
                      </div>
                    </div>
                  </div>
                </CardHeader>
              </Card>

              <CompanyMetrics company={{
                  ...staticInfo,
                  currentPrice: companyData.currentPrice,
                  previousClose: companyData.previousClose,
                  volume: "N/A",
                  marketCap: "N/A"
              }} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2 bg-gradient-to-br from-card to-card/80 border-border/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <span>Live Quantum Trading Chart</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ForecastChart data={chartData} companyName={staticInfo.name} />
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-card to-card/80 border-border/50">
                <CardHeader>
                  <CardTitle className="text-lg">Trading Insights</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 rounded-lg bg-primary/5 border border-primary/20">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-2 h-2 rounded-full bg-primary"></div>
                      <span className="font-medium text-sm">Live Signal</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        companyData.insights.signal === 'BUY' ? 'text-success' : 'text-destructive'
                    }`}>
                      {companyData.insights.signal}
                    </div>
                    <p className="text-xs text-muted-foreground">Quantum trading recommendation</p>
                  </div>

                  <div className="p-4 rounded-lg bg-accent/5 border border-accent/20">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-2 h-2 rounded-full bg-accent"></div>
                      <span className="font-medium text-sm">Confidence Level</span>
                    </div>
                    <div className="text-2xl font-bold text-foreground">
                      {companyData.insights.confidence}%
                    </div>
                    <p className="text-xs text-muted-foreground">Quantum certainty</p>
                  </div>

                  <div className="p-4 rounded-lg bg-warning/5 border border-warning/20">
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-2 h-2 rounded-full bg-warning"></div>
                      <span className="font-medium text-sm">Risk Factor</span>
                    </div>
                    <div className={`text-2xl font-bold ${
                        companyData.insights.risk === 'High' ? 'text-destructive' : 
                        companyData.insights.risk === 'Medium' ? 'text-warning' : 'text-success'
                    }`}>
                      {companyData.insights.risk}
                    </div>
                    <p className="text-xs text-muted-foreground">Market volatility</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LiveTradingPage;
