import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend, ReferenceLine } from "recharts";

interface ForecastChartProps {
  data: Array<{
    date: string;
    historical: number | null;
    predicted: number | null;
    confidenceUpper: number | null;
    confidenceLower: number | null;
  }>;
  companyName: string;
}

const ForecastChart = ({ data, companyName }: ForecastChartProps) => {
  // Calculate average price for reference line
  const validPrices = data.filter(d => d.historical !== null || d.predicted !== null);
  const avgPrice = validPrices.length > 0 
    ? validPrices.reduce((sum, d) => sum + (d.historical || d.predicted || 0), 0) / validPrices.length 
    : 0;

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card/95 backdrop-blur-sm border border-border rounded-xl p-4 shadow-xl">
          <p className="text-sm font-semibold text-foreground mb-3 border-b border-border/50 pb-2">{label}</p>
          {payload.map((entry: any, index: number) => {
            if (entry.value === null || entry.value === undefined) return null;
            return (
              <div key={index} className="flex items-center justify-between space-x-4 text-sm py-1">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: entry.color }}
                  />
                  <span className="text-muted-foreground">{entry.name}:</span>
                </div>
                <span className="font-semibold text-foreground">
                  ${entry.value?.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      );
    }
    return null;
  };

  const CustomLegend = ({ payload }: any) => {
    return (
      <div className="flex justify-center gap-6 mt-4 flex-wrap">
        {payload?.map((entry: any, index: number) => (
          <div key={index} className="flex items-center space-x-2">
            <div 
              className="w-4 h-1 rounded-full"
              style={{ 
                backgroundColor: entry.color,
                boxShadow: `0 0 8px ${entry.color}40`
              }}
            />
            <span className="text-sm text-muted-foreground">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="w-full h-96 relative">
      {/* Chart Title & Info */}
      <div className="absolute top-0 left-0 z-10 flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-muted-foreground">Live Data</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {companyName && `${companyName} • `}Quantum VQC Prediction
        </div>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 40, right: 30, left: 20, bottom: 20 }}>
          <defs>
            {/* Historical gradient - Cyan/Blue */}
            <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.4}/>
              <stop offset="50%" stopColor="#06b6d4" stopOpacity={0.15}/>
              <stop offset="100%" stopColor="#0891b2" stopOpacity={0}/>
            </linearGradient>
            
            {/* Predicted gradient - Purple/Violet */}
            <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#a855f7" stopOpacity={0.4}/>
              <stop offset="50%" stopColor="#9333ea" stopOpacity={0.15}/>
              <stop offset="100%" stopColor="#7c3aed" stopOpacity={0}/>
            </linearGradient>
            
            {/* Confidence band gradient */}
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.15}/>
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0}/>
            </linearGradient>

            {/* Glow filters */}
            <filter id="glowCyan" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            <filter id="glowPurple" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="hsl(225 15% 25%)" 
            opacity={0.5}
            vertical={false}
          />
          
          <XAxis 
            dataKey="date" 
            stroke="hsl(210 20% 60%)"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: 'hsl(225 15% 25%)' }}
            tick={{ fill: 'hsl(210 20% 60%)' }}
            dy={10}
          />
          
          <YAxis 
            stroke="hsl(210 20% 60%)"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: 'hsl(225 15% 25%)' }}
            tick={{ fill: 'hsl(210 20% 60%)' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
            dx={-10}
            domain={['auto', 'auto']}
          />
          
          {/* Average price reference line */}
          {avgPrice > 0 && (
            <ReferenceLine 
              y={avgPrice} 
              stroke="hsl(210 20% 50%)" 
              strokeDasharray="8 4"
              strokeOpacity={0.5}
              label={{ 
                value: `Avg: $${avgPrice.toFixed(0)}`, 
                position: 'right',
                fill: 'hsl(210 20% 60%)',
                fontSize: 10
              }}
            />
          )}
          
          <Tooltip content={<CustomTooltip />} />
          <Legend content={<CustomLegend />} />
          
          {/* Confidence interval - Upper band */}
          <Area
            type="monotone"
            dataKey="confidenceUpper"
            stroke="none"
            fill="url(#confidenceGradient)"
            fillOpacity={1}
            name="Confidence Band"
          />
          
          {/* Confidence interval - Lower band */}
          <Area
            type="monotone"
            dataKey="confidenceLower"
            stroke="none"
            fill="url(#confidenceGradient)"
            fillOpacity={1}
          />
          
          {/* Historical data - Primary line */}
          <Area
            type="monotone"
            dataKey="historical"
            stroke="#22d3ee"
            strokeWidth={3}
            fill="url(#historicalGradient)"
            connectNulls={false}
            name="Historical Price"
            dot={{ fill: '#22d3ee', strokeWidth: 0, r: 3 }}
            activeDot={{ 
              fill: '#22d3ee', 
              stroke: '#fff', 
              strokeWidth: 2, 
              r: 6,
              filter: 'url(#glowCyan)'
            }}
          />
          
          {/* Predicted data - Secondary line */}
          <Area
            type="monotone"
            dataKey="predicted"
            stroke="#a855f7"
            strokeWidth={3}
            strokeDasharray="8 4"
            fill="url(#predictedGradient)"
            connectNulls={false}
            name="Quantum Prediction"
            dot={{ fill: '#a855f7', strokeWidth: 0, r: 3 }}
            activeDot={{ 
              fill: '#a855f7', 
              stroke: '#fff', 
              strokeWidth: 2, 
              r: 6,
              filter: 'url(#glowPurple)'
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;