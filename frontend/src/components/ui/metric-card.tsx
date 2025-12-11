import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { cn } from "../../lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: string | number;
  trend?: "up" | "down" | "neutral";
  icon?: React.ReactNode;
  className?: string;
}

export function MetricCard({ title, value, change, trend, icon, className }: MetricCardProps) {
  return (
    <Card className={cn("bg-card/80 border-border/60 hover:border-primary/40 hover:bg-card/90 transition-all duration-300", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-foreground/80">{title}</CardTitle>
        {icon && <div className="text-primary">{icon}</div>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-foreground">
          {value}
        </div>
        {change !== undefined && (
          <p className={cn(
            "text-xs mt-1 font-medium",
            trend === "up" ? "text-green-400" : "",
            trend === "down" ? "text-red-400" : "",
            trend === "neutral" || !trend ? "text-muted-foreground" : ""
          )}>
            {typeof change === 'number' ? `${change > 0 ? '+' : ''}${change}%` : change}
          </p>
        )}
      </CardContent>
    </Card>
  );
}