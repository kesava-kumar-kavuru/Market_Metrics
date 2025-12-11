import { 
  Home, 
  TrendingUp, 
  BarChart3, 
  Activity,
  Cpu,
  Layers
} from "lucide-react";
import { NavLink } from "react-router-dom";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "../components/ui/sidebar";

const navigationItems = [
  { title: "Dashboard", url: "/dashboard", icon: Home },
  { title: "Predictions", url: "/predictions", icon: TrendingUp },
  { title: "Market Data", url: "/market-data", icon: BarChart3 },
  { title: "Quantum Predictor", url: "/quantum-predictor", icon: Cpu },
  { title: "Volatility Regime", url: "/volatility-regime", icon: Layers },
  { title: "Live Trading", url: "/live-trading", icon: Activity },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";

  const getNavClassName = ({ isActive }: { isActive: boolean }) =>
    isActive 
      ? "bg-primary/15 text-primary font-medium border-l-2 border-primary rounded-r-lg" 
      : "text-muted-foreground hover:text-foreground hover:bg-muted/50 hover:translate-x-1 transition-all duration-200 ease-out rounded-lg";

  return (
    <Sidebar
      className={`${collapsed ? "w-14" : "w-56"} border-r border-border/30 bg-background/95 backdrop-blur-sm`}
      collapsible="icon"
    >
      <SidebarContent className="bg-transparent px-2">
        {/* Logo Section */}
        <div className="px-2 py-5 border-b border-border/20 mb-2">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 gradient-quantum rounded-xl flex items-center justify-center shadow-lg shadow-primary/20">
              <span className="text-white font-bold text-sm">Q</span>
            </div>
            {!collapsed && (
              <div>
                <h1 className="text-base font-semibold text-foreground tracking-tight">
                  QuantumTrader
                </h1>
                <p className="text-[10px] text-muted-foreground/70 uppercase tracking-wider">ML Predictor</p>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <SidebarGroup className="px-1">
          <SidebarGroupContent>
            <SidebarMenu className="space-y-1">
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild className="h-10">
                    <NavLink
                      to={item.url}
                      className={getNavClassName}
                      end
                    >
                      <item.icon className="h-4 w-4 shrink-0" />
                      {!collapsed && <span className="text-sm">{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Footer */}
        {!collapsed && (
          <div className="mt-auto px-3 py-4 border-t border-border/20">
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground/60">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>System Online</span>
            </div>
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  );
}