import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

export function QuantumCircuit() {
  return (
    <Card className="border-border/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-foreground">
          <div className="w-2 h-2 bg-primary rounded-full animate-quantum-pulse"></div>
          Variational Quantum Classifier (VQC)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Quantum Circuit Visualization */}
          <div className="bg-muted/20 rounded-lg p-4 border border-primary/20">
            <div className="space-y-3">
              {/* Qubit lines */}
              {[0, 1, 2, 3].map((qubit) => (
                <div key={qubit} className="flex items-center space-x-2">
                  <span className="text-xs text-muted-foreground w-6">|{qubit}⟩</span>
                  <div className="flex-1 h-px bg-primary/40 relative">
                    {/* Quantum gates */}
                    <div className="absolute top-1/2 left-8 -translate-y-1/2 w-6 h-6 bg-purple-500/80 rounded border-2 border-purple-300 flex items-center justify-center text-xs font-bold text-white">
                      H
                    </div>
                    <div className="absolute top-1/2 left-20 -translate-y-1/2 w-6 h-6 bg-cyan-500/80 rounded border-2 border-cyan-300 flex items-center justify-center text-xs font-bold text-white">
                      R
                    </div>
                    <div className="absolute top-1/2 left-32 -translate-y-1/2 w-6 h-6 bg-emerald-500/80 rounded border-2 border-emerald-300 flex items-center justify-center text-xs font-bold text-white">
                      M
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Circuit Parameters */}
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Depth</div>
              <div className="font-semibold text-cyan-400">4</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Parameters</div>
              <div className="font-semibold text-purple-400">16</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-muted-foreground">Qubits</div>
              <div className="font-semibold text-emerald-400">4</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}