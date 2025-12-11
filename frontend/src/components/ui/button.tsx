import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-white hover:bg-primary/80 hover:shadow-md hover:shadow-primary/25 active:scale-[0.98]",
        destructive:
          "bg-red-600 text-white hover:bg-red-500 hover:shadow-md hover:shadow-red-500/25 active:scale-[0.98]",
        outline:
          "border-2 border-primary/50 bg-transparent text-primary hover:bg-primary/10 hover:border-primary active:scale-[0.98]",
        secondary:
          "bg-muted text-foreground border border-border hover:bg-muted/80 hover:border-border/80 active:scale-[0.98]",
        ghost: "text-foreground hover:bg-muted/50 hover:text-primary active:scale-[0.98]",
        link: "text-primary underline-offset-4 hover:underline hover:text-primary/80",
        quantum: "bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:from-cyan-400 hover:to-purple-500 hover:shadow-lg hover:shadow-purple-500/30 transform hover:scale-[1.02] active:scale-[0.98] font-semibold",
        financial: "bg-gradient-to-r from-emerald-500 to-cyan-500 text-white hover:from-emerald-400 hover:to-cyan-400 hover:shadow-lg hover:shadow-emerald-500/30 transform hover:scale-[1.02] active:scale-[0.98] font-semibold",
        neural: "bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-400 hover:to-pink-400 hover:shadow-lg hover:shadow-purple-500/30 transform hover:scale-[1.02] active:scale-[0.98] font-semibold",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }