import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { cx } from '@/lib/cx'

type Variant = 'primary' | 'secondary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  children: ReactNode
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: 'bg-accent text-accent-contrast hover:bg-accent-strong',
  secondary:
    'bg-surface-raised text-text border border-border hover:border-border-strong',
  ghost: 'text-text-muted hover:text-text hover:bg-surface-raised',
}

/** Fixed class strings per variant -- no variant composition across component boundaries, so
 * tailwind-merge buys nothing here. See components/ui's design-token rationale in index.css. */
export function Button({ variant = 'primary', className, children, ...rest }: ButtonProps) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-console px-3 py-1.5',
        'font-sans text-sm font-medium transition-colors',
        'disabled:pointer-events-none disabled:opacity-50',
        VARIANT_CLASSES[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}
