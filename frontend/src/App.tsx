import { DecisionsPanel } from '@/components/console/DecisionsPanel'
import { Hero } from '@/components/console/Hero'
import { MetricsPanel } from '@/components/metrics/MetricsPanel'
import { RingsPanel } from '@/components/rings/RingsPanel'
import { ScoringWidget } from '@/components/scoring/ScoringWidget'
import { Button } from '@/components/ui/Button'
import { AuthProvider } from '@/hooks/useAuth'
import { useTheme } from '@/hooks/useTheme'

/**
 * Application shell — the risk-operations console. See index.css for the design-token
 * brainstorm (colour, type, layout, signature element) this was built from, per Phase 8
 * step 1.
 */
export default function App() {
  const { effectiveTheme, toggle } = useTheme()

  return (
    <AuthProvider>
      <div className="min-h-dvh bg-bg">
        <header className="border-b border-border bg-surface">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-sm font-semibold tracking-tight text-text">
                RiskIQ
              </span>
              <span className="font-sans text-xs text-text-faint">risk-ops console</span>
            </div>
            <Button
              variant="ghost"
              onClick={toggle}
              aria-label={
                effectiveTheme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
              }
            >
              {effectiveTheme === 'dark' ? 'Light' : 'Dark'}
            </Button>
          </div>
        </header>

        <main aria-label="RiskIQ" className="mx-auto max-w-6xl space-y-8 px-4 py-6">
          <Hero />
          <ScoringWidget />
          <DecisionsPanel />
          <RingsPanel />
          <MetricsPanel />
        </main>
      </div>
    </AuthProvider>
  )
}
