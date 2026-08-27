import { useCallback, useEffect, useState } from 'react'

type ThemeChoice = 'dark' | 'light'

const STORAGE_KEY = 'riskiq-theme'

function readStoredChoice(): ThemeChoice | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored === 'dark' || stored === 'light' ? stored : null
  } catch {
    // Private browsing / site-data-blocked. No explicit choice persists; system preference
    // (and the dark @theme default) still apply via CSS regardless.
    return null
  }
}

function systemPrefersLight(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: light)').matches
}

/** What index.css actually renders right now: the explicit choice if there is one, otherwise
 * dark unless the system prefers light -- the same rule the CSS itself encodes. Kept as one
 * function so the toggle button's label and its click handler can never disagree. */
function resolveEffectiveTheme(choice: ThemeChoice | null): ThemeChoice {
  if (choice !== null) return choice
  return systemPrefersLight() ? 'light' : 'dark'
}

/**
 * The (enhancement-pass) dark/light toggle. Dark is the CSS default -- see index.css's
 * rationale -- so this hook's only job is to apply an *explicit* choice by setting
 * `data-theme` on the document root, and to persist that one choice for next visit. A
 * `null` choice (nothing stored yet) means "follow system preference", which the CSS already
 * does without any attribute present.
 */
export function useTheme() {
  const [choice, setChoiceState] = useState<ThemeChoice | null>(() => readStoredChoice())

  useEffect(() => {
    const root = document.documentElement
    if (choice === null) {
      root.removeAttribute('data-theme')
    } else {
      root.setAttribute('data-theme', choice)
    }
  }, [choice])

  const setChoice = useCallback((next: ThemeChoice) => {
    setChoiceState(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // Best-effort persistence; the choice still applies for this session either way.
    }
  }, [])

  const effectiveTheme = resolveEffectiveTheme(choice)

  const toggle = useCallback(() => {
    setChoice(effectiveTheme === 'dark' ? 'light' : 'dark')
  }, [effectiveTheme, setChoice])

  return { choice, effectiveTheme, toggle }
}
