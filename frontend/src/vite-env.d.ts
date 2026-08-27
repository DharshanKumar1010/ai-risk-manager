/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin. Set by docker-compose in development; required at build time for a
   * deployed frontend, since Vite inlines `import.meta.env.VITE_*` values into the bundle. */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
