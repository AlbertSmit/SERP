import js from '@eslint/js'
import globals from 'globals'

// Selected standalone rules from kaliber-code-intelligence/eslint.config.js.
// No Kaliber ESLint presets, plugins, or repository-local rules are loaded.
export default [
  { ignores: ['node_modules/**', 'dist/**', 'build/**', '.cache/**'] },
  {
    files: ['**/*.{js,mjs,cjs}'],
    ...js.configs.recommended,
    languageOptions: {
      ecmaVersion: 'latest',
      globals: globals.node,
    },
    rules: {
      ...js.configs.recommended.rules,
      'func-style': ['error', 'declaration'],
      'no-console': ['warn', { allow: ['log', 'warn', 'error'] }],
      'no-unused-vars': ['error', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      }],
      complexity: ['warn', { max: 10, variant: 'modified' }],
      'max-depth': ['warn', 2],
      'max-nested-callbacks': ['warn', 3],
      'max-params': ['warn', 4],
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
]
