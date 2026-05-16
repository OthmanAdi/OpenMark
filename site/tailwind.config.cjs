/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{astro,html,js,jsx,ts,tsx,md,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink:    '#0f172a',
        muted:  '#64748b',
        rule:   '#e2e8f0',
        accent: '#6366f1',
        bg:     '#ffffff',
        panel:  '#f8fafc',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Inter', 'Segoe UI', 'sans-serif'],
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      typography: ({ theme }) => ({
        DEFAULT: {
          css: {
            color: theme('colors.ink'),
            a: {
              color: theme('colors.accent'),
              textDecoration: 'none',
              '&:hover': { textDecoration: 'underline' },
            },
            h1: { color: theme('colors.ink'), fontWeight: '700' },
            h2: { color: theme('colors.ink'), fontWeight: '600' },
            h3: { color: theme('colors.ink'), fontWeight: '600' },
            blockquote: {
              borderLeftColor: theme('colors.accent'),
              color: theme('colors.muted'),
              fontStyle: 'normal',
            },
            code: {
              color: theme('colors.ink'),
              backgroundColor: theme('colors.panel'),
              padding: '0.125rem 0.375rem',
              borderRadius: '0.25rem',
              fontWeight: '500',
            },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
          },
        },
      }),
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
