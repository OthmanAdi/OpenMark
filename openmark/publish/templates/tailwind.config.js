/**
 * Tailwind config for the email templates.
 *
 * Maizzle scans these files for class names + inlines into the output HTML.
 * Email clients (Outlook especially) don't support modern CSS features, so
 * we stick to a deliberately conservative subset: simple colors, padding,
 * font-family, font-size, basic borders.
 */

module.exports = {
  content: [
    './src/**/*.html',
    './src/**/*.njk',
  ],
  theme: {
    extend: {
      colors: {
        ink:    '#0f172a',   // dark text
        muted:  '#64748b',   // secondary text
        rule:   '#e2e8f0',   // section divider
        accent: '#6366f1',   // link / hook
        bg:     '#ffffff',
        panel:  '#f8fafc',
      },
      fontFamily: {
        sans:  ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
      },
      maxWidth: {
        email: '640px',
      },
    },
  },
  corePlugins: {
    // Email clients can't do these — drop them so the inlined CSS stays small
    preflight: false,
    container: false,
    backgroundOpacity: false,
    textOpacity: false,
    placeholderOpacity: false,
    ringWidth: false,
    ringColor: false,
    boxShadow: false,
  },
};
