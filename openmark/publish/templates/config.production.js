/**
 * Production Maizzle config — minified + shorthand CSS for outbound sends.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'pathe';

const __dirname = dirname(fileURLToPath(import.meta.url));
const payload = JSON.parse(
  readFileSync(resolve(__dirname, 'data/payload.json'), 'utf-8'),
);

/** @type {import('@maizzle/framework').Config} */
export default {
  build: {
    content: ['emails/**/*.html'],
    output: {
      path: 'build_production',
      extension: 'html',
    },
    summary: true,
  },
  components: {
    root: './',
    folders: ['components', 'layouts'],
    tag: 'component',
    tagPrefix: 'x-',
    attribute: 'src',
    fileExtension: 'html',
  },
  locals: {
    data: payload,
  },
  inlineCSS: true,
  removeUnusedCSS: true,
  shorthandInlineCSS: true,
  prettify: false,
};
