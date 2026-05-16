/**
 * Maizzle v5 base config (dev / local). The Python publish layer writes
 * `data/payload.json` just before invoking `maizzle build`; we read that
 * synchronously and expose it as `locals.data` so templates can do
 * {{ data.title }}, <each loop="s in data.sections">, etc.
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
      path: 'build_local',
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
  // Passed to posthtml-expressions — available as {{ data.* }} in templates
  locals: {
    data: payload,
  },
  inlineCSS: true,
  removeUnusedCSS: true,
  shorthandInlineCSS: false,
  prettify: false,
};
