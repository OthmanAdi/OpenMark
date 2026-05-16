/**
 * GET /api/unsubscribe?t=<token>
 *
 * One-click unsubscribe. Email footers point here. Idempotent.
 */

import type { Env } from '../_lib/d1';
import { findByUnsubscribeToken, setUnsubscribed } from '../_lib/d1';

interface PagesContext {
  request: Request;
  env: Env;
}

export const onRequestGet = async (ctx: PagesContext): Promise<Response> => {
  const { request, env } = ctx;
  const url = new URL(request.url);
  const token = url.searchParams.get('t') || '';
  const baseUrl = env.PUBLISH_BASE_URL || url.origin;

  if (!token) {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=missing-token`);
  }

  const sub = await findByUnsubscribeToken(env, token);
  if (!sub) {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=invalid-token`);
  }
  if (sub.status !== 'unsubscribed') {
    await setUnsubscribed(env, sub.id);
  }
  return redirect(`${baseUrl}/unsubscribe/`);
};

function redirect(url: string, status = 303): Response {
  return new Response(null, {
    status,
    headers: {
      Location: url,
      'Cache-Control': 'no-store',
    },
  });
}
