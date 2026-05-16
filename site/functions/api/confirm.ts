/**
 * GET /api/confirm?t=<token>
 *
 * Clicked from the confirm-link email. Flips a pending row to active.
 * Idempotent — clicking twice is fine.
 */

import type { Env } from '../_lib/d1';
import { findByConfirmToken, setActive } from '../_lib/d1';

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

  const sub = await findByConfirmToken(env, token);
  if (!sub) {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=invalid-token`);
  }
  if (sub.status === 'unsubscribed' || sub.status === 'bounced') {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=token-revoked`);
  }
  if (sub.status === 'pending') {
    await setActive(env, sub.id);
  }
  return redirect(`${baseUrl}/confirmed/`);
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
