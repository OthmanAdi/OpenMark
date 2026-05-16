/**
 * POST /api/subscribe
 *
 * Accepts form-urlencoded { email, source? } from the SubscribeForm component.
 * Inserts a pending subscriber, sends a confirm-link email via Resend, then
 * redirects the browser back to /subscribe?status=sent so the page can show
 * a "check your inbox" state.
 *
 * Errors are surfaced via ?status=error&reason=<short>. The form falls back
 * to a default error message if the page doesn't read the query string.
 */

import type { Env } from '../_lib/d1';
import {
  addOrRecycle,
  isValidEmail,
  normalizeEmail,
} from '../_lib/d1';
import {
  confirmEmailBody,
  sendEmail,
} from '../_lib/resend';

interface PagesContext {
  request: Request;
  env: Env;
  waitUntil: (p: Promise<unknown>) => void;
}

export const onRequestPost = async (ctx: PagesContext): Promise<Response> => {
  const { request, env } = ctx;
  const baseUrl = env.PUBLISH_BASE_URL || new URL(request.url).origin;
  const publication = env.PUBLICATION_NAME || 'OpenMark';
  const fromEmail = env.PUBLISH_FROM_EMAIL;

  let email = '';
  let source: string | null = 'site';

  // Parse form body
  try {
    const contentType = request.headers.get('content-type') || '';
    if (contentType.includes('application/x-www-form-urlencoded') || contentType.includes('multipart/form-data')) {
      const form = await request.formData();
      email = String(form.get('email') || '').trim();
      source = String(form.get('source') || 'site').trim().slice(0, 32) || null;
    } else if (contentType.includes('application/json')) {
      const body = await request.json<{ email?: string; source?: string }>();
      email = (body.email || '').trim();
      source = (body.source || 'site').slice(0, 32) || null;
    }
  } catch {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=bad-body`, 303);
  }

  if (!isValidEmail(email)) {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=bad-email`, 303);
  }
  if (!fromEmail) {
    return redirect(`${baseUrl}/subscribe/?status=error&reason=server-not-configured`, 303);
  }

  // Add or recycle the pending subscriber
  let sub;
  try {
    sub = await addOrRecycle(env, email, source);
  } catch (err) {
    console.error('subscribe: db error', err);
    return redirect(`${baseUrl}/subscribe/?status=error&reason=db`, 303);
  }

  if (sub.status === 'active') {
    // Already opted-in; treat as success
    return redirect(`${baseUrl}/subscribe/?status=already`, 303);
  }

  // Send the confirm-link email (Resend)
  const { subject, html, text } = confirmEmailBody(baseUrl, publication, sub.confirm_token);
  const sendResult = await sendEmail(env, {
    to: normalizeEmail(email),
    subject,
    html,
    text,
    from: fromEmail,
    replyTo: fromEmail,
  });

  if (!sendResult.ok) {
    console.error('subscribe: resend error', sendResult.error);
    return redirect(`${baseUrl}/subscribe/?status=error&reason=send-failed`, 303);
  }

  return redirect(`${baseUrl}/subscribe/?status=sent`, 303);
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
