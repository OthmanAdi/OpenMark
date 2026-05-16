/**
 * Resend HTTP send wrapper for the Workers runtime.
 *
 * The Workers runtime has fetch() built in. We POST to Resend's /emails
 * endpoint with Bearer auth. No npm dependency on the resend SDK.
 */

import type { Env } from './d1';

export interface SendArgs {
  to: string;
  subject: string;
  html: string;
  text?: string;
  from: string;
  replyTo?: string;
}

export async function sendEmail(env: Env, args: SendArgs): Promise<{ ok: boolean; id?: string; error?: string }> {
  if (!env.RESEND_API_KEY) {
    return { ok: false, error: 'RESEND_API_KEY missing on the Worker env' };
  }
  const body: Record<string, unknown> = {
    from: args.from,
    to: [args.to],
    subject: args.subject,
    html: args.html,
  };
  if (args.text) body.text = args.text;
  if (args.replyTo) body.reply_to = args.replyTo;

  const resp = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    return { ok: false, error: `${resp.status}: ${text.slice(0, 200)}` };
  }
  const data = await resp.json<{ id?: string }>();
  return { ok: true, id: data.id };
}


/**
 * Build the confirm-link email body. Inline HTML + plain-text fallback.
 * Kept dependency-free; the visual chrome is the Maizzle-rendered newsletter,
 * not this transactional email.
 */
export function confirmEmailBody(
  baseUrl: string,
  publicationName: string,
  confirmToken: string,
): { subject: string; html: string; text: string } {
  const link = `${baseUrl.replace(/\/$/, '')}/api/confirm?t=${confirmToken}`;
  const subject = `Confirm your ${publicationName} subscription`;
  const html = `<!doctype html><html lang="en"><body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;color:#0f172a;background:#ffffff;margin:0;padding:32px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width:560px;margin:0 auto;">
      <tr><td>
        <h1 style="margin:0 0 16px;font-size:24px;line-height:30px;font-weight:700;">Almost in.</h1>
        <p style="margin:0 0 16px;font-size:16px;line-height:24px;">
          Click below to confirm your subscription to <strong>${publicationName}</strong>.
          Without this click your address never lands on the list.
        </p>
        <p style="margin:24px 0;">
          <a href="${link}" style="display:inline-block;background:#0f172a;color:#ffffff;padding:12px 20px;border-radius:6px;text-decoration:none;font-weight:600;">
            Confirm subscription
          </a>
        </p>
        <p style="margin:24px 0 0;font-size:14px;line-height:20px;color:#64748b;">
          Or paste this URL into your browser:<br>
          <a href="${link}" style="color:#6366f1;word-break:break-all;">${link}</a>
        </p>
        <p style="margin:32px 0 0;font-size:12px;line-height:16px;color:#94a3b8;">
          Didn't ask for this? Ignore the email. You'll never hear from us again.
        </p>
      </td></tr>
    </table>
  </body></html>`;
  const text = [
    'Almost in.',
    '',
    `Click to confirm your subscription to ${publicationName}:`,
    link,
    '',
    "Didn't ask for this? Ignore the email and you'll never hear from us again.",
  ].join('\n');
  return { subject, html, text };
}
