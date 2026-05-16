"""
Newsletter + website + LinkedIn publish layer.

Single-source-of-truth pipeline: a structured composer output (LinkedInPost,
NewsletterEssay, NewsletterRoundup, NewsletterComparison, NewsletterAnalytical)
flows into THREE renderers:

    publish_issue(composer_output, channels=...)
        ├── web:       site/src/content/newsletters/<slug>.mdx           (Astro)
        ├── email:     Maizzle-rendered HTML  →  Resend send batch
        └── linkedin:  LinkedInPost  →  Voyager API post via li_at cookie

Subscribers live in `data/subscribers.db` (SQLite). Double-opt-in via Resend.

All cheap. Resend = 3000 emails/month free. Cloudflare Pages = free hosting.
LinkedIn = your cookie. Domain = the one outlay (~$10/yr).
"""

from openmark.publish.subscribers import (
    Subscriber,
    add_subscriber,
    confirm_subscriber,
    unsubscribe,
    list_active,
    init_subscribers_db,
)
from openmark.publish.resend_client import (
    ResendClient,
    send_one,
    send_batch,
)
from openmark.publish.linkedin_post import (
    LinkedInVoyagerClient,
    post_to_linkedin,
)

__all__ = [
    "Subscriber",
    "add_subscriber",
    "confirm_subscriber",
    "unsubscribe",
    "list_active",
    "init_subscribers_db",
    "ResendClient",
    "send_one",
    "send_batch",
    "LinkedInVoyagerClient",
    "post_to_linkedin",
]
