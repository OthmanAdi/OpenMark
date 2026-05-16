"""
LinkedIn Voyager-API poster — same auth path as raindrop-mission/linkedin_fetch.py.

LinkedIn has no public API for personal-account posting unless you go through
Marketing Developer Platform approval (weeks). This module uses the same
li_at + JSESSIONID cookie path Ahmad's bookmark fetcher already relies on,
calling the internal Voyager endpoint.

Risk: aggressive usage can trigger LinkedIn account flags. Mitigations:
    - Only fire on explicit human approval (Daily Post tab)
    - Cap at 3 posts/day
    - Random 5-25s sleep before each call
    - Genuine User-Agent (current Chrome)

Env vars (in .env):
    LINKEDIN_LI_AT       = <your li_at cookie value>
    LINKEDIN_JSESSIONID  = <your JSESSIONID, e.g. "ajax:8368053795747001162">

If either is missing, the client raises on construction.

Public surface:
    LinkedInVoyagerClient(li_at, jsessionid).post_text(text, anchor_url=None)
    post_to_linkedin(text, anchor_url=None)   # module-level convenience
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


log = logging.getLogger("openmark.publish.linkedin")


VOYAGER_BASE = "https://www.linkedin.com/voyager/api"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)


@dataclass
class PostResult:
    ok: bool
    urn: str | None              # the activity URN (acts as the post id)
    permalink: str | None        # https://www.linkedin.com/feed/update/urn:li:activity:<id>
    error: str | None = None


class LinkedInVoyagerClient:
    """
    Minimal Voyager-API client. Reuses one httpx.Client across calls.

    Endpoint reverse-engineered from the LinkedIn web app. The schema for
    /contentcreation/normShares is the one the browser uses when you click
    "Post" on a status update. It works as long as the li_at cookie is valid.
    """

    def __init__(
        self,
        li_at: str | None = None,
        jsessionid: str | None = None,
        *,
        timeout: float = 30.0,
        max_per_day: int = 3,
    ) -> None:
        self.li_at = li_at or os.environ.get("LINKEDIN_LI_AT", "")
        self.jsessionid = jsessionid or os.environ.get("LINKEDIN_JSESSIONID", "")
        if not self.li_at:
            raise RuntimeError("LINKEDIN_LI_AT cookie missing. Set in .env.")
        if not self.jsessionid:
            raise RuntimeError("LINKEDIN_JSESSIONID cookie missing. Set in .env.")
        self.timeout = timeout
        self.max_per_day = max_per_day
        self._client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                base_url=VOYAGER_BASE,
                timeout=self.timeout,
                cookies={
                    "li_at": self.li_at,
                    "JSESSIONID": self.jsessionid,
                },
                headers={
                    "accept": "application/vnd.linkedin.normalized+json+2.1",
                    "accept-language": "en-US,en;q=0.9",
                    "content-type": "application/json; charset=UTF-8",
                    # csrf-token must match the JSESSIONID value verbatim
                    "csrf-token": self.jsessionid.strip('"'),
                    "user-agent": UA,
                    "x-li-lang": "en_US",
                    "x-restli-protocol-version": "2.0.0",
                    "origin": "https://www.linkedin.com",
                    "referer": "https://www.linkedin.com/feed/",
                },
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    def post_text(
        self,
        text: str,
        *,
        anchor_url: str | None = None,
        sleep_jitter_s: tuple[float, float] = (5.0, 25.0),
    ) -> PostResult:
        """
        Post a plain-text update to your LinkedIn feed.

        anchor_url, if provided, becomes the article-card preview attached to
        the post. LinkedIn auto-unfurls URLs in the text too, so passing
        anchor_url is optional — keeping the URL inline in `text` works.
        """
        if not text or not text.strip():
            return PostResult(ok=False, urn=None, permalink=None, error="empty text")

        # Polite jitter — looks human, reduces flag risk
        if sleep_jitter_s:
            lo, hi = sleep_jitter_s
            sleep_s = random.uniform(lo, hi)
            log.info(f"[linkedin] jitter sleep {sleep_s:.1f}s before post")
            time.sleep(sleep_s)

        payload: dict[str, Any] = {
            "visibleToConnectionsOnly": False,
            "externalAudienceProviders": [],
            "commentaryV2": {
                "text": text,
                "attributes": [],
            },
            "origin": "FEED",
            "allowedCommentersScope": "ALL",
            "postState": "PUBLISHED",
            "media": [],
        }

        # Optional article-card preview
        if anchor_url:
            payload["media"] = [{
                "category": "ARTICLE",
                "mediaUrn": "",
                "originalUrl": anchor_url,
            }]

        try:
            resp = self._http().post("/contentcreation/normShares", json=payload)
            resp.raise_for_status()
            data = resp.json()
            urn = data.get("updateUrn") or data.get("urn") or data.get("activityUrn")
            permalink = _urn_to_permalink(urn) if urn else None
            log.info(f"[linkedin] posted urn={urn} permalink={permalink}")
            return PostResult(ok=True, urn=urn, permalink=permalink)
        except httpx.HTTPStatusError as e:
            err = f"{e.response.status_code}: {e.response.text[:300]}"
            log.warning(f"[linkedin] post failed: {err}")
            return PostResult(ok=False, urn=None, permalink=None, error=err)
        except Exception as e:
            log.warning(f"[linkedin] post exception: {e!r}")
            return PostResult(ok=False, urn=None, permalink=None, error=repr(e))


def _urn_to_permalink(urn: str) -> str | None:
    """urn:li:activity:7123456789  ->  https://www.linkedin.com/feed/update/urn:li:activity:7123456789"""
    if not urn:
        return None
    return f"https://www.linkedin.com/feed/update/{urn}"


def post_to_linkedin(text: str, anchor_url: str | None = None) -> PostResult:
    return LinkedInVoyagerClient().post_text(text, anchor_url=anchor_url)
