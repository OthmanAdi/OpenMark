"""
Python wrapper around the Maizzle v5 build pipeline.

Maizzle is a Node.js tool that compiles Tailwind-classed HTML email templates
into bulletproof inline-styled email HTML. We drive it from Python by:

    1. Writing the per-issue payload to `templates/data/payload.json`
    2. Running `npx maizzle build production` in `templates/`
    3. Reading `templates/build_production/<template>.html` back

Why subprocess and not a Python email-templating library: Maizzle's value is
the inlining + CSS purge + Outlook hacks that the JS ecosystem has battle-
tested for years. Re-implementing them in Python is a yak shave we don't need.

Public:
    render_email(template_name, payload) -> str  # compiled HTML
    render_email_to_file(template_name, payload, out_path) -> Path
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path


log = logging.getLogger("openmark.publish.maizzle")


TEMPLATES_DIR = Path(__file__).parent / "templates"
PAYLOAD_PATH = TEMPLATES_DIR / "data" / "payload.json"
BUILD_DIR = TEMPLATES_DIR / "build_production"


class MaizzleError(RuntimeError):
    """Raised when the Maizzle subprocess fails."""


def _resolve_npx() -> str:
    """Find npx on PATH. Windows ships .cmd; Linux/Mac ship bare 'npx'."""
    for cand in ("npx.cmd", "npx"):
        path = shutil.which(cand)
        if path:
            return path
    raise MaizzleError(
        "npx not found on PATH. Install Node.js (https://nodejs.org) and "
        "ensure 'npx' is reachable. Maizzle is a Node.js-based tool."
    )


def _ensure_node_modules() -> None:
    """Run `npm install` inside templates/ if node_modules is missing."""
    nm = TEMPLATES_DIR / "node_modules" / "@maizzle" / "framework"
    if nm.exists():
        return
    log.info("[maizzle] node_modules missing; running npm install (~30s)")
    proc = subprocess.run(
        [shutil.which("npm.cmd") or shutil.which("npm") or "npm", "install"],
        cwd=str(TEMPLATES_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",         # Maizzle/npm print ANSI escapes; force UTF-8 on Windows
        errors="replace",
        timeout=600,
    )
    if proc.returncode != 0:
        raise MaizzleError(
            f"npm install failed (rc={proc.returncode}):\n"
            f"STDOUT: {proc.stdout[-800:]}\nSTDERR: {proc.stderr[-800:]}"
        )


def _write_payload(payload: dict) -> None:
    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_build(env: str = "production") -> None:
    npx = _resolve_npx()
    proc = subprocess.run(
        [npx, "maizzle", "build", env],
        cwd=str(TEMPLATES_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",         # Maizzle prints ANSI escapes; force UTF-8 on Windows
        errors="replace",
        timeout=120,
    )
    if proc.returncode != 0:
        raise MaizzleError(
            f"maizzle build failed (rc={proc.returncode}):\n"
            f"STDOUT: {proc.stdout[-800:]}\nSTDERR: {proc.stderr[-800:]}"
        )


def render_email(template_name: str, payload: dict) -> str:
    """
    Build the named template and return its compiled HTML.

    `template_name` must match a file under `templates/emails/`, without the
    `.html` extension (e.g. 'newsletter'). The build always uses production
    config so styles are inlined + minified.
    """
    if not template_name or "/" in template_name or "\\" in template_name:
        raise ValueError(f"invalid template_name: {template_name!r}")

    _ensure_node_modules()
    _write_payload(payload)
    _run_build("production")

    compiled = BUILD_DIR / f"{template_name}.html"
    if not compiled.exists():
        raise MaizzleError(
            f"expected compiled template not found at {compiled}. "
            f"build_production/ contents: "
            f"{[p.name for p in BUILD_DIR.iterdir()] if BUILD_DIR.exists() else 'MISSING'}"
        )
    return compiled.read_text(encoding="utf-8")


def render_email_to_file(template_name: str, payload: dict, out_path: str | os.PathLike) -> Path:
    """Convenience: render and persist to a specific path."""
    html = render_email(template_name, payload)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
