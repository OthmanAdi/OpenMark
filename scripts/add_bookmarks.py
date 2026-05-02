"""
OpenMark — add bookmarks from CLI.

Usage:
  python scripts/add_bookmarks.py https://github.com/something
  python scripts/add_bookmarks.py url1.com url2.com url3.com
  python scripts/add_bookmarks.py bookmarks.html
  python scripts/add_bookmarks.py export.json
  python scripts/add_bookmarks.py links.txt
  python scripts/add_bookmarks.py urls.txt --no-fetch   (skip title fetching, faster)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from openmark.pipeline.injector import (
    extract_urls_from_text, urls_to_items,
    parse_html_file, parse_json_file, parse_txt_file,
    run_injection,
)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python scripts/add_bookmarks.py <url|file> [<url|file> ...]")
        print("       python scripts/add_bookmarks.py bookmarks.html")
        print("       python scripts/add_bookmarks.py urls.txt --no-fetch")
        sys.exit(1)

    fetch_titles = "--no-fetch" not in args
    args = [a for a in args if not a.startswith("--")]

    items = []
    for arg in args:
        if os.path.isfile(arg):
            ext = os.path.splitext(arg)[1].lower()
            print(f"Parsing file: {arg}")
            if ext == ".html" or ext == ".htm":
                items.extend(parse_html_file(arg))
            elif ext == ".json":
                items.extend(parse_json_file(arg))
            else:
                items.extend(parse_txt_file(arg, fetch_titles=fetch_titles))
        elif arg.startswith("http://") or arg.startswith("https://"):
            print(f"URL: {arg}")
            items.extend(urls_to_items([arg], fetch_titles=fetch_titles))
        else:
            # Try to extract URLs from the string
            extracted = extract_urls_from_text(arg)
            if extracted:
                items.extend(urls_to_items(extracted, fetch_titles=fetch_titles))
            else:
                print(f"Skipping: not a URL or file — '{arg}'")

    if not items:
        print("No items found.")
        sys.exit(0)

    print(f"\nFound {len(items)} items. Checking for duplicates...")
    stats = run_injection(items)

    print(f"\n{'=' * 40}")
    print(f"DONE")
    print(f"  Total input:    {stats['total']}")
    print(f"  New (added):    {stats['new']}")
    print(f"  Skipped (dupe): {stats['skipped']}")
    if stats.get("error"):
        print(f"  Error: {stats['error']}")
    print("=" * 40)
    if stats["new"] > 0:
        print("Bookmarks immediately searchable in the app.")


if __name__ == "__main__":
    main()
