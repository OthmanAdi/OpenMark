import importlib.util as u
for m in ["httpx", "markdownify", "ddgs", "duckduckgo_search", "selectolax", "readability", "bs4", "lxml"]:
    print(f"{m}: {'OK' if u.find_spec(m) else 'MISSING'}")
