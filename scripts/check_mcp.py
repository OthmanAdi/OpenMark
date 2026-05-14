import sys, os, asyncio
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

from openmark.mcp import server as srv

async def main():
    tools = await srv.mcp.list_tools()
    print(f"MCP module imported OK")
    print(f"Tools registered: {len(tools)}")
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "key", "?")
        desc = (getattr(t, "description", "") or "").split("\n")[0][:90]
        print(f"  - {name}: {desc}")

asyncio.run(main())
