import sys
sys.path.insert(0, r'C:\Users\oasrvadmin\Documents\OpenMark')
sys.stdout.reconfigure(encoding='utf-8')
from openmark.stores import neo4j_store

rows = neo4j_store.query(
    "MATCH (b:Bookmark) WHERE b.source = $source RETURN b.url AS url LIMIT 8",
    {"source": "linkedin"},
)
print("Sample existing LinkedIn URLs in Neo4j:")
for r in rows:
    print("  ", r["url"][:140])

c = neo4j_store.query(
    "MATCH (b:Bookmark) WHERE b.source = $source RETURN count(b) AS n",
    {"source": "linkedin"},
)
print(f"\nTotal linkedin in DB: {c[0]['n']}")
