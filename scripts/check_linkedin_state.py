import sys
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")
from openmark.stores import neo4j_store

c = neo4j_store.query("MATCH (b:Bookmark) WHERE b.source='linkedin' RETURN count(b) AS n")
print(f"Total linkedin in DB: {c[0]['n']}")

# Group by activity_urn to see duplicates
dupes = neo4j_store.query("""
    MATCH (b:Bookmark) WHERE b.source='linkedin' AND b.activity_urn IS NOT NULL
    WITH b.activity_urn AS a, count(b) AS n, collect(b.url)[..2] AS sample_urls
    WHERE n > 1
    RETURN a, n, sample_urls
    ORDER BY n DESC LIMIT 5
""")
print(f"Activity URNs with duplicates: {len(dupes)}")
for d in dupes[:5]:
    print(f"  activity={d['a']} count={d['n']}")
    for u in d["sample_urls"]:
        print(f"    {u[:120]}")

# Without dupes
unique = neo4j_store.query("""
    MATCH (b:Bookmark) WHERE b.source='linkedin' AND b.activity_urn IS NOT NULL
    RETURN count(DISTINCT b.activity_urn) AS n
""")
print(f"\nUnique LinkedIn activity_urns: {unique[0]['n']}")

# Sample url shapes
sample = neo4j_store.query("""
    MATCH (b:Bookmark) WHERE b.source='linkedin'
    RETURN b.url AS url LIMIT 5
""")
print("\nSample urls in DB:")
for s in sample:
    print(f"  {s['url'][:140]}")
