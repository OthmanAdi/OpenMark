import sys
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")
from openmark.stores import neo4j_store

print("--- bookmark count ---")
r = neo4j_store.query("MATCH (b:Bookmark) RETURN count(b) AS n")
print("bookmarks:", r[0]["n"] if r else "?")

print("\n--- indexes ---")
r = neo4j_store.query("SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties")
for row in r:
    print(" ", row)

print("\n--- node labels ---")
r = neo4j_store.query("CALL db.labels() YIELD label RETURN label")
for row in r: print(" ", row["label"])

print("\n--- current database ---")
r = neo4j_store.query("CALL db.info() YIELD name, creationDate")
print(r)
