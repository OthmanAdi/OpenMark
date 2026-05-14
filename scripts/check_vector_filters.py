"""Sanity-test vector_search with category + source filters after the
pre-filter fix in neo4j_store.vector_search."""
import sys
sys.path.insert(0, r"C:\Users\oasrvadmin\Documents\OpenMark")
sys.stdout.reconfigure(encoding="utf-8")

from openmark.embeddings.factory import get_embedder
from openmark.stores import neo4j_store

embedder = get_embedder()
q = embedder.embed_query("AI agents and LLM tooling")

print("\n=== unfiltered ===")
hits = neo4j_store.vector_search(q, n=3)
print(f"{len(hits)} hits")
for h in hits:
    print(f"  - {h.get('source','?'):15} | {h.get('category','?'):24} | {h['title'][:70]}")

print("\n=== source=linkedin ===")
hits = neo4j_store.vector_search(q, n=3, source="linkedin")
print(f"{len(hits)} hits")
for h in hits:
    print(f"  - {h.get('source','?'):15} | {h.get('category','?'):24} | {h['title'][:70]}")

print("\n=== category='Agent Development' ===")
hits = neo4j_store.vector_search(q, n=3, category="Agent Development")
print(f"{len(hits)} hits")
for h in hits:
    print(f"  - {h.get('source','?'):15} | {h.get('category','?'):24} | {h['title'][:70]}")
