"""
Neo4j store — knowledge graph.

Nodes:   Bookmark, Tag, Category, Source, Domain
Edges:   TAGGED, IN_CATEGORY, FROM_SOURCE, FROM_DOMAIN, SIMILAR_TO, CO_OCCURS_WITH
"""

import re
from urllib.parse import urlparse
from neo4j import GraphDatabase
from openmark import config


def get_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


def setup_constraints(driver):
    """Create uniqueness constraints once."""
    constraints = [
        "CREATE CONSTRAINT bookmark_url IF NOT EXISTS FOR (b:Bookmark) REQUIRE b.url IS UNIQUE",
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT source_name IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    ]
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for cypher in constraints:
            try:
                session.run(cypher)
            except Exception as e:
                print(f"  Constraint (already exists or error): {e}")
    print("Constraints ready.")


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def ingest(items: list[dict], driver=None):
    """Write all nodes and relationships to Neo4j."""
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    setup_constraints(driver)

    total = len(items)
    batch_size = 200

    print(f"Neo4j ingesting {total} items...")

    for start in range(0, total, batch_size):
        batch = items[start:start + batch_size]

        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.execute_write(_write_batch, batch)

        print(f"  Neo4j wrote {min(start + batch_size, total)}/{total}")

    print("Building tag co-occurrence edges...")
    _build_tag_cooccurrence(driver)

    print("Neo4j ingestion complete.")

    if own_driver:
        driver.close()


def _write_batch(tx, batch: list[dict]):
    for item in batch:
        url      = item["url"]
        title    = item["title"][:500]
        category = item["category"]
        tags     = item["tags"]
        score    = float(item["score"])
        source   = item["source"]
        domain   = extract_domain(url)

        # Bookmark node
        tx.run("""
            MERGE (b:Bookmark {url: $url})
            SET b.title = $title, b.score = $score
        """, url=url, title=title, score=score)

        # Category node + relationship
        tx.run("""
            MERGE (c:Category {name: $cat})
            WITH c
            MATCH (b:Bookmark {url: $url})
            MERGE (b)-[:IN_CATEGORY]->(c)
        """, cat=category, url=url)

        # Source node + relationship
        tx.run("""
            MERGE (s:Source {name: $src})
            WITH s
            MATCH (b:Bookmark {url: $url})
            MERGE (b)-[:FROM_SOURCE]->(s)
        """, src=source, url=url)

        # Domain node + relationship
        if domain and domain != "unknown":
            tx.run("""
                MERGE (d:Domain {name: $domain})
                WITH d
                MATCH (b:Bookmark {url: $url})
                MERGE (b)-[:FROM_DOMAIN]->(d)
            """, domain=domain, url=url)

        # Tag nodes + relationships
        for tag in tags:
            if not tag:
                continue
            tx.run("""
                MERGE (t:Tag {name: $tag})
                WITH t
                MATCH (b:Bookmark {url: $url})
                MERGE (b)-[:TAGGED]->(t)
            """, tag=tag, url=url)


def _build_tag_cooccurrence(driver):
    """
    For each bookmark with multiple tags, create CO_OCCURS_WITH edges between tags.
    Weight = number of bookmarks where both tags appear together.
    """
    with driver.session(database=config.NEO4J_DATABASE) as session:
        session.run("""
            MATCH (b:Bookmark)-[:TAGGED]->(t1:Tag)
            MATCH (b)-[:TAGGED]->(t2:Tag)
            WHERE t1.name < t2.name
            MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
        """)
    print("  Tag co-occurrence edges built.")


def add_similar_to_edges(similar_pairs: list[tuple[str, str, float]], driver=None):
    """
    Write SIMILAR_TO edges derived from ChromaDB nearest-neighbor search.
    similar_pairs = [(url_a, url_b, similarity_score), ...]
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    with driver.session(database=config.NEO4J_DATABASE) as session:
        for url_a, url_b, score in similar_pairs:
            session.run("""
                MATCH (a:Bookmark {url: $url_a})
                MATCH (b:Bookmark {url: $url_b})
                MERGE (a)-[r:SIMILAR_TO]-(b)
                SET r.score = $score
            """, url_a=url_a, url_b=url_b, score=score)

    print(f"  SIMILAR_TO: {len(similar_pairs)} edges written.")

    if own_driver:
        driver.close()


def query(cypher: str, params: dict | None = None) -> list[dict]:
    """Run arbitrary Cypher and return results as list of dicts."""
    driver = get_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run(cypher, params or {})
        rows = [dict(r) for r in result]
    driver.close()
    return rows


def get_stats() -> dict:
    rows = query("""
        MATCH (b:Bookmark) WITH count(b) AS bookmarks
        MATCH (t:Tag)      WITH bookmarks, count(t) AS tags
        MATCH (c:Category) WITH bookmarks, tags, count(c) AS categories
        RETURN bookmarks, tags, categories
    """)
    return rows[0] if rows else {}


def find_similar(url: str, limit: int = 10) -> list[dict]:
    return query("""
        MATCH (b:Bookmark {url: $url})-[r:SIMILAR_TO]-(other:Bookmark)
        RETURN other.url AS url, other.title AS title, r.score AS similarity
        ORDER BY r.score DESC LIMIT $limit
    """, {"url": url, "limit": limit})


def find_by_tag(tag: str, limit: int = 20) -> list[dict]:
    return query("""
        MATCH (b:Bookmark)-[:TAGGED]->(t:Tag {name: $tag})
        RETURN b.url AS url, b.title AS title, b.score AS score
        ORDER BY b.score DESC LIMIT $limit
    """, {"tag": tag.lower(), "limit": limit})


def find_tag_cluster(tag: str, hops: int = 2, limit: int = 30) -> list[dict]:
    """Follow CO_OCCURS_WITH edges to find related tags and their bookmarks."""
    return query(f"""
        MATCH (t:Tag {{name: $tag}})-[:CO_OCCURS_WITH*1..{hops}]-(related:Tag)
        MATCH (b:Bookmark)-[:TAGGED]->(related)
        RETURN DISTINCT b.url AS url, b.title AS title, b.score AS score, related.name AS via_tag
        ORDER BY b.score DESC LIMIT $limit
    """, {"tag": tag.lower(), "limit": limit})
