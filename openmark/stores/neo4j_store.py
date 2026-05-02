"""
Neo4j store — Graph RAG knowledge graph.

Nodes:   Bookmark, Tag, Category, Domain, Community
Edges:   TAGGED, IN_CATEGORY, FROM_DOMAIN, SIMILAR_TO, CO_OCCURS_WITH, IN_COMMUNITY

Vector index on Bookmark.embedding (1024-dim cosine) enables combined
semantic + graph-structure search in a single Cypher query.
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


def setup_schema(driver):
    """Constraints + vector index. Safe to re-run (all IF NOT EXISTS)."""
    constraints = [
        "CREATE CONSTRAINT bookmark_url IF NOT EXISTS FOR (b:Bookmark) REQUIRE b.url IS UNIQUE",
        "CREATE CONSTRAINT tag_name     IF NOT EXISTS FOR (t:Tag)      REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT cat_name     IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT domain_name  IF NOT EXISTS FOR (d:Domain)   REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT comm_id      IF NOT EXISTS FOR (c:Community) REQUIRE c.louvain_id IS UNIQUE",
    ]
    vector_index = f"""
        CREATE VECTOR INDEX bookmark_embedding IF NOT EXISTS
        FOR (b:Bookmark) ON (b.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {config.pplx_dimension()},
            `vector.similarity_function`: 'cosine'
        }}}}
    """
    with driver.session(database=config.NEO4J_DATABASE) as session:
        for cypher in constraints:
            try:
                session.run(cypher)
            except Exception as e:
                print(f"  Constraint skip: {e}")
        try:
            session.run(vector_index)
            print("  Vector index ready (bookmark_embedding, 1024-dim cosine)")
        except Exception as e:
            print(f"  Vector index skip: {e}")
    print("Schema ready.")


def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def ingest(items: list[dict], embeddings: list[list[float]] | None = None, driver=None):
    """
    Write all nodes and relationships to Neo4j.
    If embeddings provided (must match items length), stores them on Bookmark nodes.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    setup_schema(driver)

    total = len(items)
    batch_size = 200

    print(f"Neo4j ingesting {total} items...")

    for start in range(0, total, batch_size):
        batch = items[start:start + batch_size]
        batch_embeds = None
        if embeddings is not None:
            batch_embeds = embeddings[start:start + batch_size]

        with driver.session(database=config.NEO4J_DATABASE) as session:
            session.execute_write(_write_batch, batch, batch_embeds)

        print(f"  Neo4j wrote {min(start + batch_size, total)}/{total}")

    print("Building tag co-occurrence edges...")
    _build_tag_cooccurrence(driver)

    print("Neo4j ingestion complete.")

    if own_driver:
        driver.close()


def _write_batch(tx, batch: list[dict], embeddings: list | None):
    for i, item in enumerate(batch):
        url      = item["url"]
        title    = item["title"][:500]
        category = item["category"] or "Entertainment & Other"
        tags     = item["tags"]
        score    = float(item["score"])
        source   = item["source"]
        domain   = extract_domain(url)
        embed    = embeddings[i] if embeddings else None

        # Bookmark node — denormalize source+category for fast WHERE filtering
        if embed is not None:
            tx.run("""
                MERGE (b:Bookmark {url: $url})
                SET b.title = $title, b.score = $score,
                    b.source = $source, b.category = $category,
                    b.embedding = $embedding
            """, url=url, title=title, score=score,
                 source=source, category=category, embedding=embed)
        else:
            tx.run("""
                MERGE (b:Bookmark {url: $url})
                SET b.title = $title, b.score = $score,
                    b.source = $source, b.category = $category
            """, url=url, title=title, score=score,
                 source=source, category=category)

        # Category node + edge
        tx.run("""
            MERGE (c:Category {name: $cat})
            WITH c MATCH (b:Bookmark {url: $url})
            MERGE (b)-[:IN_CATEGORY]->(c)
        """, cat=category, url=url)

        # Domain node + edge
        if domain and domain != "unknown":
            tx.run("""
                MERGE (d:Domain {name: $domain})
                WITH d MATCH (b:Bookmark {url: $url})
                MERGE (b)-[:FROM_DOMAIN]->(d)
            """, domain=domain, url=url)

        # Tag nodes + edges
        for tag in tags:
            if not tag:
                continue
            tx.run("""
                MERGE (t:Tag {name: $tag})
                WITH t MATCH (b:Bookmark {url: $url})
                MERGE (b)-[:TAGGED]->(t)
            """, tag=tag, url=url)


def _build_tag_cooccurrence(driver):
    with driver.session(database=config.NEO4J_DATABASE) as session:
        session.run("""
            MATCH (b:Bookmark)-[:TAGGED]->(t1:Tag)
            MATCH (b)-[:TAGGED]->(t2:Tag)
            WHERE t1.name < t2.name
            MERGE (t1)-[r:CO_OCCURS_WITH]-(t2)
            ON CREATE SET r.count = 1
            ON MATCH  SET r.count = r.count + 1
        """)
    print("  Tag co-occurrence edges built.")


def build_similar_to_edges(driver=None):
    """
    Build SIMILAR_TO edges using the Neo4j vector index.
    For each bookmark, finds its 6 nearest embedding neighbors.
    Pure Cypher — no Python loop needed.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    print("Building SIMILAR_TO edges via vector index (this takes a few minutes)...")
    with driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run("""
            MATCH (b:Bookmark)
            WHERE b.embedding IS NOT NULL
            MATCH (similar)
              SEARCH similar IN (
                VECTOR INDEX bookmark_embedding
                FOR b.embedding
                LIMIT 6
              ) SCORE AS score
            WHERE similar.url <> b.url AND score > 0.75
            MERGE (b)-[r:SIMILAR_TO]-(similar)
            SET r.score = score
            RETURN count(r) AS edges
        """)
        n = result.single()["edges"]
        print(f"  SIMILAR_TO: {n} edges built.")

    if own_driver:
        driver.close()


def setup_louvain(driver=None):
    """
    Run GDS Louvain community detection on SIMILAR_TO graph.
    Requires Neo4j GDS plugin installed.
    Creates Community nodes and IN_COMMUNITY edges.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            # Clean up old projection if exists
            try:
                session.run("CALL gds.graph.drop('bookmark-graph', false)")
            except Exception:
                pass

            # Project Bookmark nodes + SIMILAR_TO edges (undirected, weighted)
            session.run("""
                CALL gds.graph.project(
                    'bookmark-graph',
                    'Bookmark',
                    {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}}
                )
            """)

            # Run Louvain, write community ID back to Bookmark nodes
            result = session.run("""
                CALL gds.louvain.write('bookmark-graph', {
                    writeProperty: 'community',
                    relationshipWeightProperty: 'score'
                })
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """)
            row = result.single()
            print(f"  Louvain: {row['communityCount']} communities (modularity={row['modularity']:.3f})")

            # Drop projection (frees RAM)
            session.run("CALL gds.graph.drop('bookmark-graph', false)")

            # Create Community nodes + IN_COMMUNITY edges
            session.run("""
                MATCH (b:Bookmark) WHERE b.community IS NOT NULL
                MERGE (c:Community {louvain_id: b.community})
                MERGE (b)-[:IN_COMMUNITY]->(c)
            """)
            result = session.run("MATCH (c:Community) RETURN count(c) AS n")
            print(f"  Community nodes created: {result.single()['n']}")

    except Exception as e:
        print(f"  Louvain failed (GDS not installed?): {e}")

    if own_driver:
        driver.close()


# ── Search functions ───────────────────────────────────────────────────────────

def vector_search(
    query_embedding: list[float],
    n: int = 10,
    category: str | None = None,
    source: str | None = None,
) -> list[dict]:
    """
    Combined vector + graph search in a single Cypher query.
    Returns bookmarks with tags and similar URLs attached.
    """
    filters = []
    params: dict = {"embedding": query_embedding, "n": n}
    if category:
        filters.append("b.category = $category")
        params["category"] = category
    if source:
        filters.append("b.source = $source")
        params["source"] = source

    search_where = f"WHERE {' AND '.join(filters)}" if filters else ""

    # Cypher 25 SEARCH syntax (Neo4j 2026+) — pre-filters inside ANN for efficiency
    cypher = f"""
        MATCH (b)
          SEARCH b IN (
            VECTOR INDEX bookmark_embedding
            FOR $embedding
            {search_where}
            LIMIT $n
          ) SCORE AS score
        OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
        OPTIONAL MATCH (b)-[:SIMILAR_TO]->(s:Bookmark)
        OPTIONAL MATCH (b)-[:IN_COMMUNITY]->(comm:Community)
        WITH b, score,
             collect(DISTINCT t.name)[..10] AS tags,
             collect(DISTINCT s.url)[..3]   AS similar_urls,
             comm.louvain_id                AS community_id
        RETURN b.url      AS url,
               b.title    AS title,
               b.score    AS bm_score,
               b.source   AS source,
               b.category AS category,
               score       AS similarity,
               tags, similar_urls, community_id
        ORDER BY similarity DESC
    """
    return query(cypher, params)


def graph_expand(url: str) -> str:
    """
    Expand a bookmark node: its tags, co-occurring tags, similar bookmarks, community.
    Returns a formatted string ready for the agent.
    """
    results = query("""
        MATCH (b:Bookmark {url: $url})
        OPTIONAL MATCH (b)-[:TAGGED]->(t:Tag)
        OPTIONAL MATCH (b)-[:SIMILAR_TO]->(s:Bookmark)
        OPTIONAL MATCH (b)-[:IN_COMMUNITY]->(comm:Community)
        OPTIONAL MATCH (comm)<-[:IN_COMMUNITY]-(peer:Bookmark)
        WHERE peer.url <> $url
        WITH b,
             collect(DISTINCT t.name)    AS tags,
             collect(DISTINCT s.title)[..5]   AS similar_titles,
             collect(DISTINCT s.url)[..5]     AS similar_urls,
             comm.louvain_id                  AS community_id,
             collect(DISTINCT peer.title)[..5] AS community_peers
        RETURN b.title AS title, b.category AS category, tags,
               similar_titles, similar_urls,
               community_id, community_peers
    """, {"url": url})

    if not results:
        return f"Bookmark not found in graph: {url}"

    r = results[0]
    lines = [
        f"Bookmark: {r['title']}",
        f"Category: {r['category']}",
        f"Tags: {', '.join(r['tags']) or 'none'}",
        f"Similar bookmarks:",
    ]
    for title, sim_url in zip(r["similar_titles"], r["similar_urls"]):
        lines.append(f"  - {title}  {sim_url}")
    if r["community_id"] is not None:
        lines.append(f"Community #{r['community_id']} members (sample):")
        for peer in r["community_peers"]:
            lines.append(f"  - {peer}")
    return "\n".join(lines)


def search_by_community(query_embedding: list[float], n: int = 15) -> list[dict]:
    """
    Find the top matching community by vector search, then return all bookmarks in it.
    """
    # Step 1: find the community of the closest bookmark
    seed_results = vector_search(query_embedding, n=1)
    if not seed_results or seed_results[0].get("community_id") is None:
        return []

    community_id = seed_results[0]["community_id"]

    return query("""
        MATCH (c:Community {louvain_id: $cid})<-[:IN_COMMUNITY]-(b:Bookmark)
        RETURN b.url AS url, b.title AS title, b.category AS category,
               b.source AS source, b.score AS bm_score
        ORDER BY b.score DESC
        LIMIT $n
    """, {"cid": community_id, "n": n})


# ── Utility ────────────────────────────────────────────────────────────────────

def query(cypher: str, params: dict | None = None) -> list[dict]:
    driver = get_driver()
    with driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run(cypher, params or {})
        rows = [dict(r) for r in result]
    driver.close()
    return rows


def get_stats() -> dict:
    rows = query("""
        MATCH (b:Bookmark) WITH count(b) AS bookmarks
        OPTIONAL MATCH (t:Tag)       WITH bookmarks, count(t) AS tags
        OPTIONAL MATCH (c:Category)  WITH bookmarks, tags, count(c) AS categories
        OPTIONAL MATCH (comm:Community) WITH bookmarks, tags, categories, count(comm) AS communities
        RETURN bookmarks, tags, categories, communities
    """)
    return rows[0] if rows else {}


# Legacy helpers (kept for agent tool fallback)

def add_similar_to_edges(similar_pairs: list[tuple[str, str, float]], driver=None):
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
    return query(f"""
        MATCH (t:Tag {{name: $tag}})-[:CO_OCCURS_WITH*1..{hops}]-(related:Tag)
        MATCH (b:Bookmark)-[:TAGGED]->(related)
        RETURN DISTINCT b.url AS url, b.title AS title,
               b.score AS score, related.name AS via_tag
        ORDER BY b.score DESC LIMIT $limit
    """, {"tag": tag.lower(), "limit": limit})
