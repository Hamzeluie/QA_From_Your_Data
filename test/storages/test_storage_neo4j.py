import pytest

pytestmark = [pytest.mark.integration, pytest.mark.neo4j]


class TestNeo4jEntityStore:
    def test_upsert_and_find_by_canonical(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"Einstein_{tid}",
            label="PER",
            aliases=[f"Albert Einstein_{tid}", f"Einstein_{tid}"],
            summary="Physicist",
            source="test",
            related_to=[],
            context_indicators=["physics"],
        )
        node = neo4j_store.find_by_canonical(f"Einstein_{tid}")
        assert node is not None
        assert node["canonical"] == f"Einstein_{tid}"
        assert node["label"] == "PER"

    def test_find_by_alias(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"Curie_{tid}",
            label="PER",
            aliases=[f"Marie Curie_{tid}"],
            summary="Chemist",
            source="test",
        )
        hit = neo4j_store.find_by_alias(f"marie curie_{tid}".lower())
        assert hit is not None
        assert hit["canonical"] == f"Curie_{tid}"

    def test_find_candidates_exact(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"Newton_{tid}",
            label="PER",
            aliases=[f"Isaac Newton_{tid}"],
            summary="Mathematician",
            source="test",
        )
        cands = neo4j_store.find_candidates_cypher(f"Isaac Newton_{tid}")
        assert len(cands) >= 1
        assert any(c.canonical == f"Newton_{tid}" and c.match_method == "exact_match" for c in cands)

    def test_find_candidates_partial(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"GalileoGalilei_{tid}",
            label="PER",
            aliases=[f"Galileo Galilei_{tid}"],
            summary="Astronomer",
            source="test",
        )
        cands = neo4j_store.find_candidates_cypher(f"Galilei_{tid}")
        assert any(c.match_method == "partial_match" for c in cands)

    def test_merge_entities(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"Keep_{tid}", label="PER", aliases=[f"KeepAlias_{tid}"],
            summary="Keep summary", source="test",
        )
        neo4j_store.upsert_entity(
            canonical=f"Absorb_{tid}", label="PER", aliases=[f"AbsorbAlias_{tid}"],
            summary="Absorb summary", source="test",
        )
        neo4j_store.merge_entities(f"Absorb_{tid}", f"Keep_{tid}")
        assert neo4j_store.find_by_canonical(f"Absorb_{tid}") is None
        keep = neo4j_store.find_by_canonical(f"Keep_{tid}")
        assert f"absorbalias_{tid}" in [a.lower() for a in keep["aliases"]]

    def test_outbox_created_on_upsert(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"OutboxTest_{tid}", label="PER", aliases=[f"OB_{tid}"],
            summary="x", source="test",
        )
        pending = neo4j_store.get_pending_outbox(limit=10)
        assert any(f"OutboxTest_{tid}" == p.canonical for p in pending)

    def test_create_relation(self, neo4j_store, tid):
        neo4j_store.upsert_entity(
            canonical=f"Subj_{tid}", label="PER", aliases=[f"S_{tid}"],
            summary="s", source="test",
        )
        neo4j_store.upsert_entity(
            canonical=f"Obj_{tid}", label="LOC", aliases=[f"O_{tid}"],
            summary="o", source="test",
        )
        neo4j_store.create_relation(
            subject=f"Subj_{tid}",
            predicate="born_in",
            obj=f"Obj_{tid}",
            doc_id=f"doc_{tid}",
            confidence=0.95,
            provisional=False,
            relation_id=f"rel_{tid}",
            evidence=["He was born there."],
        )