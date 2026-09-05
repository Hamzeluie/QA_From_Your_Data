import pytest

pytestmark = pytest.mark.integration


class TestStorageFactory:
    def test_init_all_runs_without_error(self, factory):
        assert factory.neo4j is not None
        assert factory.postgres is not None
        assert factory.qdrant is not None
        assert factory.es is not None
        assert factory.redis is not None

    def test_end_to_end_write_and_read(self, factory, tid, mock_embedder):
        from shared.data_classes import Entity

        factory.neo4j.upsert_entity(
            canonical=f"FactoryEnt_{tid}",
            label="PER",
            aliases=[f"FE_{tid}"],
            summary="Factory test entity",
            source="test",
        )
        factory.es.index_entity(
            canonical=f"FactoryEnt_{tid}",
            aliases=[f"FE_{tid}"],
            summary="Factory test entity",
            context_indicators=[],
            related_to=[],
            label="PER",
        )
        vec = mock_embedder.encode(["Factory test entity"])[0].tolist()
        factory.qdrant.upsert_entity_summary(
            f"FactoryEnt_{tid}", vec, {"label": "PER"}
        )
        factory.redis.set_alias(f"fe_{tid}".lower(), f"FactoryEnt_{tid}")

        factory.postgres.create_document(tid, "user_1")
        factory.postgres.log_mention(
            Entity(doc_id=tid, 
                               chunk_id="c1", 
                               canonical_name=f"E_{tid}",
                               text="test", 
                               label="PER", 
                               mention_sentence="test",
                               start=0, 
                               end=1, 
                               confidence=1.0,
                               status="RESOLVED"))

        assert factory.neo4j.find_by_canonical(f"FactoryEnt_{tid}") is not None
        assert factory.redis.get_alias(f"fe_{tid}".lower()) == f"FactoryEnt_{tid}"