import pytest
import time

pytestmark = [pytest.mark.integration, pytest.mark.elasticsearch]


class TestElasticsearchEntitySearch:
    def test_index_and_search_alias(self, es_store, tid):
        es_store.index_entity(
            canonical=f"ES_Ent_{tid}",
            aliases=[f"Alias1_{tid}", f"Alias2_{tid}"],
            summary="A famous scientist",
            context_indicators=["science"],
            related_to=[],
            label="PER",
        )
        time.sleep(1.5)  # ES refresh
        hits = es_store.search_aliases(f"Alias1_{tid}")
        assert any(h["canonical"] == f"ES_Ent_{tid}" for h in hits)

    def test_remove_entity(self, es_store, tid):
        es_store.index_entity(
            canonical=f"Rem_{tid}", aliases=[f"R_{tid}"],
            summary="x", context_indicators=[], related_to=[], label="PER",
        )
        time.sleep(1.5)
        es_store.remove_entity(f"Rem_{tid}")
        time.sleep(1.0)
        hits = es_store.search_aliases(f"R_{tid}")
        assert not any(h["canonical"] == f"Rem_{tid}" for h in hits)

    def test_index_chunk(self, es_store, tid):
        es_store.index_chunk(tid, "c1", "This is a chunk.", "user_1")