import pytest

pytestmark = [pytest.mark.integration, pytest.mark.clickhouse]


class TestClickHouseStateStore:
    def test_create_and_get_document_state(self, clickhouse_store, tid):
        clickhouse_store.create_document(tid, "user_1")
        state = clickhouse_store.get_document_state(tid)
        assert state is not None
        assert state["doc_id"] == tid
        assert state["status"] == "uploaded"

    def test_transition_state(self, clickhouse_store, tid):
        clickhouse_store.create_document(tid, "user_1")
        ok = clickhouse_store.transition_state(tid, "uploaded", "ner_done")
        assert ok is True
        state = clickhouse_store.get_document_state(tid)
        assert state["status"] == "ner_done"

    def test_transition_state_wrong_expected_fails(self, clickhouse_store, tid):
        clickhouse_store.create_document(tid, "user_1")
        ok = clickhouse_store.transition_state(tid, "indexed", "re_done")
        assert ok is False

    def test_enqueue_and_resolve_unresolved(self, clickhouse_store, tid):
        clickhouse_store.create_document(tid, "user_1")
        clickhouse_store.enqueue_unresolved(
            resolution_id=f"res_{tid}",
            doc_id=tid,
            entity_row={
                "original_text": "Foo",
                "entity_label": "PER",
                "mention_sentence": "Foo is here.",
                "start": 0,
                "end": 3,
                "confidence": 0.5,
            },
            candidates_json='[]',
        )
        clickhouse_store.resolve_unresolved(f"res_{tid}", f"CanonicalFoo_{tid}", "operator_1")
        # ClickHouse async ALTER; we can't easily read back without flush, so just assert no exception

    def test_log_mention(self, clickhouse_store, tid):
        clickhouse_store.log_mention(
            doc_id=tid, chunk_id="c1", canonical_name=f"E_{tid}",
            original_text="e", entity_label="PER", mention_sentence="s",
            start=0, end=1, confidence=1.0, source="test",
        )

    def test_insert_relations(self, clickhouse_store, tid):
        from shared.data_classes import Relation
        rels = [
            Relation(
                doc_id=tid, subject=f"S_{tid}", subject_label="PER",
                predicate="knows", object=f"O_{tid}", object_label="PER",
                mention_sentence="x", confidence=0.9, source="test",
                relation_id=f"r_{tid}",
            )
        ]
        clickhouse_store.insert_relations(rels)

    def test_get_documents_by_state(self, clickhouse_store, tid):
        clickhouse_store.create_document(tid, "user_1")
        docs = clickhouse_store.get_documents_by_state("uploaded", limit=10)
        assert any(d["doc_id"] == tid for d in docs)
        
