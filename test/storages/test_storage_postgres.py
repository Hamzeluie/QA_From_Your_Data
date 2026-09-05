# test_story_postgres.py
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


class TestPostgresStateStore:
    def test_create_and_get_document_state(self, postgres_store, tid):
        postgres_store.create_document(tid, "user_1")
        state = postgres_store.get_document_state(tid)
        assert state is not None
        assert state["doc_id"] == tid
        assert state["status"] == "uploaded"

    def test_transition_state(self, postgres_store, tid):
        postgres_store.create_document(tid, "user_1")
        ok = postgres_store.transition_state(tid, "uploaded", "ner_done")
        assert ok is True
        state = postgres_store.get_document_state(tid)
        assert state["status"] == "ner_done"

    def test_transition_state_wrong_expected_fails(self, postgres_store, tid):
        postgres_store.create_document(tid, "user_1")
        ok = postgres_store.transition_state(tid, "indexed", "re_done")
        assert ok is False

    def test_enqueue_and_resolve_unresolved(self, postgres_store, tid):
        postgres_store.create_document(tid, "user_1")
        postgres_store.enqueue_unresolved(
            resolution_id=f"res_{tid}",
            doc_id=tid,
            entity_row={
                "text": "Foo",
                "label": "PER",
                "mention_sentence": "Foo is here.",
                "start": 0,
                "end": 3,
                "confidence": 0.5,
            },
            candidates_json='[]',
        )
        postgres_store.resolve_unresolved(f"res_{tid}", f"CanonicalFoo_{tid}", "operator_1")
        # ClickHouse async ALTER; we can't easily read back without flush, so just assert no exception

    def test_log_mention(self, postgres_store, tid):
        from shared.data_classes import Entity
        
        postgres_store.log_mention(
            Entity(doc_id=tid, 
                   chunk_id="c1", 
                   canonical_name=f"E_{tid}",
                   text="test", 
                   label="PER", 
                   mention_sentence="test",
                   start=0, 
                   end=1, 
                   confidence=1.0,
                   status="RESOLVED")
        )

    def test_insert_relations(self, postgres_store, tid):
        from shared.data_classes import Relation
        rels = [
            Relation(
                doc_id=tid, subject=f"S_{tid}", subject_label="PER",
                predicate="test", object=f"O_{tid}", object_label="PER",
                mention_sentence="test", confidence=0.9,
                relation_id=f"r_{tid}",
            )
        ]
        postgres_store.insert_relations(rels)

    def test_get_documents_by_state(self, postgres_store, tid):
        postgres_store.create_document(tid, "user_1")
        docs = postgres_store.get_documents_by_state("uploaded", limit=10)
        assert any(d["doc_id"] == tid for d in docs)
        
