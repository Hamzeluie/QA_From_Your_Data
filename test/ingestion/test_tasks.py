import pytest
from unittest.mock import patch

from ingestion.tasks import ingest_document_task, poll_outbox_task, backfill_document_task

pytestmark = pytest.mark.integration


class TestCeleryTasks:
    def test_ingest_document_task_eager(self, tid, factory):
        # Celery eager mode executes task inline
        with patch('ingestion.tasks._get_relation_extractor') as mock_get:
            mock_get.return_value = lambda text, entities: []
            with patch('ingestion.pipeline.IngestionPipeline.run') as mock_run:
                mock_run.return_value = {"status": "success", "doc_id": tid}
                result = ingest_document_task.apply(args=[tid, "raw text", "user_1"])
                assert result.successful()
                assert result.result["status"] == "success"

    def test_poll_outbox_task_runs(self, tid, factory):
        # Seed an outbox event
        factory.neo4j.upsert_entity(
            canonical=f"TaskEnt_{tid}", label="PER",
            aliases=[f"TE_{tid}"], summary="Task summary", source="test",
        )
        result = poll_outbox_task.apply(args=[10])
        assert result.successful()
        assert isinstance(result.result, dict)
        assert "processed" in result.result

    def test_backfill_document_task(self, tid):
        with patch('ingestion.pipeline.IngestionPipeline.run') as mock_run:
            mock_run.return_value = {"status": "success", "doc_id": tid}
            result = backfill_document_task.apply(args=[tid, "raw text"])
            assert result.successful()