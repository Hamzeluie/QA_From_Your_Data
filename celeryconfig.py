beat_schedule = {
    "outbox-every-30s": {
        "task": "ingestion.tasks.poll_outbox_task",
        "schedule": 30.0,
    },
    "merge-every-5m": {
        "task": "ingestion.tasks.merge_entities_task",
        "schedule": 300.0,
    },
}