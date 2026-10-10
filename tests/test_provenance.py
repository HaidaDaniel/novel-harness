import json

from novel_harness.run_operations import _stages_match
from novel_harness.runners.base import RunnerResult


def test_new_run_and_attempt_freeze_reconstructable_provenance(service, source):
    svc, fake = service
    run_id = svc.create_run(runner="test", chapters=[1])
    manifest = svc.repository.run(run_id)["manifest"]
    stage = manifest["stages"][0]
    provenance = stage["execution_provenance"]

    assert provenance["schema_version"] == 1
    assert provenance["requested"]["model"] == {"id": "fake-model"}
    assert provenance["resolved"] == {"id": "fake-model"}
    assert provenance["completeness"] == "complete"
    assert provenance["material"]["pipeline"] == "direct"
    assert provenance["material"]["prompt_sha256"] == stage["prompt_sha256"]
    assert "api_key" not in json.dumps(provenance)

    svc.execute(run_id)
    attempt = svc.repository.attempts(
        run_id, 1, manifest["chapters"][0]["chunks"][0]["id"], stage["name"]
    )[0]
    assert attempt["execution_provenance"]["resolved"] == {
        "id": "fake-model",
        "source": "runner-reported",
    }
    assert svc.status(run_id)["runners"][0]["execution_provenance"] == provenance
    assert fake.calls


def test_missing_runtime_identity_is_explicit_and_not_comparable(service, source):
    svc, _ = service

    class UnknownRunner:
        def run(self, job):
            return RunnerResult("Русский текст для проверки. " * 8, finish_reason="stop")

    svc.runner_factory = lambda _: UnknownRunner()
    run_id = svc.create_run(runner="test", chapters=[1])
    manifest = svc.repository.run(run_id)["manifest"]
    svc.execute(run_id)
    chunk_id = manifest["chapters"][0]["chunks"][0]["id"]
    attempt = svc.repository.attempts(run_id, 1, chunk_id, manifest["stages"][0]["name"])[0]

    provenance = attempt["execution_provenance"]
    assert provenance["resolved"] == {
        "status": "unknown",
        "reason": "runner did not report a model",
    }
    assert provenance["completeness"] == "unknown"
    stage = manifest["stages"][0]
    unknown_stage = {
        **stage,
        "execution_provenance": {
            **stage["execution_provenance"],
            "resolved": {"status": "unknown"},
            "completeness": "unknown",
        },
    }
    assert not _stages_match([unknown_stage], [unknown_stage])


def test_legacy_stage_without_provenance_remains_compatible(service, source):
    svc, _ = service
    run_id = svc.create_run(runner="test", chapters=[1])
    manifest = svc.repository.run(run_id)["manifest"]
    legacy = {**manifest["stages"][0]}
    legacy.pop("execution_provenance")
    current = {**legacy, "execution_provenance": manifest["stages"][0]["execution_provenance"]}

    assert _stages_match([legacy], [current])
