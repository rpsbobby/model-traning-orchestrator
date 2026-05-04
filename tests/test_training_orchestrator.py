from __future__ import annotations

import io

import pytest
import responses

from source.training_orchestrator import (
    TrainingHttpOrchestrator,
    TrainingOrchestratorError,
)


def make_orchestrator(
    raw_log_stream: io.StringIO | None = None,
    event_stream: io.StringIO | None = None,
) -> TrainingHttpOrchestrator:
    return TrainingHttpOrchestrator.from_host_port(
        url="http://trainer",
        port=8000,
        raw_log_stream=raw_log_stream or io.StringIO(),
        event_stream=event_stream or io.StringIO(),
        poll_interval_seconds=0,
        request_timeout_seconds=1,
        echo_logs_to_stdout=False,
        health_wait_timeout_seconds=1,
    )


@responses.activate
def test_start_training_object_detection_posts_expected_endpoint() -> None:
    raw_logs = io.StringIO()
    events = io.StringIO()
    orchestrator = make_orchestrator(raw_logs, events)

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/object_detection",
        json={
            "ok": True,
            "backend": "yolo",
            "accepted": True,
            "job_id": "abc123",
        },
        status=202,
    )

    result = orchestrator.start_training(
        project_type="ObjectDetection",
        config_path=None,
        additional_args_path=None,
    )

    assert result["job_id"] == "abc123"
    assert orchestrator.job_id == "abc123"
    assert orchestrator.backend == "yolo"

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == (
        "http://trainer:8000/training/start/object_detection"
    )


@responses.activate
def test_start_training_sends_query_params() -> None:
    orchestrator = make_orchestrator()

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/object_detection",
        json={
            "ok": True,
            "backend": "yolo",
            "accepted": True,
            "job_id": "abc123",
        },
        status=202,
    )

    with pytest.raises(TrainingOrchestratorError):
        orchestrator.start_training(
            project_type="ObjectDetection",
            config_path="/missing/config.yaml",
            additional_args_path="/missing/args.txt",
        )


@responses.activate
def test_start_training_rejects_non_202() -> None:
    orchestrator = make_orchestrator()

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/classification",
        json={"detail": "bad request"},
        status=400,
    )

    with pytest.raises(TrainingOrchestratorError, match="Training start failed"):
        orchestrator.start_training(project_type="Classification")


def test_start_training_rejects_unknown_project_type() -> None:
    orchestrator = make_orchestrator()

    with pytest.raises(TrainingOrchestratorError, match="Unsupported project type"):
        orchestrator.start_training(project_type="BadProjectType")


@responses.activate
def test_poll_logs_once_updates_offset_and_writes_nothing_directly() -> None:
    orchestrator = make_orchestrator()
    orchestrator.job_id = "abc123"

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "yolo",
            "offset": 0,
            "next_offset": 42,
            "chunk": "epoch 1 loss 0.5\n",
            "done": False,
        },
        status=200,
    )

    payload = orchestrator.poll_logs_once()

    assert payload["chunk"] == "epoch 1 loss 0.5\n"
    assert orchestrator.log_offset == 42

    assert responses.calls[0].request.url == (
        "http://trainer:8000/training/jobs/abc123/logs?offset=0&limit=262144"
    )


@responses.activate
def test_poll_logs_until_done_writes_raw_logs_and_final_status() -> None:
    raw_logs = io.StringIO()
    events = io.StringIO()
    orchestrator = make_orchestrator(raw_logs, events)
    orchestrator.job_id = "abc123"

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "yolo",
            "offset": 0,
            "next_offset": 10,
            "chunk": "epoch 1\n",
            "done": False,
        },
        status=200,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "yolo",
            "offset": 10,
            "next_offset": 20,
            "chunk": "epoch 2\n",
            "done": True,
        },
        status=200,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "yolo",
            "state": "done",
            "pid": 123,
            "exit_code": 0,
            "error": "",
        },
        status=200,
    )

    final_status = orchestrator.poll_logs_until_done()

    assert final_status["state"] == "done"
    assert raw_logs.getvalue() == "epoch 1\nepoch 2\n"
    assert "training_finished" in events.getvalue()


@responses.activate
def test_poll_logs_until_done_cancels_on_traceback() -> None:
    raw_logs = io.StringIO()
    events = io.StringIO()
    orchestrator = make_orchestrator(raw_logs, events)
    orchestrator.job_id = "abc123"

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "yolo",
            "offset": 0,
            "next_offset": 25,
            "chunk": "Traceback (most recent call last):\nboom\n",
            "done": False,
        },
        status=200,
    )

    responses.add(
        responses.POST,
        "http://trainer:8000/training/jobs/abc123/cancel",
        json={
            "ok": True,
            "job_id": "abc123",
            "cancelled": True,
        },
        status=200,
    )

    with pytest.raises(TrainingOrchestratorError, match="Traceback"):
        orchestrator.poll_logs_until_done()

    assert "Traceback" in raw_logs.getvalue()


@responses.activate
def test_run_starts_then_polls_until_done_without_health_check() -> None:
    raw_logs = io.StringIO()
    events = io.StringIO()
    orchestrator = make_orchestrator(raw_logs, events)

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/classification",
        json={
            "ok": True,
            "backend": "classification",
            "accepted": True,
            "job_id": "abc123",
        },
        status=202,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "classification",
            "offset": 0,
            "next_offset": 12,
            "chunk": "training...\n",
            "done": True,
        },
        status=200,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "classification",
            "state": "done",
            "pid": 123,
            "exit_code": 0,
            "error": "",
        },
        status=200,
    )

    final_status = orchestrator.run(
        project_type="Classification",
        wait_for_health=False,
    )

    assert final_status["state"] == "done"
    assert raw_logs.getvalue() == "training...\n"

@responses.activate
def test_start_training_sends_existing_config_query_params(tmp_path) -> None:
    orchestrator = make_orchestrator()

    config = tmp_path / "config.yaml"
    args = tmp_path / "args.txt"
    config.write_text("config", encoding="utf-8")
    args.write_text("args", encoding="utf-8")

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/object_detection",
        json={
            "ok": True,
            "backend": "yolo",
            "accepted": True,
            "job_id": "abc123",
        },
        status=202,
    )

    orchestrator.start_training(
        project_type="ObjectDetection",
        config_path=str(config),
        additional_args_path=str(args),
    )

    request_url = responses.calls[0].request.url

    assert "config_path=" in request_url
    assert "additional_args_path=" in request_url

@responses.activate
def test_run_waits_for_health_before_starting_training() -> None:
    raw_logs = io.StringIO()
    events = io.StringIO()
    _orchestrator = make_orchestrator(raw_logs, events)

    _orchestrator = TrainingHttpOrchestrator.from_host_port(
        url="http://trainer",
        port=8000,
        raw_log_stream=raw_logs,
        event_stream=events,
        poll_interval_seconds=0,
        request_timeout_seconds=1,
        echo_logs_to_stdout=False,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/health",
        json={"ok": True},
        status=200,
    )

    responses.add(
        responses.POST,
        "http://trainer:8000/training/start/classification",
        json={
            "ok": True,
            "backend": "classification",
            "accepted": True,
            "job_id": "abc123",
        },
        status=202,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123/logs",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "classification",
            "offset": 0,
            "next_offset": 12,
            "chunk": "training...\n",
            "done": True,
        },
        status=200,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/training/jobs/abc123",
        json={
            "ok": True,
            "job_id": "abc123",
            "backend": "classification",
            "state": "done",
            "pid": 123,
            "exit_code": 0,
            "error": "",
        },
        status=200,
    )

    final_status = _orchestrator.run(project_type="Classification")

    assert final_status["state"] == "done"

    assert responses.calls[0].request.method == "GET"
    assert responses.calls[0].request.url == "http://trainer:8000/health"

    assert responses.calls[1].request.method == "POST"
    assert responses.calls[1].request.url == (
        "http://trainer:8000/training/start/classification"
    )

    assert "health_check_ok" in events.getvalue()

@responses.activate
def test_wait_for_health_times_out_when_service_never_becomes_healthy() -> None:
    orchestrator = TrainingHttpOrchestrator.from_host_port(
        url="http://trainer",
        port=8000,
        raw_log_stream=io.StringIO(),
        event_stream=io.StringIO(),
        poll_interval_seconds=0,
        request_timeout_seconds=1,
        echo_logs_to_stdout=False,
        health_wait_timeout_seconds=0.01,
    )

    responses.add(
        responses.GET,
        "http://trainer:8000/health",
        json={"ok": False},
        status=503,
    )

    with pytest.raises(TrainingOrchestratorError, match="did not become healthy"):
        orchestrator.wait_for_health()