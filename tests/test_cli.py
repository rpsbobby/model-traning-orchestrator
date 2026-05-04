from __future__ import annotations

import json
from argparse import Namespace

import pytest

from source import cli


def make_args(tmp_path, **overrides) -> Namespace:
    values = {
        "url": "http://trainer",
        "port": 8000,
        "project_type": "Classification",
        "data_dir": str(tmp_path),
        "config_path": None,
        "additional_args_path": None,
        "poll_interval": 0.0,
        "timeout": 1.0,
        "log_limit": 256 * 1024,
        "max_poll_errors": 5,
        "no_stdout": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parse_args_minimum_required(tmp_path) -> None:
    args = cli.parse_args(
        [
            "--project-type",
            "Classification",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert args.url == "http://localhost"
    assert args.port == 8000
    assert args.project_type == "Classification"
    assert args.data_dir == str(tmp_path)
    assert args.config_path is None
    assert args.additional_args_path is None


def test_parse_args_object_detection_with_paths(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    args_path = tmp_path / "args.txt"

    args = cli.parse_args(
        [
            "--url",
            "http://localhost",
            "--port",
            "8000",
            "--project-type",
            "ObjectDetection",
            "--data-dir",
            str(tmp_path),
            "--config-path",
            str(config_path),
            "--additional-args-path",
            str(args_path),
            "--poll-interval",
            "0.5",
            "--timeout",
            "2",
            "--log-limit",
            "1024",
            "--max-poll-errors",
            "3",
            "--no-stdout",
        ]
    )

    assert args.url == "http://localhost"
    assert args.port == 8000
    assert args.project_type == "ObjectDetection"
    assert args.data_dir == str(tmp_path)
    assert args.config_path == str(config_path)
    assert args.additional_args_path == str(args_path)
    assert args.poll_interval == 0.5
    assert args.timeout == 2
    assert args.log_limit == 1024
    assert args.max_poll_errors == 3
    assert args.no_stdout is True


def test_parse_args_rejects_unknown_project_type(tmp_path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--project-type",
                "BadProject",
                "--data-dir",
                str(tmp_path),
            ]
        )


def test_run_from_args_success_creates_output_files(tmp_path, monkeypatch, capsys) -> None:
    class FakeOrchestrator:
        def __init__(self, *args, **kwargs) -> None:
            self.raw_log_stream = kwargs["raw_log_stream"]
            self.event_stream = kwargs["event_stream"]

        @classmethod
        def from_host_port(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def run(self, project_type, config_path=None, additional_args_path=None):
            self.raw_log_stream.write("epoch 1\n")
            self.event_stream.write('{"event": "training_started"}\n')
            return {
                "ok": True,
                "job_id": "abc123",
                "backend": "classification",
                "state": "done",
                "pid": 123,
                "exit_code": 0,
                "error": "",
            }

    monkeypatch.setattr(cli, "TrainingHttpOrchestrator", FakeOrchestrator)

    args = make_args(tmp_path)
    exit_code = cli.run_from_args(args)

    assert exit_code == 0

    assert (tmp_path / "training.log").read_text(encoding="utf-8") == "epoch 1\n"
    assert (tmp_path / "training_events.jsonl").read_text(encoding="utf-8") == (
        '{"event": "training_started"}\n'
    )

    final_status = json.loads(
        (tmp_path / "training_final_status.json").read_text(encoding="utf-8")
    )

    assert final_status["state"] == "done"
    assert final_status["job_id"] == "abc123"

    captured = capsys.readouterr()
    assert "Training finished" in captured.out
    assert "training.log" in captured.out
    assert captured.err == ""


def test_run_from_args_non_done_state_returns_1(tmp_path, monkeypatch, capsys) -> None:
    class FakeOrchestrator:
        @classmethod
        def from_host_port(cls, *args, **kwargs):
            return cls()

        def run(self, project_type, config_path=None, additional_args_path=None):
            return {
                "ok": True,
                "job_id": "abc123",
                "backend": "classification",
                "state": "failed",
                "pid": 123,
                "exit_code": 1,
                "error": "boom",
            }

    monkeypatch.setattr(cli, "TrainingHttpOrchestrator", FakeOrchestrator)

    args = make_args(tmp_path)
    exit_code = cli.run_from_args(args)

    assert exit_code == 1

    final_status = json.loads(
        (tmp_path / "training_final_status.json").read_text(encoding="utf-8")
    )

    assert final_status["state"] == "failed"

    captured = capsys.readouterr()
    assert "Training ended with non-success state: failed" in captured.err


def test_run_from_args_orchestrator_error_writes_error_file(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FakeOrchestrator:
        @classmethod
        def from_host_port(cls, *args, **kwargs):
            return cls()

        def run(self, project_type, config_path=None, additional_args_path=None):
            raise cli.TrainingOrchestratorError("server exploded")

    monkeypatch.setattr(cli, "TrainingHttpOrchestrator", FakeOrchestrator)

    args = make_args(tmp_path)
    exit_code = cli.run_from_args(args)

    assert exit_code == 1

    error_file = tmp_path / "training_orchestrator_error.txt"
    assert error_file.read_text(encoding="utf-8") == "server exploded\n"

    captured = capsys.readouterr()
    assert "Training orchestration failed: server exploded" in captured.err
    assert "Error saved to:" in captured.err


def test_run_from_args_unexpected_error_writes_error_file(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FakeOrchestrator:
        @classmethod
        def from_host_port(cls, *args, **kwargs):
            return cls()

        def run(self, project_type, config_path=None, additional_args_path=None):
            raise ValueError("bad value")

    monkeypatch.setattr(cli, "TrainingHttpOrchestrator", FakeOrchestrator)

    args = make_args(tmp_path)
    exit_code = cli.run_from_args(args)

    assert exit_code == 1

    error_file = tmp_path / "training_orchestrator_error.txt"
    assert error_file.read_text(encoding="utf-8") == "bad value\n"

    captured = capsys.readouterr()
    assert "Unexpected orchestration failure: bad value" in captured.err
    assert "Error saved to:" in captured.err