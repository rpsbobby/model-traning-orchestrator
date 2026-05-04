from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from source.training_orchestrator import (
    TrainingHttpOrchestrator,
    TrainingOrchestratorError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start and poll model training through the Model Runtime Engine HTTP API.",
    )

    parser.add_argument(
        "--url",
        default="http://localhost",
        help="Trainer base URL without port. Default: http://localhost",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Trainer HTTP port. Default: 8000",
    )

    parser.add_argument(
        "--project-type",
        required=True,
        choices=sorted(TrainingHttpOrchestrator.PROJECT_TYPE_TO_START_PATH.keys()),
        help="Training project type to start.",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Mounted project/data directory where logs and events will be saved.",
    )

    parser.add_argument(
        "--config-path",
        default=None,
        help="Absolute training config path. Passed as query param config_path.",
    )

    parser.add_argument(
        "--additional-args-path",
        default=None,
        help="Absolute additional args path. Only supported by ObjectDetection/YOLO.",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds. Default: 1.0",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout in seconds. Default: 10.0",
    )

    parser.add_argument(
        "--log-limit",
        type=int,
        default=256 * 1024,
        help="Max log bytes to fetch per request. Default: 262144",
    )

    parser.add_argument(
        "--max-poll-errors",
        type=int,
        default=5,
        help="Maximum consecutive polling errors before failing. Default: 5",
    )

    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Do not echo training logs to stdout.",
    )

    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = data_dir / "training.log"
    events_path = data_dir / "training_events.jsonl"
    final_status_path = data_dir / "training_final_status.json"
    error_path = data_dir / "training_orchestrator_error.txt"

    try:
        with (
            raw_log_path.open("w", encoding="utf-8") as raw_log_stream,
            events_path.open("w", encoding="utf-8") as event_stream,
        ):
            orchestrator = TrainingHttpOrchestrator.from_host_port(
                url=args.url,
                port=args.port,
                raw_log_stream=raw_log_stream,
                event_stream=event_stream,
                poll_interval_seconds=args.poll_interval,
                request_timeout_seconds=args.timeout,
                log_limit_bytes=args.log_limit,
                max_poll_errors=args.max_poll_errors,
                echo_logs_to_stdout=not args.no_stdout,
            )

            final_status = orchestrator.run(
                project_type=args.project_type,
                config_path=args.config_path,
                additional_args_path=args.additional_args_path,
            )

        final_status_path.write_text(
            json.dumps(final_status, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        print()
        print("Training finished")
        print(f"Raw log:      {raw_log_path}")
        print(f"Events:       {events_path}")
        print(f"Final status: {final_status_path}")

        state = str(final_status.get("state", "")).lower()
        if state and state != "done":
            print(f"Training ended with non-success state: {state}", file=sys.stderr)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130

    except TrainingOrchestratorError as exc:
        error_path.write_text(str(exc) + "\n", encoding="utf-8")
        print(f"Training orchestration failed: {exc}", file=sys.stderr)
        print(f"Error saved to: {error_path}", file=sys.stderr)
        return 1

    except Exception as exc:
        error_path.write_text(str(exc) + "\n", encoding="utf-8")
        print(f"Unexpected orchestration failure: {exc}", file=sys.stderr)
        print(f"Error saved to: {error_path}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_from_args(args)