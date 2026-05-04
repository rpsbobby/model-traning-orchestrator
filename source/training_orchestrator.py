from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, IO

import requests


class TrainingOrchestratorError(RuntimeError):
    pass


class TrainingHttpOrchestrator:
    PROJECT_TYPE_TO_START_PATH = {
        "Classification": "/training/start/classification",
        "ObjectDetection": "/training/start/object_detection",
        "Segmentation": "/training/start/segmentation",
        "InstanceSegmentation": "/training/start/instance_segmentation",
        "SemanticSegmentation": "/training/start/semantic_segmentation",
        "AnomalyDetection": "/training/start/anomaly_detection",
    }

    def __init__(
        self,
        base_url: str,
        raw_log_stream: IO[str],
        event_stream: IO[str] | None = None,
        poll_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 10.0,
        log_limit_bytes: int = 256 * 1024,
        max_poll_errors: int = 5,
        echo_logs_to_stdout: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.raw_log_stream = raw_log_stream
        self.event_stream = event_stream
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.log_limit_bytes = log_limit_bytes
        self.max_poll_errors = max_poll_errors
        self.echo_logs_to_stdout = echo_logs_to_stdout

        self.job_id: str | None = None
        self.backend: str | None = None
        self.project_type: str | None = None
        self.log_offset = 0
        self.poll_error_count = 0

    @classmethod
    def from_host_port(
        cls,
        url: str,
        port: int,
        raw_log_stream: IO[str],
        event_stream: IO[str] | None = None,
        **kwargs: Any,
    ) -> TrainingHttpOrchestrator:
        return cls(
            base_url=f"{url.rstrip('/')}:{port}",
            raw_log_stream=raw_log_stream,
            event_stream=event_stream,
            **kwargs,
        )

    def run(
        self,
        project_type: str,
        config_path: str | None = None,
        additional_args_path: str | None = None,
    ) -> dict[str, Any]:
        self.start_training(
            project_type=project_type,
            config_path=config_path,
            additional_args_path=additional_args_path,
        )
        return self.poll_logs_until_done()

    def start_training(
        self,
        project_type: str,
        config_path: str | None = None,
        additional_args_path: str | None = None,
    ) -> dict[str, Any]:
        start_path = self._start_path_for_project_type(project_type)
        start_url = f"{self.base_url}{start_path}"

        params: dict[str, str] = {}

        if config_path:
            params["config_path"] = self._validate_absolute_file(config_path, "config_path")

        if additional_args_path:
            params["additional_args_path"] = self._validate_absolute_file(
                additional_args_path,
                "additional_args_path",
            )

        self._write_event(
            "training_start_request",
            {
                "project_type": project_type,
                "url": start_url,
                "params": params,
            },
        )

        try:
            response = requests.post(
                start_url,
                params=params,
                timeout=self.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TrainingOrchestratorError(f"Failed to call training start endpoint: {exc}") from exc

        if response.status_code != 202:
            raise TrainingOrchestratorError(
                f"Training start failed. HTTP {response.status_code}: {response.text}"
            )

        payload = self._read_json_response(response)

        if not payload.get("ok"):
            raise TrainingOrchestratorError(f"Training start response was not ok: {payload}")

        job_id = payload.get("job_id")
        if not job_id:
            raise TrainingOrchestratorError(f"Training start response did not include job_id: {payload}")

        self.job_id = str(job_id)
        self.backend = str(payload.get("backend", ""))
        self.project_type = project_type
        self.log_offset = 0
        self.poll_error_count = 0

        self._write_event(
            "training_started",
            {
                "job_id": self.job_id,
                "backend": self.backend,
                "project_type": self.project_type,
                "response": payload,
            },
        )

        return payload

    def poll_logs_until_done(self) -> dict[str, Any]:
        job_id = self._require_job_id()

        self._write_event(
            "training_log_polling_started",
            {
                "job_id": job_id,
                "offset": self.log_offset,
                "limit": self.log_limit_bytes,
            },
        )

        while True:
            try:
                logs_payload = self.poll_logs_once()
                self.poll_error_count = 0
            except requests.RequestException as exc:
                self.poll_error_count += 1

                self._write_event(
                    "training_log_poll_error",
                    {
                        "job_id": job_id,
                        "error": str(exc),
                        "poll_error_count": self.poll_error_count,
                    },
                )

                if self.poll_error_count >= self.max_poll_errors:
                    raise TrainingOrchestratorError(
                        "Lost connection to training service after repeated polling failures"
                    ) from exc

                backoff_seconds = min(
                    self.poll_interval_seconds * (2**self.poll_error_count),
                    4.0,
                )
                time.sleep(backoff_seconds)
                continue

            chunk = str(logs_payload.get("chunk", ""))
            done = bool(logs_payload.get("done", False))

            if chunk:
                self._write_raw_log(chunk)

                detected_error = self._detect_error(chunk)
                if detected_error:
                    self._write_event(
                        "training_error_detected",
                        {
                            "job_id": job_id,
                            "error": detected_error,
                        },
                    )

                    try:
                        self.cancel_training()
                    finally:
                        raise TrainingOrchestratorError(detected_error)

            self._write_event(
                "training_log_poll",
                {
                    "job_id": job_id,
                    "offset": logs_payload.get("offset"),
                    "next_offset": logs_payload.get("next_offset"),
                    "chunk_bytes": len(chunk.encode("utf-8", errors="replace")),
                    "done": done,
                },
            )

            if done:
                final_status = self.get_training_status()

                self._write_event(
                    "training_finished",
                    {
                        "job_id": job_id,
                        "status": final_status,
                    },
                )

                return final_status

            time.sleep(self.poll_interval_seconds)

    def poll_logs_once(self) -> dict[str, Any]:
        job_id = self._require_job_id()
        logs_url = f"{self.base_url}/training/jobs/{job_id}/logs"

        response = requests.get(
            logs_url,
            params={
                "offset": self.log_offset,
                "limit": self.log_limit_bytes,
            },
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()

        payload = self._read_json_response(response)

        if not payload.get("ok"):
            raise TrainingOrchestratorError(f"Log response was not ok: {payload}")

        self.log_offset = int(payload.get("next_offset", self.log_offset))

        return payload

    def get_training_status(self) -> dict[str, Any]:
        job_id = self._require_job_id()
        status_url = f"{self.base_url}/training/jobs/{job_id}"

        try:
            response = requests.get(
                status_url,
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TrainingOrchestratorError(f"Failed to fetch training status: {exc}") from exc

        return self._read_json_response(response)

    def cancel_training(self) -> dict[str, Any] | None:
        if not self.job_id:
            return None

        cancel_url = f"{self.base_url}/training/jobs/{self.job_id}/cancel"

        try:
            response = requests.post(
                cancel_url,
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self._write_event(
                "training_cancel_failed",
                {
                    "job_id": self.job_id,
                    "error": str(exc),
                },
            )
            return None

        payload = self._read_json_response(response)

        self._write_event(
            "training_cancel_requested",
            {
                "job_id": self.job_id,
                "response": payload,
            },
        )

        return payload

    def _start_path_for_project_type(self, project_type: str) -> str:
        path = self.PROJECT_TYPE_TO_START_PATH.get(project_type)

        if not path:
            supported = ", ".join(sorted(self.PROJECT_TYPE_TO_START_PATH))
            raise TrainingOrchestratorError(
                f"Unsupported project type: {project_type!r}. Supported values: {supported}"
            )

        return path

    def _write_raw_log(self, chunk: str) -> None:
        self.raw_log_stream.write(chunk)
        self.raw_log_stream.flush()

        if self.echo_logs_to_stdout:
            print(chunk, end="", flush=True)

    def _write_event(self, event_name: str, payload: dict[str, Any]) -> None:
        if self.event_stream is None:
            return

        event = {
            "event": event_name,
            "timestamp": time.time(),
            **payload,
        }

        self.event_stream.write(json.dumps(event, default=str) + "\n")
        self.event_stream.flush()

    def _detect_error(self, chunk: str) -> str | None:
        if "CUDA out of memory" in chunk:
            return (
                "Error: out of GPU memory.\n\n"
                "Solution 1: Close the main Acumen app.\n"
                "Solution 2: Reduce the batch size.\n"
                "Solution 3: Reboot the system.\n\n"
                "If none of the above solve the issue, contact Ash Technologies."
            )

        if "Traceback" in chunk:
            return chunk

        return None

    def _require_job_id(self) -> str:
        if not self.job_id:
            raise TrainingOrchestratorError("No active training job. Call start_training() first.")

        return self.job_id

    @staticmethod
    def _read_json_response(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise TrainingOrchestratorError(
                f"Expected JSON response but got: {response.text}"
            ) from exc

        if not isinstance(payload, dict):
            raise TrainingOrchestratorError(f"Expected JSON object response but got: {payload!r}")

        return payload

    @staticmethod
    def _validate_absolute_file(path: str, label: str) -> str:
        candidate = Path(path)

        if not candidate.is_absolute():
            raise TrainingOrchestratorError(f"{label} must be an absolute path: {path}")

        if not candidate.is_file():
            raise TrainingOrchestratorError(f"{label} file does not exist: {path}")

        return str(candidate)