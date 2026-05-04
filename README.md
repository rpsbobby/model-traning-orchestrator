# Model Training Orchestrator

Small Python HTTP orchestrator for remotely triggering and monitoring model training inside a Kubernetes pod.

The orchestrator talks to the Model Runtime Engine training API, starts a training job, polls training logs, and saves outputs beside the mounted project/data directory.

## Purpose

This app is intended to run alongside the training container, for example:

```text
pod:
  - model-runtime-engine   # training API / trainer
  - training-orchestrator  # this app
```

The orchestrator:

```text
1. Sends a training start request
2. Receives a job_id
3. Polls training logs by offset
4. Saves raw training logs
5. Saves structured orchestration events
6. Saves final job status
```

## Supported training types

| Project type | HTTP start endpoint |
|---|---|
| `Classification` | `/training/start/classification` |
| `ObjectDetection` | `/training/start/object_detection` |
| `Segmentation` | `/training/start/segmentation` |
| `InstanceSegmentation` | `/training/start/instance_segmentation` |
| `SemanticSegmentation` | `/training/start/semantic_segmentation` |
| `AnomalyDetection` | `/training/start/anomaly_detection` |

## Project layout

```text
.
├── main.py
├── pyproject.toml
├── source
│   ├── __init__.py
│   ├── cli.py
│   └── training_orchestrator.py
└── tests
    ├── test_cli.py
    └── test_training_orchestrator.py
```

## Setup

This project uses `uv`.

```bash
uv sync
```

For development dependencies:

```bash
uv sync --dev
```

## Running tests

```bash
uv run pytest
```

## Usage

### Object detection / YOLO

```bash
uv run python main.py \
  --url http://localhost \
  --port 8000 \
  --project-type ObjectDetection \
  --data-dir /data/my_project \
  --config-path /data/my_project/config.yaml \
  --additional-args-path /data/my_project/args.txt
```

### Classification

```bash
uv run python main.py \
  --url http://localhost \
  --port 8000 \
  --project-type Classification \
  --data-dir /data/my_project \
  --config-path /data/my_project/config.yaml
```

### Segmentation

```bash
uv run python main.py \
  --url http://localhost \
  --port 8000 \
  --project-type Segmentation \
  --data-dir /data/my_project \
  --config-path /data/my_project/config.yaml
```

## Output files

The orchestrator writes output files into `--data-dir`.

```text
training.log
training_events.jsonl
training_final_status.json
training_orchestrator_error.txt
```

### `training.log`

Raw trainer log output returned by:

```text
GET /training/jobs/<job_id>/logs
```

### `training_events.jsonl`

Structured newline-delimited JSON events emitted by the orchestrator.

Example:

```json
{"event": "training_started", "timestamp": 1777900000.123, "job_id": "abc123", "backend": "yolo"}
{"event": "training_log_poll", "timestamp": 1777900001.456, "job_id": "abc123", "offset": 0, "next_offset": 4096, "chunk_bytes": 4096, "done": false}
{"event": "training_finished", "timestamp": 1777900042.789, "job_id": "abc123", "status": {"state": "done"}}
```

### `training_final_status.json`

Final status returned by:

```text
GET /training/jobs/<job_id>
```

Example:

```json
{
  "ok": true,
  "job_id": "abc123",
  "backend": "yolo",
  "state": "done",
  "pid": 123,
  "exit_code": 0,
  "error": ""
}
```

### `training_orchestrator_error.txt`

Created only when orchestration fails.

## CLI options

```text
--url                   Trainer base URL without port. Default: http://localhost
--port                  Trainer HTTP port. Default: 8000
--project-type          Training project type to start
--data-dir              Mounted project/data directory for output files
--config-path           Absolute training config path
--additional-args-path  Absolute additional args path, only supported by ObjectDetection/YOLO
--poll-interval         Polling interval in seconds. Default: 1.0
--timeout               HTTP request timeout in seconds. Default: 10.0
--log-limit             Max log bytes per polling request. Default: 262144
--max-poll-errors       Max consecutive polling errors before failing. Default: 5
--no-stdout             Do not echo trainer logs to stdout
```

## API flow

Start training:

```text
POST /training/start/<project-type>
```

The server returns:

```json
{
  "ok": true,
  "backend": "yolo",
  "accepted": true,
  "job_id": "abc123"
}
```

Poll logs:

```text
GET /training/jobs/abc123/logs?offset=0&limit=262144
```

The server returns:

```json
{
  "ok": true,
  "job_id": "abc123",
  "backend": "yolo",
  "offset": 0,
  "next_offset": 4096,
  "chunk": "...",
  "done": false
}
```

Fetch final status:

```text
GET /training/jobs/abc123
```

Cancel on detected fatal trainer error:

```text
POST /training/jobs/abc123/cancel
```

## Development

Run all tests:

```bash
uv run pytest
```

Run a single test file:

```bash
uv run pytest tests/test_training_orchestrator.py
```

Run CLI tests only:

```bash
uv run pytest tests/test_cli.py
```

## Notes

When the orchestrator runs in the same Kubernetes pod as the trainer container, use:

```text
--url http://localhost
--port 8000
```

Containers in the same pod share the pod network namespace, so the trainer API should be reachable on localhost if it is bound correctly.
