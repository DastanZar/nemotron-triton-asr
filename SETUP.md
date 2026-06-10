# AI Setup Guide — Nemotron 3.5 ASR Triton Stack

## For Kilo / AI Assistants

This guide is for AI assistants setting up this project on a new server.

### Prerequisites

- Ubuntu 22.04+ with NVIDIA GPU (L40S recommended, 48GB VRAM)
- Docker with NVIDIA Container Toolkit
- Docker Compose v2+
- Python 3.12+ (for client testing)
- 50GB+ free disk space
- SSH access with PEM key

### Step 1: Clone the Repo

```bash
cd /home/ubuntu
git clone https://github.com/DastanZar/nemotron-triton-asr.git
cd nemotron-triton-asr
```

### Step 2: Build and Start

```bash
docker compose build
docker compose up -d
```

First build takes ~10-15 minutes. Triton image is ~9GB (includes NeMo, PyTorch, CUDA).

### Step 3: Verify

```bash
# Check containers are running
docker ps --format '{{.Names}} {{.Status}}'

# Health check
curl http://localhost:8000/healthz
# Should return: {"ok":true}

# Test transcription
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/home/ubuntu/nemotron-triton/test_audio/english/en_00.wav"
```

### Step 4: Access Dashboard

Forward port 8000 via VS Code or SSH tunnel, then open `http://localhost:8000`.

### Architecture

```
Browser → WebSocket → Gateway (8000) → HTTP → Triton (8001)
                                               ↓
                                       model.py (NeMo Python backend)
                                       Loads nvidia/nemotron-3.5-asr-streaming-0.6b
                                       Manages per-stream cache state
                                       Dynamic batching (4,8,16,32)
```

### Key Files to Understand

| File | Purpose |
|---|---|
| `docker-compose.yml` | Two services: triton + gateway. Mounts test_audio and cache |
| `deploy/Dockerfile.triton` | Triton 25.05 base, installs NeMo from source |
| `deploy/Dockerfile.gateway` | Python 3.12 slim, FastAPI + tritonclient |
| `model_repository/nemotron_streaming/config.pbtxt` | Triton config: inputs, outputs, dynamic_batching |
| `model_repository/nemotron_streaming/1/model.py` | Core: loads NeMo model, manages StreamState, runs batched transcribe |
| `gateway/app.py` | FastAPI routes: /, /healthz, /v1/files, /v1/transcribe, /v1/stream |
| `gateway/dashboard.html` | Browser UI for FLEURS benchmarking |

### Model Parameters (config.pbtxt)

```
MODEL_NAME: nvidia/nemotron-3.5-asr-streaming-0.6b
ATT_CONTEXT_SIZE: [56,3]        # 560ms chunks
STRIP_LANG_TAGS: true
MAX_SESSION_IDLE_SEC: 900
max_batch_size: 64
preferred_batch_size: [4, 8, 16, 32]
max_queue_delay_microseconds: 2000
```

### StreamState (model.py)

Each active stream maintains:
```python
@dataclass
class StreamState:
    stream_id: str
    target_lang: str
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    audio_buffer: np.ndarray | None = None
    previous_hypotheses: Any = None
    previous_pred_out: Any = None
    last_text: str = ""
    updated_at: float = 0.0
```

Sessions are auto-expired after `MAX_SESSION_IDLE_SEC` (900s).

### How model.py Works

1. **initialize()**: Downloads model from HuggingFace, loads NeMo, sets up GPU
2. **execute()**: Receives batched requests from Triton, groups by `(target_lang, is_end)`
3. **_run_group()**: For each group:
   - Sets inference prompt for target language
   - Creates/restores stream state per stream_id
   - Accumulates audio chunks in buffer
   - Runs `asr_model.transcribe()` on full accumulated audio
   - Returns transcript, is_final, language
4. **_expire_idle_sessions()**: Removes stale sessions

### API Details

#### POST /v1/transcribe
- Multipart form: `file` (audio), `target_lang` (optional), `chunk_ms` (optional)
- Returns: `{stream_id, text, is_final, language}`

#### WebSocket /v1/stream
- Client sends JSON: `{stream_id, target_lang, sample_rate, encoding, is_start, is_end, audio_b64}`
- Server responds: `{stream_id, text, is_final, language}`
- Encoding: `pcm_s16le` (Int16 PCM) or `float32le`

#### GET /v1/files
- Returns: `{english: [file1, ...], hindi: [file1, ...]}`

### Modifying the Stack

#### Change model parameters
Edit `model_repository/nemotron_streaming/config.pbtxt` and restart:
```bash
docker compose restart triton
```

#### Add new languages
1. Add test audio files to `test_audio/<lang>/`
2. The dashboard auto-discovers languages from the directory structure
3. Add language option to `gateway/app.py` and `dashboard.html`

#### Change chunk sizes
Edit the `DEFAULT_CHUNK_MS` environment variable in `docker-compose.yml` or the chunk size selector in the dashboard.

### Disk Space

| Component | Size |
|---|---|
| Triton image | ~9GB |
| HuggingFace cache | ~1.2GB |
| Test audio files | ~50MB |
| **Total** | ~10.5GB |

### Common Issues

| Issue | Fix |
|---|---|
| `no space left on device` | Remove old Docker images: `docker system prune -af` |
| Model download fails | Check internet connectivity inside container |
| Triton won't start | Check GPU: `nvidia-smi`, verify CUDA driver |
| Gateway 503 | Triton not ready yet, wait 30s after `docker compose up` |
| WebSocket 403 | Browser caching old dashboard, hard refresh |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRITON_URL` | `triton:8001` | Triton HTTP endpoint |
| `MODEL_NAME` | `nemotron_streaming` | Triton model name |
| `DEFAULT_CHUNK_MS` | `320` | Default chunk size in ms |
| `AUDIO_DIR` | `/data/test_audio` | Path to audio files inside container |
| `HF_HOME` | `/models/.hf-cache` | HuggingFace cache directory |
| `MODEL_NAME` (triton) | `nvidia/nemotron-3.5-asr-streaming-0.6b` | HuggingFace model ID |
| `TARGET_LANG_DEFAULT` | `auto` | Default language |
| `ATT_CONTEXT_SIZE` | `[56,3]` | NeMo attention context size |
| `STRIP_LANG_TAGS` | `true` | Strip language tags from output |
| `MAX_SESSION_IDLE_SEC` | `900` | Session expiry time |
| `LOG_LEVEL` | `INFO` | Logging level |
