# Context Carry — Nemotron 3.5 ASR Triton Stack

## What This Is

A production-grade streaming ASR deployment for **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** using Triton Inference Server + FastAPI gateway. This replaces the earlier NIM-based approach which had streaming/gRPC bugs.

## Architecture

```
Browser/Client → WebSocket → FastAPI Gateway (port 8000) → HTTP → Triton Server (port 8001)
                                                      ↓
                                              Python Backend (model.py)
                                              NeMo ASR model on GPU
```

### Two Services (Docker Compose)
| Service | Container | Port | Role |
|---|---|---|---|
| **triton** | `nemotron-triton` | 8001 (HTTP), 8002 (gRPC), 8003 (metrics) | Inference server with NeMo Python backend |
| **gateway** | `nemotron-gateway` | 8000 | FastAPI app: WebSocket + HTTP API + Dashboard |

### Key Files
```
nemotron-triton/
├── docker-compose.yml                     # Orchestrates both services
├── deploy/
│   ├── Dockerfile.triton                  # Triton 25.05 + NeMo from source
│   └── Dockerfile.gateway                 # Python 3.12 + FastAPI
├── model_repository/
│   └── nemotron_streaming/
│       ├── config.pbtxt                   # Triton model config (dynamic_batching)
│       └── 1/model.py                     # Python backend: loads NeMo model, manages stream state
├── gateway/
│   ├── app.py                             # FastAPI: /v1/stream (WS), /v1/transcribe (POST), /v1/files, /healthz
│   └── dashboard.html                     # Browser dashboard for FLEURS benchmarking
├── client/
│   └── concurrent_stream_client.py        # CLI client for concurrent WebSocket streaming
├── test_audio/                            # FLEURS test clips
│   ├── english/                           # 32 WAV files
│   └── hindi/                             # 15 WAV files
└── cache/hf/                              # HuggingFace model cache (persistent)
```

## How Triton Approach Differs from NIM

| Aspect | NIM (old) | Triton (current) |
|---|---|---|
| Model serving | Black-box NIM container | Triton Python backend + NeMo |
| Stream state | NIM internal, no control | `StreamState` dataclass with cache tensors |
| Batching | Opaque | Triton `dynamic_batching` (preferred_batch_size: 4,8,16,32) |
| API | gRPC streaming (buggy) | FastAPI WebSocket + HTTP |
| Dependencies | NGC API key, TRT engines | NeMo from source, HuggingFace cache |
| Dashboard | Custom nim_dashboard.html | FLEURS benchmark dashboard |

## Model Details

- **Model**: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- **att_context_size**: `[56,3]` (560ms chunks)
- **STRIP_LANG_TAGS**: true
- **MAX_SESSION_IDLE_SEC**: 900
- **dynamic_batching**: max 64, preferred [4,8,16,32], 2ms queue delay

## Server Info

- **Current server**: 13.220.228.60 (AWS, L40S GPU, 48GB VRAM)
- **Project path**: `/home/ubuntu/nemotron-triton/`
- **Old project (untouched)**: `/home/ubuntu/nemotron-asr-benchmark/`
- **GitHub PAT**: stored in `~/.netrc` or environment variable
- **NGC API key**: stored in environment variable `NGC_API_KEY`

## Dashboard Features

- **Language picker**: English, Hindi, Auto-detect
- **Concurrency slider**: 1-50 simultaneous streams
- **Chunk size**: 80ms / 160ms / 320ms / 560ms
- **File browser**: Lists all FLEURS test clips from server
- **Select All / Deselect All / Select 1-N**: Quick file selection
- **Real-time results**: Transcripts appear as they're processed
- **Stats bar**: Done / Total / Errors / Avg Time
- **Logs tab**: Full transcription progress

## Known Issues

1. **Browser cache**: Hard refresh (Ctrl+Shift+R) after any dashboard changes
2. **WAV parsing**: Dashboard parses WAV headers manually (AudioContext.decodeAudioData is unreliable)
3. **Port forwarding**: Use VS Code Ports tab to forward port 8000 for browser access

## Commands

```bash
# Check status
docker ps --format '{{.Names}} {{.Status}}'

# Health check
curl http://localhost:8000/healthz

# View logs
docker compose -f /home/ubuntu/nemotron-triton/docker-compose.yml logs -f

# Restart
docker compose -f /home/ubuntu/nemotron-triton/docker-compose.yml restart

# Stop
docker compose -f /home/ubuntu/nemotron-triton/docker-compose.yml down

# Test transcription
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/home/ubuntu/nemotron-triton/test_audio/english/en_00.wav"
```

## What Was Done Today

1. Cloned colleague's Triton-based ASR project from 54.158.0.249
2. Analyzed architecture differences (Triton vs NIM)
3. Deployed on current server in `/home/ubuntu/nemotron-triton/`
4. Fixed NIM streaming bug (force_close thread) — now irrelevant, Triton approach works
5. Built custom dashboard with FLEURS file browser and concurrency control
6. Fixed WAV parsing bug (AudioContext → manual WAV header parser)
7. Added `/v1/files` API endpoint for dashboard file listing
8. Mounted test_audio directory into gateway container
9. Pushed to GitHub as new repo
