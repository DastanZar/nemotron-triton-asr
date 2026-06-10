# Nemotron ASR Streaming on Triton

This project deploys `nvidia/nemotron-3.5-asr-streaming-0.6b` behind NVIDIA Triton and exposes a simple HTTP/WebSocket API on port `8000`.

The stack is split into two services:

- `triton`: Triton Inference Server with a Python backend model that loads the NeMo checkpoint and keeps per-stream cache state.
- `gateway`: FastAPI app that accepts audio streams, chunks them, and forwards them to Triton.

## Layout

- `docker-compose.yml`: runtime orchestration for EC2
- `deploy/bootstrap_remote.sh`: one-shot setup script for the AWS host
- `model_repository/nemotron_streaming/1/model.py`: Triton Python backend
- `gateway/app.py`: public API on port `8000`
- `client/concurrent_stream_client.py`: configurable concurrent STT load/client runner

## API

### WebSocket

`ws://<host>:8000/v1/stream`

The client sends JSON messages:

```json
{
  "stream_id": "call-001",
  "target_lang": "auto",
  "sample_rate": 16000,
  "is_start": true,
  "is_end": false,
  "audio_b64": "<base64 PCM16 or float32 bytes>",
  "encoding": "pcm_s16le"
}
```

Server responses:

```json
{
  "stream_id": "call-001",
  "text": "partial or final transcript",
  "is_final": false,
  "language": "auto"
}
```

### HTTP file transcription

`POST http://<host>:8000/v1/transcribe`

Multipart form:

- `file`: WAV/FLAC/OGG audio file
- `target_lang`: optional, defaults to `auto`
- `chunk_ms`: optional, defaults to `320`

## Client examples

```bash
python client/concurrent_stream_client.py \
  --server ws://54.158.0.249:8000/v1/stream \
  --inputs /path/to/a.wav /path/to/b.wav \
  --concurrency 2 \
  --target-lang auto \
  --chunk-ms 320
```

## Notes

- The upstream model card states this model was released on Hugging Face on `June 4, 2026`.
- The model card recommends NeMo cache-aware streaming with `att_context_size` values that map to chunk sizes of `80ms`, `160ms`, `320ms`, `560ms`, and `1120ms`.
- Triton `dynamic_batching` is enabled so chunk requests from concurrent streams can be co-scheduled on the GPU.
