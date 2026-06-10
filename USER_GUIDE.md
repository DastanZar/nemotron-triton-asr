# User Guide — Nemotron 3.5 ASR Triton Dashboard

## Quick Start

### 1. Access the Dashboard

**Via VS Code Remote-SSH:**
1. Open VS Code and connect to the server via Remote-SSH
2. Go to the **Ports** tab (bottom panel)
3. Click **Forward a Port** → enter `8000`
4. Open `http://localhost:8000` in your laptop browser

**Via SSH tunnel:**
```bash
ssh -L 8000:localhost:8000 -i AWS-Ramesh.pem ubuntu@13.220.228.60
```
Then open `http://localhost:8000` in your browser.

### 2. Run a Benchmark

1. **Select Language**: English, Hindi, or Auto-detect
2. **Set Concurrency**: Use the slider (1-50 simultaneous streams)
3. **Select Files**: Check/uncheck individual files, or use:
   - **Select All** — select every file
   - **Deselect All** — clear selection
   - **Select 1-N** — enter a number to select first N files
4. **Click Run Benchmark**
5. Watch results appear in real-time on the **Results** tab
6. Switch to **Logs** tab to see detailed progress

### 3. Understand the Results

Each result card shows:
- **File name** (e.g., `en_00`)
- **Language** detected
- **FINAL** badge — final transcription
- **Transcript text**

The stats bar shows:
- **Done** — files completed
- **Total** — files selected
- **Errors** — failed transcriptions
- **Avg Time** — average processing time per file

## Available Test Files

| Language | Location | Count |
|---|---|---|
| English | `/test_audio/english/` | 32 WAV files |
| Hindi | `/test_audio/hindi/` | 15 WAV files |

All files are 16-bit PCM WAV, mono, 16kHz.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/healthz` | GET | Health check (`{"ok": true}`) |
| `/v1/files` | GET | List available audio files |
| `/v1/transcribe` | POST | Transcribe a single file |
| `/v1/stream` | WebSocket | Real-time streaming transcription |

### Transcribe via curl

**English:**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/home/ubuntu/nemotron-triton/test_audio/english/en_00.wav" \
  -F "target_lang=en"
```

**Hindi:**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/home/ubuntu/nemotron-triton/test_audio/hindi/hindi_01.wav" \
  -F "target_lang=hi"
```

**Auto-detect:**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/home/ubuntu/nemotron-triton/test_audio/english/en_00.wav"
```

## Troubleshooting

| Problem | Solution |
|---|---|
| Dashboard shows "Disconnected" | Check if containers are running: `docker ps` |
| Dashboard shows nothing after clicking Run | Hard refresh: `Ctrl+Shift+R` / `Cmd+Shift+R` |
| `Connection refused` on port 8000 | Port forward not set up. Add port 8000 in VS Code Ports tab |
| Transcription returns empty text | Audio file might be too short or corrupted. Check file with `ffprobe` |
| Slow transcription | Reduce concurrency or increase chunk size |

## Restart / Stop

```bash
cd /home/ubuntu/nemotron-triton

# Restart both services
docker compose restart

# Stop both services
docker compose down

# View live logs
docker compose logs -f

# View only Triton logs
docker compose logs -f triton

# View only Gateway logs
docker compose logs -f gateway
```
