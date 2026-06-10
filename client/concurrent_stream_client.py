import argparse
import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


TARGET_SR = 16000


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def iter_chunks(audio: np.ndarray, chunk_ms: int):
    chunk_samples = max(1, int(TARGET_SR * chunk_ms / 1000))
    for start in range(0, len(audio), chunk_samples):
        yield audio[start : start + chunk_samples]


async def stream_file(server: str, audio_path: Path, target_lang: str, chunk_ms: int, realtime: bool):
    audio = load_audio(audio_path)
    stream_id = f"{audio_path.stem}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    last_text = ""

    async with websockets.connect(server, max_size=8 * 1024 * 1024) as ws:
        chunks = list(iter_chunks(audio, chunk_ms))
        for index, chunk in enumerate(chunks):
            payload = {
                "stream_id": stream_id,
                "target_lang": target_lang,
                "sample_rate": TARGET_SR,
                "encoding": "pcm_s16le",
                "is_start": index == 0,
                "is_end": index == len(chunks) - 1,
                "audio_b64": base64.b64encode((chunk * 32767.0).astype(np.int16).tobytes()).decode("utf-8"),
            }
            await ws.send(json.dumps(payload))
            response = json.loads(await ws.recv())
            last_text = response.get("text", "")
            if realtime and index < len(chunks) - 1:
                await asyncio.sleep(chunk_ms / 1000.0)

    elapsed = time.perf_counter() - started
    return {
        "stream_id": stream_id,
        "file": str(audio_path),
        "elapsed_sec": round(elapsed, 3),
        "text": last_text,
    }


async def bounded_gather(coros, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def runner(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(runner(coro) for coro in coros))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="WebSocket endpoint, e.g. ws://host:8000/v1/stream")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input audio files")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--target-lang", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    coros = [
        stream_file(
            server=args.server,
            audio_path=Path(path),
            target_lang=args.target_lang,
            chunk_ms=args.chunk_ms,
            realtime=args.realtime,
        )
        for path in args.inputs
    ]
    results = await bounded_gather(coros, concurrency=args.concurrency)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
