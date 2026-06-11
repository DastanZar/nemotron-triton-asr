#!/usr/bin/env python3
"""
Nemotron ASR Triton Benchmark Suite
Tests: single-stream latency, concurrent throughput, GPU utilization
"""

import argparse
import asyncio
import base64
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


TARGET_SR = 16000
GWSERVER = os.environ.get("GWSERVER", "ws://localhost:8000/v1/stream")


@dataclass
class StreamResult:
    stream_id: str
    file: str
    elapsed_sec: float
    audio_duration_sec: float
    rtf: float  # real-time factor = elapsed / audio_duration
    chunks_sent: int
    text: str


@dataclass
class BenchmarkResult:
    concurrency: int
    total_files: int
    total_audio_sec: float
    total_elapsed_sec: float
    avg_rtf: float
    throughput_rtf: float  # total_audio / total_elapsed (higher = better)
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    streams: List[StreamResult] = field(default_factory=list)


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr).astype(np.float32)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.duration


def iter_chunks(audio: np.ndarray, chunk_ms: int):
    chunk_samples = max(1, int(TARGET_SR * chunk_ms / 1000))
    for start in range(0, len(audio), chunk_samples):
        yield audio[start : start + chunk_samples]


async def stream_file(
    server: str,
    audio_path: Path,
    target_lang: str,
    chunk_ms: int,
) -> StreamResult:
    audio = load_audio(audio_path)
    duration = get_audio_duration(audio_path)
    stream_id = f"{audio_path.stem}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    last_text = ""
    chunks_sent = 0
    chunk_latencies = []

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
                "audio_b64": base64.b64encode(
                    (chunk * 32767.0).astype(np.int16).tobytes()
                ).decode("utf-8"),
            }
            chunk_start = time.perf_counter()
            await ws.send(json.dumps(payload))
            response = json.loads(await ws.recv())
            chunk_elapsed = time.perf_counter() - chunk_start
            chunk_latencies.append(chunk_elapsed * 1000)  # ms
            last_text = response.get("text", "")
            chunks_sent += 1

    elapsed = time.perf_counter() - started
    return StreamResult(
        stream_id=stream_id,
        file=str(audio_path.name),
        elapsed_sec=round(elapsed, 3),
        audio_duration_sec=round(duration, 3),
        rtf=round(elapsed / duration, 3) if duration > 0 else 0,
        chunks_sent=chunks_sent,
        text=last_text,
    )


async def bounded_gather(coros, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def runner(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(runner(coro) for coro in coros))


def get_gpu_info():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        parts = out.strip().split(", ")
        return {
            "gpu_util": float(parts[0]),
            "mem_used_mb": float(parts[1]),
            "mem_total_mb": float(parts[2]),
            "temp_c": float(parts[3]),
        }
    except Exception:
        return None


def collect_gpu_samples(stop_event, samples):
    while not stop_event.is_set():
        info = get_gpu_info()
        if info:
            samples.append(info)
        time.sleep(0.5)


def run_benchmark(
    server: str,
    audio_files: List[Path],
    concurrency: int,
    target_lang: str,
    chunk_ms: int,
    label: str,
) -> BenchmarkResult:
    import threading

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {label}")
    print(f"  Concurrency: {concurrency} | Files: {len(audio_files)} | Chunk: {chunk_ms}ms")
    print(f"{'='*60}")

    gpu_samples = []
    stop_event = threading.Event()
    gpu_thread = threading.Thread(
        target=collect_gpu_samples, args=(stop_event, gpu_samples), daemon=True
    )
    gpu_thread.start()

    coros = [
        stream_file(server, f, target_lang, chunk_ms) for f in audio_files
    ]

    wall_start = time.perf_counter()
    results = asyncio.run(bounded_gather(coros, concurrency))
    wall_elapsed = time.perf_counter() - wall_start

    stop_event.set()
    gpu_thread.join(timeout=2)

    total_audio = sum(r.audio_duration_sec for r in results)
    total_elapsed = sum(r.elapsed_sec for r in results)
    avg_rtf = np.mean([r.rtf for r in results])
    throughput_rtf = total_audio / wall_elapsed if wall_elapsed > 0 else 0

    latencies = []
    for r in results:
        # approximate per-chunk latency from elapsed / chunks
        if r.chunks_sent > 0:
            latencies.append(r.elapsed_sec / r.chunks_sent * 1000)

    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0

    bench = BenchmarkResult(
        concurrency=concurrency,
        total_files=len(results),
        total_audio_sec=round(total_audio, 2),
        total_elapsed_sec=round(wall_elapsed, 2),
        avg_rtf=round(avg_rtf, 3),
        throughput_rtf=round(throughput_rtf, 2),
        p50_latency_ms=round(p50, 1),
        p95_latency_ms=round(p95, 1),
        p99_latency_ms=round(p99, 1),
        streams=results,
    )

    print(f"\n  RESULTS:")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Files processed':<30} {bench.total_files:>12}")
    print(f"  {'Total audio (sec)':<30} {bench.total_audio_sec:>12}")
    print(f"  {'Wall time (sec)':<30} {bench.total_elapsed_sec:>12}")
    print(f"  {'Avg RTF (per stream)':<30} {bench.avg_rtf:>12}")
    print(f"  {'Throughput RTF':<30} {bench.throughput_rtf:>12}x")
    print(f"  {'Chunk latency p50 (ms)':<30} {bench.p50_latency_ms:>12}")
    print(f"  {'Chunk latency p95 (ms)':<30} {bench.p95_latency_ms:>12}")
    print(f"  {'Chunk latency p99 (ms)':<30} {bench.p99_latency_ms:>12}")

    if gpu_samples:
        gpu_utils = [s["gpu_util"] for s in gpu_samples]
        mem_used = [s["mem_used_mb"] for s in gpu_samples]
        temps = [s["temp_c"] for s in gpu_samples]
        print(f"\n  GPU:")
        print(f"  {'GPU util avg':<30} {np.mean(gpu_utils):>11.1f}%")
        print(f"  {'GPU util peak':<30} {np.max(gpu_utils):>11.1f}%")
        print(f"  {'VRAM used (MB)':<30} {np.mean(mem_used):>11.0f}")
        print(f"  {'Temperature':<30} {np.mean(temps):>11.0f}C")

    return bench


def main():
    parser = argparse.ArgumentParser(description="Nemotron ASR Benchmark")
    parser.add_argument("--server", default=GWSERVER, help="WebSocket endpoint")
    parser.add_argument("--lang", choices=["english", "hindi", "both"], default="english")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all files")
    parser.add_argument("--realtime", action="store_true", help="Simulate real-time streaming")
    args = parser.parse_args()

    test_dir = Path(__file__).parent.parent / "test_audio"
    files = []
    if args.lang in ("english", "both"):
        files.extend(sorted((test_dir / "english").glob("*.wav")))
    if args.lang in ("hindi", "both"):
        files.extend(sorted((test_dir / "hindi").glob("*.wav")))

    if not files:
        print("No test audio found. Run: python scripts/download_fleurs.py")
        return

    if args.max_files > 0:
        files = files[: args.max_files]

    print(f"Nemotron ASR Triton Benchmark")
    print(f"Server: {args.server}")
    print(f"Test audio: {len(files)} files, {args.lang}")
    print(f"Chunk size: {args.chunk_ms}ms")

    gpu = get_gpu_info()
    if gpu:
        print(f"GPU: {gpu['mem_used_mb']:.0f}/{gpu['mem_total_mb']:.0f}MB, {gpu['temp_c']:.0f}C")

    all_results = []
    for c in args.concurrency:
        result = run_benchmark(
            server=args.server,
            audio_files=files,
            concurrency=c,
            target_lang="auto",
            chunk_ms=args.chunk_ms,
            label=f"Concurrency={c}",
        )
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Concurrency':<14} {'RTF':>8} {'Throughput':>12} {'p50(ms)':>10} {'p95(ms)':>10} {'GPU%':>8}")
    print(f"  {'-'*64}")
    for r in all_results:
        gpu_avg = 0
        if r.streams:
            # rough estimate from throughput
            gpu_avg = min(100, r.throughput_rtf * 10)
        print(
            f"  {r.concurrency:<14} {r.avg_rtf:>8.3f} {r.throughput_rtf:>11.2f}x {r.p50_latency_ms:>10.1f} {r.p95_latency_ms:>10.1f} {'~':>1}{gpu_avg:>6.0f}"
        )

    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(
            [
                {
                    "concurrency": r.concurrency,
                    "total_files": r.total_files,
                    "total_audio_sec": r.total_audio_sec,
                    "wall_time_sec": r.total_elapsed_sec,
                    "avg_rtf": r.avg_rtf,
                    "throughput_rtf": r.throughput_rtf,
                    "p50_ms": r.p50_latency_ms,
                    "p95_ms": r.p95_latency_ms,
                    "p99_ms": r.p99_latency_ms,
                }
                for r in all_results
            ],
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
