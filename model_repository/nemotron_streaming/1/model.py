import ast
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from nemo.collections.asr.parts.utils.transcribe_utils import setup_model


LOGGER = logging.getLogger("nemotron_streaming")


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _decode_scalar_string(tensor) -> str:
    value = tensor.as_numpy().reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_scalar_bool(tensor) -> bool:
    value = tensor.as_numpy().reshape(-1)[0]
    return bool(value)


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


class TritonPythonModel:
    def initialize(self, args):
        if isinstance(args, str):
            args = json.loads(args)
        model_config = args["model_config"]
        if isinstance(model_config, str):
            model_config = json.loads(model_config)
        params = model_config.get("parameters", {})

        self.model_name = self._parameter(params, "MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.default_target_lang = self._parameter(params, "TARGET_LANG_DEFAULT", "auto")
        self.att_context_size = ast.literal_eval(self._parameter(params, "ATT_CONTEXT_SIZE", "[56,3]"))
        self.strip_lang_tags = _as_bool(self._parameter(params, "STRIP_LANG_TAGS", "true"))
        self.max_session_idle_sec = int(self._parameter(params, "MAX_SESSION_IDLE_SEC", "900"))

        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading model %s on %s", self.model_name, self.device)

        cfg = type("Cfg", (), {"model_path": None, "pretrained_name": self.model_name})()
        self.asr_model, _ = setup_model(cfg=cfg, map_location=self.device)
        if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
        if hasattr(self.asr_model, "set_inference_prompt"):
            self.asr_model.set_inference_prompt(self.default_target_lang)
            self.asr_model.decoding.set_strip_lang_tags(self.strip_lang_tags)

        self.asr_model = self.asr_model.to(device=self.device, dtype=torch.float32)
        self.asr_model.eval()
        self.sessions: dict[str, StreamState] = {}

    def execute(self, requests):
        self._expire_idle_sessions()
        responses = []
        grouped = {}

        for index, request in enumerate(requests):
            stream_id = _decode_scalar_string(pb_utils.get_input_tensor_by_name(request, "STREAM_ID"))
            target_lang = _decode_scalar_string(pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")) or self.default_target_lang
            is_end = _decode_scalar_bool(pb_utils.get_input_tensor_by_name(request, "IS_END"))
            group_key = (target_lang, is_end)
            grouped.setdefault(group_key, []).append((index, request, stream_id, target_lang, is_end))

        output_map = {}
        for (target_lang, is_end), items in grouped.items():
            batch_result = self._run_group(items=items, target_lang=target_lang, is_end=is_end)
            output_map.update(batch_result)

        for index in range(len(requests)):
            transcript, is_final, language = output_map[index]
            outputs = [
                pb_utils.Tensor("TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=object)),
                pb_utils.Tensor("IS_FINAL", np.array([is_final], dtype=bool)),
                pb_utils.Tensor("LANGUAGE", np.array([language.encode("utf-8")], dtype=object)),
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))

        return responses

    def finalize(self):
        self.sessions.clear()

    def _run_group(self, items, target_lang: str, is_end: bool):
        if hasattr(self.asr_model, "set_inference_prompt"):
            self.asr_model.set_inference_prompt(target_lang)
            self.asr_model.decoding.set_strip_lang_tags(self.strip_lang_tags)

        states = []

        for _, request, stream_id, _, _ in items:
            is_start = _decode_scalar_bool(pb_utils.get_input_tensor_by_name(request, "IS_START"))
            chunk = pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNK").as_numpy().astype(np.float32).reshape(-1)
            length = int(pb_utils.get_input_tensor_by_name(request, "AUDIO_LENGTH").as_numpy().reshape(-1)[0])
            if is_start or stream_id not in self.sessions:
                self.sessions[stream_id] = self._new_state(stream_id=stream_id, target_lang=target_lang)
            state = self.sessions[stream_id]
            state.target_lang = target_lang
            state.updated_at = time.time()
            audio_slice = chunk[:length]
            if state.audio_buffer is None or is_start:
                state.audio_buffer = audio_slice.copy()
            else:
                state.audio_buffer = np.concatenate([state.audio_buffer, audio_slice])
            states.append(state)

        audio_batch = [state.audio_buffer.astype(np.float32, copy=False) for state in states]
        with torch.inference_mode():
            transcribed_texts = self.asr_model.transcribe(
                audio_batch,
                batch_size=len(audio_batch),
                target_lang=target_lang,
                verbose=False,
            )
        texts = self._extract_texts(transcribed_texts)

        results = {}
        for idx, (request_index, _, stream_id, _, _) in enumerate(items):
            state = self.sessions[stream_id]
            state.last_text = texts[idx]
            results[request_index] = (texts[idx], is_end, target_lang)
            if is_end:
                self.sessions.pop(stream_id, None)
        return results

    def _new_state(self, stream_id: str, target_lang: str) -> StreamState:
        cache_last_channel, cache_last_time, cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(
            batch_size=1
        )
        return StreamState(
            stream_id=stream_id,
            target_lang=target_lang,
            cache_last_channel=cache_last_channel.to(self.device),
            cache_last_time=cache_last_time.to(self.device),
            cache_last_channel_len=cache_last_channel_len.to(self.device),
            updated_at=time.time(),
        )

    def _expire_idle_sessions(self):
        now = time.time()
        stale_ids = [
            stream_id
            for stream_id, state in self.sessions.items()
            if now - state.updated_at > self.max_session_idle_sec
        ]
        for stream_id in stale_ids:
            self.sessions.pop(stream_id, None)

    @staticmethod
    def _extract_texts(transcribed_texts) -> list[str]:
        texts = []
        for item in transcribed_texts:
            if hasattr(item, "text"):
                texts.append(item.text)
            else:
                texts.append(str(item))
        return texts

    @staticmethod
    def _parameter(params, key: str, default: str) -> str:
        value = params.get(key)
        if not value:
            return default
        return value["string_value"]
