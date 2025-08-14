from __future__ import annotations

import asyncio
import inspect
import contextlib
import json
import ast
import base64
import dataclasses
import uuid
import time
import os
import threading
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ..log import logger

# ElevenLabs Conversational AI uses PCM 16kHz mono
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class _RealtimeOptions:
    agent_id: str
    api_key: str | None
    modalities: list[Literal["text", "audio"]]
    temperature: float | None
    tool_choice: llm.ToolChoice | None
    user_id: str | None
    dynamic_variables: dict[str, Any] | None
    conversation_config_override: dict[str, Any] | None
    extra_body: dict[str, Any] | None
    # TODO: add dynamic_variables, language, voice override as needed


class RealtimeModel(llm.RealtimeModel):
    """
    ElevenLabs realtime model wrapper matching LiveKit's llm.RealtimeModel interface.

    This scaffolds the class and capabilities. The actual streaming logic will be
    implemented by wrapping elevenlabs.conversational_ai.conversation.Conversation.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        api_key: str | None = None,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        user_id: str | None = None,
        dynamic_variables: dict[str, Any] | None = None,
        conversation_config_override: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        modalities_val = modalities if modalities is not NOT_GIVEN else ["text", "audio"]
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,          # handled by ElevenLabs VAD/barge-in
                user_transcription=True,       # transcription events provided by SDK
                auto_tool_reply_generation=True,
                audio_output=("audio" in modalities_val),
            )
        )

        self._opts = _RealtimeOptions(
            agent_id=agent_id,
            api_key=api_key,
            modalities=modalities_val,
            temperature=(None if temperature is NOT_GIVEN else float(temperature)),
            tool_choice=(None if tool_choice is NOT_GIVEN else tool_choice),
            user_id=user_id,
            dynamic_variables=dynamic_variables,
            conversation_config_override=conversation_config_override,
            extra_body=extra_body,
        )

    def session(self) -> RealtimeSession:
        return RealtimeSession(self)

    async def aclose(self) -> None:  # pragma: no cover - nothing to cleanup yet
        return None


class RealtimeSession(llm.RealtimeSession[Any]):
    """
    ElevenLabs realtime session scaffolding.

    TODO: Wire ElevenLabs SDK Conversation and audio bridging.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._tool_choice: llm.ToolChoice | None = realtime_model._opts.tool_choice
        self._closed = False
        # Placeholders for upcoming SDK objects
        self._conversation = None  # type: ignore[var-annotated]
        self._audio_task: asyncio.Task[None] | None = None
        # Prevent re-starts after an explicit end_conversation()
        self._ended: bool = False

        # ElevenLabs SDK objects
        self._client = None  # type: ignore[var-annotated]
        self._audio_if: _LiveKitAudioInterface | None = None

        # Audio input resampler to 16k mono
        self._resampler: rtc.AudioResampler | None = None

        # Active generation state (created lazily on first agent output)
        self._gen_audio_q: asyncio.Queue[rtc.AudioFrame] | None = None
        self._gen_text_q: asyncio.Queue[str] | None = None
        self._gen_func_q: asyncio.Queue[llm.FunctionCall] | None = None
        self._gen_close_sent = False
        self._gen_idle_handle: asyncio.TimerHandle | None = None
        self._gen_idle_timeout_s = 0.8  # close generation after idle
        self._loop = asyncio.get_event_loop()
        # Future used by generate_reply to resolve on first generation
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        # Client tools bridging ElevenLabs <-> LiveKit
        self._client_tools = None  # created at start
        self._registered_tool_names: set[str] = set()
        # Pending tool results (bridge FunctionCall -> FunctionCallOutput)
        self._pending_tool_results: dict[str, concurrent.futures.Future[Any]] = {}
        self._pending_tool_lock = threading.Lock()
        # Primary timeout for tool bridging; configurable via env
        self._tool_timeout_s: float = float(os.getenv("LIVEKIT_ELEVENLABS_TOOL_TIMEOUT_S", "75.0"))
        # Extra grace for near-deadline results (helps avoid races)
        self._tool_timeout_grace_s: float = float(os.getenv("LIVEKIT_ELEVENLABS_TOOL_TIMEOUT_GRACE_S", "8.0"))
        # Dedup maps to coalesce duplicate invocations while one is in-flight
        # key -> call_id, where key = f"{tool}:{json.dumps(sorted params)}"
        self._pending_tool_by_key: dict[str, str] = {}
        # call_id -> key (for cleanup on resolution)
        self._call_id_to_key: dict[str, str] = {}
        # call_id -> perf_counter() timestamp when invocation was emitted
        self._tool_call_start_ts: dict[str, float] = {}
        # feature toggle: deduplicate identical tool invocations while one is in-flight
        self._dedup_enable: bool = os.getenv("LIVEKIT_ELEVENLABS_DEDUP_TOOLS", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Conversation initiation data (can be updated before session starts)
        self._init_dynamic_vars: dict[str, Any] | None = (
            realtime_model._opts.dynamic_variables.copy() if realtime_model._opts.dynamic_variables else None
        )
        self._init_config_override: dict[str, Any] | None = (
            realtime_model._opts.conversation_config_override.copy() if realtime_model._opts.conversation_config_override else None
        )
        self._init_extra_body: dict[str, Any] | None = (
            realtime_model._opts.extra_body.copy() if realtime_model._opts.extra_body else None
        )
        logger.debug("Initialized ElevenLabs RealtimeSession scaffold")

    # Properties
    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    # Update methods
    async def update_instructions(self, instructions: str) -> None:
        # TODO: map to SDK (if supported); otherwise, store and apply on reconnect
        logger.debug("update_instructions called (not yet wired): %s", instructions)

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self._chat_ctx = chat_ctx.copy()
        logger.debug("chat_ctx updated: %d items", len(self._chat_ctx.items))

        # Bridge: resolve pending tool results when FunctionCallOutput arrives from AgentActivity
        try:
            for item in self._chat_ctx.items:
                if isinstance(item, llm.FunctionCallOutput):
                    call_id = item.call_id
                    fut: concurrent.futures.Future[Any] | None = None
                    with self._pending_tool_lock:
                        fut = self._pending_tool_results.pop(call_id, None)
                        key = self._call_id_to_key.pop(call_id, None)
                        if key is not None:
                            self._pending_tool_by_key.pop(key, None)
                        started = self._tool_call_start_ts.pop(call_id, None)
                    if fut is None:
                        logger.debug(
                            "late tool result arrived with no pending waiter (call_id=%s, is_error=%s)",
                            call_id,
                            getattr(item, "is_error", None),
                        )
                        continue
                    if not fut.done():
                        # parse JSON output back to Python object when possible
                        parsed: Any
                        try:
                            parsed = json.loads(item.output)
                        except Exception:
                            # try to parse python-literal-like strings (e.g. "{'k': 'v'}")
                            try:
                                literal = ast.literal_eval(item.output)
                                # ensure it's JSON-serializable
                                json.dumps(literal)
                                parsed = literal
                            except Exception:
                                parsed = item.output

                        if item.is_error:
                            fut.set_exception(RuntimeError(str(parsed)))
                        else:
                            fut.set_result(parsed)
                        try:
                            latency_ms = None
                            if started is not None:
                                latency_ms = int((time.perf_counter() - started) * 1000)
                            logger.debug(
                                "resolved tool result (call_id=%s, is_error=%s, latency_ms=%s): %s",
                                call_id,
                                item.is_error,
                                latency_ms,
                                parsed,
                            )
                        except Exception:
                            # ensure logging doesn't break resolution flow
                            pass
        except Exception:
            logger.exception("failed to resolve pending tool results from chat_ctx")

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool | Any]) -> None:
        self._tools = llm.ToolContext(tools)
        logger.debug("tools updated: %d", len(self._tools.function_tools))
        # If session already running and client tools available, register only new tool names
        if self._conversation is not None and self._client_tools is not None:
            try:
                self._register_tools_into_client_tools(register_only_new=True)
            except Exception:
                logger.exception("failed to register updated tools into ElevenLabs ClientTools")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if tool_choice is not NOT_GIVEN:
            self._tool_choice = tool_choice
            logger.debug("tool_choice updated: %s", self._tool_choice)

    # Plugin-specific updates (dynamic variables & config overrides)
    async def update_dynamic_variables(self, dynamic_variables: dict[str, Any], *, merge: bool = True) -> None:
        if self._conversation is not None:
            logger.warning("dynamic_variables updated after session start; will apply on next session")
        if merge and self._init_dynamic_vars:
            self._init_dynamic_vars.update(dynamic_variables)
        else:
            self._init_dynamic_vars = dynamic_variables.copy()
        logger.debug("dynamic_variables set: keys=%s", list(self._init_dynamic_vars.keys()) if self._init_dynamic_vars else [])

    async def update_conversation_config_override(self, override: dict[str, Any], *, merge: bool = True) -> None:
        if self._conversation is not None:
            logger.warning("conversation_config_override updated after session start; will apply on next session")
        if merge and self._init_config_override:
            self._init_config_override.update(override)
        else:
            self._init_config_override = override.copy()
        logger.debug("conversation_config_override set")

    # Audio / Video input
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        # Ensure conversation started before sending any audio
        self._ensure_started()

        # Resample/remix to 16k mono if necessary
        frames: list[rtc.AudioFrame]
        if (
            frame.sample_rate != SAMPLE_RATE
            or frame.num_channels != NUM_CHANNELS
            or not self._resampler
        ):
            if not self._resampler:
                # LiveKit AudioResampler signature: (input_rate, output_rate, *, num_channels=1, quality=...)
                # Resample rate; we'll downmix to mono separately if needed.
                self._resampler = rtc.AudioResampler(
                    frame.sample_rate,
                    SAMPLE_RATE,
                    num_channels=frame.num_channels,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )
            frames = list(self._resampler.push(frame))
        else:
            frames = [frame]

        if not self._audio_if:
            return

        for f in frames:
            # Convert to bytes and feed to SDK input callback
            if f.num_channels == NUM_CHANNELS:
                pcm_bytes = f.data.tobytes()
            else:
                # Downmix to mono by averaging channels
                import array

                samples = f.samples_per_channel
                ch = f.num_channels
                pcm = memoryview(f.data).cast("h")
                out = array.array("h", [0] * samples)
                for i in range(samples):
                    base = i * ch
                    ssum = 0
                    for c in range(ch):
                        ssum += int(pcm[base + c])
                    # clamp to int16
                    val = ssum // ch
                    if val > 32767:
                        val = 32767
                    elif val < -32768:
                        val = -32768
                    out[i] = val
                pcm_bytes = out.tobytes()
            self._audio_if.push_input(pcm_bytes)

    def push_video(self, frame: rtc.VideoFrame) -> None:  # noqa: ARG002
        # ElevenLabs realtime is audio-first; ignore for now
        pass

    # Generation control
    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # For ElevenLabs, the server manages turn-taking. We return a Future
        # that resolves when the next generation is created.
        if self._pending_generation_fut is None or self._pending_generation_fut.done():
            self._pending_generation_fut = self._loop.create_future()
        # Ensure session is started unless explicitly ended
        if not self._ended:
            self._ensure_started()
        else:
            logger.debug("generate_reply called after end_conversation; suppressing session start")
        return self._pending_generation_fut

    def commit_audio(self) -> None:
        # ElevenLabs handles VAD/turn-taking; no explicit commit needed
        pass

    def clear_audio(self) -> None:
        # TODO: if SDK exposes clearing input buffer, wire it
        pass

    def interrupt(self) -> None:
        # Map to stopping current output stream locally; server handles barge-in on new input
        self._close_active_generation()

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:  # noqa: ARG002
        # ElevenLabs handles VAD/turn-taking; no explicit truncate needed
        pass

    def end_conversation(self) -> None:
        """
        Gracefully end the underlying ElevenLabs conversation session without
        closing this LiveKit RealtimeSession. A new conversation will be
        created automatically on next audio via _ensure_started().
        """
        try:
            if self._conversation is not None:
                self._conversation.end_session()
        except Exception:
            logger.exception("error while ending ElevenLabs conversation session")
        finally:
            # Mark as explicitly ended and ensure local teardown mirrors SDK callback behavior
            self._ended = True
            self._on_sdk_end_session()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            with contextlib.suppress(Exception):  # type: ignore[name-defined]
                await self._audio_task
        # End ElevenLabs conversation session if started
        try:
            if self._conversation is not None:
                self._conversation.end_session()
        except Exception:
            logger.exception("error while ending ElevenLabs conversation session")
        logger.debug("ElevenLabs RealtimeSession closed")

    # Optional override: forward user activity to SDK to keep session alive
    def start_user_activity(self) -> None:
        try:
            if self._conversation is not None:
                self._conversation.register_user_activity()
        except Exception:
            logger.exception("error while registering user activity on ElevenLabs session")

    # ---- Internal helpers ----
    def _ensure_started(self) -> None:
        if self._conversation is not None:
            return
        if self._ended:
            logger.debug("suppressing ElevenLabs session restart after end_conversation")
            return

        # Lazy imports to avoid hard dependency if not used
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs.conversational_ai.conversation import (
                Conversation,
                ConversationInitiationData,
                ClientTools as ELClientTools,
            )
        except Exception as e:  # pragma: no cover - import-time errors
            logger.exception("Failed to import ElevenLabs SDK: %s", e)
            raise

        # Audio interface bridging LiveKit <-> ElevenLabs
        self._audio_if = _LiveKitAudioInterface(
            on_output=self._on_sdk_output,
            on_interrupt=self._on_sdk_interrupt,
        )

        api_key = self.realtime_model._opts.api_key
        try:
            self._client = ElevenLabs(api_key=api_key) if api_key else ElevenLabs()
        except TypeError:
            # Fallback for older SDKs: API key picked from env automatically
            self._client = ElevenLabs()

        # Conversation callbacks
        def _on_agent_response(text: str) -> None:
            # Text chunks from the agent (best-effort stream)
            self._ensure_generation_open(modalities=["audio", "text"])
            if self._gen_text_q is not None:
                self._loop.call_soon_threadsafe(self._gen_text_q.put_nowait, text)

        def _on_agent_response_correction(original: str, corrected: str) -> None:
            # Emit corrected text as a new token
            _ = original  # unused
            _on_agent_response(corrected)

        def _on_user_transcript(text: str) -> None:
            # Best-effort: we don't know if final; mark as interim
            try:
                self._loop.call_soon_threadsafe(
                    self.emit,
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=str(uuid.uuid4()), transcript=text, is_final=False
                    ),
                )
            except Exception:
                logger.exception("error scheduling user transcript event")

        requires_auth = bool(api_key)
        try:
            # Build initiation data, including user_id (needed for handshake payload)
            init_cfg = ConversationInitiationData(
                extra_body=self._init_extra_body,
                conversation_config_override=self._init_config_override,
                dynamic_variables=self._init_dynamic_vars,
                user_id=self.realtime_model._opts.user_id,
            )

            # Prepare client tools and register LiveKit tools
            self._client_tools = ELClientTools()
            self._registered_tool_names.clear()
            self._register_tools_into_client_tools(register_only_new=False)

            self._conversation = Conversation(
                self._client,
                self.realtime_model._opts.agent_id,
                self.realtime_model._opts.user_id,
                requires_auth=requires_auth,
                audio_interface=self._audio_if,
                config=init_cfg,
                client_tools=self._client_tools,
                callback_agent_response=_on_agent_response,
                callback_agent_response_correction=_on_agent_response_correction,
                callback_user_transcript=_on_user_transcript,
                callback_latency_measurement=None,
                callback_end_session=self._on_sdk_end_session,
            )
            self._conversation.start_session()
        except Exception:
            logger.exception("failed to start ElevenLabs conversation session")
            raise

    # Internal: register LiveKit tools into ElevenLabs ClientTools
    def _register_tools_into_client_tools(self, *, register_only_new: bool) -> None:
        if self._client_tools is None:
            return
        tools_map = self._tools.function_tools if self._tools else {}
        for name, tool in tools_map.items():
            if register_only_new and name in self._registered_tool_names:
                continue
            handler = self._make_tool_handler(tool, name)
            try:
                # Always register sync handlers; async tools are executed on our asyncio loop
                self._client_tools.register(name, handler, is_async=False)
                self._registered_tool_names.add(name)
                logger.debug("registered client tool: %s (async=%s)", name, asyncio.iscoroutinefunction(tool))
            except Exception as e:
                # duplicate or invalid handlers
                logger.warning("skipped registering tool '%s': %s", name, e)

    def _make_tool_handler(self, tool: Any, tool_name: str) -> Callable[[dict[str, Any]], Any]:
        RESERVED_KEYS = {"tool_call_id"}

        def _sanitize(p: dict[str, Any]) -> dict[str, Any]:
            if not p:
                return {}
            return {k: v for k, v in p.items() if k not in RESERVED_KEYS}

        def _to_json_safe(obj: Any) -> Any:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                pass

            # dataclasses -> dict
            if dataclasses.is_dataclass(obj):
                return _to_json_safe(dataclasses.asdict(obj))

            # pydantic models / similar
            if hasattr(obj, "model_dump"):
                try:
                    return _to_json_safe(obj.model_dump())
                except Exception:
                    pass

            # containers
            if isinstance(obj, dict):
                return {str(k): _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_to_json_safe(v) for v in obj]

            # bytes -> utf-8 or base64 string
            if isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")
                except Exception:
                    return base64.b64encode(obj).decode("ascii")

            # primitives or fallback to str
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)

        def _to_provider_result(obj: Any) -> str:
            """Return a plain string result for the provider.
            ElevenLabs may expect tool results as strings; convert any object accordingly.
            """
            if isinstance(obj, str):
                return obj
            try:
                # Prefer JSON text representation to preserve structure
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                return str(obj)

        def _call_tool_sync(params: dict[str, Any]) -> Any:
            # NOTE: We do NOT execute the tool here. We bridge the provider's tool call
            # to LiveKit's AgentActivity by emitting a FunctionCall into the function stream
            # and then blocking until the corresponding FunctionCallOutput is provided via
            # update_chat_ctx(). This preserves the canonical RunContext injection path.
            clean = _sanitize(params)
            # Determine call id from params or generate one
            call_id = str(params.get("tool_call_id") or uuid.uuid4())
            # Build a dedup key from tool name and canonicalized params
            try:
                dedup_key = f"{tool_name}:{json.dumps(clean, sort_keys=True, separators=(',', ':'))}"
            except Exception:
                # fallback if params are not JSON serializable; use str()
                dedup_key = f"{tool_name}:{str(clean)}"
            logger.debug(
                "client tool '%s' invoked (call_id=%s) with params: %s", tool_name, call_id, clean
            )

            try:
                # Check for an in-flight identical invocation and coalesce if found
                if self._dedup_enable:
                    with self._pending_tool_lock:
                        existing_call_id = self._pending_tool_by_key.get(dedup_key)
                        existing_fut = (
                            self._pending_tool_results.get(existing_call_id)
                            if existing_call_id is not None
                            else None
                        )
                        existing_started = (
                            self._tool_call_start_ts.get(existing_call_id)
                            if existing_call_id is not None
                            else None
                        )
                        if existing_fut is not None and not existing_fut.done():
                            logger.debug(
                                "coalescing duplicate tool '%s' (call_id=%s) into existing (call_id=%s)",
                                tool_name,
                                call_id,
                                existing_call_id,
                            )
                            try:
                                # use remaining time budget from the original call start
                                remaining = self._tool_timeout_s
                                if existing_started is not None:
                                    elapsed = time.perf_counter() - existing_started
                                    remaining = max(0.1, self._tool_timeout_s - elapsed)
                                result_obj = existing_fut.result(timeout=remaining)
                                # ensure provider-friendly string
                                result_obj = _to_provider_result(_to_json_safe(result_obj))
                                logger.debug(
                                    "duplicate tool '%s' returning bridged result from existing (call_id=%s)",
                                    tool_name,
                                    existing_call_id,
                                )
                                return result_obj
                            except concurrent.futures.TimeoutError:
                                # final grace window to catch very late resolutions
                                try:
                                    result_obj = existing_fut.result(timeout=self._tool_timeout_grace_s)
                                    result_obj = _to_provider_result(_to_json_safe(result_obj))
                                    logger.debug(
                                        "duplicate tool '%s' received result in grace window (call_id=%s)",
                                        tool_name,
                                        existing_call_id,
                                    )
                                    return result_obj
                                except Exception as e:
                                    logger.error(
                                        "duplicate tool '%s' bridged error in grace window (call_id=%s): %s",
                                        tool_name,
                                        existing_call_id,
                                        e,
                                    )
                                    return f"error: {str(e)} (call_id={existing_call_id})"
                                except concurrent.futures.TimeoutError:
                                    logger.error(
                                        "duplicate tool '%s' timed out waiting for existing result (call_id=%s)",
                                        tool_name,
                                        existing_call_id,
                                    )
                                    return "error: tool execution timed out"
                            except Exception as e:
                                # Any error propagated from the original call
                                logger.error(
                                    "duplicate tool '%s' bridged error from existing (call_id=%s): %s",
                                    tool_name,
                                    existing_call_id,
                                    e,
                                )
                                return f"error: {str(e)} (call_id={existing_call_id})"
                        elif existing_call_id is not None:
                            # stale mapping; clean it up
                            self._pending_tool_by_key.pop(dedup_key, None)

                # Prepare waiter BEFORE emitting the FunctionCall to avoid a race where
                # update_chat_ctx resolves the result before the future is registered.
                fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
                with self._pending_tool_lock:
                    self._pending_tool_results[call_id] = fut
                    if self._dedup_enable:
                        self._pending_tool_by_key[dedup_key] = call_id
                        self._call_id_to_key[call_id] = dedup_key
                    self._tool_call_start_ts[call_id] = time.perf_counter()

                # Emit FunctionCall into the generation function stream and wait for
                # the bridged result via update_chat_ctx(), so tools receive RunContext.
                self._ensure_generation_open(modalities=["text"])  # type: ignore[arg-type]
                if self._gen_func_q is not None:
                    fnc = llm.FunctionCall(
                        call_id=call_id,
                        name=tool_name,
                        arguments=json.dumps(clean),
                    )
                    self._loop.call_soon_threadsafe(self._gen_func_q.put_nowait, fnc)
                else:
                    logger.warning("function queue is not available when invoking tool '%s'", tool_name)

                try:
                    result_obj = fut.result(timeout=self._tool_timeout_s)
                    result_obj = _to_provider_result(_to_json_safe(result_obj))
                    logger.debug(
                        "client tool '%s' completed (call_id=%s) with bridged result: %s",
                        tool_name,
                        call_id,
                        result_obj,
                    )
                    return result_obj
                except concurrent.futures.TimeoutError:
                    # give a small grace window to catch very late results
                    try:
                        result_obj = fut.result(timeout=self._tool_timeout_grace_s)
                        result_obj = _to_provider_result(_to_json_safe(result_obj))
                        logger.debug(
                            "client tool '%s' received result in grace window (call_id=%s): %s",
                            tool_name,
                            call_id,
                            result_obj,
                        )
                        return result_obj
                    except concurrent.futures.TimeoutError:
                        pass
                    with self._pending_tool_lock:
                        self._pending_tool_results.pop(call_id, None)
                        key = self._call_id_to_key.pop(call_id, None)
                        if key is not None:
                            self._pending_tool_by_key.pop(key, None)
                        self._tool_call_start_ts.pop(call_id, None)
                    logger.error("tool '%s' timed out waiting for result (call_id=%s)", tool_name, call_id)
                    return "error: tool execution timed out"
                except Exception as e:
                    # Propagate structured error to provider instead of generic bridging failure
                    logger.error(
                        "client tool '%s' raised bridged error (call_id=%s): %s",
                        tool_name,
                        call_id,
                        e,
                    )
                    return f"error: {str(e)} (call_id={call_id})"
            except Exception:
                logger.exception(
                    "error while bridging tool '%s' invocation (call_id=%s)", tool_name, call_id
                )
                return {"error": "tool bridging failed"}

        return _call_tool_sync

    def _on_sdk_output(self, audio: bytes) -> None:
        # Called from ElevenLabs thread; schedule into asyncio loop
        try:
            self._ensure_generation_open(modalities=["audio"])  # type: ignore[arg-type]

            # Create AudioFrame from PCM16 mono @ 16kHz
            samples = len(audio) // 2
            frame = rtc.AudioFrame(
                audio,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples,
            )

            if self._gen_audio_q is not None:
                self._loop.call_soon_threadsafe(self._gen_audio_q.put_nowait, frame)

            # Restart idle timer
            if self._gen_idle_handle and not self._gen_idle_handle.cancelled():
                self._gen_idle_handle.cancel()
            self._gen_idle_handle = self._loop.call_later(
                self._gen_idle_timeout_s, self._close_active_generation
            )
        except Exception:
            logger.exception("error while handling SDK output audio")

    def _on_sdk_interrupt(self) -> None:
        # Called by SDK to stop audio output (barge-in)
        self._close_active_generation()
        try:
            self._loop.call_soon_threadsafe(
                self.emit, "input_speech_started", llm.InputSpeechStartedEvent()
            )
        except Exception:
            logger.exception("error scheduling input_speech_started")

    def _on_sdk_end_session(self) -> None:
        # Conversation ended
        self._close_active_generation()
        try:
            if self._audio_if is not None:
                self._audio_if.stop()
        except Exception:
            logger.exception("error while stopping audio interface on session end")
        # Allow a new conversation to be created on next activity
        self._conversation = None

    def _ensure_generation_open(self, *, modalities: list[str]) -> None:
        if self._gen_audio_q is not None and self._gen_text_q is not None and self._gen_func_q is not None:
            return

        self._gen_audio_q = asyncio.Queue()
        self._gen_text_q = asyncio.Queue()
        self._gen_func_q = asyncio.Queue()
        self._gen_close_sent = False

        msg_id = str(uuid.uuid4())

        q_audio = self._gen_audio_q
        q_text = self._gen_text_q
        q_func = self._gen_func_q

        async def _audio_iter():
            assert q_audio is not None
            while True:
                item = await q_audio.get()
                if item is None:  # type: ignore[comparison-overlap]
                    break
                yield item

        async def _text_iter():
            assert q_text is not None
            while True:
                item = await q_text.get()
                if item is None:  # type: ignore[comparison-overlap]
                    break
                yield item

        async def _func_iter():
            assert q_func is not None
            while True:
                item = await q_func.get()
                if item is None:  # type: ignore[comparison-overlap]
                    break
                yield item

        # Compute modalities now and expose as a reusable awaitable (Future) so
        # it can be awaited multiple times safely by the consumer.
        mods: list[Literal["text", "audio"]]
        if "text" in modalities:
            mods = ["text", "audio"]
        else:
            mods = ["audio"]
        mods_future: asyncio.Future[list[Literal["text", "audio"]]] = self._loop.create_future()
        if not mods_future.done():
            mods_future.set_result(mods)

        async def _one_message():
            yield llm.MessageGeneration(
                message_id=msg_id,
                text_stream=_text_iter(),
                audio_stream=_audio_iter(),
                modalities=mods_future,
            )

        async def _function_stream():
            async for fc in _func_iter():
                yield fc

        def _emit_generation_created_on_loop():
            # Build the event on the loop thread and emit
            generation_ev = llm.GenerationCreatedEvent(
                message_stream=_one_message(),
                function_stream=_function_stream(),
                user_initiated=False,
            )
            # Resolve any pending generate_reply future
            if self._pending_generation_fut is not None and not self._pending_generation_fut.done():
                self._pending_generation_fut.set_result(generation_ev)
            self.emit("generation_created", generation_ev)

        # Schedule emission on the asyncio loop for thread-safety
        self._loop.call_soon_threadsafe(_emit_generation_created_on_loop)

    def _close_active_generation(self) -> None:
        if self._gen_close_sent:
            return
        self._gen_close_sent = True
        if self._gen_idle_handle and not self._gen_idle_handle.cancelled():
            self._gen_idle_handle.cancel()
            self._gen_idle_handle = None
        if self._gen_audio_q is not None:
            self._loop.call_soon_threadsafe(self._gen_audio_q.put_nowait, None)  # type: ignore[arg-type]
        if self._gen_text_q is not None:
            self._loop.call_soon_threadsafe(self._gen_text_q.put_nowait, None)  # type: ignore[arg-type]
        if self._gen_func_q is not None:
            self._loop.call_soon_threadsafe(self._gen_func_q.put_nowait, None)  # type: ignore[arg-type]
        # Allow future generations to open
        self._gen_audio_q = None
        self._gen_text_q = None
        self._gen_func_q = None


# ---- Audio Interface adapter ----
class _LiveKitAudioInterface:
    """Bridges LiveKit audio frames to ElevenLabs SDK AudioInterface.

    This class mirrors the subset of the SDK's AudioInterface used by Conversation.
    It will be constructed inside the RealtimeSession and receives input audio via
    push_input(); output() is called by the SDK with 16-bit PCM @ 16kHz.
    """

    # These attributes mirror DefaultAudioInterface constants for reference
    INPUT_FRAMES_PER_BUFFER = 4000  # 250ms @ 16kHz
    OUTPUT_FRAMES_PER_BUFFER = 1000  # 62.5ms @ 16kHz

    def __init__(
        self,
        *,
        on_output: Callable[[bytes], None],
        on_interrupt: Callable[[], None],
    ) -> None:
        self._on_output = on_output
        self._on_interrupt = on_interrupt
        self._input_cb: Callable[[bytes], None] | None = None

    # Methods expected by ElevenLabs AudioInterface
    def start(self, input_callback: Callable[[bytes], None]):
        self._input_cb = input_callback

    def stop(self):
        self._input_cb = None

    def output(self, audio: bytes):
        self._on_output(audio)

    def interrupt(self):
        self._on_interrupt()

    # Called by LiveKit session
    def push_input(self, audio: bytes) -> None:
        if self._input_cb is not None:
            try:
                self._input_cb(audio)
            except Exception:
                logger.exception("error while pushing input audio to SDK")
