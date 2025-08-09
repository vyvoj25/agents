from __future__ import annotations

import asyncio
import contextlib
import uuid
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
                auto_tool_reply_generation=False,
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

        # ElevenLabs SDK objects
        self._client = None  # type: ignore[var-annotated]
        self._audio_if: _LiveKitAudioInterface | None = None

        # Audio input resampler to 16k mono
        self._resampler: rtc.AudioResampler | None = None

        # Active generation state (created lazily on first agent output)
        self._gen_audio_q: asyncio.Queue[rtc.AudioFrame] | None = None
        self._gen_text_q: asyncio.Queue[str] | None = None
        self._gen_close_sent = False
        self._gen_idle_handle: asyncio.TimerHandle | None = None
        self._gen_idle_timeout_s = 0.8  # close generation after idle
        self._loop = asyncio.get_event_loop()
        # Future used by generate_reply to resolve on first generation
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        # Client tools bridging ElevenLabs <-> LiveKit
        self._client_tools = None  # created at start
        self._registered_tool_names: set[str] = set()
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
        # Ensure session is started
        self._ensure_started()
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
        # TODO: emulate by restarting conversation / updating context
        raise NotImplementedError("truncate not implemented for ElevenLabs yet")

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

        is_async = asyncio.iscoroutinefunction(tool)

        def _call_tool_sync(params: dict[str, Any]) -> Any:
            clean = _sanitize(params)
            logger.debug("client tool '%s' invoked (%s) with params: %s", tool_name, "async" if is_async else "sync", clean)
            try:
                import inspect

                def _call_kwargs() -> Any:
                    # Try kwargs directly
                    return tool(**clean)

                def _call_single_named_dict(param_name: str) -> Any:
                    return tool(**{param_name: clean})

                def _call_positional_dict() -> Any:
                    return tool(clean)

                # Prefer kwargs when signature can accept them
                sig = inspect.signature(tool)
                params_list = [p for p in sig.parameters.values() if p.name != "self"]
                has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params_list)
                # Determine if all provided keys are acceptable as kwargs
                acceptable_kw_names = {p.name for p in params_list if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )}
                can_use_kwargs = has_var_kw or all(k in acceptable_kw_names for k in clean.keys())

                # Try kwargs path first if feasible
                if can_use_kwargs:
                    try:
                        if is_async:
                            coro = _call_kwargs()
                            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                            result = fut.result()
                        else:
                            result = _call_kwargs()
                        logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                        return result
                    except TypeError as e:
                        logger.debug("kwargs call failed for tool '%s': %s", tool_name, e)

                # Next, try mixed binding: positional in declared order + keyword-only
                try:
                    used_keys: set[str] = set()
                    args_list: list[Any] = []
                    kw_only: dict[str, Any] = {}
                    for p in params_list:
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            if p.name in clean:
                                args_list.append(clean[p.name])
                                used_keys.add(p.name)
                        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
                            if p.name in clean:
                                kw_only[p.name] = clean[p.name]
                                used_keys.add(p.name)

                    # Remaining keys go into kwargs only if **kwargs is accepted
                    remaining = {k: v for k, v in clean.items() if k not in used_keys}
                    if has_var_kw:
                        kw_all = {**kw_only, **remaining}
                    else:
                        kw_all = kw_only

                    if is_async:
                        coro = tool(*args_list, **kw_all)
                        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                        result = fut.result()
                    else:
                        result = tool(*args_list, **kw_all)
                    logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                    return result
                except TypeError as e:
                    logger.debug("pos+kw binding failed for tool '%s': %s", tool_name, e)

                # Next, try passing the entire dict to a single named parameter
                # If there is exactly one non-self parameter, use its name
                non_self_params = [p for p in params_list if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )]
                tried_param_names: set[str] = set()
                if len(non_self_params) == 1:
                    pname = non_self_params[0].name
                    tried_param_names.add(pname)
                    try:
                        if is_async:
                            coro = _call_single_named_dict(pname)
                            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                            result = fut.result()
                        else:
                            result = _call_single_named_dict(pname)
                        logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                        return result
                    except TypeError as e:
                        logger.debug("single-dict param '%s' failed for tool '%s': %s", pname, tool_name, e)

                # Try common parameter names
                for pname in ("parameters", "params", "payload", "data"):
                    if pname in tried_param_names:
                        continue
                    try:
                        if is_async:
                            coro = _call_single_named_dict(pname)
                            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                            result = fut.result()
                        else:
                            result = _call_single_named_dict(pname)
                        logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                        return result
                    except TypeError:
                        continue

                # Finally, as a last resort, pass as positional dict only if the function allows positional args
                allows_positional = any(p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                        for p in params_list)
                if allows_positional:
                    if is_async:
                        coro = _call_positional_dict()
                        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
                        result = fut.result()
                    else:
                        result = _call_positional_dict()
                    logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                    return result

                # Give up with a clear error
                raise TypeError(f"unable to bind client tool '{tool_name}' with provided parameters: {list(clean.keys())}")
                logger.debug("client tool '%s' completed with result: %s", tool_name, result)
                return result
            except Exception:
                logger.exception("client tool '%s' raised an exception", tool_name)
                raise

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

    def _ensure_generation_open(self, *, modalities: list[str]) -> None:
        if self._gen_audio_q is not None and self._gen_text_q is not None:
            return

        self._gen_audio_q = asyncio.Queue()
        self._gen_text_q = asyncio.Queue()
        self._gen_close_sent = False

        msg_id = str(uuid.uuid4())

        async def _audio_iter():
            assert self._gen_audio_q is not None
            while True:
                item = await self._gen_audio_q.get()
                if item is None:  # type: ignore[comparison-overlap]
                    break
                yield item

        async def _text_iter():
            assert self._gen_text_q is not None
            while True:
                item = await self._gen_text_q.get()
                if item is None:  # type: ignore[comparison-overlap]
                    break
                yield item

        async def _modalities() -> list[Literal["text", "audio"]]:
            # If we received any text, both will be present; otherwise audio-only
            return [m for m in ["text" if "text" in modalities else None, "audio"] if m]  # type: ignore[list-item]

        async def _one_message():
            yield llm.MessageGeneration(
                message_id=msg_id,
                text_stream=_text_iter(),
                audio_stream=_audio_iter(),
                modalities=_modalities(),
            )

        async def _empty_func_stream():
            if False:
                yield None  # pragma: no cover

        def _emit_generation_created_on_loop():
            # Build the event on the loop thread and emit
            generation_ev = llm.GenerationCreatedEvent(
                message_stream=_one_message(),
                function_stream=_empty_func_stream(),
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
        # Allow future generations to open
        self._gen_audio_q = None
        self._gen_text_q = None
        # Reset for next generation
        self._gen_audio_q = None
        self._gen_text_q = None
        self._gen_close_sent = False


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
