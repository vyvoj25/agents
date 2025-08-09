API REFERENCE
Agent WebSockets
WSS

wss://api.elevenlabs.io
/v1/convai/conversation
Handshake
URL	wss://api.elevenlabs.io//v1/convai/conversation
Method	GET
Status	101 Switching Protocols
Try it
Messages

{"type":"conversation_initiation_client_data","conversation_config_override":{"agent":{"prompt":{"prompt":"You are a helpful customer support agent named Alexis."},"first_message":"Hi, I'm Alexis from ElevenLabs support. How can I help you today?","language":"en"},"tts":{"voice_id":"21m00Tcm4TlvDq8ikWAM"}},"custom_llm_extra_body":{"temperature":0.7,"max_tokens":150},"dynamic_variables":{"user_name":"John","account_type":"premium"}}
publish

{
  "type": "conversation_initiation_client_data",
  "conversation_config_override": {
    "agent": {
      "prompt": {
        "prompt": "You are a helpful customer support agent named Alexis."
      },
      "first_message": "Hi, I'm Alexis from ElevenLabs support. How can I help you today?",
      "language": "en"
    },
    "tts": {
      "voice_id": "21m00Tcm4TlvDq8ikWAM"
    }
  },
  "custom_llm_extra_body": {
    "temperature": 0.7,
    "max_tokens": 150
  },
  "dynamic_variables": {
    "user_name": "John",
    "account_type": "premium"
  }
}

{"type":"conversation_initiation_metadata","conversation_initiation_metadata_event":{"conversation_id":"conv_123456789","agent_output_audio_format":"pcm_16000","user_input_audio_format":"pcm_16000"}}
subscribe

{
  "type": "conversation_initiation_metadata",
  "conversation_initiation_metadata_event": {
    "conversation_id": "conv_123456789",
    "agent_output_audio_format": "pcm_16000",
    "user_input_audio_format": "pcm_16000"
  }
}

{"user_audio_chunk":"base64EncodedAudioData=="}
publish

{
  "user_audio_chunk": "base64EncodedAudioData=="
}

{"type":"vad_score","vad_score_event":{"vad_score":0.95}}
subscribe

{
  "type": "vad_score",
  "vad_score_event": {
    "vad_score": 0.95
  }
}

{"type":"user_transcript","user_transcription_event":{"user_transcript":"I need help with my voice cloning project."}}
subscribe

{
  "type": "user_transcript",
  "user_transcription_event": {
    "user_transcript": "I need help with my voice cloning project."
  }
}

{"type":"internal_tentative_agent_response","tentative_agent_response_internal_event":{"tentative_agent_response":"I'd be happy to help with your voice cloning project..."}}
subscribe

{
  "type": "internal_tentative_agent_response",
  "tentative_agent_response_internal_event": {
    "tentative_agent_response": "I'd be happy to help with your voice cloning project..."
  }
}

{"type":"agent_response","agent_response_event":{"agent_response":"I'd be happy to help with your voice cloning project. Could you tell me what specific aspects you need assistance with?"}}
subscribe

{
  "type": "agent_response",
  "agent_response_event": {
    "agent_response": "I'd be happy to help with your voice cloning project. Could you tell me what specific aspects you need assistance with?"
  }
}

{"type":"audio","audio_event":{"audio_base_64":"base64EncodedAudioResponse==","event_id":1}}
subscribe

{
  "type": "audio",
  "audio_event": {
    "audio_base_64": "base64EncodedAudioResponse==",
    "event_id": 1
  }
}

{"type":"ping","ping_event":{"event_id":12345,"ping_ms":50}}
subscribe

{
  "type": "ping",
  "ping_event": {
    "event_id": 12345,
    "ping_ms": 50
  }
}

{"type":"pong","event_id":12345}
publish

{
  "type": "pong",
  "event_id": 12345
}

{"type":"client_tool_call","client_tool_call":{"tool_name":"check_account_status","tool_call_id":"tool_call_123","parameters":{"user_id":"user_123"}}}
subscribe

{
  "type": "client_tool_call",
  "client_tool_call": {
    "tool_name": "check_account_status",
    "tool_call_id": "tool_call_123",
    "parameters": {
      "user_id": "user_123"
    }
  }
}

{"type":"client_tool_result","tool_call_id":"tool_call_123","result":"Account is active and in good standing","is_error":false}
publish

{
  "type": "client_tool_result",
  "tool_call_id": "tool_call_123",
  "result": "Account is active and in good standing",
  "is_error": false
}

{"type":"contextual_update","text":"User is viewing the pricing page"}
publish

{
  "type": "contextual_update",
  "text": "User is viewing the pricing page"
}

{"type":"user_message","text":"I would like to upgrade my account"}
publish

{
  "type": "user_message",
  "text": "I would like to upgrade my account"
}

{"type":"user_activity"}
publish

{
  "type": "user_activity"
}
Establish a WebSocket connection for real-time conversations with an AI agent.

Handshake
WSS

wss://api.elevenlabs.io
/v1/convai/conversation

Query parameters
agent_id
any
Required
The unique identifier for the voice to use in the TTS process.
Send
User Audio Chunk
object
Required

Hide 1 properties
user_audio_chunk
string
Optional
Base64-encoded audio data chunk from user input.

OR
Pong
object
Required

Hide 2 properties
type
"pong"
Required
event_id
integer
Optional
The ID of the ping event being responded to.
OR
Conversation Initiation Client Data
object
Required

Hide 4 properties
type
"conversation_initiation_client_data"
Required
conversation_config_override
object
Optional
Override settings for conversation behavior

Hide 2 properties
agent
object
Optional
Configuration for the AI agent's behavior

Hide 3 properties
prompt
object
Optional
System prompt configuration

Hide 1 properties
prompt
string
Optional
Custom system prompt to guide agent behavior.
first_message
string
Optional
Initial message the agent should use to start the conversation.
language
string
Optional
Preferred language code for the conversation.
tts
object
Optional
Text-to-speech configuration


Hide 1 properties
voice_id
string
Optional
ID of the voice to use for text-to-speech synthesis.

custom_llm_extra_body
object
Optional
Additional LLM configuration parameters

Hide 2 properties
temperature
double
Optional
Temperature parameter controlling response randomness.
max_tokens
integer
Optional
Maximum number of tokens allowed in LLM responses.
dynamic_variables
map from strings to strings or doubles or integers or booleans
Optional
Dictionary of dynamic variables to be used in the conversation. Keys are the dynamic variable names which must be strings, values can be strings, numbers, integers, or booleans.

Hide 4 variants
string
Required
OR
double
Required
OR
integer
Required
OR
boolean
Required
OR
Client Tool Result
object
Required

Hide 4 properties
type
"client_tool_result"
Required
tool_call_id
string
Optional
Unique identifier of the tool call being responded to.
result
string
Optional
Result data from the tool execution.
is_error
boolean
Optional
Flag indicating if the tool execution encountered an error.
OR
Contextual Update
object
Required

Hide 2 properties
type
"contextual_update"
Required
text
string
Required
Contextual information to be added to the conversation state.
OR
User Message
object
Required

Hide 2 properties
type
"user_message"
Required
text
string
Optional
Text message content from the user.
OR
User Activity
object
Required

Hide 1 properties
type
"user_activity"
Required
Receive
Conversation Initiation Metadata
object
Required

Hide 2 properties
type
"conversation_initiation_metadata"
Optional
Defaults to conversation_initiation_metadata
conversation_initiation_metadata_event
object
Optional
Initial conversation metadata

Hide 3 properties
conversation_id
string
Optional
Unique identifier for the conversation session.
agent_output_audio_format
string
Optional
Audio format specification for agent's speech output.
user_input_audio_format
string
Optional
Audio format specification for user's speech input.
OR
User Transcript
object
Required

Hide 2 properties
type
"user_transcript"
Optional
Defaults to user_transcript
user_transcription_event
object
Optional
Transcription event data

Hide 1 properties
user_transcript
string
Optional
Transcribed text from user's speech input.
OR
Agent Response
object
Required

Hide 2 properties
type
"agent_response"
Required
agent_response_event
object
Optional
Agent response event data

Hide 1 properties
agent_response
string
Required
Text content of the agent's response.
OR
Agent Response Correction
object
Required

Hide 2 properties
type
"agent_response_correction"
Required
agent_response_correction_event
object
Optional
Agent response correction event data

Hide 2 properties
original_agent_response
string
Required
The original agent response before correction
corrected_agent_response
string
Required
The corrected agent response after truncation or interruption
OR
Audio Response
object
Required

Hide 2 properties
type
"audio"
Required
audio_event
object
Optional
Audio event data

Hide 2 properties
audio_base_64
string
Optional
Base64-encoded audio data of agent’s speech.

event_id
integer
Optional
Sequential identifier for the audio chunk.
OR
Interruption
object
Required

Hide 2 properties
type
"interruption"
Required
interruption_event
object
Optional
Interruption event data

Hide 1 properties
event_id
integer
Optional
ID of the event that was interrupted.
OR
Ping
object
Required

Hide 2 properties
type
"ping"
Required
ping_event
object
Optional
Ping event data

Hide 2 properties
event_id
integer
Optional
Unique identifier for the ping event.
ping_ms
integer
Optional
Measured round-trip latency in milliseconds.

OR
Client Tool Call
object
Required

Hide 2 properties
type
"client_tool_call"
Required
client_tool_call
object
Optional
Tool call request data

Hide 3 properties
tool_name
string
Optional
Identifier of the tool to be executed.
tool_call_id
string
Optional
Unique identifier for this tool call request.
parameters
map from strings to any
Optional
Tool-specific parameters for the execution request.

OR
Contextual Update
object
Required

Hide 2 properties
type
"contextual_update"
Required
text
string
Required
Contextual information to be added to the conversation state.
OR
VAD Score
object
Required

Hide 2 properties
type
"vad_score"
Required
vad_score_event
object
Optional
VAD event data

Hide 1 properties
vad_score
double
Required
>=0
<=1
Voice activity detection confidence score between 0 and 1
OR
Internal Tentative Agent Response
object
Required

Hide 2 properties
type
"internal_tentative_agent_response"
Required
tentative_agent_response_internal_event
object
Optional
Preliminary event data containing agent's tentative response

Hide 1 properties
tentative_agent_response
string
Required
Preliminary text from the agent
