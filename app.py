from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import base64
import tempfile
import os
import wave
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, AsyncGenerator
from faster_whisper import WhisperModel
from pydub import AudioSegment
import numpy as np
import httpx

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Faster-Whisper model (using tiny model for fastest real-time processing)
model = None

def load_whisper_model():
    global model
    if model is None:
        print("Loading Faster-Whisper model (tiny)...")
        # Use tiny model for speed, device="cpu" for compatibility
        # For better accuracy but slower, use "base" or "small"
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Model loaded!")
    return model

# Single-thread executor for Whisper so transcription doesn't block the event loop
_whisper_executor = ThreadPoolExecutor(max_workers=1)

def _transcribe_wav_sync(whisper_model, wav_path: str):
    """Synchronous transcription (run in thread). Returns (text,)."""
    segments, info = whisper_model.transcribe(
        wav_path,
        language="en",
        task="transcribe",
        beam_size=1,
        best_of=1,
        patience=1.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())
    return " ".join(text_parts).strip()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Clear conversation history when page loads
    clear_conversation_history()
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    whisper_model = load_whisper_model()

    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            audio_data = json.loads(data)

            # Ignore new_line_marker (client-only hint)
            if audio_data.get("type") == "new_line_marker":
                continue

            # Handle PCM audio data (raw audio, better for real-time)
            if audio_data.get("type") == "audio_pcm":
                try:
                    # Decode base64 PCM data
                    pcm_bytes = base64.b64decode(audio_data["data"])
                    sample_rate = audio_data.get("sampleRate", 16000)
                    channels = audio_data.get("channels", 1)
                    
                    # Check minimum size (at least 2KB for 2 seconds at 16kHz)
                    if len(pcm_bytes) < 2048:
                        print(f"PCM data too small ({len(pcm_bytes)} bytes), skipping")
                        continue
                    
                    # Convert PCM bytes to numpy array
                    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                    
                    # Normalize to float32 [-1, 1] for Whisper
                    audio_float = pcm_array.astype(np.float32) / 32768.0
                    
                    # Save to temporary WAV file
                    wav_path = None
                    try:
                        import wave
                        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

                        with wave.open(wav_path, 'wb') as wav_file:
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(pcm_bytes)

                        # Run Whisper in thread so event loop stays free (receive + Gemini can run)
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(
                            _whisper_executor,
                            _transcribe_wav_sync,
                            whisper_model,
                            wav_path
                        )

                        if text:
                            print(f"Transcription: {text}")
                            should_new_line = audio_data.get("shouldNewLine", False)
                            try:
                                await manager.send_personal_message(
                                    json.dumps({
                                        "type": "transcription",
                                        "text": text,
                                        "newLine": should_new_line
                                    }),
                                    websocket
                                )
                            except WebSocketDisconnect:
                                raise
                    finally:
                        # Clean up temporary file
                        if wav_path and os.path.exists(wav_path):
                            try:
                                os.unlink(wav_path)
                            except:
                                pass
                                
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"Error processing PCM audio: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        await manager.send_personal_message(
                            json.dumps({"type": "error", "message": f"Lỗi xử lý audio: {str(e)}"}),
                            websocket
                        )
                    except WebSocketDisconnect:
                        raise

            # Legacy WebM support (fallback)
            elif audio_data.get("type") == "audio":
                # Decode base64 audio data
                try:
                    audio_bytes = base64.b64decode(audio_data["data"])
                except Exception as e:
                    print(f"Error decoding base64: {e}")
                    continue
                
                # Check minimum file size (at least 2KB to ensure valid webm file with header)
                if len(audio_bytes) < 2048:
                    print(f"Audio data too small ({len(audio_bytes)} bytes), skipping")
                    continue
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Check file size after writing (need at least 2KB for valid webm)
                    file_size = os.path.getsize(tmp_path)
                    if file_size < 2048:
                        print(f"File too small after writing ({file_size} bytes), skipping")
                        continue
                    
                    # Convert webm to wav using pydub
                    # Try different formats in case webm doesn't work
                    audio = None
                    formats_to_try = ["webm", "ogg", "opus"]
                    
                    for fmt in formats_to_try:
                        try:
                            audio = AudioSegment.from_file(tmp_path, format=fmt)
                            print(f"Successfully loaded audio as {fmt}, duration: {len(audio)}ms")
                            break
                        except Exception as e:
                            # Only print error if it's the last format
                            if fmt == formats_to_try[-1]:
                                print(f"Failed to load as {fmt}: {str(e)[:200]}")
                            continue
                    
                    if audio is None:
                        print("Could not load audio file in any format, skipping")
                        continue
                    
                    # Check if audio has content (at least 200ms for valid transcription)
                    if len(audio) < 200:
                        print(f"Audio too short ({len(audio)}ms), skipping")
                        continue
                    
                    # Export to wav format that Whisper can process
                    wav_path = tmp_path.replace(".webm", ".wav")
                    audio.export(wav_path, format="wav")

                    # Run Whisper in thread so event loop stays free
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        _whisper_executor,
                        _transcribe_wav_sync,
                        whisper_model,
                        wav_path
                    )

                    if text:
                        try:
                            await manager.send_personal_message(
                                json.dumps({"type": "transcription", "text": text}),
                                websocket
                            )
                        except WebSocketDisconnect:
                            raise
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        await manager.send_personal_message(
                            json.dumps({"type": "error", "message": f"Lỗi xử lý audio: {str(e)}"}),
                            websocket
                        )
                    except WebSocketDisconnect:
                        raise
                finally:
                    # Clean up temporary files
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    wav_path = tmp_path.replace(".webm", ".wav")
                    if os.path.exists(wav_path):
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

# --- Chat API (separate from transcript WebSocket) ---
GEMINI_API_KEY = "AIzaSyD8dh5XHrE1PfS9fY-h_6zdEAuGLJyrQvE"

# Conversation history file
HISTORY_FILE = "conversation_history.json"

SYSTEM_PROMPT = """You are an AI Interview Coach helping candidates prepare concise, professional interview responses.

## CRITICAL INSTRUCTIONS - READ CAREFULLY:

### Response Length (MANDATORY):
- **MAXIMUM**: ONE single paragraph only (3-5 sentences, 80-120 words)
- **NO EXCEPTIONS**: Never write more than one paragraph
- **Be EXTREMELY concise**: Every word must add value
- If you write more than 120 words, you have FAILED

### Response Structure:
1. **Direct Answer** (1 sentence): Answer the question immediately - no preamble
2. **Key Evidence** (2-3 sentences): ONE specific example from resume with concrete results/metrics
3. **Connection to Role** (1 sentence): Brief tie-back to the job requirements

### Style Guidelines:
- **CLEAR & SIMPLE**: Use everyday professional language - no corporate jargon
- **CONFIDENT**: Active voice, strong verbs (achieved, led, built, improved)
- **AUTHENTIC**: Conversational tone that sounds natural when spoken
- **SPECIFIC**: Include numbers, percentages, or measurable outcomes

### Content Rules:
- ✅ Base examples ONLY on resume details provided
- ✅ Connect skills to job description requirements
- ✅ Review conversation history - don't repeat previous examples
- ✅ Use STAR method (Situation-Task-Action-Result) compressed into 2-3 sentences
- ❌ NO bullet points, NO lists, NO multiple paragraphs
- ❌ NO vague statements without evidence
- ❌ NO clichés ("I'm a perfectionist", "I work too hard")

### Common Question Types:
- **Behavioral ("Tell me about a time...")**: Use STAR - emphasize Action & Result
- **Technical**: Briefly explain approach + specific achievement
- **"Why this role/company?"**: Connect 1-2 resume strengths to JD requirements
- **Strengths/Weaknesses**: Name it + brief example + what you learned

## FINAL CHECK BEFORE RESPONDING:
1. ✓ Is it ONE paragraph only?
2. ✓ Is it under 120 words?
3. ✓ Does it directly answer the question?
4. ✓ Did I include specific evidence from the resume?
5. ✓ Can it be spoken naturally in 60 seconds?

If ANY answer is NO, rewrite to be MORE CONCISE.

**Remember**: The candidate needs a quick, memorable response they can deliver smoothly in an interview. BREVITY and CLARITY are more important than completeness.
"""

# Helper functions for conversation history
def load_conversation_history():
    """Load conversation history from JSON file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []

def save_conversation_history(history):
    """Save conversation history to JSON file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

def add_to_history(role, content):
    """Add a message to conversation history"""
    history = load_conversation_history()
    history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })
    save_conversation_history(history)
    return history

def clear_conversation_history():
    """Clear all conversation history"""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return True
    except Exception as e:
        print(f"Error clearing conversation history: {e}")
        return False

def format_history_for_gemini(history):
    """Convert our history format to Gemini's expected format"""
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "human" else "model"
        gemini_history.append({
            "role": role,
            "parts": [msg["content"]]
        })
    return gemini_history

def _run_gemini_into_queue(question: str, jd: str, resume_text: Optional[str], image_base64: Optional[str], image_mime: str, out_queue: queue.Queue):
    """Sync: run Gemini streaming and put ("chunk", delta) or ("done", None) or ("error", msg) into out_queue."""
    try:
        from google import genai
        from google.genai import types
        
        # Create client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Load conversation history
        history = load_conversation_history()
        
        # Build context with resume and JD
        context_parts = []
        if resume_text and resume_text.strip():
            context_parts.append(f"**Resume:**\n{resume_text.strip()}\n")
        if jd and jd.strip():
            context_parts.append(f"**Job Description:**\n{jd.strip()}\n")

        # Create system instruction with context
        system_instruction = SYSTEM_PROMPT
        if context_parts:
            system_instruction += "\n## Context for this interview:\n" + "\n".join(context_parts)

        # Convert history to contents format
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "human" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            ))
        
        # Add current question
        question_parts = [types.Part.from_text(text=question)]
        if image_base64:
            try:
                # Decode base64 to bytes for inline data
                import base64 as b64
                image_bytes = b64.b64decode(image_base64)
                question_parts.append(types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_mime
                ))
            except Exception as e:
                print(f"Warning: Could not add image: {e}")
        
        contents.append(types.Content(
            role="user",
            parts=question_parts
        ))

        # Generate content with streaming
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        
        accumulated_text = ""
        
        # Check if response has text attribute
        if hasattr(response, 'text'):
            accumulated_text = response.text
            out_queue.put(("chunk", accumulated_text))
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            accumulated_text += part.text
                            out_queue.put(("chunk", part.text))
        
        # Save conversation to history (send accumulated_text back for saving)
        out_queue.put(("done", accumulated_text))
        
        # Close client
        client.close()
        
    except ImportError as e:
        out_queue.put(("error", f"Google Gen AI SDK not available: {str(e)}"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_queue.put(("error", str(e)))

async def stream_chat_response(question: str, jd: str, resume_text: Optional[str], image_base64: Optional[str], image_mime: str):
    """Async generator: yields NDJSON lines for chat stream."""
    out_queue = queue.Queue()
    thread = threading.Thread(
        target=_run_gemini_into_queue,
        args=(question, jd, resume_text, image_base64, image_mime, out_queue)
    )
    thread.start()
    loop = asyncio.get_event_loop()
    
    accumulated_response = ""

    while True:
        kind, payload = await loop.run_in_executor(None, out_queue.get)
        if kind == "chunk":
            accumulated_response += payload
            yield json.dumps({"type": "chunk", "text": payload}) + "\n"
        elif kind == "done":
            # Use accumulated response from streaming, fallback to payload if needed
            final_response = accumulated_response if accumulated_response else payload
            
            # Add human question to history
            add_to_history("human", question)
            # Add AI response to history (full response from Gemini)
            if final_response:
                add_to_history("AI", final_response)
            
            yield json.dumps({"type": "done"}) + "\n"
            break
        elif kind == "error":
            yield json.dumps({"type": "error", "message": payload}) + "\n"
            break

@app.post("/api/chat-stream")
async def chat_stream(request: Request):
    """Chat API: accept question (and optional jd, resume, image), stream Gemini response as NDJSON."""
    try:
        body = await request.json()
        question = (body.get("question") or "").strip()
        jd = (body.get("jd") or "").strip()
        resume_text = (body.get("resume") or "").strip() or None
        image_base64 = body.get("image")
        image_mime = body.get("image_mime") or "image/jpeg"

        if not question:
            return JSONResponse({"error": "Question is required"}, status_code=400)

        return StreamingResponse(
            stream_chat_response(question, jd, resume_text, image_base64, image_mime),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/clear-history")
async def clear_history_endpoint():
    """Clear conversation history"""
    try:
        success = clear_conversation_history()
        if success:
            return {"status": "success", "message": "Conversation history cleared"}
        else:
            return JSONResponse({"error": "Failed to clear history"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/get-history")
async def get_history_endpoint():
    """Get current conversation history"""
    try:
        history = load_conversation_history()
        return {"history": history, "count": len(history)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Mount static files
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

