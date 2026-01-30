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

SYSTEM_PROMPT = """You are an interview assistant helping me answer questions.

Based on my resume help me craft responses that:

1. Keep answers SHORT and FOCUSED

2. Answer the question DIRECTLY without extra information

3. Use SIMPLE, CLEAR English words (avoid complex vocabulary)

4. Write in PARAGRAPH format (not bullet points)

5. Highlight my RELEVANT experience and achievements

6. Sound CONFIDENT but NATURAL

Format: Give me one concise paragraph answer that I can easily read and use.

Additional Guidelines:
- Base your answers on the information provided in the resume and job description
- If the question is about experience or skills, reference specific details from the resume
- If the question is about why you're a good fit, connect resume qualifications to job requirements
- Keep answers clear, confident, and interview-appropriate

"""

def _run_gemini_into_queue(question: str, jd: str, resume_text: Optional[str], image_base64: Optional[str], image_mime: str, out_queue: queue.Queue):
    """Sync: run Gemini streaming and put ("chunk", delta) or ("done", None) or ("error", msg) into out_queue."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        context_parts = []
        if resume_text and resume_text.strip():
            context_parts.append(f"Resume:\n{resume_text.strip()}\n")
        if jd and jd.strip():
            context_parts.append(f"Job Description:\n{jd.strip()}\n")

        full_prompt = SYSTEM_PROMPT
        if context_parts:
            full_prompt += "\n".join(context_parts) + "\n"
        full_prompt += f"\nQuestion: {question}\n\nPlease provide a helpful answer based on the above context."

        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        content_parts = [full_prompt]
        if image_base64:
            try:
                content_parts.append({"inline_data": {"mime_type": image_mime, "data": image_base64}})
            except Exception:
                pass

        response = model.generate_content(content_parts, stream=True)
        accumulated_text = ""

        try:
            chunks_iter = iter(response)
        except TypeError:
            chunks_iter = None

        if chunks_iter is not None:
            for chunk in chunks_iter:
                chunk_text = None
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif hasattr(chunk, 'parts') and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, 'text'):
                            chunk_text = part.text
                            break
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    chunk_text = part.text
                                    break
                            break

                if chunk_text:
                    new_text = chunk_text
                    if accumulated_text:
                        if new_text.startswith(accumulated_text):
                            delta = new_text[len(accumulated_text):]
                        else:
                            min_len = min(len(accumulated_text), len(new_text))
                            common_len = sum(1 for i in range(min_len) if accumulated_text[i] == new_text[i])
                            delta = new_text[common_len:]
                    else:
                        delta = new_text
                    if delta:
                        accumulated_text = new_text
                        out_queue.put(("chunk", delta))
        else:
            full_text = getattr(response, 'text', None)
            if callable(full_text):
                full_text = full_text() or ''
            else:
                full_text = full_text or ''
            if not full_text and hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                    full_text = ''.join(getattr(p, 'text', '') or '' for p in cand.content.parts)
            if full_text:
                out_queue.put(("chunk", full_text))
        out_queue.put(("done", None))
    except ImportError:
        out_queue.put(("error", "Google Generative AI SDK not available"))
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

    while True:
        kind, payload = await loop.run_in_executor(None, out_queue.get)
        if kind == "chunk":
            yield json.dumps({"type": "chunk", "text": payload}) + "\n"
        elif kind == "done":
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

async def process_gemini_streaming(question: str, jd: str, resume_text: Optional[str], websocket: WebSocket, image_base64: Optional[str] = None, image_mime: str = "image/jpeg", send_lock: Optional[asyncio.Lock] = None):
    """Process Gemini streaming and send chunks via WebSocket"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        system_prompt = """You are an interview assistant helping me answer questions.

Based on my resume help me craft responses that:

1. Keep answers SHORT and FOCUSED

2. Answer the question DIRECTLY without extra information

3. Use SIMPLE, CLEAR English words (avoid complex vocabulary)

4. Write in PARAGRAPH format (not bullet points)

5. Highlight my RELEVANT experience and achievements

6. Sound CONFIDENT but NATURAL

Format: Give me one concise paragraph answer that I can easily read and use.

Additional Guidelines:
- Base your answers on the information provided in the resume and job description
- If the question is about experience or skills, reference specific details from the resume
- If the question is about why you're a good fit, connect resume qualifications to job requirements
- Keep answers clear, confident, and interview-appropriate

"""
        context_parts = []
        if resume_text and resume_text.strip():
            context_parts.append(f"Resume:\n{resume_text.strip()}\n")
        if jd and jd.strip():
            context_parts.append(f"Job Description:\n{jd.strip()}\n")
        full_prompt = system_prompt
        if context_parts:
            full_prompt += "\n".join(context_parts) + "\n"
        full_prompt += f"\nQuestion: {question}\n\nPlease provide a helpful answer based on the above context."
        model_name = "gemini-2.5-flash"
        try:
            model = genai.GenerativeModel(model_name)
            content_parts = [full_prompt]
            if image_base64:
                try:
                    image_part = {"inline_data": {"mime_type": image_mime, "data": image_base64}}
                    content_parts.append(image_part)
                    print("Image added to Gemini request")
                except Exception as img_err:
                    print(f"Warning: Could not add image: {img_err}. Continuing without image.")
            
            print(f"Starting Gemini streaming via WebSocket...")
            response = model.generate_content(content_parts, stream=True)
            accumulated_text = ""
            chunk_count = 0

            try:
                response_stream = iter(response)
            except TypeError:
                response_stream = None

            if response_stream is not None:
                for chunk in response_stream:
                    chunk_count += 1
                    chunk_text = None
                    if hasattr(chunk, 'text'):
                        chunk_text = chunk.text
                    elif hasattr(chunk, 'parts') and chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text'):
                                chunk_text = part.text
                                break
                    elif hasattr(chunk, 'candidates') and chunk.candidates:
                        for candidate in chunk.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text'):
                                            chunk_text = part.text
                                            break
                    if chunk_text:
                        new_text = chunk_text
                        if accumulated_text:
                            if new_text.startswith(accumulated_text):
                                delta = new_text[len(accumulated_text):]
                            else:
                                min_len = min(len(accumulated_text), len(new_text))
                                common_len = sum(1 for i in range(min_len) if accumulated_text[i] == new_text[i])
                                delta = new_text[common_len:]
                        else:
                            delta = new_text
                        if delta:
                            accumulated_text = new_text
                            print(f"Chunk #{chunk_count}: Sending delta ({len(delta)} chars) via WebSocket")
                            if send_lock:
                                async with send_lock:
                                    await manager.send_personal_message(
                                        json.dumps({"type": "gemini_chunk", "chunk": delta}),
                                        websocket
                                    )
                            else:
                                await manager.send_personal_message(
                                    json.dumps({"type": "gemini_chunk", "chunk": delta}),
                                    websocket
                                )
            else:
                full_text = getattr(response, 'text', None)
                if callable(full_text):
                    full_text = full_text() or ''
                else:
                    full_text = full_text or ''
                if not full_text and hasattr(response, 'candidates') and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                        full_text = ''.join(getattr(p, 'text', '') or '' for p in cand.content.parts)
                if full_text:
                    if send_lock:
                        async with send_lock:
                            await manager.send_personal_message(
                                json.dumps({"type": "gemini_chunk", "chunk": full_text}),
                                websocket
                            )
                    else:
                        await manager.send_personal_message(
                            json.dumps({"type": "gemini_chunk", "chunk": full_text}),
                            websocket
                        )

            print(f"Streaming complete. Total chunks: {chunk_count}")
            if send_lock:
                async with send_lock:
                    await manager.send_personal_message(
                        json.dumps({"type": "gemini_done"}),
                        websocket
                    )
            else:
                await manager.send_personal_message(
                    json.dumps({"type": "gemini_done"}),
                    websocket
                )

        except Exception as e:
            error_msg = str(e)
            print(f"Error in Gemini streaming: {error_msg}")
            import traceback
            traceback.print_exc()
            if send_lock:
                async with send_lock:
                    await manager.send_personal_message(
                        json.dumps({"type": "gemini_error", "error": error_msg}),
                        websocket
                    )
            else:
                await manager.send_personal_message(
                    json.dumps({"type": "gemini_error", "error": error_msg}),
                    websocket
                )

    except ImportError:
        if send_lock:
            async with send_lock:
                await manager.send_personal_message(
                    json.dumps({"type": "gemini_error", "error": "Google Generative AI SDK not available"}),
                    websocket
                )
        else:
            await manager.send_personal_message(
                json.dumps({"type": "gemini_error", "error": "Google Generative AI SDK not available"}),
                websocket
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        if send_lock:
            async with send_lock:
                await manager.send_personal_message(
                    json.dumps({"type": "gemini_error", "error": str(e)}),
                    websocket
                )
        else:
            await manager.send_personal_message(
                json.dumps({"type": "gemini_error", "error": str(e)}),
                websocket
            )

@app.post("/api/ask-gemini")
async def ask_gemini(
    question: str = Form(...),
    jd: str = Form(""),
    resume: Optional[UploadFile] = File(None)
):
    """Ask Gemini AI using Google Generative AI SDK with resume and JD context"""
    try:
        if not question:
            return {"error": "Question is required"}
        
        # Try using Google Generative AI SDK
        try:
            import google.generativeai as genai
            
            # Configure the API key
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Build the prompt with context
            system_prompt = """You are an interview assistant helping me answer questions.

Based on my resume help me craft responses that:

1. Keep answers SHORT and FOCUSED

2. Answer the question DIRECTLY without extra information

3. Use SIMPLE, CLEAR English words (avoid complex vocabulary)

4. Write in PARAGRAPH format (not bullet points)

5. Highlight my RELEVANT experience and achievements

6. Sound CONFIDENT but NATURAL

Format: Give me one concise paragraph answer that I can easily read and use.

Additional Guidelines:
- Base your answers on the information provided in the resume and job description
- If the question is about experience or skills, reference specific details from the resume
- If the question is about why you're a good fit, connect resume qualifications to job requirements
- Keep answers clear, confident, and interview-appropriate

"""
            
            context_parts = []
            
            # Add JD context if provided
            if jd and jd.strip():
                context_parts.append(f"Job Description:\n{jd.strip()}\n")
            
            # Add resume context if provided
            resume_file_obj = None
            uploaded_file = None
            if resume:
                try:
                    # Read PDF file
                    resume_bytes = await resume.read()
                    
                    # Save to temp file first
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(resume_bytes)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Upload file to Gemini and get file URI
                        uploaded_file = genai.upload_file(
                            path=tmp_path,
                            mime_type="application/pdf"
                        )
                        
                        # Wait for file to be processed
                        import time
                        max_wait = 30  # Max 30 seconds
                        wait_time = 0
                        while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                            time.sleep(1)
                            wait_time += 1
                            uploaded_file = genai.get_file(uploaded_file.name)
                        
                        if uploaded_file.state.name == "ACTIVE":
                            context_parts.append(f"Resume: [PDF file uploaded - {resume.filename}]")
                            resume_file_obj = uploaded_file
                            print(f"✅ Resume uploaded successfully: {resume.filename}")
                        else:
                            print(f"Warning: Resume file processing failed: {uploaded_file.state.name}. Continuing without resume.")
                            resume_file_obj = None
                    except Exception as upload_error:
                        # If upload fails (e.g., API key issue), continue without resume
                        error_msg = str(upload_error)
                        if "API key" in error_msg or "expired" in error_msg.lower():
                            print(f"⚠️ Resume upload failed due to API key issue. Continuing without resume file.")
                        else:
                            print(f"⚠️ Resume upload failed: {error_msg}. Continuing without resume file.")
                        resume_file_obj = None
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                except Exception as e:
                    print(f"⚠️ Error processing resume: {str(e)}. Continuing without resume.")
                    resume_file_obj = None
            
            # Build the full prompt
            full_prompt = system_prompt
            if context_parts:
                full_prompt += "\n".join(context_parts) + "\n"
            full_prompt += f"\nQuestion: {question}\n\nPlease provide a helpful answer based on the above context."
            
            # Try different model names
            model_names = [
                "gemini-3-flash-preview",  # Most stable and fast
            ]
            
            # Find working model and prepare content
            working_model = None
            content_parts = None
            
            for model_name in model_names:
                try:
                    print(f"Trying model with SDK: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    
                    # Prepare content parts
                    content_parts = [full_prompt]
                    
                    # Add resume file if available
                    if resume_file_obj:
                        content_parts.append(resume_file_obj)
                    
                    # Test if model works (quick test without streaming)
                    try:
                        test_response = model.generate_content([full_prompt])
                        if test_response:
                            working_model = model
                            print(f"✅ Model {model_name} is working, starting streaming...")
                            break
                    except:
                        # If test fails, try with full content
                        working_model = model
                        print(f"✅ Using model {model_name} for streaming...")
                        break
                except Exception as e:
                    print(f"❌ {model_name} failed with SDK: {str(e)}")
                    continue
            
            if not working_model or not content_parts:
                return {"error": "All models failed with SDK"}
            
            # Stream response from Gemini
            async def stream_gemini_response() -> AsyncGenerator[str, None]:
                try:
                    print("Starting Gemini streaming...")
                    response = working_model.generate_content(content_parts, stream=True)
                    accumulated_text = ""
                    chunk_count = 0
                    try:
                        response_stream = iter(response)
                    except TypeError:
                        response_stream = None
                    if response_stream is not None:
                        for chunk in response_stream:
                            chunk_count += 1
                            print(f"Received chunk #{chunk_count}")
                            chunk_text = None
                            if hasattr(chunk, 'text'):
                                chunk_text = chunk.text
                            elif hasattr(chunk, 'parts') and chunk.parts:
                                for part in chunk.parts:
                                    if hasattr(part, 'text'):
                                        chunk_text = part.text
                                        break
                            elif hasattr(chunk, 'candidates') and chunk.candidates:
                                for candidate in chunk.candidates:
                                    if hasattr(candidate, 'content') and candidate.content:
                                        if hasattr(candidate.content, 'parts'):
                                            for part in candidate.content.parts:
                                                if hasattr(part, 'text'):
                                                    chunk_text = part.text
                                                    break
                                            break
                            if chunk_text:
                                new_text = chunk_text
                                if accumulated_text:
                                    if new_text.startswith(accumulated_text):
                                        delta = new_text[len(accumulated_text):]
                                    else:
                                        min_len = min(len(accumulated_text), len(new_text))
                                        common_len = sum(1 for i in range(min_len) if accumulated_text[i] == new_text[i])
                                        delta = new_text[common_len:]
                                else:
                                    delta = new_text
                                if delta:
                                    accumulated_text = new_text
                                    print(f"Chunk #{chunk_count}: Sending delta ({len(delta)} chars): '{delta[:30]}...'")
                                    chunk_json = json.dumps({'chunk': delta})
                                    yield f"data: {chunk_json}\n\n"
                    else:
                        full_text = getattr(response, 'text', None)
                        if callable(full_text):
                            full_text = full_text() or ''
                        else:
                            full_text = full_text or ''
                        if not full_text and hasattr(response, 'candidates') and response.candidates:
                            cand = response.candidates[0]
                            if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                                full_text = ''.join(getattr(p, 'text', '') or '' for p in cand.content.parts)
                        if full_text:
                            yield f"data: {json.dumps({'chunk': full_text})}\n\n"
                    print(f"Streaming complete. Total chunks: {chunk_count}")
                    # Send completion signal
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error in streaming: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
            
            return StreamingResponse(
                stream_gemini_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable buffering
                }
            )
            
        except ImportError:
            return {"error": "Google Generative AI SDK not available. Please install: pip install google-generativeai"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Mount static files with cache control
# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

