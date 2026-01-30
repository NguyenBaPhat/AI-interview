from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import tempfile
import os
import wave
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
                        
                        # Transcribe using Faster-Whisper (English, optimized for real-time)
                        segments, info = whisper_model.transcribe(
                            wav_path,
                            language="en",  # English
                            task="transcribe",
                            beam_size=1,    # Faster decoding
                            best_of=1,      # No beam search
                            patience=1.0,   # Lower patience for speed
                            condition_on_previous_text=False,  # Don't wait for context
                            vad_filter=True,  # Voice Activity Detection to avoid silence
                            vad_parameters=dict(min_silence_duration_ms=500)  # Detect silence
                        )
                        
                        # Collect all segments
                        text_parts = []
                        for segment in segments:
                            text_parts.append(segment.text.strip())
                        
                        text = " ".join(text_parts).strip()
                        
                        if text:
                            print(f"Transcription: {text}")
                            # Check if this should start a new line (from client VAD detection)
                            should_new_line = audio_data.get("shouldNewLine", False)
                            
                            # Send transcription back to client
                            await manager.send_personal_message(
                                json.dumps({
                                    "type": "transcription", 
                                    "text": text,
                                    "newLine": should_new_line  # Tell frontend to start new line
                                }),
                                websocket
                            )
                    finally:
                        # Clean up temporary file
                        if wav_path and os.path.exists(wav_path):
                            try:
                                os.unlink(wav_path)
                            except:
                                pass
                                
                except Exception as e:
                    print(f"Error processing PCM audio: {e}")
                    import traceback
                    traceback.print_exc()
                    # Send error message to client
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": f"Lỗi xử lý audio: {str(e)}"}),
                        websocket
                    )
            
            # Handle Gemini AI request via WebSocket
            elif audio_data.get("type") == "ask_gemini":
                try:
                    question = audio_data.get("question", "")
                    jd = audio_data.get("jd", "")
                    resume_base64 = audio_data.get("resume", None)
                    
                    if not question:
                        await manager.send_personal_message(
                            json.dumps({"type": "gemini_error", "error": "Question is required"}),
                            websocket
                        )
                        continue
                    
                    # Process Gemini streaming
                    await process_gemini_streaming(question, jd, resume_base64, websocket)
                    
                except Exception as e:
                    print(f"Error processing Gemini request: {e}")
                    import traceback
                    traceback.print_exc()
                    await manager.send_personal_message(
                        json.dumps({"type": "gemini_error", "error": str(e)}),
                        websocket
                    )
            
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
                    
                    # Transcribe using Faster-Whisper (legacy WebM support)
                    segments, info = whisper_model.transcribe(
                        wav_path,
                        language="en",  # English
                        task="transcribe",
                        beam_size=1,
                        best_of=1,
                        patience=1.0,
                        condition_on_previous_text=False,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Collect all segments
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text.strip())
                    
                    text = " ".join(text_parts).strip()
                    
                    if text:
                        # Send transcription back to client
                        await manager.send_personal_message(
                            json.dumps({"type": "transcription", "text": text}),
                            websocket
                        )
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    import traceback
                    traceback.print_exc()
                    # Send error message to client
                    await manager.send_personal_message(
                        json.dumps({"type": "error", "message": f"Lỗi xử lý audio: {str(e)}"}),
                        websocket
                    )
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

# Gemini API endpoint
GEMINI_API_KEY = "AIzaSyD8dh5XHrE1PfS9fY-h_6zdEAuGLJyrQvE"

async def process_gemini_streaming(question: str, jd: str, resume_base64: Optional[str], websocket: WebSocket):
    """Process Gemini streaming and send chunks via WebSocket"""
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
        if resume_base64:
            try:
                # Decode base64 resume
                resume_bytes = base64.b64decode(resume_base64)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(resume_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # Upload file to Gemini
                    uploaded_file = genai.upload_file(
                        path=tmp_path,
                        mime_type="application/pdf"
                    )
                    
                    # Wait for file to be processed
                    import time
                    max_wait = 30
                    wait_time = 0
                    while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                        time.sleep(1)
                        wait_time += 1
                        uploaded_file = genai.get_file(uploaded_file.name)
                    
                    if uploaded_file.state.name == "ACTIVE":
                        context_parts.append(f"Resume: [PDF file uploaded]")
                        resume_file_obj = uploaded_file
                        print(f"✅ Resume uploaded successfully")
                    else:
                        print(f"Warning: Resume file processing failed: {uploaded_file.state.name}")
                        resume_file_obj = None
                except Exception as upload_error:
                    error_msg = str(upload_error)
                    if "API key" in error_msg or "expired" in error_msg.lower():
                        print(f"⚠️ Resume upload failed due to API key issue. Continuing without resume.")
                    else:
                        print(f"⚠️ Resume upload failed: {error_msg}. Continuing without resume.")
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
        
        # Get model
        model_name = "gemini-3-flash-preview"
        try:
            model = genai.GenerativeModel(model_name)
            
            # Prepare content parts
            content_parts = [full_prompt]
            if resume_file_obj:
                content_parts.append(resume_file_obj)
            
            print(f"Starting Gemini streaming via WebSocket...")
            
            # Stream response from Gemini
            response_stream = model.generate_content(
                content_parts,
                stream=True
            )
            
            accumulated_text = ""
            chunk_count = 0
            
            for chunk in response_stream:
                chunk_count += 1
                
                # Get text from chunk
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
                    # Extract delta
                    new_text = chunk_text
                    
                    if accumulated_text:
                        if new_text.startswith(accumulated_text):
                            delta = new_text[len(accumulated_text):]
                        else:
                            # Find common prefix
                            min_len = min(len(accumulated_text), len(new_text))
                            common_len = 0
                            for i in range(min_len):
                                if accumulated_text[i] == new_text[i]:
                                    common_len += 1
                                else:
                                    break
                            delta = new_text[common_len:]
                    else:
                        delta = new_text
                    
                    if delta:
                        accumulated_text = new_text
                        print(f"Chunk #{chunk_count}: Sending delta ({len(delta)} chars) via WebSocket")
                        
                        # Send chunk via WebSocket immediately
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "gemini_chunk",
                                "chunk": delta
                            }),
                            websocket
                        )
            
            print(f"Streaming complete. Total chunks: {chunk_count}")
            
            # Send completion signal
            await manager.send_personal_message(
                json.dumps({
                    "type": "gemini_done"
                }),
                websocket
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in Gemini streaming: {error_msg}")
            import traceback
            traceback.print_exc()
            await manager.send_personal_message(
                json.dumps({
                    "type": "gemini_error",
                    "error": error_msg
                }),
                websocket
            )
            
    except ImportError:
        await manager.send_personal_message(
            json.dumps({
                "type": "gemini_error",
                "error": "Google Generative AI SDK not available"
            }),
            websocket
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        await manager.send_personal_message(
            json.dumps({
                "type": "gemini_error",
                "error": str(e)
            }),
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
                    # Use generate_content with stream=True for streaming
                    response_stream = working_model.generate_content(
                        content_parts,
                        stream=True
                    )
                    
                    accumulated_text = ""
                    chunk_count = 0
                    
                    for chunk in response_stream:
                        chunk_count += 1
                        print(f"Received chunk #{chunk_count}")
                        
                        # Gemini streaming: each chunk contains the FULL text so far, not just delta
                        # We need to extract only the new part
                        chunk_text = None
                        
                        # Try to get text from chunk
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
                            # Gemini returns cumulative text, so we need to extract delta
                            new_text = chunk_text
                            
                            # Calculate delta: new text minus what we already have
                            if accumulated_text:
                                # Find the longest common prefix
                                if new_text.startswith(accumulated_text):
                                    delta = new_text[len(accumulated_text):]
                                else:
                                    # Text doesn't match - might be a new response or error
                                    # Try to find where it diverges
                                    min_len = min(len(accumulated_text), len(new_text))
                                    common_len = 0
                                    for i in range(min_len):
                                        if accumulated_text[i] == new_text[i]:
                                            common_len += 1
                                        else:
                                            break
                                    delta = new_text[common_len:]
                            else:
                                # First chunk
                                delta = new_text
                            
                            if delta:
                                accumulated_text = new_text
                                print(f"Chunk #{chunk_count}: Sending delta ({len(delta)} chars): '{delta[:30]}...'")
                                
                                # Send chunk as Server-Sent Events format
                                chunk_json = json.dumps({'chunk': delta})
                                yield f"data: {chunk_json}\n\n"
                    
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

