let mediaRecorder = null;
let audioChunks = [];
let websocket = null;
let isRecording = false;
let audioContext = null;
let sourceNode = null;
let stream = null;
let newLineCheckInterval = null; // Global interval for checking silence

const shareBtn = document.getElementById('shareBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const status = document.getElementById('status');
const transcription = document.getElementById('transcription');
const chatMessages = document.getElementById('chatMessages');
const voiceIndicator = document.getElementById('voiceIndicator');
const questionInput = document.getElementById('questionInput');
const askAiBtn = document.getElementById('askAiBtn');
const resumeInput = document.getElementById('resumeInput');
const jdInput = document.getElementById('jdInput');
const chatImageUpload = document.getElementById('chatImageUpload');
const chatImageBtn = document.getElementById('chatImageBtn');
const chatImageName = document.getElementById('chatImageName');

// Store chat messages and lines for tracking
let chatLines = []; // Array to store each line's text
let currentLineIndex = -1; // Index of current line being updated
let chatImageFile = null; // Store image for current chat message

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
        updateStatus('Đã kết nối', 'connected');
    };
    
    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // Handle transcription messages only (chat uses separate /api/chat-stream)
            if (data.type === 'transcription' && data.text) {
                if (data.newLine) {
                    forceNewLine = true;
                    currentLineElement = null;
                    currentLineText = '';
                }
                appendTranscription(data.text);
            } else if (data.type === 'error') {
                console.error('Server error:', data.message);
                updateStatus('Lỗi: ' + data.message, '');
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Lỗi kết nối', '');
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        updateStatus('Đã ngắt kết nối', '');
    };
}

// Update status display
function updateStatus(message, className = '') {
    status.textContent = message;
    status.className = `status ${className}`;
}

// Display transcription - append to current line or create new line
let currentLineText = '';
let currentLineElement = null;

// Global flag to force new line (set by VAD when 3s silence detected)
let forceNewLine = false;

function scrollTranscriptToBottom() {
    requestAnimationFrame(() => {
        transcription.scrollTop = transcription.scrollHeight;
    });
}

function scrollChatToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

function appendTranscription(text) {
    if (!text || !text.trim()) return;
    
    const placeholder = transcription.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    const textTrimmed = text.trim();
    
    // Check if we should start a new line
    // Force new line if flag is set, or if current line is empty
    const shouldStartNewLine = forceNewLine || !currentLineElement || currentLineElement.textContent.trim() === '';
    
    if (shouldStartNewLine) {
        console.log('Creating new line for:', textTrimmed);
        
        // Create new chat message (user message)
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message user';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble user new-sentence';
        bubble.textContent = textTrimmed;
        
        messageDiv.appendChild(bubble);
        transcription.appendChild(messageDiv);
        
        // Track this line
        currentLineIndex++;
        chatLines[currentLineIndex] = textTrimmed;
        currentLineElement = bubble;
        currentLineText = textTrimmed;
        
        // Reset force new line flag
        forceNewLine = false;
        scrollTranscriptToBottom();
        // Remove animation class after animation
        setTimeout(() => {
            if (bubble) {
                bubble.classList.remove('new-sentence');
            }
        }, 400);
    } else {
        // Append to current line (real-time updates)
        currentLineText += ' ' + textTrimmed;
        currentLineElement.textContent = currentLineText;
        
        // Update tracked line
        if (currentLineIndex >= 0) {
            chatLines[currentLineIndex] = currentLineText;
        }
        scrollTranscriptToBottom();
    }
}

function displayNewSentence(sentence) {
    // Create new paragraph for this sentence
    const p = document.createElement('p');
    p.textContent = sentence;
    p.className = 'new-sentence';
    transcription.appendChild(p);
    scrollTranscriptToBottom();
    // Remove animation class after animation completes
    setTimeout(() => {
        p.classList.remove('new-sentence');
    }, 400);
}

// Clear transcription
function clearTranscription() {
    transcription.innerHTML = '<div class="placeholder">Bản ghi sẽ hiển thị ở đây...</div>';
    voiceIndicator.style.display = 'none';
    currentLineElement = null;
    currentLineText = '';
    forceNewLine = false;
    chatLines = [];
    currentLineIndex = -1;
}

// Ask AI function
let isProcessingAI = false; // Flag to prevent duplicate calls

async function askAI() {
    // Prevent duplicate calls
    if (isProcessingAI) {
        console.log('AI request already in progress, ignoring...');
        return;
    }
    
    const question = questionInput.value.trim();
    
    // Get the latest line text if question is empty
    let questionText = question;
    let useExistingMessage = false;
    
    if (!questionText && chatLines.length > 0) {
        // Get the last line (most recent)
        questionText = chatLines[chatLines.length - 1];
        useExistingMessage = true; // Don't create new user message, use existing one
    }
    
    if (!questionText) {
        alert('Vui lòng nhập câu hỏi hoặc đợi có bản ghi âm thanh.');
        return;
    }
    
    // Set processing flag
    isProcessingAI = true;
    
    // Disable button
    askAiBtn.disabled = true;
    askAiBtn.textContent = '⏳ Đang xử lý...';
    
    // Remove placeholder from chat panel if exists
    const chatPlaceholder = chatMessages.querySelector('.placeholder');
    if (chatPlaceholder) {
        chatPlaceholder.remove();
    }
    
    // Only show user question in chat if it's a new question (not from transcription)
    if (!useExistingMessage) {
        // Show user question in chat (only for manually entered questions)
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'chat-message user';
        const userBubble = document.createElement('div');
        userBubble.className = 'message-bubble user';
        userBubble.textContent = questionText;
        userMessageDiv.appendChild(userBubble);
        chatMessages.appendChild(userMessageDiv);
    }
    
    // Create AI response placeholder
    const aiMessageDiv = document.createElement('div');
    aiMessageDiv.className = 'chat-message ai';
    const aiBubble = document.createElement('div');
    aiBubble.className = 'message-bubble ai streaming';
    aiBubble.textContent = '🤖 Đang suy nghĩ...';
    aiMessageDiv.appendChild(aiBubble);
    chatMessages.appendChild(aiMessageDiv);
    scrollChatToBottom();

    // Clear input ngay sau khi gửi Ask AI
    questionInput.value = '';
    chatImageFile = null;
    chatImageUpload.value = '';
    chatImageName.textContent = '';
    chatImageBtn.classList.remove('has-image');

    try {
        const resumeText = resumeInput.value.trim();
        let imageBase64 = null;
        let imageMime = 'image/jpeg';
        if (chatImageFile) {
            const imgB64 = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const dataUrl = reader.result;
                    const [header, b64] = dataUrl.split(',');
                    const mime = (header.match(/data:([^;]+)/) || [])[1] || 'image/jpeg';
                    resolve({ base64: b64, mime: mime });
                };
                reader.onerror = reject;
                reader.readAsDataURL(chatImageFile);
            });
            imageBase64 = imgB64.base64;
            imageMime = imgB64.mime;
        }

        aiBubble.textContent = '';

        const response = await fetch('/api/chat-stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: questionText,
                jd: jdInput.value.trim(),
                resume: resumeText || undefined,
                image: imageBase64,
                image_mime: imageMime
            })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: response.statusText }));
            throw new Error(err.error || response.statusText);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let streamEnded = false;

        while (!streamEnded) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);
                    if (data.type === 'chunk' && data.text) {
                        aiBubble.textContent += data.text;
                        scrollChatToBottom();
                    } else if (data.type === 'done') {
                        aiBubble.classList.remove('streaming');
                        streamEnded = true;
                        break;
                    } else if (data.type === 'error') {
                        aiBubble.textContent = `❌ Lỗi: ${data.message || data.error || 'Unknown'}`;
                        aiBubble.classList.remove('streaming');
                        streamEnded = true;
                        break;
                    }
                } catch (e) {
                    console.error('Parse NDJSON:', e);
                }
            }
            if (streamEnded) break;
        }
        if (buffer.trim() && !streamEnded) {
            try {
                const data = JSON.parse(buffer);
                if (data.type === 'chunk' && data.text) aiBubble.textContent += data.text;
                if (data.type === 'error') aiBubble.textContent = `❌ Lỗi: ${data.message || data.error}`;
            } catch (_) {}
        }
        aiBubble.classList.remove('streaming');
    } catch (error) {
        console.error('Error asking AI:', error);
        aiBubble.textContent = `❌ Lỗi: ${error.message}`;
        aiBubble.classList.remove('streaming');
    }

    isProcessingAI = false;
    askAiBtn.disabled = false;
    askAiBtn.textContent = '🤖 Ask AI';
    scrollChatToBottom();
}

// Chat image upload: button triggers file input
chatImageBtn.addEventListener('click', () => {
    chatImageUpload.click();
});

chatImageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn file ảnh (JPEG, PNG, WebP,...).');
            chatImageUpload.value = '';
            return;
        }
        chatImageFile = file;
        chatImageName.textContent = file.name;
        chatImageBtn.classList.add('has-image');
    } else {
        chatImageFile = null;
        chatImageName.textContent = '';
        chatImageBtn.classList.remove('has-image');
    }
});

// Check if mediaDevices is available
function checkMediaDevicesSupport() {
    // Check for modern API
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return true;
    }
    
    // Check for legacy API (deprecated but might work)
    if (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia) {
        console.warn('Using legacy getUserMedia API');
        return 'legacy';
    }
    
    return false;
}

// Check if getDisplayMedia is supported
function checkDisplayMediaSupport() {
    if (!checkMediaDevicesSupport()) {
        return false;
    }
    return typeof navigator.mediaDevices.getDisplayMedia === 'function';
}

// Check if getUserMedia is supported
function checkUserMediaSupport() {
    const support = checkMediaDevicesSupport();
    if (support === true) {
        return typeof navigator.mediaDevices.getUserMedia === 'function';
    }
    return support === 'legacy';
}

// Get user media with legacy API support
async function getUserMediaLegacy(constraints) {
    return new Promise((resolve, reject) => {
        const getUserMedia = navigator.getUserMedia || 
                           navigator.webkitGetUserMedia || 
                           navigator.mozGetUserMedia || 
                           navigator.msGetUserMedia;
        
        if (!getUserMedia) {
            reject(new Error('getUserMedia is not supported'));
            return;
        }
        
        getUserMedia.call(navigator, constraints, resolve, reject);
    });
}

// Start screen/audio sharing
async function startSharing() {
    console.log('startSharing called');
    console.log('navigator:', navigator);
    console.log('navigator.mediaDevices:', navigator.mediaDevices);
    console.log('window.location.protocol:', window.location.protocol);
    
    // First check if mediaDevices is available at all
    if (!checkMediaDevicesSupport()) {
        let errorMsg = 'Trình duyệt của bạn không hỗ trợ truy cập microphone/camera.\n\n';
        errorMsg += 'Vui lòng:\n';
        errorMsg += '1. Sử dụng trình duyệt hiện đại (Chrome, Firefox, Edge)\n';
        
        // Check if it's HTTP (not HTTPS or localhost)
        if (window.location.protocol === 'http:' && !window.location.hostname.includes('localhost') && !window.location.hostname.includes('127.0.0.1')) {
            errorMsg += '2. ⚠️ QUAN TRỌNG: Bạn đang dùng HTTP. Một số trình duyệt yêu cầu HTTPS để truy cập microphone.\n';
            errorMsg += '   Vui lòng truy cập qua HTTPS hoặc localhost (http://localhost:8000)\n';
        } else {
            errorMsg += '2. Đảm bảo bạn đang truy cập qua HTTPS hoặc localhost\n';
        }
        
        errorMsg += '3. Kiểm tra cài đặt quyền của trình duyệt';
        
        alert(errorMsg);
        updateStatus('Trình duyệt không hỗ trợ', '');
        return;
    }
    
    try {
        // Check if getDisplayMedia is available
        if (!checkDisplayMediaSupport()) {
            console.log('getDisplayMedia not supported, checking getUserMedia');
            
            // Check if getUserMedia is available
            if (!checkUserMediaSupport()) {
                alert(
                    'Trình duyệt không hỗ trợ truy cập microphone.\n\n' +
                    'Vui lòng sử dụng trình duyệt hiện đại (Chrome, Firefox, Edge) và đảm bảo bạn đang truy cập qua HTTPS hoặc localhost.'
                );
                updateStatus('Trình duyệt không hỗ trợ', '');
                return;
            }
            
            // Fallback to microphone if getDisplayMedia is not available
            const useMic = confirm(
                'Trình duyệt không hỗ trợ chia sẻ màn hình với âm thanh.\n\n' +
                'Bạn có muốn sử dụng microphone thay thế không?\n\n' +
                'Nhấn OK để dùng microphone, Cancel để hủy.'
            );
            
            if (!useMic) {
                console.log('User cancelled microphone request');
                return;
            }
            
            console.log('Requesting microphone access...');
            updateStatus('Đang yêu cầu quyền truy cập microphone...', '');
            
            // Use microphone instead
            try {
                const mediaSupport = checkMediaDevicesSupport();
                if (mediaSupport === true) {
                    // Modern API
                    stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            sampleRate: 16000
                        }
                    });
                } else if (mediaSupport === 'legacy') {
                    // Legacy API
                    console.log('Using legacy getUserMedia API');
                    stream = await getUserMediaLegacy({
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });
                } else {
                    throw new Error('getUserMedia is not supported');
                }
                console.log('Microphone access granted, stream:', stream);
                updateStatus('Đã có quyền truy cập microphone', 'connected');
            } catch (micError) {
                console.error('Microphone error:', micError);
                if (micError.name === 'NotAllowedError' || micError.name === 'PermissionDeniedError') {
                    alert('Quyền truy cập microphone bị từ chối. Vui lòng cấp quyền trong cài đặt trình duyệt.');
                } else if (micError.name === 'NotFoundError' || micError.name === 'DevicesNotFoundError') {
                    alert('Không tìm thấy microphone. Vui lòng kiểm tra thiết bị của bạn.');
                } else {
                    alert('Lỗi khi truy cập microphone: ' + micError.message);
                }
                updateStatus('Lỗi truy cập microphone', '');
                return;
            }
        } else {
            console.log('Requesting screen share with audio...');
            updateStatus('Đang yêu cầu chia sẻ màn hình...', '');
            // Request screen share with audio
            stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
            console.log('Screen share granted, stream:', stream);
        }
        
        // Check if audio track is available (only for screen share)
        console.log('Checking audio tracks...');
        const audioTracks = stream.getAudioTracks();
        const videoTracks = stream.getVideoTracks();
        console.log('Audio tracks:', audioTracks.length, 'Video tracks:', videoTracks.length);
        
        // If using screen share and no audio, warn user
        if (videoTracks.length > 0 && audioTracks.length === 0) {
            alert('Không có âm thanh được chia sẻ. Vui lòng chọn "Chia sẻ tab" và bật âm thanh trong dialog.');
            stream.getTracks().forEach(track => track.stop());
            return;
        }
        
        // If no audio at all, error
        if (audioTracks.length === 0) {
            alert('Không thể truy cập âm thanh. Vui lòng kiểm tra quyền truy cập microphone.');
            stream.getTracks().forEach(track => track.stop());
            return;
        }
        
        // Create audio-only stream for processing
        const audioOnlyStream = new MediaStream();
        audioTracks.forEach(track => {
            audioOnlyStream.addTrack(track);
            console.log('Added audio track:', track.id, 'enabled:', track.enabled, 'readyState:', track.readyState);
        });
        
        // Verify audio tracks are ready
        if (audioOnlyStream.getAudioTracks().length === 0) {
            alert('Không có audio track để ghi âm.');
            stream.getTracks().forEach(track => track.stop());
            updateStatus('Lỗi: Không có audio track', '');
            return;
        }
        
        // Setup AudioContext for raw PCM audio capture (better for real-time)
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000  // Whisper uses 16kHz
            });
            sourceNode = audioContext.createMediaStreamSource(audioOnlyStream);
            
            // Create ScriptProcessorNode for capturing raw audio data with VAD
            const bufferSize = 4096;
            const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            let audioBuffer = [];
            let isSpeaking = false;
            let silenceStartTime = null;
            let lastSendTime = Date.now();
            const SILENCE_THRESHOLD = 0.015; // Volume threshold for silence
            const SILENCE_DURATION = 3000; // 3 seconds of silence = new line
            const CHUNK_DURATION = 2000; // Send chunks every 2 seconds for real-time
            const sampleRate = audioContext.sampleRate;
            const samplesPerChunk = Math.floor(sampleRate * CHUNK_DURATION / 1000);
            let shouldNewLine = false; // Flag to mark next transcription as new line
            
            // VAD: Calculate RMS (Root Mean Square) for volume detection
            function calculateRMS(buffer) {
                let sum = 0;
                for (let i = 0; i < buffer.length; i++) {
                    sum += buffer[i] * buffer[i];
                }
                return Math.sqrt(sum / buffer.length);
            }
            
            processor.onaudioprocess = (event) => {
                if (!isRecording) return;
                
                const inputData = event.inputBuffer.getChannelData(0);
                const volume = calculateRMS(inputData);
                
                // Convert Float32Array to Int16Array for PCM
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Voice Activity Detection for new line detection
                if (volume > SILENCE_THRESHOLD) {
                    // Voice detected
                    if (!isSpeaking) {
                        isSpeaking = true;
                        voiceIndicator.style.display = 'flex';
                        console.log('Voice activity detected');
                    }
                    // Reset silence timer when voice is detected
                    if (silenceStartTime !== null) {
                        silenceStartTime = null;
                    }
                    
                    // Add to audio buffer
                    audioBuffer.push(...Array.from(pcmData));
                    
                } else {
                    // Silence detected
                    if (isSpeaking) {
                        // We were speaking, now silence - start timer
                        if (silenceStartTime === null) {
                            silenceStartTime = Date.now();
                            console.log('Silence started, waiting for', SILENCE_DURATION, 'ms');
                        }
                        
                        const silenceDuration = Date.now() - silenceStartTime;
                        
                        if (silenceDuration >= SILENCE_DURATION && !shouldNewLine) {
                            // 3 seconds of silence = mark for new line
                            console.log('Long silence detected (', silenceDuration, 'ms) - marking for new line');
                            shouldNewLine = true;
                            isSpeaking = false;
                            voiceIndicator.style.display = 'none';
                            
                            // Send a signal immediately to force new line on next transcription
                            if (websocket && websocket.readyState === WebSocket.OPEN) {
                                // Send a marker message to ensure new line happens
                                websocket.send(JSON.stringify({
                                    type: 'new_line_marker'
                                }));
                                console.log('Sent new line marker');
                            }
                        }
                    }
                }
                
                // Send audio chunks continuously for real-time (every 2 seconds or when buffer is full)
                const now = Date.now();
                const timeSinceLastSend = now - lastSendTime;
                
                if (audioBuffer.length >= samplesPerChunk || 
                    (timeSinceLastSend >= CHUNK_DURATION && audioBuffer.length > 0)) {
                    
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                        const chunkToSend = audioBuffer.slice(0, samplesPerChunk);
                        audioBuffer = audioBuffer.slice(samplesPerChunk);
                        
                        // Convert array to Int16Array, then to Uint8Array for base64 encoding
                        const int16Array = new Int16Array(chunkToSend);
                        const uint8Array = new Uint8Array(int16Array.buffer);
                        
                        // Convert to base64
                        let binary = '';
                        for (let i = 0; i < uint8Array.length; i++) {
                            binary += String.fromCharCode(uint8Array[i]);
                        }
                        const base64PCM = btoa(binary);
                        
                        console.log('Sending real-time audio chunk, samples:', chunkToSend.length, 'shouldNewLine:', shouldNewLine);
                        websocket.send(JSON.stringify({
                            type: 'audio_pcm',
                            data: base64PCM,
                            sampleRate: sampleRate,
                            channels: 1,
                            shouldNewLine: shouldNewLine  // Tell server if this should start new line
                        }));
                        
                        // If shouldNewLine was true, set the global flag for next transcription
                        if (shouldNewLine) {
                            console.log('Setting forceNewLine flag for next transcription');
                            forceNewLine = true;
                            shouldNewLine = false; // Reset after setting global flag
                        }
                        lastSendTime = now;
                    }
                }
            };
            
            // Connect processor
            sourceNode.connect(processor);
            processor.connect(audioContext.destination);
            
            // Set up interval to check for silence and force new line
            newLineCheckInterval = setInterval(() => {
                if (!isRecording) {
                    clearInterval(newLineCheckInterval);
                    return;
                }
                
                if (isSpeaking && silenceStartTime !== null) {
                    const silenceDuration = Date.now() - silenceStartTime;
                    if (silenceDuration >= SILENCE_DURATION && !shouldNewLine) {
                        console.log('Interval check: Long silence detected (', silenceDuration, 'ms) - forcing new line');
                        shouldNewLine = true;
                        forceNewLine = true;
                        isSpeaking = false;
                        voiceIndicator.style.display = 'none';
                    }
                }
            }, 500); // Check every 500ms
            
            console.log('AudioContext setup complete, using PCM streaming with VAD');
            console.log('Sample rate:', sampleRate);
            console.log('VAD settings: Silence threshold:', SILENCE_THRESHOLD, 'Silence duration:', SILENCE_DURATION, 'ms');
            
        } catch (ctxError) {
            console.error('Could not create AudioContext:', ctxError);
            alert('Lỗi khi tạo AudioContext: ' + ctxError.message);
            stream.getTracks().forEach(track => track.stop());
            updateStatus('Lỗi tạo AudioContext', '');
            return;
        }
        
        // PCM streaming is handled by AudioContext processor above
        // No need for MediaRecorder when using PCM streaming
        
        // Check WebSocket connection
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not connected, waiting...');
            updateStatus('Đang chờ kết nối...', '');
            
            // Wait for WebSocket to connect
            const checkConnection = setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    clearInterval(checkConnection);
                    console.log('WebSocket connected, starting PCM streaming');
                    isRecording = true;
                    shareBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                    updateStatus('Đang ghi âm...', 'recording');
                } else if (websocket && websocket.readyState === WebSocket.CLOSED) {
                    clearInterval(checkConnection);
                    alert('Không thể kết nối đến server. Vui lòng refresh trang.');
                    stream.getTracks().forEach(track => track.stop());
                    updateStatus('Lỗi kết nối', '');
                }
            }, 100);
            
            // Timeout after 5 seconds
            setTimeout(() => {
                clearInterval(checkConnection);
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                    alert('Không thể kết nối đến server sau 5 giây. Vui lòng refresh trang.');
                    stream.getTracks().forEach(track => track.stop());
                    updateStatus('Lỗi kết nối', '');
                }
            }, 5000);
            
            return;
        }
        
        // Start PCM streaming (AudioContext processor is already set up)
        console.log('Starting PCM audio streaming...');
        console.log('Stream active:', stream.active);
        console.log('Audio tracks active:', audioTracks.map(t => ({id: t.id, enabled: t.enabled, readyState: t.readyState, muted: t.muted})));
        
        // Ensure tracks are enabled and not muted
        audioTracks.forEach(track => {
            if (!track.enabled) {
                track.enabled = true;
            }
        });
        
        isRecording = true;
        shareBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        updateStatus('Đang ghi âm...', 'recording');
        
        // Handle when user stops sharing (for screen share)
        if (videoTracks.length > 0) {
            stream.getVideoTracks()[0].onended = () => {
                stopRecording();
            };
        }
        
        // Handle when audio track ends
        audioTracks[0].onended = () => {
            stopRecording();
        };
        
    } catch (error) {
        console.error('Error starting share:', error);
        alert('Lỗi khi bắt đầu chia sẻ: ' + error.message);
        updateStatus('Lỗi', '');
    }
}

// Stop recording
function stopRecording() {
    isRecording = false;
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    if (audioContext) {
        // Disconnect processor
        if (sourceNode) {
            sourceNode.disconnect();
        }
        audioContext.close();
        audioContext = null;
        sourceNode = null;
    }
    
    // Clear interval if exists
    if (newLineCheckInterval) {
        clearInterval(newLineCheckInterval);
        newLineCheckInterval = null;
    }
    
    // Reset transcription tracking
    voiceIndicator.style.display = 'none';
    forceNewLine = false;
    
    shareBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    updateStatus('Đã dừng ghi âm', 'connected');
}

// Event listeners
shareBtn.addEventListener('click', startSharing);
stopBtn.addEventListener('click', stopRecording);
clearBtn.addEventListener('click', clearTranscription);
askAiBtn.addEventListener('click', askAI);

// Allow Enter key to submit
questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        askAI();
    }
});

// Initialize WebSocket on page load
initWebSocket();

