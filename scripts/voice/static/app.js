// Browser client for voice_server_gemini.py
// - Captures mic at 16kHz mono PCM16, streams over WS as binary frames.
// - Receives 24kHz PCM16 audio chunks (base64 in JSON), queues + plays.
// - Renders transcripts + tool calls in #log.

const $ = (id) => document.getElementById(id);
const logEl = $("log");
const statusEl = $("status");
const connectBtn = $("connect");
const micBtn = $("mic");
const openBtn = $("opentoggle");
const textIn = $("textin");

let ws = null;
let audioCtx = null;          // 16k capture context
let playCtx = null;           // 24k playback context
let micStream = null;
let micNode = null;
let micProcessor = null;
let openMic = false;
let micActive = false;
let playQueueTime = 0;
let playSources = [];         // active AudioBufferSourceNodes, for interrupt clear

function logLine(cls, txt) {
  const span = document.createElement("div");
  span.className = cls;
  span.textContent = txt;
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(s, cls) {
  statusEl.textContent = s;
  statusEl.className = cls || "";
}

// --- Resample Float32 [-1,1] -> Int16 PCM at 16kHz ----------------------
function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function downsampleTo16k(input, inputRate) {
  if (inputRate === 16000) return input;
  const ratio = inputRate / 16000;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const idx = i * ratio;
    const lo = Math.floor(idx), hi = Math.min(input.length - 1, lo + 1);
    const frac = idx - lo;
    out[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return out;
}

// --- Playback ------------------------------------------------------------
function ensurePlayCtx() {
  if (!playCtx) {
    playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
  }
  if (playCtx.state === "suspended") playCtx.resume();
  return playCtx;
}

function playPcm16Chunk(b64, sampleRate) {
  const ctx = ensurePlayCtx();
  const sr = sampleRate || 24000;
  const bin = atob(b64);
  const len = bin.length / 2;
  const buf = ctx.createBuffer(1, len, sr);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < len; i++) {
    const lo = bin.charCodeAt(i * 2);
    const hi = bin.charCodeAt(i * 2 + 1);
    let v = (hi << 8) | lo;
    if (v >= 0x8000) v -= 0x10000;
    ch[i] = v / 0x8000;
  }
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  if (playQueueTime < now) playQueueTime = now;
  src.start(playQueueTime);
  playQueueTime += buf.duration;
  playSources.push(src);
  src.onended = () => {
    const idx = playSources.indexOf(src);
    if (idx >= 0) playSources.splice(idx, 1);
  };
}

function stopPlayback() {
  for (const s of playSources) {
    try { s.stop(); } catch {}
    try { s.disconnect(); } catch {}
  }
  playSources = [];
  if (playCtx) playQueueTime = playCtx.currentTime;
}

// --- Mic capture ---------------------------------------------------------
async function startMic() {
  if (micActive) return;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  micNode = audioCtx.createMediaStreamSource(micStream);
  // ScriptProcessorNode is deprecated but widely supported; AudioWorklet is
  // nicer but heavier to set up. For a local dev tool this is fine.
  micProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
  micProcessor.onaudioprocess = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const inBuf = ev.inputBuffer.getChannelData(0);
    const down = downsampleTo16k(inBuf, audioCtx.sampleRate);
    const pcm = floatTo16BitPCM(down);
    ws.send(pcm.buffer);
  };
  micNode.connect(micProcessor);
  micProcessor.connect(audioCtx.destination);
  micActive = true;
  micBtn.classList.add("active");
}

function stopMic(sendEnd = false) {
  if (!micActive) return;
  try { micProcessor.disconnect(); } catch {}
  try { micNode.disconnect(); } catch {}
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micProcessor = null; micNode = null; micStream = null;
  micActive = false;
  micBtn.classList.remove("active");
  // Only signal audio_stream_end for push-to-talk release; for open-mic toggle
  // we just stop capture silently — server VAD already handles turn boundaries.
  if (sendEnd && ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "audio_end" }));
  }
}

// --- WS ------------------------------------------------------------------
function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/audio";
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  setStatus("connecting...");
  ws.onopen = () => { setStatus("connected", "live"); logLine("sys", "[ws open]"); };
  ws.onclose = (ev) => {
    setStatus("disconnected");
    logLine("sys", "[ws close " + ev.code + "]");
    stopMic();
    micBtn.disabled = true; openBtn.disabled = true;
  };
  ws.onerror = (e) => { setStatus("error", "err"); logLine("err", "[ws error]"); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return; // server sends JSON only
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "ready") {
      logLine("sys", "[gemini ready: " + msg.model + "]");
      micBtn.disabled = false; openBtn.disabled = false;
    } else if (msg.type === "audio") {
      const sr = (msg.mime && msg.mime.match(/rate=(\d+)/)) ? parseInt(RegExp.$1, 10) : 24000;
      playPcm16Chunk(msg.data, sr);
    } else if (msg.type === "user_transcript") {
      logLine("me", "you> " + msg.text);
    } else if (msg.type === "model_transcript") {
      logLine("ai", "ai>  " + msg.text);
    } else if (msg.type === "model_text") {
      logLine("ai", "ai(text)> " + msg.text);
    } else if (msg.type === "tool_call") {
      logLine("tool", "tool> " + msg.name + "(" + JSON.stringify(msg.args) + ")");
    } else if (msg.type === "tool_result") {
      logLine("tool", "  <= " + msg.preview);
    } else if (msg.type === "turn_complete") {
      logLine("sys", "[turn complete]");
    } else if (msg.type === "interrupted") {
      logLine("sys", "[interrupted — barge-in]");
      stopPlayback();
    } else if (msg.type === "error") {
      logLine("err", "[err] " + msg.msg);
    }
  };
}

// --- UI wiring -----------------------------------------------------------
connectBtn.onclick = () => { ensurePlayCtx(); connect(); };

// Push-to-talk: pressing+holding starts mic, releasing stops AND flushes
micBtn.addEventListener("mousedown", () => startMic());
micBtn.addEventListener("mouseup",   () => { if (!openMic) stopMic(/*sendEnd*/ true); });
micBtn.addEventListener("mouseleave",() => { if (!openMic) stopMic(/*sendEnd*/ true); });
micBtn.addEventListener("touchstart",(e)=>{ e.preventDefault(); startMic(); });
micBtn.addEventListener("touchend",  (e)=>{ e.preventDefault(); if (!openMic) stopMic(/*sendEnd*/ true); });

openBtn.onclick = () => {
  openMic = !openMic;
  openBtn.textContent = "Open mic: " + (openMic ? "ON" : "off");
  // Open-mic: do NOT send audio_end on toggle off; server VAD handles turns.
  if (openMic) startMic(); else stopMic(/*sendEnd*/ false);
};

textIn.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && textIn.value.trim() && ws && ws.readyState === WebSocket.OPEN) {
    const t = textIn.value.trim();
    logLine("me", "you(text)> " + t);
    ws.send(JSON.stringify({ type: "text", text: t }));
    textIn.value = "";
  }
});
