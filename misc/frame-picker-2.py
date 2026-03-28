#!/usr/bin/env python3
"""Frame Picker — Extract video frames, process them via generate.py, detect shot boundaries."""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

PORT = 8777
GENERATE_PY = Path(__file__).resolve().parent.parent / "generate.py"

MODES = {
    "depth": {},
    "lineart": {"--model": ["canny", "teed"]},
    "normalmap": {},
    "sketch": {},
    "openpose": {"--pose-mode": ["wholebody", "body", "bodyhand", "bodyface"]},
    "segment": {"--output-layer": ["foreground", "background", "both"]},
    "upscale": {"--model": ["4x", "2x", "anime"]},
}

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
}

# Server state
current_video_path = None
sbd_progress = {"percent": 0, "running": False}


def _get_duration(video_path):
    """Get video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def detect_shots(video_path, threshold=0.3):
    """Run ffmpeg scene detection with progress tracking, return cached or fresh results."""
    global sbd_progress
    path = Path(video_path)
    stat = path.stat()
    cache_file = path.parent / f"{path.name}.shots"

    # Validate cache: exists + newer than video
    if cache_file.exists() and cache_file.stat().st_mtime >= stat.st_mtime:
        print(f"[sbd] cache hit: {cache_file}")
        return json.loads(cache_file.read_text())

    duration = _get_duration(path)
    print(f"[sbd] analyzing: {path.name} (threshold={threshold}, duration={duration:.0f}s)")

    cmd = [
        "ffmpeg", "-i", str(path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ]

    sbd_progress = {"percent": 0, "running": True}
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)

    shots = [0.0]  # first frame is always shot start
    for line in proc.stderr:
        # Parse ffmpeg progress: time=HH:MM:SS.ms (only allow upward movement)
        if duration > 0:
            tm = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if tm:
                cur = int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + float(tm.group(3))
                pct = min(99, int(cur / duration * 100))
                if pct > sbd_progress["percent"]:
                    sbd_progress["percent"] = pct

        # Parse shot boundaries from showinfo
        if "showinfo" in line:
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                t = float(m.group(1))
                if t > 0.01:
                    shots.append(t)

    proc.wait()
    sbd_progress = {"percent": 100, "running": False}

    data = {"shots": shots, "threshold": threshold, "file": path.name}
    cache_file.write_text(json.dumps(data))
    print(f"[sbd] found {len(shots)} shots, cached: {cache_file}")
    return data


HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Frame Picker</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #1a1a1a; color: #eee; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 30px 40px; }
  h1 { font-size: 1.4em; margin-bottom: 16px; font-weight: 500; }
  #drop-zone {
    width: 100%; max-width: 1400px;
    border: 2px dashed #555; border-radius: 12px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    transition: border-color 0.2s;
    margin-bottom: 20px; font-size: 1.1em; color: #888;
    padding: 30px 20px; gap: 16px;
  }
  #drop-zone.hidden { display: none; }
  .path-row {
    display: flex; gap: 8px; align-items: center; width: 100%; max-width: 700px;
  }
  .path-row span { color: #666; font-size: 0.85em; white-space: nowrap; }
  #path-input {
    flex: 1; background: #2a2a2a; color: #eee; border: 1px solid #555; border-radius: 6px;
    padding: 8px 12px; font-size: 0.9em; font-family: monospace;
  }
  #path-input:focus { outline: none; border-color: #4af; }
  #path-input::placeholder { color: #555; }
  #btn-load-path {
    background: #2a7; color: #fff; border: 1px solid #2a7; border-radius: 6px;
    padding: 8px 16px; cursor: pointer; font-size: 0.9em; white-space: nowrap;
  }
  #btn-load-path:hover { background: #3b8; }
  #player-area { display: none; width: 100%; max-width: 1400px; }
  #player-area.visible { display: flex; flex-direction: column; align-items: center; }
  video { width: 100%; max-height: calc(100vh - 280px); border-radius: 8px; background: #000; object-fit: contain; }
  canvas { display: none; }
  .controls { width: 100%; margin-top: 12px; }
  .scrubber-wrap { position: relative; width: 100%; }
  #scrubber { width: 100%; cursor: pointer; accent-color: #4af; position: relative; z-index: 1; }
  #shot-markers {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none; z-index: 2;
  }
  .shot-marker {
    position: absolute; top: 2px; bottom: 2px; width: 2px;
    background: #f80; border-radius: 1px; opacity: 0.85;
  }
  .info-row { display: flex; justify-content: space-between; align-items: center; width: 100%; margin-top: 8px; font-size: 0.9em; color: #aaa; }
  .btn-row { display: flex; gap: 10px; margin-top: 14px; flex-wrap: nowrap; justify-content: center; align-items: center; }
  button {
    background: #333; color: #eee; border: 1px solid #555; border-radius: 6px;
    padding: 8px 18px; cursor: pointer; font-size: 0.95em; transition: background 0.15s;
  }
  button:hover { background: #444; }
  button.primary { background: #2a7; border-color: #2a7; color: #fff; }
  button.primary:hover { background: #3b8; }
  button.primary.processing { background: #a72; border-color: #a72; pointer-events: none; }
  select { background: #333; color: #eee; border: 1px solid #555; border-radius: 6px; padding: 6px 10px; font-size: 0.95em; }
  .frame-nav { display: flex; gap: 6px; align-items: center; }
  .frame-nav button { padding: 8px 12px; font-size: 1.1em; }
  #filmstrip {
    display: flex; gap: 3px; width: 100%; margin-bottom: 8px;
    cursor: ew-resize;
  }
  #filmstrip .thumb {
    flex: 1; min-width: 0; border-radius: 3px;
    border: 2px solid transparent; transition: border-color 0.15s;
    position: relative; background: #000;
  }
  #filmstrip .thumb.active { border-color: #4af; }
  #filmstrip .thumb img { width: 100%; display: block; border-radius: 2px; }
  #filmstrip .thumb span {
    position: absolute; bottom: 1px; left: 0; right: 0;
    text-align: center; font-size: 0.6em; color: #ccc;
    background: rgba(0,0,0,0.6); padding: 1px 0;
  }
  #status { margin-top: 10px; font-size: 0.85em; color: #888; min-height: 1.2em; }
  #shot-info { color: #f80; }
  #sbd-overlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 100;
    flex-direction: column; align-items: center; justify-content: center; gap: 16px;
  }
  #sbd-overlay.active { display: flex; }
  #sbd-overlay .label { font-size: 1.1em; color: #eee; }
  #sbd-progress-wrap {
    width: 400px; height: 6px; background: #333; border-radius: 3px; overflow: hidden;
  }
  #sbd-progress-bar {
    height: 100%; width: 0%; background: #f80; border-radius: 3px;
    transition: width 0.4s ease;
  }
</style>
</head>
<body>

<h1>Frame Picker</h1>


<div id="drop-zone">
  <div class="path-row">
    <button id="btn-pick-file" style="background:#2a7;color:#fff;border:1px solid #2a7;border-radius:6px;padding:10px 28px;cursor:pointer;font-size:1.05em;">Datei wählen</button>
  </div>
  <div class="path-row" style="margin-top:4px;">
    <span>oder Pfad:</span>
    <input type="text" id="path-input" placeholder="/pfad/zum/video.mp4" autocomplete="off">
    <button id="btn-load-path">Laden</button>
  </div>
</div>

<div id="player-area">
  <div id="filmstrip"></div>
  <video id="video"></video>
  <canvas id="canvas"></canvas>
  <div class="controls">
    <div class="scrubber-wrap">
      <input type="range" id="scrubber" min="0" max="10000" value="0" step="1">
      <div id="shot-markers"></div>
    </div>
    <div class="info-row">
      <span id="time-display">0:00.000 / 0:00</span>
      <span><span id="frame-info"></span> <span id="shot-info"></span></span>
    </div>
  </div>
  <div class="btn-row">
    <div class="frame-nav">
      <button id="btn-prev-shot" title="Prev Shot">[</button>
      <button id="btn-prev-10" title="-10 Frames">-10</button>
      <button id="btn-prev" title="-1 Frame">-1</button>
      <button id="btn-play">Play</button>
      <button id="btn-next" title="+1 Frame">+1</button>
      <button id="btn-next-10" title="+10 Frames">+10</button>
      <button id="btn-next-shot" title="Next Shot">]</button>
    </div>
    <select id="mode-select">
      <option value="">Frame</option>
      <option value="depth">Depth</option>
      <option value="lineart">Lineart</option>
      <option value="normalmap">Normalmap</option>
      <option value="sketch">Sketch</option>
      <option value="openpose">OpenPose</option>
      <option value="segment">Segment</option>
      <option value="upscale">Upscale</option>
    </select>
    <select id="submode-select" style="display:none"></select>
    <select id="format-select">
      <option value="png">PNG</option>
      <option value="jpeg">JPEG (95%)</option>
      <option value="webp">WebP</option>
    </select>
    <button id="btn-save" class="primary" title="Frame speichern">💾</button>
    <button id="btn-new" title="Neue Datei">📂</button>
  </div>
  <div id="status"></div>
</div>

<div id="sbd-overlay">
  <div class="label">Shots werden erkannt...</div>
  <div id="sbd-progress-wrap"><div id="sbd-progress-bar"></div></div>
</div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const timeDisplay = document.getElementById('time-display');
const frameInfo = document.getElementById('frame-info');
const shotInfo = document.getElementById('shot-info');
const dropZone = document.getElementById('drop-zone');
const playerArea = document.getElementById('player-area');
const formatSelect = document.getElementById('format-select');
const modeSelect = document.getElementById('mode-select');
const submodeSelect = document.getElementById('submode-select');
const pathInput = document.getElementById('path-input');
const btnLoadPath = document.getElementById('btn-load-path');
const shotMarkersEl = document.getElementById('shot-markers');

const SUBMODES = {
  openpose: ['wholebody', 'body', 'bodyhand', 'bodyface'],
  lineart: ['canny', 'teed'],
  upscale: ['4x', '2x', 'anime'],
  segment: ['foreground', 'background', 'both'],
};

modeSelect.addEventListener('change', () => {
  const subs = SUBMODES[modeSelect.value];
  if (subs) {
    submodeSelect.innerHTML = subs.map(s => `<option value="${s}">${s}</option>`).join('');
    submodeSelect.style.display = '';
  } else {
    submodeSelect.style.display = 'none';
    submodeSelect.innerHTML = '';
  }
});
const btnPlay = document.getElementById('btn-play');
const btnSave = document.getElementById('btn-save');
const btnNew = document.getElementById('btn-new');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const btnPrev10 = document.getElementById('btn-prev-10');
const btnNext10 = document.getElementById('btn-next-10');
const btnPrevShot = document.getElementById('btn-prev-shot');
const btnNextShot = document.getElementById('btn-next-shot');
const status = document.getElementById('status');

let fileName = 'video';
let isSeeking = false;
let videoPath = null;  // set when loaded via path input
let shotBoundaries = [];
const FRAME_DURATION = 1 / 30;

// Audio scrubbing via Web Audio API
let audioCtx = null;
let audioBuffer = null;
let scrubSource = null;
const GRAIN_DURATION = 0.06;

function playScrubGrain(time) {
  if (!audioBuffer || !audioCtx) return;
  if (scrubSource) { try { scrubSource.stop(); } catch(e) {} }
  scrubSource = audioCtx.createBufferSource();
  scrubSource.buffer = audioBuffer;
  const gain = audioCtx.createGain();
  gain.gain.value = 0.8;
  gain.gain.setValueAtTime(0.8, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + GRAIN_DURATION);
  scrubSource.connect(gain);
  gain.connect(audioCtx.destination);
  const offset = Math.max(0, Math.min(time, audioBuffer.duration - GRAIN_DURATION));
  scrubSource.start(0, offset, GRAIN_DURATION);
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3);
  return `${m}:${sec.padStart(6, '0')}`;
}

function updateDisplay() {
  const t = video.currentTime;
  const d = video.duration || 0;
  timeDisplay.textContent = `${formatTime(t)} / ${formatTime(d)}`;
  const approxFrame = Math.round(t / FRAME_DURATION);
  frameInfo.textContent = `~Frame ${approxFrame}`;
  if (!isSeeking) {
    scrubber.value = d ? (t / d) * 10000 : 0;
  }
  // Shot info
  if (shotBoundaries.length > 0) {
    let idx = 0;
    for (let i = 0; i < shotBoundaries.length; i++) {
      if (shotBoundaries[i] <= t + 0.05) idx = i;
    }
    shotInfo.textContent = `| Shot ${idx + 1}/${shotBoundaries.length}`;
  } else {
    shotInfo.textContent = '';
  }
}

// --- Shot navigation ---
function prevShot() {
  if (!shotBoundaries.length) return;
  const t = video.currentTime;
  for (let i = shotBoundaries.length - 1; i >= 0; i--) {
    if (shotBoundaries[i] < t - 0.15) {
      video.currentTime = shotBoundaries[i];
      updateDisplay();
      return;
    }
  }
  video.currentTime = 0;
  updateDisplay();
}

function nextShot() {
  if (!shotBoundaries.length) return;
  const t = video.currentTime;
  for (let i = 0; i < shotBoundaries.length; i++) {
    if (shotBoundaries[i] > t + 0.15) {
      video.currentTime = shotBoundaries[i];
      updateDisplay();
      return;
    }
  }
}

btnPrevShot.addEventListener('click', prevShot);
btnNextShot.addEventListener('click', nextShot);

function renderShotMarkers() {
  shotMarkersEl.innerHTML = '';
  const dur = video.duration;
  if (!dur || !shotBoundaries.length) return;
  shotBoundaries.forEach(t => {
    if (t < 0.01) return; // skip 0.0 marker
    const marker = document.createElement('div');
    marker.className = 'shot-marker';
    marker.style.left = `${(t / dur) * 100}%`;
    shotMarkersEl.appendChild(marker);
  });
}

const sbdOverlay = document.getElementById('sbd-overlay');
const sbdLabel = sbdOverlay.querySelector('.label');
const sbdBar = document.getElementById('sbd-progress-bar');

async function detectShots(path) {
  status.textContent = 'Shots werden erkannt...';
  sbdLabel.textContent = 'Shots werden erkannt... 0%';
  sbdBar.style.width = '0%';
  sbdOverlay.classList.add('active');

  // Poll progress
  const pollId = setInterval(async () => {
    try {
      const r = await fetch('/sbd-progress');
      const p = await r.json();
      sbdLabel.textContent = `Shots werden erkannt... ${p.percent}%`;
      sbdBar.style.width = p.percent + '%';
    } catch(e) {}
  }, 500);

  try {
    const resp = await fetch('/detect-shots', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    shotBoundaries = data.shots || [];
    renderShotMarkers();
    status.textContent = `${shotBoundaries.length} Shots erkannt`;
    updateDisplay();
  } catch (err) {
    status.textContent = `SBD Fehler: ${err.message}`;
  } finally {
    clearInterval(pollId);
    sbdOverlay.classList.remove('active');
  }
}

// --- Load video from path ---
async function loadFromPath(path) {
  if (!path.trim()) return;
  status.textContent = 'Video wird geladen...';
  try {
    const resp = await fetch('/load-video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path.trim() }),
    });
    if (!resp.ok) throw new Error(await resp.text());

    videoPath = path.trim();
    localStorage.setItem('fp_videoPath', videoPath);
    fileName = path.trim().split('/').pop().replace(/\.[^.]+$/, '');
    shotBoundaries = [];
    shotMarkersEl.innerHTML = '';

    // Reset audio scrub (no File object for path-loaded videos)
    audioCtx = null;
    audioBuffer = null;

    video.onloadedmetadata = () => {
      video.currentTime = 0;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      dropZone.classList.add('hidden');
      playerArea.classList.add('visible');
      updateDisplay();
      generateFilmstrip();
      // Auto-detect shots
      detectShots(videoPath);
    };
    video.src = '/video?t=' + Date.now();
    video.load();
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
  }
}

// Path input: don't trigger drop zone click
pathInput.addEventListener('click', e => e.stopPropagation());
btnLoadPath.addEventListener('click', e => { e.stopPropagation(); loadFromPath(pathInput.value); });
pathInput.addEventListener('keydown', e => {
  e.stopPropagation();
  if (e.key === 'Enter') { e.preventDefault(); loadFromPath(pathInput.value); }
});

// Native file picker via server-side osascript
const btnPickFile = document.getElementById('btn-pick-file');
btnPickFile.addEventListener('click', async e => {
  e.stopPropagation();
  btnPickFile.textContent = '...';
  try {
    const resp = await fetch('/pick-file', { method: 'POST' });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    if (data.path) {
      pathInput.value = data.path;
      loadFromPath(data.path);
    }
  } catch (err) {
    status.textContent = `Picker: ${err.message}`;
  } finally {
    btnPickFile.textContent = 'Datei wählen';
  }
});

const filmstrip = document.getElementById('filmstrip');
const THUMB_COUNT = 10;

function generateFilmstrip() {
  filmstrip.innerHTML = '';
  const dur = video.duration;
  if (!dur) return;
  const thumbCanvas = document.createElement('canvas');
  const thumbCtx = thumbCanvas.getContext('2d');
  const aspect = video.videoWidth / video.videoHeight;
  thumbCanvas.width = 160;
  thumbCanvas.height = Math.round(160 / aspect);

  const tempVideo = document.createElement('video');
  tempVideo.muted = true;
  tempVideo.preload = 'auto';
  tempVideo.src = video.src;

  let i = 0;
  const timestamps = Array.from({length: THUMB_COUNT}, (_, idx) => dur * (idx + 0.5) / THUMB_COUNT);

  tempVideo.addEventListener('loadeddata', () => { seekNext(); });

  function seekNext() {
    if (i >= THUMB_COUNT) return;
    tempVideo.currentTime = timestamps[i];
  }

  tempVideo.addEventListener('seeked', () => {
    if (i >= THUMB_COUNT) return;
    thumbCtx.drawImage(tempVideo, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const img = document.createElement('img');
    img.src = thumbCanvas.toDataURL('image/jpeg', 0.7);
    const ts = timestamps[i];
    const label = document.createElement('span');
    label.textContent = formatTime(ts);
    const div = document.createElement('div');
    div.className = 'thumb';
    div.appendChild(img);
    div.appendChild(label);
    div.dataset.time = ts;
    filmstrip.appendChild(div);
    i++;
    seekNext();
  });
}

// Skimmer: mouse movement over filmstrip scrubs video + audio
let skimPending = false;
let skimLatest = -1;
filmstrip.addEventListener('mousemove', e => {
  const rect = filmstrip.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const t = ratio * video.duration;
  skimLatest = t;
  playScrubGrain(t);
  const thumbs = filmstrip.querySelectorAll('.thumb');
  const idx = Math.min(Math.floor(ratio * THUMB_COUNT), THUMB_COUNT - 1);
  thumbs.forEach((th, i) => th.classList.toggle('active', i === idx));
  if (!skimPending) {
    skimPending = true;
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked);
      skimPending = false;
      if (Math.abs(video.currentTime - skimLatest) > 0.01) {
        skimPending = true;
        video.currentTime = skimLatest;
        video.addEventListener('seeked', onSeeked, { once: true });
      }
    };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.currentTime = t;
  }
});
filmstrip.addEventListener('mouseleave', () => {
  if (scrubSource) { try { scrubSource.stop(); } catch(e) {} }
  btnPlay.textContent = 'Play';
  filmstrip.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
});

// Drop zone — no drag & drop (browser can't provide file path for SBD)

// Scrubber
scrubber.addEventListener('input', () => {
  isSeeking = true;
  video.pause();
  btnPlay.textContent = 'Play';
  video.currentTime = (scrubber.value / 10000) * video.duration;
});
scrubber.addEventListener('change', () => { isSeeking = false; });

video.addEventListener('timeupdate', updateDisplay);
video.addEventListener('seeked', updateDisplay);

// Play/Pause
btnPlay.addEventListener('click', () => {
  if (video.paused) {
    video.play();
    btnPlay.textContent = 'Pause';
  } else {
    video.pause();
    btnPlay.textContent = 'Play';
  }
});
video.addEventListener('ended', () => { btnPlay.textContent = 'Play'; });

// Frame stepping
function stepFrames(n) {
  video.pause();
  btnPlay.textContent = 'Play';
  video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + n * FRAME_DURATION));
}
btnPrev.addEventListener('click', () => stepFrames(-1));
btnNext.addEventListener('click', () => stepFrames(1));
btnPrev10.addEventListener('click', () => stepFrames(-10));
btnNext10.addEventListener('click', () => stepFrames(10));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  // Don't intercept when typing in path input
  if (document.activeElement === pathInput) return;
  if (!playerArea.classList.contains('visible')) return;
  document.activeElement.blur();
  if (e.key === 'ArrowLeft') { e.preventDefault(); e.stopPropagation(); stepFrames(e.shiftKey ? -10 : -1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); e.stopPropagation(); stepFrames(e.shiftKey ? 10 : 1); }
  if (e.key === ' ') { e.preventDefault(); btnPlay.click(); }
  if (e.key === 's' || e.key === 'S') { e.preventDefault(); btnSave.click(); }
  if (e.key === '[') { e.preventDefault(); prevShot(); }
  if (e.key === ']') { e.preventDefault(); nextShot(); }
}, true);

// Save frame
btnSave.addEventListener('click', async () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  const fmt = formatSelect.value;
  const mode = modeSelect.value;
  const timeStr = video.currentTime.toFixed(3).replace('.', '_');

  if (!mode) {
    // No mode: client-side download as before
    const mimeType = fmt === 'jpeg' ? 'image/jpeg' : fmt === 'webp' ? 'image/webp' : 'image/png';
    const quality = fmt === 'jpeg' ? 0.95 : undefined;
    const ext = fmt === 'jpeg' ? 'jpg' : fmt;
    canvas.toBlob(blob => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${fileName}_frame_${timeStr}.${ext}`;
      a.click();
      URL.revokeObjectURL(a.href);
    }, mimeType, quality);
    return;
  }

  // With mode: send to server for processing
  btnSave.classList.add('processing');
  btnSave.textContent = 'Wait';
  status.textContent = `${mode} wird verarbeitet...`;

  const dataUrl = canvas.toDataURL('image/png');
  try {
    const resp = await fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        mode: mode,
        submode: submodeSelect.value || '',
        filename: `${fileName}_frame_${timeStr}_${mode}`,
      }),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err);
    }
    const blob = await resp.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const ext = resp.headers.get('X-Extension') || 'png';
    a.download = `${fileName}_frame_${timeStr}_${mode}.${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
    status.textContent = `${mode} fertig.`;
  } catch (err) {
    status.textContent = `Fehler: ${err.message}`;
  } finally {
    btnSave.classList.remove('processing');
    btnSave.textContent = '💾';
  }
});

// New file
btnNew.addEventListener('click', () => {
  video.pause();
  video.src = '';
  videoPath = null;
  localStorage.removeItem('fp_videoPath');
  shotBoundaries = [];
  shotMarkersEl.innerHTML = '';
  btnPlay.textContent = 'Play';
  playerArea.classList.remove('visible');
  dropZone.classList.remove('hidden');
  pathInput.value = '';
  status.textContent = '';
  shotInfo.textContent = '';
});

// Auto-restore last video on page load
const savedPath = localStorage.getItem('fp_videoPath');
if (savedPath) {
  pathInput.value = savedPath;
  loadFromPath(savedPath);
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # quiet

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_HEAD(self):
        """Handle HEAD requests (browser sends these for video probing)."""
        if self.path.startswith("/video"):
            self._serve_video(head_only=True)
        else:
            payload = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            payload = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith("/video"):
            self._serve_video()
        elif self.path == "/sbd-progress":
            self._send_json(sbd_progress)
        else:
            self.send_error(404)

    def _serve_video(self, head_only=False):
        global current_video_path
        if current_video_path is None or not current_video_path.exists():
            self.send_error(404, "No video loaded")
            return

        file_size = current_video_path.stat().st_size
        ext = current_video_path.suffix.lower()
        ct = CONTENT_TYPES.get(ext, "application/octet-stream")
        range_header = self.headers.get("Range")

        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                if not head_only:
                    with open(current_video_path, "rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                return

        # Full file
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        if not head_only:
            with open(current_video_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b""
        body = json.loads(raw) if raw else {}

        if self.path == "/process":
            self._handle_process(body)
        elif self.path == "/load-video":
            self._handle_load_video(body)
        elif self.path == "/detect-shots":
            self._handle_detect_shots(body)
        elif self.path == "/pick-file":
            self._handle_pick_file()
        else:
            self.send_error(404)

    def _send_json(self, data, status_code=200):
        payload = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text_error(self, msg, status_code=400):
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(msg.encode())

    def _handle_pick_file(self):
        """Open native macOS file picker via osascript, return selected path."""
        script = (
            'set f to choose file of type {"public.movie"} '
            'with prompt "Video wählen"\n'
            'return POSIX path of f'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            # User cancelled or error
            self._send_json({"path": None})
            return
        path = result.stdout.strip()
        self._send_json({"path": path})

    def _handle_load_video(self, body):
        global current_video_path
        path = Path(body.get("path", "")).expanduser()
        if not path.exists():
            self._send_text_error(f"Datei nicht gefunden: {path}", 404)
            return
        if not path.is_file():
            self._send_text_error(f"Kein File: {path}", 400)
            return

        current_video_path = path
        print(f"[frame-picker] video loaded: {path}")
        self._send_json({"ok": True})

    def _handle_detect_shots(self, body):
        path = body.get("path", "")
        if not path or not Path(path).exists():
            self._send_text_error("Video-Pfad ungültig", 400)
            return

        threshold = body.get("threshold", 0.3)
        data = detect_shots(path, threshold)
        self._send_json(data)

    def _handle_process(self, body):
        mode = body.get("mode", "")
        submode = body.get("submode", "")
        image_data = body.get("image", "")
        base_name = body.get("filename", "frame")

        if mode not in MODES:
            self.send_error(400, f"Unknown mode: {mode}")
            return

        # Decode base64 PNG from data URL
        header, b64 = image_data.split(",", 1)
        img_bytes = base64.b64decode(b64)

        # Canvas always produces RGBA PNGs — strip alpha (video frames have no transparency)
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        input_path = Path(tempfile.mktemp(suffix=".png"))
        input_path.write_bytes(img_bytes)

        # segment+both → output is a directory, others → single file
        is_multi = (mode == "segment" and submode == "both")
        if is_multi:
            output_path = Path(tempfile.mkdtemp(prefix="fp_seg_"))
        else:
            output_path = Path("/tmp") / f"{base_name}.png"

        cmd = [
            sys.executable, str(GENERATE_PY),
            "image", mode,
            "--images", str(input_path),
            "-o", str(output_path),
            "--screen-log-format", "json",
        ]

        # Append sub-mode argument if present
        if submode:
            submode_flags = MODES.get(mode, {})
            for flag in submode_flags:
                if submode in submode_flags[flag]:
                    cmd.extend([flag, submode])
                    break

        print(f"[frame-picker] {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            input_path.unlink(missing_ok=True)

        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            self._send_text_error(err, 500)
            return

        if is_multi:
            # Zip all output PNGs
            import zipfile
            zip_path = Path("/tmp") / f"{base_name}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for f in sorted(output_path.glob("*.png")):
                    zf.write(f, f.name)
            # Clean up temp dir
            import shutil
            shutil.rmtree(output_path, ignore_errors=True)

            if not zip_path.exists():
                self._send_text_error("No output files produced", 500)
                return

            data = zip_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("X-Extension", "zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            if not output_path.exists():
                self._send_text_error("No output file produced", 500)
                return

            data = output_path.read_bytes()
            ext = output_path.suffix.lstrip(".")

            self.send_response(200)
            self.send_header("Content-Type", f"image/{ext}")
            self.send_header("X-Extension", ext)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    server = ThreadedHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[frame-picker] {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[frame-picker] stopped")
        server.server_close()


if __name__ == "__main__":
    main()
