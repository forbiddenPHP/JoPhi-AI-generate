#!/usr/bin/env python3
"""Frame Picker — Extract video frames and optionally process them via generate.py."""

import base64
import json
import os
import subprocess
import sys
import tempfile
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
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
    width: 100%; max-width: 1400px; height: 200px;
    border: 2px dashed #555; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: border-color 0.2s;
    margin-bottom: 20px; font-size: 1.1em; color: #888;
  }
  #drop-zone.drag-over { border-color: #4af; color: #4af; }
  #drop-zone.hidden { display: none; }
  #player-area { display: none; width: 100%; max-width: 1400px; }
  #player-area.visible { display: flex; flex-direction: column; align-items: center; }
  video { width: 100%; max-height: calc(100vh - 280px); border-radius: 8px; background: #000; object-fit: contain; }
  canvas { display: none; }
  .controls { width: 100%; margin-top: 12px; }
  #scrubber { width: 100%; cursor: pointer; accent-color: #4af; }
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
  #file-input { display: none; }
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
</style>
</head>
<body>

<h1>Frame Picker</h1>

<input type="file" id="file-input" accept="video/*">

<div id="drop-zone">Video hierher ziehen oder klicken zum Öffnen</div>

<div id="player-area">
  <div id="filmstrip"></div>
  <video id="video"></video>
  <canvas id="canvas"></canvas>
  <div class="controls">
    <input type="range" id="scrubber" min="0" max="10000" value="0" step="1">
    <div class="info-row">
      <span id="time-display">0:00.000 / 0:00</span>
      <span id="frame-info"></span>
    </div>
  </div>
  <div class="btn-row">
    <div class="frame-nav">
      <button id="btn-prev-10" title="-10 Frames">⏪</button>
      <button id="btn-prev" title="-1 Frame">◀</button>
      <button id="btn-play">▶</button>
      <button id="btn-next" title="+1 Frame">▶▏</button>
      <button id="btn-next-10" title="+10 Frames">⏩</button>
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

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const timeDisplay = document.getElementById('time-display');
const frameInfo = document.getElementById('frame-info');
const dropZone = document.getElementById('drop-zone');
const playerArea = document.getElementById('player-area');
const fileInput = document.getElementById('file-input');
const formatSelect = document.getElementById('format-select');
const modeSelect = document.getElementById('mode-select');
const submodeSelect = document.getElementById('submode-select');

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
const status = document.getElementById('status');

let fileName = 'video';
let isSeeking = false;
const FRAME_DURATION = 1 / 30;

// Audio scrubbing via Web Audio API
let audioCtx = null;
let audioBuffer = null;
let scrubSource = null;
const GRAIN_DURATION = 0.06;

function initAudioScrub(file) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const reader = new FileReader();
  reader.onload = () => {
    audioCtx.decodeAudioData(reader.result, buf => { audioBuffer = buf; });
  };
  reader.readAsArrayBuffer(file);
}

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
}

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
  btnPlay.textContent = '▶';
  filmstrip.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
});

function loadFile(file) {
  if (!file || !file.type.startsWith('video/')) return;
  fileName = file.name.replace(/\.[^.]+$/, '');
  initAudioScrub(file);
  const url = URL.createObjectURL(file);
  video.src = url;
  video.load();
  video.onloadedmetadata = () => {
    video.currentTime = 0;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    dropZone.classList.add('hidden');
    playerArea.classList.add('visible');
    updateDisplay();
    generateFilmstrip();
  };
}

// Drop zone
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) loadFile(fileInput.files[0]); });

// Scrubber
scrubber.addEventListener('input', () => {
  isSeeking = true;
  video.pause();
  btnPlay.textContent = '▶';
  video.currentTime = (scrubber.value / 10000) * video.duration;
});
scrubber.addEventListener('change', () => { isSeeking = false; });

video.addEventListener('timeupdate', updateDisplay);
video.addEventListener('seeked', updateDisplay);

// Play/Pause
btnPlay.addEventListener('click', () => {
  if (video.paused) {
    video.play();
    btnPlay.textContent = '⏸';
  } else {
    video.pause();
    btnPlay.textContent = '▶';
  }
});
video.addEventListener('ended', () => { btnPlay.textContent = '▶'; });

// Frame stepping
function stepFrames(n) {
  video.pause();
  btnPlay.textContent = '▶';
  video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + n * FRAME_DURATION));
}
btnPrev.addEventListener('click', () => stepFrames(-1));
btnNext.addEventListener('click', () => stepFrames(1));
btnPrev10.addEventListener('click', () => stepFrames(-10));
btnNext10.addEventListener('click', () => stepFrames(10));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (!playerArea.classList.contains('visible')) return;
  document.activeElement.blur();
  if (e.key === 'ArrowLeft') { e.preventDefault(); e.stopPropagation(); stepFrames(e.shiftKey ? -10 : -1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); e.stopPropagation(); stepFrames(e.shiftKey ? 10 : 1); }
  if (e.key === ' ') { e.preventDefault(); btnPlay.click(); }
  if (e.key === 's' || e.key === 'S') { e.preventDefault(); btnSave.click(); }
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
  btnSave.textContent = '⏳';
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
  btnPlay.textContent = '▶';
  playerArea.classList.remove('visible');
  dropZone.classList.remove('hidden');
  fileInput.value = '';
  status.textContent = '';
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/process":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
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
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(err.encode())
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
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"No output files produced")
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
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"No output file produced")
                return

            data = output_path.read_bytes()
            ext = output_path.suffix.lstrip(".")

            self.send_response(200)
            self.send_header("Content-Type", f"image/{ext}")
            self.send_header("X-Extension", ext)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
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
