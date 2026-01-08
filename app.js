/* ============================
   Config
   ============================ */

const TRACKS = [
  "audio/track1.wav",
  "audio/track2.wav",
  "audio/track3.wav"
];

const FEATURE_LIST = [
  "centroid",
  "energy",
  "flux",
  "flatness",
  "rolloff",
  "spread",
  "entropy",
  "crest",
  "slope",
  "density"
];

const BASE_GAIN = {
  centroid: 1,
  energy: 300,
  flux: 600,
  flatness: 1,
  rolloff: 1,
  spread: 1,
  entropy: 1,
  crest: 0.001,
  slope: -1000000,
  density: 1000
};

const VISUAL = {
  blurRadius: 3.6,
  blurIterations: 3, // kept for compatibility, not used in this version
  threshold: 246,
  edgeCushion: 10
};

let currentXFeature = "flux";
let currentYFeature = "density";

let xScale = 8;
let yScale = 2;
let inertia = 0.97;
let smoothX = null;
let smoothY = null;

/* ============================
   DOM + canvas
   ============================ */

const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d");

const playBtn = document.getElementById("play-btn");
const volumeSlider = document.getElementById("volume-slider");

function resizeDrawCanvas() {
  const size = drawCanvas.clientWidth;
  drawCanvas.width = size;
  drawCanvas.height = size;
}
resizeDrawCanvas();
window.addEventListener("resize", resizeDrawCanvas);

/* Offscreen canvases */
const lowResSizeX = 512;
const lowResSizeY = 480;
const lowCanvas = document.createElement("canvas");
lowCanvas.width = lowResSizeX;
lowCanvas.height = lowResSizeY;
const lowCtx = lowCanvas.getContext("2d");

const blurCanvas = document.createElement("canvas");
blurCanvas.width = lowResSizeX;
blurCanvas.height = lowResSizeY;
const blurCtx = blurCanvas.getContext("2d");

function clearMainCanvas() {
  drawCtx.fillStyle = "#ffffff";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}
function clearLowCanvas() {
  lowCtx.fillStyle = "#ffffff";
  lowCtx.fillRect(0, 0, lowCanvas.width, lowCanvas.height);
}
function clearBlurCanvas() {
  blurCtx.clearRect(0, 0, blurCanvas.width, blurCanvas.height);
}

clearMainCanvas();
clearLowCanvas();
clearBlurCanvas();

/* ============================
   Audio setup
   ============================ */

const fftSize = 2048;

let audioContext = null;
let analyser = null;
let sourceNode = null;
let audioBuffer = null;
let volumeNode = null;

let freqData = null;
let prevMag = null;

let trackFinished = false;


function initAudio() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    volumeNode = audioContext.createGain();
    volumeNode.gain.value = 1;

    analyser = audioContext.createAnalyser();
    analyser.fftSize = fftSize;
    analyser.smoothingTimeConstant = 0.7;

    freqData = new Float32Array(analyser.frequencyBinCount);
    prevMag = new Float32Array(analyser.frequencyBinCount);

    volumeNode.connect(audioContext.destination);
  }
}

async function loadRandomTrack() {
  initAudio();
  const chosen = TRACKS[Math.floor(Math.random() * TRACKS.length)];
  const res = await fetch(chosen);
  const arr = await res.arrayBuffer();
  audioBuffer = await audioContext.decodeAudioData(arr);
}

/* Volume slider */
volumeSlider.addEventListener("input", () => {
  if (!volumeNode) return;
  const v = parseFloat(volumeSlider.value);
  if (!isNaN(v)) {
    volumeNode.gain.value = v;
  }
});

/* ============================
   Feature computation
   ============================ */

const smoothed = {};
const smoothed2 = {};
FEATURE_LIST.forEach(f => smoothed[f] = 0);

function computeFeatures() {
  if (!analyser || !freqData || !audioContext) return null;

  analyser.getFloatFrequencyData(freqData);

  const n = freqData.length;
  const mags = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    mags[i] = Math.pow(10, freqData[i] / 20);
  }

  const nyquist = audioContext.sampleRate / 2;

  // Centroid
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const freq = (i / n) * nyquist;
    num += freq * mags[i];
    den += mags[i];
  }
  const centroidNorm = den > 0 ? (num / den) / nyquist : 0;

  // Energy
  let energyRaw = 0;
  for (let i = 0; i < n; i++) energyRaw += mags[i] * mags[i];
  const energy = Math.log10(1 + energyRaw);

  // Flux
  let fluxRaw = 0;
  if (!prevMag) prevMag = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const diff = mags[i] - prevMag[i];
    if (diff > 0) fluxRaw += diff * diff;
  }
  prevMag.set(mags);
  const flux = Math.log10(1 + 10 * fluxRaw);

  // Flatness
  let geo = 0, arith = 0;
  for (let i = 0; i < n; i++) {
    const m = mags[i] + 1e-12;
    geo += Math.log(m);
    arith += m;
  }
  geo = Math.exp(geo / n);
  arith /= n;
  const flatness = geo / (arith + 1e-12);

  // Rolloff
  const totalMag = mags.reduce((a, b) => a + b, 0);
  let cumulative = 0;
  let rolloff = 1;
  for (let i = 0; i < n; i++) {
    cumulative += mags[i];
    if (cumulative >= totalMag * 0.85) {
      rolloff = i / n;
      break;
    }
  }

  // Spread
  const centroidHz = centroidNorm * nyquist;
  let spreadNum = 0;
  for (let i = 0; i < n; i++) {
    const freq = (i / n) * nyquist;
    const diff = freq - centroidHz;
    spreadNum += mags[i] * diff * diff;
  }
  const spread = Math.sqrt(spreadNum / (den || 1)) / nyquist;

  // Entropy
  let entropy = 0;
  for (let i = 0; i < n; i++) {
    const p = mags[i] / (totalMag + 1e-12);
    if (p > 0) entropy -= p * Math.log2(p);
  }
  entropy /= Math.log2(n);

  // Crest
  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    if (mags[i] > maxMag) maxMag = mags[i];
  }
  const crest = maxMag / (arith + 1e-12);

  // Slope
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    const x = i;
    const y = mags[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const slope = (n * sumXY - sumX * sumY) /
                (n * sumXX - sumX * sumX + 1e-12);

  // Density
  let densityRaw = 0;
  for (let i = 0; i < n; i++) densityRaw += mags[i] * mags[i];
  const density = Math.log10(1 + densityRaw);

  return {
    centroid: centroidNorm,
    energy,
    flux,
    flatness,
    rolloff,
    spread,
    entropy,
    crest,
    slope,
    density
  };
}

/* ============================
   Mapping & drawing
   ============================ */

let isPlaying = false;
let prevX = null;
let prevY = null;
const margin = 40;

function resetCursor() {
  prevX = null;
  prevY = null;
}

function ema(prev, value, inertia) {
  return inertia * prev + (1 - inertia) * value;
}

function mapToCoords(features) {
  FEATURE_LIST.forEach(f => {
  const raw = features[f];
  const postGain = raw * BASE_GAIN[f];

  // --- SAFE INITIALIZATION ---
  if (!Number.isFinite(smoothed[f])) smoothed[f] = postGain;
  if (!Number.isFinite(smoothed2[f])) smoothed2[f] = smoothed[f];

  // --- FIRST PASS (EMA1) ---
  smoothed[f] = inertia * smoothed[f] + (1 - inertia) * postGain;

  // --- SECOND PASS (EMA2) ---
  smoothed2[f] = inertia * smoothed2[f] + (1 - inertia) * smoothed[f];
});


  const xVal = smoothed2[currentXFeature] * xScale;
  const yVal = smoothed2[currentYFeature] * yScale;

  const xNorm = Math.min(1, Math.max(0, xVal));
  const yNorm = Math.min(1, Math.max(0, yVal));

  const edgeCushion = VISUAL.edgeCushion;

  const minX = margin + edgeCushion;
  const maxX = drawCanvas.width - margin - edgeCushion;
  const minY = margin + edgeCushion;
  const maxY = drawCanvas.height - margin - edgeCushion;

  const x = minX + xNorm * (maxX - minX);
  const y = maxY - yNorm * (maxY - minY);
  
  if (smoothX === null) {
  smoothX = x;
  smoothY = y;
} else {
  smoothX = inertia * smoothX + (1 - inertia) * x;
  smoothY = inertia * smoothY + (1 - inertia) * y;
}

return { x: smoothX, y: smoothY };

}

function drawLineLowRes(x, y) {
  const lx = x / drawCanvas.width * lowCanvas.width;
  const ly = y / drawCanvas.height * lowCanvas.height;

  if (prevX === null || prevY === null) {
    prevX = lx;
    prevY = ly;
    return;
  }

  const baseWidth = 0.4 * (lowCanvas.width / drawCanvas.width);

  lowCtx.strokeStyle = "#000000";
  lowCtx.lineWidth = baseWidth;
  lowCtx.lineCap = "round";
  lowCtx.beginPath();
  lowCtx.moveTo(prevX, prevY);
  lowCtx.lineTo(lx, ly);
  lowCtx.stroke();

  lowCtx.lineWidth = baseWidth * 0.4;
  lowCtx.beginPath();
  lowCtx.moveTo(prevX, prevY);
  lowCtx.lineTo(lx, ly);
  lowCtx.stroke();

  prevX = lx;
  prevY = ly;
}

/* ============================
   Composite render
   ============================ */

function renderComposite() {
  clearBlurCanvas();

  blurCtx.filter = VISUAL.blurRadius > 0 ? `blur(${VISUAL.blurRadius}px)` : "none";
  blurCtx.drawImage(lowCanvas, 0, 0);
  blurCtx.filter = "none";

  clearMainCanvas();
  drawCtx.imageSmoothingEnabled = true;
  drawCtx.imageSmoothingQuality = "high";

  drawCtx.drawImage(
    blurCanvas,
    0, 0, blurCanvas.width, blurCanvas.height,
    0, 0, drawCanvas.width, drawCanvas.height
  );

  const threshold = VISUAL.threshold;
  const imgData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    const v = data[i];
    const out = v > threshold ? 255 : 0;
    data[i] = data[i+1] = data[i+2] = out;
  }

  drawCtx.putImageData(imgData, 0, 0);
}

/* ============================
   Animation loop
   ============================ */

function step() {
  if (isPlaying && analyser && freqData) {
    const features = computeFeatures();
    if (features) {
      const coords = mapToCoords(features);
      drawLineLowRes(coords.x, coords.y);
    
    }
    
  }

  renderComposite();
  requestAnimationFrame(step);
}
requestAnimationFrame(step);

/* ============================
   Playback control
   ============================ */

function resetForPlayback() {
  FEATURE_LIST.forEach(f => smoothed[f] = 0);
  prevMag = null;
  resetCursor();
  clearLowCanvas();
  clearBlurCanvas();
  clearMainCanvas();
}

function stopPlayback() {
  if (sourceNode) {
    try { sourceNode.stop(); } catch (e) {}
    sourceNode.disconnect();
    sourceNode = null;
  }
  isPlaying = false;
  playBtn.textContent = "▶";
}

async function startPlayback() {
  if (!audioBuffer) return;

  initAudio();
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  stopPlayback();
  resetForPlayback();

  sourceNode = audioContext.createBufferSource();
  sourceNode.buffer = audioBuffer;

  sourceNode.connect(analyser);
  sourceNode.connect(volumeNode);

 sourceNode.onended = () => {
  isPlaying = false;
  trackFinished = true;
  playBtn.textContent = "▶";
};


  sourceNode.start(0);
  isPlaying = true;
  playBtn.textContent = "⏸";
}

/* ============================
   UI wiring
   ============================ */

playBtn.addEventListener("click", async () => {
  // If no audio loaded yet, or last track finished → load a new one
  if (!audioBuffer || trackFinished) {
    trackFinished = false;
    await loadRandomTrack();
  }

  if (!isPlaying) {
    await startPlayback();
  } else {
    stopPlayback();
  }
});


/* ============================
   Init
   ============================ */

loadRandomTrack().catch(err => {
  console.error("Error loading audio:", err);
});
