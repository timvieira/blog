title: Interactive KL Divergence Fitting
date: 2026-03-13
tags: statistics, machine-learning, probability, interactive
comments: true

In a [previous post](/blog/post/2014/10/06/kl-divergence-as-an-objective-function/),
I covered the theory behind the two directions of KL divergence for fitting a
model $q_\theta$ to a target $p$. The key takeaway: $\textbf{KL}(p \| q)$ is
*inclusive* (mean-seeking) while $\textbf{KL}(q \| p)$ is *exclusive*
(mode-seeking). But reading about it is one thing&mdash;seeing it is another.

The widget below lets you watch both directions optimize simultaneously. The
target $p$ is a Gaussian mixture (shaded region), and we fit a single Gaussian
$q$ (colored curve) by gradient descent. Drag the modes of $p$ to rearrange
them, or drag $q$ to set its starting point.

<div id="kl-widget-container">
<style>
#kl-widget-container {
    margin: 1.5em 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.4;
}
#kl-widget-container * { box-sizing: border-box; }
.kl-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-bottom: 10px;
    padding: 8px 12px;
    background: #f7f7f7;
    border-radius: 6px;
    border: 1px solid #ddd;
}
.kl-controls label {
    font-size: 13px;
    font-weight: 600;
    color: #555;
}
.kl-controls select, .kl-controls button {
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid #ccc;
    background: #fff;
    cursor: pointer;
}
.kl-controls button {
    font-weight: 600;
    min-width: 60px;
}
.kl-controls button:hover { background: #eee; }
.kl-panel-label {
    font-size: 13px;
    font-weight: 700;
    padding: 4px 0 2px 4px;
    color: #333;
}
.kl-panel-stats {
    font-size: 12px;
    padding: 0 0 2px 4px;
    color: #666;
    font-family: Menlo, Monaco, monospace;
}
canvas.kl-canvas {
    display: block;
    width: 100%;
    height: 220px;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: default;
}
.kl-speed-group {
    display: flex;
    align-items: center;
    gap: 4px;
}
.kl-speed-group input[type=range] {
    width: 80px;
}
</style>

<div class="kl-controls">
    <label>Target modes</label>
    <button id="kl-add-mode">+</button>
    <button id="kl-remove-mode">−</button>
    <label>Fit q:</label>
    <select id="kl-family">
        <option value="gaussian">Gaussian(μ, σ²)</option>
        <option value="fixedvar">Gaussian(μ, fixed σ²)</option>
    </select>
    <button id="kl-reset">Reset</button>
    <div class="kl-speed-group">
        <label>Speed:</label>
        <input type="range" id="kl-speed" min="1" max="20" value="5">
    </div>
</div>

<div class="kl-panel-label">KL(p || q) &mdash; inclusive / mean-seeking</div>
<div class="kl-panel-stats" id="kl-stats-inclusive">&nbsp;</div>
<canvas class="kl-canvas" id="kl-canvas-inclusive" height="220"></canvas>

<div style="height: 8px;"></div>

<div class="kl-panel-label">KL(q || p) &mdash; exclusive / mode-seeking</div>
<div class="kl-panel-stats" id="kl-stats-exclusive">&nbsp;</div>
<canvas class="kl-canvas" id="kl-canvas-exclusive" height="220"></canvas>

<script>
(function() {
"use strict";

// ===== Math utilities =====
const sqrt2pi = Math.sqrt(2 * Math.PI);

function gaussPdf(x, mu, sigma) {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * sqrt2pi);
}

function gaussLogPdf(x, mu, sigma) {
    const z = (x - mu) / sigma;
    return -0.5 * z * z - Math.log(sigma * sqrt2pi);
}

function mixturePdf(x, components) {
    let s = 0;
    for (const c of components) s += c.w * gaussPdf(x, c.mu, c.sigma);
    return s;
}

function mixtureLogPdf(x, components) {
    // log-sum-exp for numerical stability
    let maxLog = -Infinity;
    const logTerms = [];
    for (const c of components) {
        const lt = Math.log(c.w) + gaussLogPdf(x, c.mu, c.sigma);
        logTerms.push(lt);
        if (lt > maxLog) maxLog = lt;
    }
    let s = 0;
    for (const lt of logTerms) s += Math.exp(lt - maxLog);
    return maxLog + Math.log(s);
}

// Numerical integration by trapezoidal rule over [a, b] with n steps
function integrate(f, a, b, n) {
    const h = (b - a) / n;
    let s = 0.5 * (f(a) + f(b));
    for (let i = 1; i < n; i++) s += f(a + i * h);
    return s * h;
}

// ===== Initial target =====
const INITIAL_TARGET = [
    { mu: -2.5, sigma: 0.8, w: 0.5 },
    { mu: 2.5, sigma: 0.8, w: 0.5 }
];

// ===== State =====
const X_MIN = -7, X_MAX = 7;
const N_GRID = 500;
const N_INT = 800;  // integration grid

let stepsPerFrame = 5;
let family = 'gaussian';
let pComponents = INITIAL_TARGET.map(c => ({ ...c }));

// Two fitted distributions: inclusive and exclusive
let qInc = { mu: 0, sigma: 1.5 };
let qExc = { mu: 0, sigma: 1.5 };

// Optimal solutions shown as dashed lines
let qBruteInc = { mu: 0, sigma: 1 };  // inclusive: moment-matching (closed form)
let qBrute = { mu: 0, sigma: 1 };     // exclusive: grid search

// Learning rates
const LR_INC = 0.05;     // inclusive: simple GD (convex, always converges)
const LR_EXC = 0.08;     // exclusive: Adam base LR

// Adam state for exclusive optimizer
let adamState = null;
function resetAdam() {
    adamState = { mMu: 0, vMu: 0, mSig: 0, vSig: 0, t: 0 };
}
resetAdam();

// Canvas setup
const canvasInc = document.getElementById('kl-canvas-inclusive');
const canvasExc = document.getElementById('kl-canvas-exclusive');
const statsInc = document.getElementById('kl-stats-inclusive');
const statsExc = document.getElementById('kl-stats-exclusive');

function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
}

function initCanvases() {
    resizeCanvas(canvasInc);
    resizeCanvas(canvasExc);
}


function resetQ() {
    // Random init across the full range
    qExc.mu = (Math.random() - 0.5) * 10;   // [-5, 5]
    qExc.sigma = 0.3 + Math.random() * 2.5;  // [0.3, 2.8]
    qInc.mu = (Math.random() - 0.5) * 10;
    qInc.sigma = 0.3 + Math.random() * 2.5;
    resetAdam();
}

// ===== Brute force for exclusive KL =====
function computeKLExclusive(mu, sigma) {
    // KL(q || p) = integral q(x) [log q(x) - log p(x)] dx
    return integrate(x => {
        const qx = gaussPdf(x, mu, sigma);
        if (qx < 1e-15) return 0;
        return qx * (gaussLogPdf(x, mu, sigma) - mixtureLogPdf(x, pComponents));
    }, X_MIN, X_MAX, N_INT);
}

function computeKLInclusive(mu, sigma) {
    // KL(p || q) = integral p(x) [log p(x) - log q(x)] dx
    return integrate(x => {
        const px = mixturePdf(x, pComponents);
        if (px < 1e-15) return 0;
        return px * (mixtureLogPdf(x, pComponents) - gaussLogPdf(x, mu, sigma));
    }, X_MIN, X_MAX, N_INT);
}

// Run GD to convergence from a starting point, return {mu, sigma, kl}
function optimizeKL(klFunc, gradFunc, startMu, startSigma, fixSigma) {
    let mu = startMu, sigma = startSigma;
    const lr = 0.02;
    for (let i = 0; i < 500; i++) {
        const g = gradFunc(mu, sigma);
        mu -= lr * g.dmu;
        if (!fixSigma) {
            sigma -= lr * 0.5 * g.dsigma;
            sigma = Math.max(0.1, sigma);
        }
        // Clamp mu to reasonable range
        mu = Math.max(X_MIN + 1, Math.min(X_MAX - 1, mu));
    }
    return { mu, sigma, kl: klFunc(mu, sigma) };
}

// Analytical moments of Gaussian mixture p
function momentsMixture(mu_q) {
    let Epx = 0, Epx2 = 0;
    for (const c of pComponents) {
        Epx += c.w * c.mu;
        Epx2 += c.w * (c.sigma * c.sigma + (c.mu - mu_q) * (c.mu - mu_q));
    }
    return { Epx, Epx2 };  // Epx = E_p[x], Epx2 = E_p[(x - mu_q)^2]
}

// Gradient of KL(q||p) w.r.t. mu, sigma (analytical)
// KL(q||p) = -H(q) - E_q[log p], so gradients come from differentiating E_q[log p]
function gradKLExclusive(mu, sigma) {
    function dLogP(x) {
        const px = mixturePdf(x, pComponents);
        if (px < 1e-15) return 0;
        let dpx = 0;
        for (const c of pComponents)
            dpx += c.w * gaussPdf(x, c.mu, c.sigma) * (-(x - c.mu) / (c.sigma * c.sigma));
        return dpx / px;
    }
    const EqdLogP = integrate(x => gaussPdf(x, mu, sigma) * dLogP(x), X_MIN, X_MAX, N_INT);
    const EqzDLogP = integrate(x => gaussPdf(x, mu, sigma) * ((x - mu) / sigma) * dLogP(x), X_MIN, X_MAX, N_INT);
    return {
        dmu: -EqdLogP,
        dsigma: -1.0 / sigma - EqzDLogP
    };
}

// Gradient of KL(p||q) w.r.t. mu, sigma (fully analytical)
function gradKLInclusive(mu, sigma) {
    const sigma2 = sigma * sigma;
    const { Epx, Epx2 } = momentsMixture(mu);
    return {
        dmu: (mu - Epx) / sigma2,
        dsigma: 1.0 / sigma - Epx2 / (sigma2 * sigma)
    };
}

function computeOptimal() {
    const fixSigma = family === 'fixedvar';
    const startSigma = fixSigma ? 1.0 : 1.5;

    // === Inclusive KL(p||q): moment-matching is exact for free Gaussian ===
    if (!fixSigma) {
        const { Epx, Epx2 } = momentsMixture(0);
        qBruteInc.mu = Epx;
        // Var(p) = E_p[(x - E_p[x])^2] = E_p[x^2] - (E_p[x])^2
        // momentsMixture(0).Epx2 = E_p[x^2], so Var = Epx2 - Epx^2
        // But momentsMixture(mu_q) gives E_p[(x-mu_q)^2], so use mu_q = Epx:
        const m2 = momentsMixture(Epx);
        qBruteInc.sigma = Math.sqrt(Math.max(0.01, m2.Epx2));
    } else {
        // Fixed sigma: multi-start GD (inclusive is still convex, but do it anyway)
        let best = { kl: Infinity };
        for (const c of pComponents) {
            const r = optimizeKL(computeKLInclusive, gradKLInclusive, c.mu, 1.0, true);
            if (r.kl < best.kl) best = r;
        }
        const r0 = optimizeKL(computeKLInclusive, gradKLInclusive, 0, 1.0, true);
        if (r0.kl < best.kl) best = r0;
        qBruteInc.mu = best.mu;
        qBruteInc.sigma = best.sigma;
    }

    // === Exclusive KL(q||p): multi-start GD from each mode + overall mean ===
    let bestExc = { kl: Infinity };
    // Start from each mode of p
    for (const c of pComponents) {
        const r = optimizeKL(computeKLExclusive, gradKLExclusive, c.mu, fixSigma ? 1.0 : c.sigma, fixSigma);
        if (r.kl < bestExc.kl) bestExc = r;
    }
    // Also start from overall mean
    const meanMu = pComponents.reduce((s, c) => s + c.w * c.mu, 0);
    const r0 = optimizeKL(computeKLExclusive, gradKLExclusive, meanMu, startSigma, fixSigma);
    if (r0.kl < bestExc.kl) bestExc = r0;
    // Also start from midpoints between modes
    for (let i = 0; i < pComponents.length; i++) {
        for (let j = i + 1; j < pComponents.length; j++) {
            const mid = (pComponents[i].mu + pComponents[j].mu) / 2;
            const r = optimizeKL(computeKLExclusive, gradKLExclusive, mid, startSigma, fixSigma);
            if (r.kl < bestExc.kl) bestExc = r;
        }
    }
    qBrute.mu = bestExc.mu;
    qBrute.sigma = bestExc.sigma;
}

// ===== Gradient steps =====
function stepInclusive() {
    // KL(p || q) gradient is fully analytical for Gaussian q and mixture p.
    // d/dmu KL = (mu - E_p[x]) / sigma^2
    // d/dsigma KL = 1/sigma - E_p[(x-mu)^2] / sigma^3
    const g = gradKLInclusive(qInc.mu, qInc.sigma);
    qInc.mu -= LR_INC * g.dmu;
    if (family === 'gaussian') {
        qInc.sigma -= LR_INC * g.dsigma;
        qInc.sigma = Math.max(0.1, qInc.sigma);
    }
}

function stepExclusive() {
    // KL(q||p) = -H(q) - E_q[log p], gradient via reparameterization:
    //   dKL/dmu = -E_q[d/dx log p(x)]
    //   dKL/dsigma = -1/sigma - E_q[(x-mu)/sigma * d/dx log p(x)]

    const mu = qExc.mu, sigma = qExc.sigma;

    function dLogP(x) {
        const px = mixturePdf(x, pComponents);
        if (px < 1e-15) return 0;
        let dpx = 0;
        for (const c of pComponents)
            dpx += c.w * gaussPdf(x, c.mu, c.sigma) * (-(x - c.mu) / (c.sigma * c.sigma));
        return dpx / px;
    }

    const EqdLogP = integrate(x => gaussPdf(x, mu, sigma) * dLogP(x), X_MIN, X_MAX, N_INT);
    const gradMu = -EqdLogP;

    let gradSigma = 0;
    if (family === 'gaussian') {
        const EqzDLogP = integrate(x => gaussPdf(x, mu, sigma) * ((x - mu) / sigma) * dLogP(x), X_MIN, X_MAX, N_INT);
        gradSigma = -1.0 / sigma - EqzDLogP;
    }

    // Adam update with LR decay for smooth convergence
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    adamState.t++;
    const lr = LR_EXC / (1 + 0.002 * adamState.t);  // gentle decay

    adamState.mMu = beta1 * adamState.mMu + (1 - beta1) * gradMu;
    adamState.vMu = beta2 * adamState.vMu + (1 - beta2) * gradMu * gradMu;
    const mMuHat = adamState.mMu / (1 - Math.pow(beta1, adamState.t));
    const vMuHat = adamState.vMu / (1 - Math.pow(beta2, adamState.t));
    qExc.mu -= lr * mMuHat / (Math.sqrt(vMuHat) + eps);

    if (family === 'gaussian') {
        adamState.mSig = beta1 * adamState.mSig + (1 - beta1) * gradSigma;
        adamState.vSig = beta2 * adamState.vSig + (1 - beta2) * gradSigma * gradSigma;
        const mSigHat = adamState.mSig / (1 - Math.pow(beta1, adamState.t));
        const vSigHat = adamState.vSig / (1 - Math.pow(beta2, adamState.t));
        qExc.sigma -= lr * mSigHat / (Math.sqrt(vSigHat) + eps);
        qExc.sigma = Math.max(0.1, qExc.sigma);
    }
}

// ===== Drawing =====
function xToCanvas(x, canvas) {
    return (x - X_MIN) / (X_MAX - X_MIN) * canvas.width;
}

function canvasToX(cx, canvas) {
    return X_MIN + (cx / canvas.width) * (X_MAX - X_MIN);
}

function yToCanvas(y, canvas, yMax) {
    const margin = 20 * (window.devicePixelRatio || 1);
    const plotH = canvas.height - margin;
    return canvas.height - margin - (y / yMax) * plotH;
}

function drawPanel(canvas, q, qBruteForce, colorQ, label) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Compute max y for scaling
    let yMax = 0;
    const dx = (X_MAX - X_MIN) / N_GRID;
    for (let i = 0; i <= N_GRID; i++) {
        const x = X_MIN + i * dx;
        const py = mixturePdf(x, pComponents);
        const qy = gaussPdf(x, q.mu, q.sigma);
        yMax = Math.max(yMax, py, qy);
    }
    if (qBruteForce) {
        for (let i = 0; i <= N_GRID; i++) {
            const x = X_MIN + i * dx;
            yMax = Math.max(yMax, gaussPdf(x, qBruteForce.mu, qBruteForce.sigma));
        }
    }
    yMax *= 1.1;

    // Draw x-axis
    const margin = 20 * dpr;
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height - margin);
    ctx.lineTo(canvas.width, canvas.height - margin);
    ctx.stroke();

    // Tick marks
    ctx.fillStyle = '#999';
    ctx.font = (10 * dpr) + 'px sans-serif';
    ctx.textAlign = 'center';
    for (let t = Math.ceil(X_MIN); t <= Math.floor(X_MAX); t++) {
        const cx = xToCanvas(t, canvas);
        ctx.beginPath();
        ctx.moveTo(cx, canvas.height - margin);
        ctx.lineTo(cx, canvas.height - margin + 4 * dpr);
        ctx.stroke();
        ctx.fillText(t, cx, canvas.height - 2 * dpr);
    }

    // Draw p (filled)
    ctx.beginPath();
    ctx.moveTo(xToCanvas(X_MIN, canvas), yToCanvas(0, canvas, yMax));
    for (let i = 0; i <= N_GRID; i++) {
        const x = X_MIN + i * dx;
        ctx.lineTo(xToCanvas(x, canvas), yToCanvas(mixturePdf(x, pComponents), canvas, yMax));
    }
    ctx.lineTo(xToCanvas(X_MAX, canvas), yToCanvas(0, canvas, yMax));
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,0,0,0.08)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1.5 * dpr;
    ctx.stroke();

    // Draw brute force q (dashed) if provided
    if (qBruteForce) {
        ctx.beginPath();
        ctx.setLineDash([6 * dpr, 4 * dpr]);
        for (let i = 0; i <= N_GRID; i++) {
            const x = X_MIN + i * dx;
            const y = gaussPdf(x, qBruteForce.mu, qBruteForce.sigma);
            if (i === 0) ctx.moveTo(xToCanvas(x, canvas), yToCanvas(y, canvas, yMax));
            else ctx.lineTo(xToCanvas(x, canvas), yToCanvas(y, canvas, yMax));
        }
        ctx.strokeStyle = 'rgba(100,100,100,0.5)';
        ctx.lineWidth = 1.5 * dpr;
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw q (solid line)
    ctx.beginPath();
    for (let i = 0; i <= N_GRID; i++) {
        const x = X_MIN + i * dx;
        const y = gaussPdf(x, q.mu, q.sigma);
        if (i === 0) ctx.moveTo(xToCanvas(x, canvas), yToCanvas(y, canvas, yMax));
        else ctx.lineTo(xToCanvas(x, canvas), yToCanvas(y, canvas, yMax));
    }
    ctx.strokeStyle = colorQ;
    ctx.lineWidth = 2.5 * dpr;
    ctx.stroke();

    // Draw mode handles for p
    for (const c of pComponents) {
        const cx = xToCanvas(c.mu, canvas);
        const cy = yToCanvas(c.w * gaussPdf(c.mu, c.mu, c.sigma), canvas, yMax);
        ctx.beginPath();
        ctx.arc(cx, cy, 5 * dpr, 0, 2 * Math.PI);
        ctx.fillStyle = (hoveredHandle && hoveredHandle.target === 'p' && hoveredHandle.comp === c)
            ? 'rgba(0,0,0,0.6)' : 'rgba(0,0,0,0.25)';
        ctx.fill();
    }

    // Draw q handle
    const qcx = xToCanvas(q.mu, canvas);
    const qcy = yToCanvas(gaussPdf(q.mu, q.mu, q.sigma), canvas, yMax);
    ctx.beginPath();
    ctx.arc(qcx, qcy, 5 * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = (hoveredHandle && hoveredHandle.target === 'q' && hoveredHandle.q === q)
        ? colorQ : colorQ.replace('1)', '0.4)');
    ctx.fill();
}

// ===== Interaction =====
let hoveredHandle = null;
let dragState = null;

function getHandles(canvas, q, qBruteForce) {
    const dpr = window.devicePixelRatio || 1;
    let yMax = 0;
    const dx = (X_MAX - X_MIN) / N_GRID;
    for (let i = 0; i <= N_GRID; i++) {
        const x = X_MIN + i * dx;
        yMax = Math.max(yMax, mixturePdf(x, pComponents), gaussPdf(x, q.mu, q.sigma));
        if (qBruteForce) yMax = Math.max(yMax, gaussPdf(x, qBruteForce.mu, qBruteForce.sigma));
    }
    yMax *= 1.1;

    const handles = [];
    for (const c of pComponents) {
        handles.push({
            target: 'p', comp: c,
            cx: xToCanvas(c.mu, canvas),
            cy: yToCanvas(c.w * gaussPdf(c.mu, c.mu, c.sigma), canvas, yMax)
        });
    }
    handles.push({
        target: 'q', q: q,
        cx: xToCanvas(q.mu, canvas),
        cy: yToCanvas(gaussPdf(q.mu, q.mu, q.sigma), canvas, yMax)
    });
    return handles;
}

function hitTest(canvas, q, qBruteForce, mx, my) {
    const dpr = window.devicePixelRatio || 1;
    const handles = getHandles(canvas, q, qBruteForce);
    let best = null, bestDist = 15 * dpr;
    for (const h of handles) {
        const d = Math.sqrt((mx - h.cx) ** 2 + (my - h.cy) ** 2);
        if (d < bestDist) { bestDist = d; best = h; }
    }
    return best;
}

function getMousePos(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    return {
        x: (e.clientX - rect.left) * dpr,
        y: (e.clientY - rect.top) * dpr
    };
}

function setupInteraction(canvas, q, getQBrute) {
    canvas.addEventListener('mousemove', e => {
        const pos = getMousePos(canvas, e);
        if (dragState && dragState.canvas === canvas) {
            const newMu = canvasToX(pos.x, canvas);
            // Vertical drag adjusts sigma: dragging down = wider, up = narrower
            const dy = pos.y - dragState.startY;
            const dpr = window.devicePixelRatio || 1;
            const sigmaDelta = dy / (150 * dpr);  // scale factor for sensitivity
            const newSigma = Math.max(0.15, dragState.startSigma + sigmaDelta);
            if (dragState.handle.target === 'p') {
                dragState.handle.comp.mu = newMu;
                dragState.handle.comp.sigma = newSigma;
                computeOptimal();
            } else {
                dragState.handle.q.mu = newMu;
                if (family === 'gaussian') {
                    dragState.handle.q.sigma = newSigma;
                }
            }
            return;
        }
        const hit = hitTest(canvas, q, getQBrute(), pos.x, pos.y);
        hoveredHandle = hit;
        canvas.style.cursor = hit ? 'grab' : 'default';
    });

    canvas.addEventListener('mousedown', e => {
        const pos = getMousePos(canvas, e);
        const hit = hitTest(canvas, q, getQBrute(), pos.x, pos.y);
        if (hit) {
            const startSigma = hit.target === 'p' ? hit.comp.sigma : hit.q.sigma;
            dragState = { canvas, handle: hit, startX: pos.x, startY: pos.y, startSigma };
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        }
    });

    canvas.addEventListener('wheel', e => {
        const pos = getMousePos(canvas, e);
        const hit = hitTest(canvas, q, getQBrute(), pos.x, pos.y);
        if (hit) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.05 : -0.05;
            if (hit.target === 'p') {
                hit.comp.sigma = Math.max(0.2, hit.comp.sigma + delta);
                computeOptimal();
            } else if (family === 'gaussian') {
                hit.q.sigma = Math.max(0.1, hit.q.sigma + delta);
            }
        }
    }, { passive: false });
}

document.addEventListener('mouseup', () => {
    if (dragState) {
        dragState.canvas.style.cursor = 'default';
        dragState = null;
        resetAdam();
    }
});

document.addEventListener('mousemove', e => {
    if (dragState) {
        const pos = getMousePos(dragState.canvas, e);
        const newMu = canvasToX(pos.x, dragState.canvas);
        const dpr = window.devicePixelRatio || 1;
        const dy = pos.y - dragState.startY;
        const newSigma = Math.max(0.15, dragState.startSigma + dy / (150 * dpr));
        if (dragState.handle.target === 'p') {
            dragState.handle.comp.mu = newMu;
            dragState.handle.comp.sigma = newSigma;
            computeOptimal();
        } else {
            dragState.handle.q.mu = newMu;
            if (family === 'gaussian') {
                dragState.handle.q.sigma = newSigma;
            }
        }
    }
});

setupInteraction(canvasInc, qInc, () => qBruteInc);
setupInteraction(canvasExc, qExc, () => qBrute);

// ===== Controls =====
document.getElementById('kl-add-mode').addEventListener('click', function() {
    // Add a new mode at a random position not too close to existing ones
    let mu = (Math.random() - 0.5) * 8;
    const sigma = 0.5 + Math.random() * 0.5;
    // Reweight: equal weights across all components
    const n = pComponents.length + 1;
    const w = 1 / n;
    for (const c of pComponents) c.w = w;
    pComponents.push({ mu, sigma, w });
    resetQ();
    computeOptimal();
});

document.getElementById('kl-remove-mode').addEventListener('click', function() {
    if (pComponents.length <= 1) return;
    pComponents.pop();
    // Reweight
    const w = 1 / pComponents.length;
    for (const c of pComponents) c.w = w;
    resetQ();
    computeOptimal();
});

document.getElementById('kl-family').addEventListener('change', function() {
    family = this.value;
    if (family === 'fixedvar') {
        qInc.sigma = 1.0;
        qExc.sigma = 1.0;
    }
    computeOptimal();
});

document.getElementById('kl-reset').addEventListener('click', function() {
    resetQ();
    if (family === 'fixedvar') {
        qInc.sigma = 1.0;
        qExc.sigma = 1.0;
    }
    computeOptimal();
});

document.getElementById('kl-speed').addEventListener('input', function() {
    stepsPerFrame = parseInt(this.value);
});

// ===== Animation loop =====
function frame() {
    resizeCanvas(canvasInc);
    resizeCanvas(canvasExc);

    if (!dragState) {
        for (let i = 0; i < stepsPerFrame; i++) {
            stepInclusive();
            stepExclusive();
        }
    }

    drawPanel(canvasInc, qInc, qBruteInc, 'rgba(220, 50, 50, 1)', 'inclusive');
    drawPanel(canvasExc, qExc, qBrute, 'rgba(50, 100, 220, 1)', 'exclusive');

    // Update stats
    const klInc = computeKLInclusive(qInc.mu, qInc.sigma);
    const klExc = computeKLExclusive(qExc.mu, qExc.sigma);
    statsInc.textContent = 'KL = ' + klInc.toFixed(4) +
        '  |  μ = ' + qInc.mu.toFixed(3) +
        '  |  σ = ' + qInc.sigma.toFixed(3) +
        '  |  optimal: μ=' + qBruteInc.mu.toFixed(2) + ' σ=' + qBruteInc.sigma.toFixed(2);
    statsExc.textContent = 'KL = ' + klExc.toFixed(4) +
        '  |  μ = ' + qExc.mu.toFixed(3) +
        '  |  σ = ' + qExc.sigma.toFixed(3) +
        '  |  optimal: μ=' + qBrute.mu.toFixed(2) + ' σ=' + qBrute.sigma.toFixed(2);

    requestAnimationFrame(frame);
}

// ===== Init =====
resetQ();
computeOptimal();
initCanvases();
frame();

})();
</script>
</div>

**What to look for:**

- **Inclusive (top, red):** $q$ spreads out to cover all of $p$'s mass. It finds
  the moment-matching solution&mdash;the mean and variance of the mixture. This
  is convex, so gradient descent always converges to the global optimum.

- **Exclusive (bottom, blue):** $q$ locks onto a single mode. The dashed gray
  line shows the global optimum. The optimizer may find a different local minimum
  each time you hit Reset&mdash;the landscape is nonconvex. Try it a few times!

- **Drag the handles** on $p$'s modes (dark circles) to move them. Scroll on a
  handle to change its width. You can also drag $q$'s handle (colored circle) to
  set a new starting point.

- **Fixed-variance mode:** Switch to "Gaussian(μ, fixed σ²)" to see mode-seeking
  even more clearly&mdash;$q$ can only slide left and right.

#### Why do we care?

The choice of KL direction determines what your model *learns to ignore*:

- **Inclusive** ($\textbf{KL}(p \| q)$): $q$ must cover everywhere $p$ has
  mass, so it overshoots, spreading too wide. It would rather be wrong
  everywhere than miss a mode. This is what maximum likelihood does.

- **Exclusive** ($\textbf{KL}(q \| p)$): $q$ can safely ignore modes of $p$,
  collapsing onto just one. It would rather be precise about one thing than
  vaguely right about everything. This is what variational inference does.
