title: Discrete KL Divergence: N-gram Distillation
date: 2026-03-13
tags: statistics, machine-learning, probability, interactive
status: draft
comments: true

In the [continuous KL demo](/blog/kl-fitting-interactive/), we watched
forward and reverse KL fit a Gaussian to a mixture (see also [KL-divergence as
an objective function](/blog/kl-variational/)). Now let's move to **discrete
distributions**, framed as language model distillation.

A **teacher** assigns probabilities to 3-word phrases from an 8-word vocabulary
(512 total sequences, but only a handful have nonzero mass). A **student**
n-gram model tries to match the teacher by minimizing KL divergence. The n-gram
order controls expressiveness:

- **Unigram** ($n{=}1$): each position independent&mdash;can't capture word correlations.
- **Bigram** ($n{=}2$): each word depends on the previous&mdash;pairwise dependencies.
- **Trigram** ($n{=}3$): full joint over all three positions&mdash;can represent anything.

Watch how $\textbf{KL}(p \| q)$ (forward) and $\textbf{KL}(q \| p)$ (reverse)
produce different students, especially when the model is too simple to match the
teacher exactly.

<div id="dkl-container">
<style>
#dkl-container {
    margin: 1.5em 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.4;
}
#dkl-container * { box-sizing: border-box; }
#dkl-container p { margin: 0; }
.dkl-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
    padding: 8px 12px;
    background: #f7f7f7;
    border-radius: 6px;
    border: 1px solid #ddd;
}
.dkl-controls label {
    font-size: 13px;
    font-weight: 600;
    color: #555;
}
.dkl-controls select, .dkl-controls button {
    font-size: 13px;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid #ccc;
    background: #fff;
    cursor: pointer;
}
.dkl-controls button { font-weight: 600; }
.dkl-controls button:hover { background: #eee; }
.dkl-ngram-btn.active {
    background: #333;
    color: #fff;
    border-color: #333;
}
.dkl-ngram-btn.active:hover { background: #555; }
.dkl-sliders {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
    font-size: 13px;
}
.dkl-sliders label { font-weight: 600; color: #555; }
.dkl-sliders input[type=range] { flex: 1; min-width: 50px; }
.dkl-editor {
    margin-bottom: 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    overflow: hidden;
}
.dkl-editor-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: #f7f7f7;
    border-bottom: 1px solid #ddd;
    font-size: 13px;
    font-weight: 600;
    color: #555;
}
.dkl-editor-header .dkl-add-group {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
}
.dkl-editor-header select {
    font-size: 12px;
    padding: 2px 4px;
    border-radius: 3px;
    border: 1px solid #ccc;
}
.dkl-editor-header button {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 3px;
    border: 1px solid #ccc;
    background: #fff;
    cursor: pointer;
    font-weight: 600;
}
.dkl-editor-header button:hover { background: #eee; }
.dkl-support-list {
    padding: 4px 8px;
    max-height: 260px;
    overflow-y: auto;
}
.dkl-sup-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 0;
}
.dkl-sup-label {
    width: 110px;
    font-family: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
    font-size: 11.5px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    flex-shrink: 0;
    color: #444;
}
.dkl-sup-track {
    flex: 1;
    height: 14px;
    background: #eee;
    border-radius: 3px;
    position: relative;
    cursor: ew-resize;
}
.dkl-sup-bar {
    height: 100%;
    background: rgba(0,0,0,0.25);
    border-radius: 3px;
    min-width: 2px;
}
.dkl-sup-val {
    font-size: 11px;
    color: #888;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}
.dkl-sup-rm {
    font-size: 14px;
    color: #bbb;
    cursor: pointer;
    width: 18px;
    text-align: center;
    flex-shrink: 0;
    line-height: 1;
    border: none;
    background: none;
    padding: 0;
}
.dkl-sup-rm:hover { color: #e44; }
.dkl-panel-label {
    font-size: 13px;
    font-weight: 700;
    padding: 4px 0 2px 4px;
    color: #333;
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.dkl-kl-val {
    font-weight: 400;
    font-size: 12px;
    color: #888;
    font-family: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
}
.dkl-panel {
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 40px;
}
.dkl-bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 2px 0;
}
.dkl-bar-label {
    width: 110px;
    font-family: "SF Mono", "Fira Code", "Fira Mono", Menlo, Consolas, monospace;
    font-size: 11.5px;
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    flex-shrink: 0;
    color: #444;
}
.dkl-bar-track {
    flex: 1;
    height: 16px;
    background: #f5f5f5;
    border-radius: 3px;
    position: relative;
    overflow: hidden;
}
.dkl-bar-p {
    position: absolute;
    left: 0; top: 0;
    height: 100%;
    background: rgba(0,0,0,0.12);
    border-radius: 3px;
}
.dkl-bar-q {
    position: absolute;
    left: 0; top: 0;
    height: 100%;
    border-radius: 3px;
}
.dkl-bar-q-fwd { background: rgba(220, 50, 50, 0.5); }
.dkl-bar-q-rev { background: rgba(50, 100, 220, 0.5); }
.dkl-bar-vals {
    font-size: 11px;
    color: #888;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}
.dkl-wasted {
    font-size: 12px;
    color: #888;
    padding: 3px 4px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.dkl-wasted-bar {
    display: inline-block;
    height: 8px;
    border-radius: 2px;
    min-width: 0;
}
.dkl-wasted-bar-fwd { background: rgba(220, 50, 50, 0.4); }
.dkl-wasted-bar-rev { background: rgba(50, 100, 220, 0.4); }
.dkl-legend {
    font-size: 11px;
    font-weight: 400;
    color: #999;
}
.dkl-legend-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    vertical-align: middle;
    margin-right: 2px;
}
.dkl-panel-row {
    display: flex;
    gap: 8px;
    align-items: stretch;
}
.dkl-panel-row .dkl-panel {
    flex: 1;
    min-width: 0;
}
canvas.dkl-stats-canvas {
    display: block;
    width: 220px;
    min-height: 150px;
    flex-shrink: 0;
    border: 1px solid #ddd;
    border-radius: 4px;
}
@media (max-width: 600px) {
    .dkl-panel-row { flex-direction: column; }
    canvas.dkl-stats-canvas { width: 100%; height: 160px; }
    .dkl-sup-label, .dkl-bar-label { width: 80px; font-size: 10px; }
    .dkl-sup-val, .dkl-bar-vals { width: 32px; font-size: 10px; }
}
</style>

<div class="dkl-controls">
    <label>N-gram:</label>
    <button class="dkl-ngram-btn active" data-n="1">1</button>
    <button class="dkl-ngram-btn" data-n="2">2</button>
    <button class="dkl-ngram-btn" data-n="3">3</button>
    <label style="margin-left:6px">Preset:</label>
    <select id="dkl-preset">
        <option value="simple">Simple</option>
        <option value="correlated">Correlated</option>
        <option value="zipf">Zipf-like</option>
        <option value="ambiguous">Ambiguous</option>
    </select>
    <button id="dkl-reset">Reset</button>
</div>
<div class="dkl-sliders">
    <label>Speed:</label>
    <input type="range" id="dkl-speed" min="1" max="50" value="2">
    <label>LR:</label>
    <input type="range" id="dkl-lr" min="0" max="100" value="50">
</div>

<div class="dkl-editor">
    <div class="dkl-editor-header">
        <span>Teacher distribution</span>
        <div class="dkl-add-group">
            <select id="dkl-add-w1"></select>
            <select id="dkl-add-w2"></select>
            <select id="dkl-add-w3"></select>
            <button id="dkl-add-btn">+ Add</button>
        </div>
    </div>
    <div class="dkl-support-list" id="dkl-support-list"></div>
</div>

<div class="dkl-panel-label">
    KL(p&thinsp;||&thinsp;q) &mdash; forward / inclusive
    <span class="dkl-kl-val" id="dkl-kl-fwd"></span>
    <span class="dkl-legend">
        <span class="dkl-legend-swatch" style="background:rgba(0,0,0,0.12)"></span>p
        <span class="dkl-legend-swatch" style="background:rgba(220,50,50,0.5);margin-left:4px"></span>q
    </span>
</div>
<div class="dkl-panel-row">
<div class="dkl-panel" id="dkl-panel-fwd"></div>
<canvas class="dkl-stats-canvas" id="dkl-stats-fwd"></canvas>
</div>
<div class="dkl-wasted" id="dkl-wasted-fwd">
    Outside support: <span></span>
    <span class="dkl-wasted-bar dkl-wasted-bar-fwd"></span>
</div>

<div style="height: 8px;"></div>

<div class="dkl-panel-label">
    KL(q&thinsp;||&thinsp;p) &mdash; reverse / exclusive
    <span class="dkl-kl-val" id="dkl-kl-rev"></span>
    <span class="dkl-legend">
        <span class="dkl-legend-swatch" style="background:rgba(0,0,0,0.12)"></span>p
        <span class="dkl-legend-swatch" style="background:rgba(50,100,220,0.5);margin-left:4px"></span>q
    </span>
</div>
<div class="dkl-panel-row">
<div class="dkl-panel" id="dkl-panel-rev"></div>
<canvas class="dkl-stats-canvas" id="dkl-stats-rev"></canvas>
</div>
<div class="dkl-wasted" id="dkl-wasted-rev">
    Outside support: <span></span>
    <span class="dkl-wasted-bar dkl-wasted-bar-rev"></span>
</div>

<script>
(function() {
"use strict";

// ===== Constants =====
var V = 8, T = 3, N_SEQ = V * V * V; // 512
var VOCAB = ["the", "a", "big", "small", "cat", "dog", "sat", "ran"];
var LOG_ZERO = -20; // penalty for log(0) in reverse KL

function seqStr(s) { return VOCAB[s[0]] + " " + VOCAB[s[1]] + " " + VOCAB[s[2]]; }
function seqIdx(s) { return s[0] * V * V + s[1] * V + s[2]; }
function idxToSeq(i) { return [Math.floor(i / (V * V)), Math.floor(i / V) % V, i % V]; }

// ===== Presets =====
var PRESETS = {
    simple: [
        { seq: [0,4,6], p: 1/6 },
        { seq: [0,5,7], p: 1/6 },
        { seq: [1,4,7], p: 1/6 },
        { seq: [1,5,6], p: 1/6 },
        { seq: [2,4,6], p: 1/6 },
        { seq: [3,5,7], p: 1/6 }
    ],
    correlated: [
        { seq: [0,4,6], p: 0.20 },
        { seq: [0,4,7], p: 0.05 },
        { seq: [0,5,6], p: 0.05 },
        { seq: [0,5,7], p: 0.20 },
        { seq: [1,2,4], p: 0.125 },
        { seq: [1,2,5], p: 0.125 },
        { seq: [1,3,4], p: 0.125 },
        { seq: [1,3,5], p: 0.125 }
    ],
    zipf: [
        { seq: [0,4,6], p: 0.35 },
        { seq: [0,5,7], p: 0.15 },
        { seq: [1,4,7], p: 0.10 },
        { seq: [2,5,6], p: 0.08 },
        { seq: [3,4,7], p: 0.07 },
        { seq: [0,2,4], p: 0.06 },
        { seq: [1,5,6], p: 0.05 },
        { seq: [0,3,5], p: 0.05 },
        { seq: [2,4,7], p: 0.05 },
        { seq: [1,2,5], p: 0.04 }
    ],
    ambiguous: [
        { seq: [0,4,6], p: 0.125 },
        { seq: [0,5,6], p: 0.125 },
        { seq: [0,4,7], p: 0.125 },
        { seq: [0,5,7], p: 0.125 },
        { seq: [1,2,4], p: 0.125 },
        { seq: [1,3,5], p: 0.125 },
        { seq: [1,2,5], p: 0.125 },
        { seq: [1,3,4], p: 0.125 }
    ]
};

// ===== State =====
var support = [];
var teacher = new Float64Array(N_SEQ); // lookup: idx -> p
var ngramOrder = 1;
var stepsPerFrame = 2;
var baseLR = 0.1;
var dragging = null; // {idx, startX, startP}

// Student logits and Adam states (forward and reverse)
var logitsFwd, logitsRev;
var adamFwd, adamRev;

// ===== N-gram helpers =====
function getLogitCount(n) {
    // pos1: V logits (always), pos2: numCtx(1,n)*V, pos3: numCtx(2,n)*V
    var nc1 = 1, nc2 = (n === 1) ? 1 : V, nc3 = (n === 1) ? 1 : (n === 2) ? V : V * V;
    return V + nc2 * V + nc3 * V;
}

function getOffsets(n) {
    var o1 = 0;
    var nc2 = (n === 1) ? 1 : V;
    var o2 = V;
    var o3 = V + nc2 * V;
    return [o1, o2, o3];
}

function numCtx(pos, n) {
    if (pos === 0) return 1;
    if (pos === 1) return (n === 1) ? 1 : V;
    return (n === 1) ? 1 : (n === 2) ? V : V * V;
}

function getCtx(pos, n, x) {
    if (pos === 0) return 0;
    if (pos === 1) return (n === 1) ? 0 : x[0];
    if (n === 1) return 0;
    if (n === 2) return x[1];
    return x[0] * V + x[1];
}

// Softmax a slice of a logit array, return Float64Array of probs
function softmax(logits, off, sz) {
    var mx = -Infinity;
    for (var i = 0; i < sz; i++) { var v = logits[off + i]; if (v > mx) mx = v; }
    var out = new Float64Array(sz);
    var s = 0;
    for (var i = 0; i < sz; i++) { out[i] = Math.exp(logits[off + i] - mx); s += out[i]; }
    for (var i = 0; i < sz; i++) out[i] /= s;
    return out;
}

// Precompute all conditional distributions: probs[pos][ctx] = Float64Array(V)
function precomputeProbs(logits, n) {
    var offsets = getOffsets(n);
    var probs = [[], [], []];
    for (var pos = 0; pos < 3; pos++) {
        var nc = numCtx(pos, n);
        for (var c = 0; c < nc; c++) {
            probs[pos].push(softmax(logits, offsets[pos] + c * V, V));
        }
    }
    return probs;
}

function seqProb(probs, n, x) {
    return probs[0][0][x[0]]
         * probs[1][getCtx(1, n, x)][x[1]]
         * probs[2][getCtx(2, n, x)][x[2]];
}

function seqLogProb(probs, n, x) {
    return Math.log(probs[0][0][x[0]] + 1e-30)
         + Math.log(probs[1][getCtx(1, n, x)][x[1]] + 1e-30)
         + Math.log(probs[2][getCtx(2, n, x)][x[2]] + 1e-30);
}

// ===== Teacher lookup =====
function rebuildTeacher() {
    teacher.fill(0);
    for (var i = 0; i < support.length; i++) {
        teacher[seqIdx(support[i].seq)] = support[i].p;
    }
}

// ===== KL and gradients =====
function computeForwardKL(logits, n) {
    var probs = precomputeProbs(logits, n);
    var offsets = getOffsets(n);
    var count = getLogitCount(n);
    var grad = new Float64Array(count);
    var kl = 0;

    // Accumulate marginals for gradient
    // grad[off + c*V + v] = q(v|c) * margCtx(c) - margP(c, v)
    // where margCtx(c) = Σ_{x in support, ctx_pos(x)=c} p(x)
    //       margP(c,v) = Σ_{x in support, ctx_pos(x)=c, x_pos=v} p(x)
    var margCtx = [[], [], []];
    var margPcv = [[], [], []];
    for (var pos = 0; pos < 3; pos++) {
        var nc = numCtx(pos, n);
        for (var c = 0; c < nc; c++) {
            margCtx[pos].push(0);
            margPcv[pos].push(new Float64Array(V));
        }
    }

    for (var i = 0; i < support.length; i++) {
        var x = support[i].seq;
        var px = support[i].p;
        if (px <= 0) continue;
        kl += px * (Math.log(px) - seqLogProb(probs, n, x));
        for (var pos = 0; pos < 3; pos++) {
            var c = getCtx(pos, n, x);
            margCtx[pos][c] += px;
            margPcv[pos][c][x[pos]] += px;
        }
    }

    for (var pos = 0; pos < 3; pos++) {
        var nc = numCtx(pos, n);
        var off = offsets[pos];
        for (var c = 0; c < nc; c++) {
            var qp = probs[pos][c];
            var mc = margCtx[pos][c];
            var mp = margPcv[pos][c];
            for (var v = 0; v < V; v++) {
                grad[off + c * V + v] = qp[v] * mc - mp[v];
            }
        }
    }

    // Wasted mass
    var massInSupport = 0;
    for (var i = 0; i < support.length; i++) {
        massInSupport += seqProb(probs, n, support[i].seq);
    }

    return { kl: kl, grad: grad, wasted: 1 - massInSupport, probs: probs };
}

function computeReverseKL(logits, n) {
    var probs = precomputeProbs(logits, n);
    var offsets = getOffsets(n);
    var count = getLogitCount(n);

    // Accumulators for gradient: A[pos][c][v] and B[pos][c]
    var A = [[], [], []], B = [[], [], []];
    for (var pos = 0; pos < 3; pos++) {
        var nc = numCtx(pos, n);
        for (var c = 0; c < nc; c++) {
            A[pos].push(new Float64Array(V));
            B[pos].push(0);
        }
    }

    var kl = 0;
    for (var idx = 0; idx < N_SEQ; idx++) {
        var x = idxToSeq(idx);
        var qx = seqProb(probs, n, x);
        if (qx < 1e-30) continue;
        var logqx = Math.log(qx);
        var px = teacher[idx];
        var logpx = px > 0 ? Math.log(px) : LOG_ZERO;
        var R = logqx - logpx;
        kl += qx * R;
        var qR = qx * R;
        for (var pos = 0; pos < 3; pos++) {
            var c = getCtx(pos, n, x);
            A[pos][c][x[pos]] += qR;
            B[pos][c] += qR;
        }
    }

    // grad[c][v] = A[c][v] - q(v|c) * B[c]
    var grad = new Float64Array(count);
    for (var pos = 0; pos < 3; pos++) {
        var nc = numCtx(pos, n);
        var off = offsets[pos];
        for (var c = 0; c < nc; c++) {
            var qp = probs[pos][c];
            var bc = B[pos][c];
            for (var v = 0; v < V; v++) {
                grad[off + c * V + v] = A[pos][c][v] - qp[v] * bc;
            }
        }
    }

    // Wasted mass
    var massInSupport = 0;
    for (var i = 0; i < support.length; i++) {
        massInSupport += seqProb(probs, n, support[i].seq);
    }

    return { kl: kl, grad: grad, wasted: 1 - massInSupport, probs: probs };
}

// ===== Adam optimizer =====
function makeAdam(count) {
    return { m: new Float64Array(count), v: new Float64Array(count), t: 0 };
}

function adamStep(logits, grad, state, lr) {
    var beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    state.t++;
    var lrD = lr / (1 + 0.0005 * state.t);
    var bc1 = 1 - Math.pow(beta1, state.t);
    var bc2 = 1 - Math.pow(beta2, state.t);
    for (var i = 0; i < logits.length; i++) {
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * grad[i];
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * grad[i] * grad[i];
        logits[i] -= lrD * (state.m[i] / bc1) / (Math.sqrt(state.v[i] / bc2) + eps);
    }
}

// ===== Initialization =====
function initStudent() {
    var count = getLogitCount(ngramOrder);
    logitsFwd = new Float64Array(count);
    logitsRev = new Float64Array(count);
    for (var i = 0; i < count; i++) {
        logitsFwd[i] = (Math.random() - 0.5) * 0.2;
        logitsRev[i] = (Math.random() - 0.5) * 0.2;
    }
    adamFwd = makeAdam(count);
    adamRev = makeAdam(count);
}

function resetAdam() {
    var count = getLogitCount(ngramOrder);
    adamFwd = makeAdam(count);
    adamRev = makeAdam(count);
}

function loadPreset(name) {
    var p = PRESETS[name];
    support = p.map(function(s) { return { seq: s.seq.slice(), p: s.p }; });
    renormalize();
    rebuildTeacher();
    initStudent();
    rebuildAllDOM();
}

function renormalize() {
    var total = 0;
    for (var i = 0; i < support.length; i++) total += support[i].p;
    if (total > 0) for (var i = 0; i < support.length; i++) support[i].p /= total;
}

// ===== Sufficient statistics: n-gram counts =====
var statsFwdCanvas = document.getElementById('dkl-stats-fwd');
var statsRevCanvas = document.getElementById('dkl-stats-rev');

// Compute pooled n-gram counts from a distribution over sequences.
// For order k, extracts all k-grams from each 3-word sequence and sums.
// Returns Float64Array of size V^k.
function ngramCounts(probs, n, order) {
    var size = Math.pow(V, order);
    var counts = new Float64Array(size);
    for (var idx = 0; idx < N_SEQ; idx++) {
        var x = idxToSeq(idx);
        var qx = seqProb(probs, n, x);
        if (qx < 1e-20) continue;
        for (var start = 0; start <= T - order; start++) {
            var gi = 0;
            for (var k = 0; k < order; k++) gi = gi * V + x[start + k];
            counts[gi] += qx;
        }
    }
    return counts;
}

function teacherNgramCounts(order) {
    var size = Math.pow(V, order);
    var counts = new Float64Array(size);
    for (var i = 0; i < support.length; i++) {
        var x = support[i].seq, px = support[i].p;
        for (var start = 0; start <= T - order; start++) {
            var gi = 0;
            for (var k = 0; k < order; k++) gi = gi * V + x[start + k];
            counts[gi] += px;
        }
    }
    return counts;
}

// Decode an n-gram index back to a label string
function ngramLabel(gi, order) {
    var words = [];
    for (var k = order - 1; k >= 0; k--) {
        words[k] = VOCAB[gi % V];
        gi = Math.floor(gi / V);
    }
    return words.join(' ');
}

// Cached display list — rebuilt only when teacher or n-gram order changes
var cachedStatsItems = null;
var cachedStatsOrder = -1;

function rebuildStatsItems() {
    var order = ngramOrder;
    var pCounts = teacherNgramCounts(order);
    var items = [];
    var size = Math.pow(V, order);
    for (var gi = 0; gi < size; gi++) {
        if (pCounts[gi] > 0) {
            items.push({ gi: gi, label: ngramLabel(gi, order), p: pCounts[gi], qFwd: 0, qRev: 0 });
        }
    }
    items.sort(function(a, b) { return b.p - a.p; });
    if (items.length > 24) items.length = 24;
    cachedStatsItems = items;
    cachedStatsOrder = order;
}

function resizeCanvas(canvas) {
    var rect = canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    var w = Math.round(rect.width * dpr);
    var h = Math.round(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
}

function drawStats(canvas, qCounts, colorRgba) {
    resizeCanvas(canvas);
    if (!cachedStatsItems || cachedStatsItems.length === 0) return;
    var items = cachedStatsItems;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var dpr = window.devicePixelRatio || 1;
    var W = canvas.width, H = canvas.height;
    var titleH = 14 * dpr;
    var labelH = 50 * dpr; // space for rotated labels
    var padL = 4 * dpr, padR = 4 * dpr;
    var plotH = H - titleH - labelH;
    var plotW = W - padL - padR;
    var n = items.length;
    var groupW = plotW / n;
    var barW = Math.max(2, Math.min(groupW * 0.35, 8 * dpr));
    var gapInner = Math.max(1, barW * 0.15);
    var baseline = titleH + plotH;

    // Find max for scaling
    var maxVal = 0;
    for (var i = 0; i < n; i++) {
        var qv = qCounts[items[i].gi];
        if (items[i].p > maxVal) maxVal = items[i].p;
        if (qv > maxVal) maxVal = qv;
    }
    maxVal = Math.max(maxVal, 0.01) * 1.15;

    // Title
    ctx.fillStyle = '#888';
    ctx.font = 'bold ' + (9 * dpr) + 'px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(ngramOrder + '-gram counts: p vs q', W / 2, 11 * dpr);

    // Baseline
    ctx.strokeStyle = '#e8e8e8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, baseline);
    ctx.lineTo(W - padR, baseline);
    ctx.stroke();

    for (var i = 0; i < n; i++) {
        var center = padL + (i + 0.5) * groupW;
        var pH = (items[i].p / maxVal) * (plotH - 4 * dpr);
        var qH = (qCounts[items[i].gi] / maxVal) * (plotH - 4 * dpr);

        // Teacher bar (gray)
        ctx.fillStyle = 'rgba(0,0,0,0.2)';
        ctx.fillRect(center - barW - gapInner / 2, baseline - pH, barW, Math.max(pH, 0.5));

        // Student bar (colored)
        ctx.fillStyle = colorRgba;
        ctx.fillRect(center + gapInner / 2, baseline - qH, barW, Math.max(qH, 0.5));

        // Rotated label
        ctx.save();
        ctx.translate(center, baseline + 4 * dpr);
        ctx.rotate(-Math.PI / 3);
        ctx.fillStyle = '#999';
        ctx.font = (7.5 * dpr) + 'px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(items[i].label, 0, 0);
        ctx.restore();
    }
}

// ===== DOM references =====
var supportListEl = document.getElementById('dkl-support-list');
var panelFwdEl = document.getElementById('dkl-panel-fwd');
var panelRevEl = document.getElementById('dkl-panel-rev');
var klFwdEl = document.getElementById('dkl-kl-fwd');
var klRevEl = document.getElementById('dkl-kl-rev');
var wastedFwdEl = document.getElementById('dkl-wasted-fwd');
var wastedRevEl = document.getElementById('dkl-wasted-rev');

// Bar element references (rebuilt when support changes)
var supBarEls = [];   // [{track, bar, val}]
var fwdBarEls = [];   // [{pBar, qBar, valSpan}]
var revBarEls = [];

// Sort support by probability descending, rebuild all DOM
function rebuildAllDOM() {
    support.sort(function(a, b) { return b.p - a.p; });
    rebuildTeacher();
    rebuildStatsItems();
    rebuildSupportDOM();
    rebuildPanelDOM(panelFwdEl, 'fwd');
    rebuildPanelDOM(panelRevEl, 'rev');
}

function rebuildSupportDOM() {
    supportListEl.innerHTML = '';
    supBarEls = [];
    for (var i = 0; i < support.length; i++) {
        var row = document.createElement('div');
        row.className = 'dkl-sup-row';

        var label = document.createElement('span');
        label.className = 'dkl-sup-label';
        label.textContent = seqStr(support[i].seq);

        var track = document.createElement('div');
        track.className = 'dkl-sup-track';
        track.dataset.idx = i;

        var bar = document.createElement('div');
        bar.className = 'dkl-sup-bar';
        track.appendChild(bar);

        var val = document.createElement('span');
        val.className = 'dkl-sup-val';

        var rm = document.createElement('button');
        rm.className = 'dkl-sup-rm';
        rm.textContent = '\u00d7';
        rm.dataset.idx = i;

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(val);
        row.appendChild(rm);
        supportListEl.appendChild(row);

        supBarEls.push({ track: track, bar: bar, val: val });
    }
    updateSupportBars();
}

function updateSupportBars() {
    var maxP = 0;
    for (var i = 0; i < support.length; i++) if (support[i].p > maxP) maxP = support[i].p;
    maxP = Math.max(maxP, 0.01);
    for (var i = 0; i < support.length; i++) {
        var pct = (support[i].p / maxP) * 100;
        supBarEls[i].bar.style.width = pct + '%';
        supBarEls[i].val.textContent = support[i].p.toFixed(3);
    }
}

function rebuildPanelDOM(panelEl, direction) {
    panelEl.innerHTML = '';
    var barEls = [];
    for (var i = 0; i < support.length; i++) {
        var row = document.createElement('div');
        row.className = 'dkl-bar-row';

        var label = document.createElement('span');
        label.className = 'dkl-bar-label';
        label.textContent = seqStr(support[i].seq);

        var track = document.createElement('div');
        track.className = 'dkl-bar-track';

        var pBar = document.createElement('div');
        pBar.className = 'dkl-bar-p';

        var qBar = document.createElement('div');
        qBar.className = 'dkl-bar-q dkl-bar-q-' + direction;

        track.appendChild(pBar);
        track.appendChild(qBar);

        var vals = document.createElement('span');
        vals.className = 'dkl-bar-vals';

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(vals);
        panelEl.appendChild(row);

        barEls.push({ pBar: pBar, qBar: qBar, valSpan: vals });
    }
    if (direction === 'fwd') fwdBarEls = barEls;
    else revBarEls = barEls;
}

function updatePanelBars(barEls, probs, n, klVal, wastedMass, direction) {
    // Compute q for each support sequence
    var qVals = [];
    var maxV = 0;
    for (var i = 0; i < support.length; i++) {
        var q = seqProb(probs, n, support[i].seq);
        qVals.push(q);
        if (support[i].p > maxV) maxV = support[i].p;
        if (q > maxV) maxV = q;
    }
    maxV = Math.max(maxV, 0.001) * 1.05;

    for (var i = 0; i < support.length; i++) {
        barEls[i].pBar.style.width = (support[i].p / maxV * 100) + '%';
        barEls[i].qBar.style.width = (qVals[i] / maxV * 100) + '%';
        barEls[i].valSpan.textContent = qVals[i].toFixed(3);
    }

    // KL value
    var klEl = (direction === 'fwd') ? klFwdEl : klRevEl;
    klEl.textContent = 'KL = ' + klVal.toFixed(4);

    // Wasted mass
    var wEl = (direction === 'fwd') ? wastedFwdEl : wastedRevEl;
    var pct = Math.max(0, wastedMass * 100);
    wEl.querySelector('span').textContent = pct.toFixed(1) + '%';
    wEl.querySelector('.dkl-wasted-bar').style.width = Math.min(pct, 100) + 'px';
}

// ===== Interaction: support editor drag =====
supportListEl.addEventListener('mousedown', function(e) {
    var track = e.target.closest('.dkl-sup-track');
    if (track) {
        var idx = parseInt(track.dataset.idx);
        startDrag(idx, e.clientX, track);
        e.preventDefault();
    }
});
supportListEl.addEventListener('touchstart', function(e) {
    var track = e.target.closest('.dkl-sup-track');
    if (track && e.touches.length === 1) {
        var idx = parseInt(track.dataset.idx);
        startDrag(idx, e.touches[0].clientX, track);
        e.preventDefault();
    }
}, { passive: false });

function startDrag(idx, clientX, track) {
    dragging = { idx: idx, track: track };
    handleDragMove(clientX);
}

function handleDragMove(clientX) {
    if (!dragging) return;
    var rect = dragging.track.getBoundingClientRect();
    var frac = (clientX - rect.left) / rect.width;
    frac = Math.max(0.01, Math.min(0.98, frac));

    // Set this sequence's share, scale others proportionally
    var idx = dragging.idx;
    var oldP = support[idx].p;
    var otherTotal = 1 - oldP;
    var newP = frac; // approximate: probability proportional to bar fill
    // Scale: the bar is displayed as p / maxP, but for dragging we want to
    // set the absolute probability. Find current maxP:
    var maxP = 0;
    for (var i = 0; i < support.length; i++) if (support[i].p > maxP) maxP = support[i].p;
    newP = frac * Math.max(maxP, 0.01);
    newP = Math.max(0.005, Math.min(0.99, newP));

    support[idx].p = newP;
    renormalize();
    rebuildTeacher();
    updateSupportBars();
}

document.addEventListener('mousemove', function(e) {
    if (dragging) handleDragMove(e.clientX);
});
document.addEventListener('touchmove', function(e) {
    if (dragging && e.touches.length === 1) {
        handleDragMove(e.touches[0].clientX);
        e.preventDefault();
    }
}, { passive: false });
document.addEventListener('mouseup', function() {
    if (dragging) { dragging = null; resetAdam(); rebuildStatsItems(); }
});
document.addEventListener('touchend', function() {
    if (dragging) { dragging = null; resetAdam(); rebuildStatsItems(); }
});

// ===== Interaction: remove sequence =====
supportListEl.addEventListener('click', function(e) {
    var rm = e.target.closest('.dkl-sup-rm');
    if (rm && support.length > 1) {
        var idx = parseInt(rm.dataset.idx);
        support.splice(idx, 1);
        renormalize();
        rebuildTeacher();
        resetAdam();
        rebuildAllDOM();
    }
});

// ===== Interaction: add sequence =====
var addW1 = document.getElementById('dkl-add-w1');
var addW2 = document.getElementById('dkl-add-w2');
var addW3 = document.getElementById('dkl-add-w3');
[addW1, addW2, addW3].forEach(function(sel) {
    for (var i = 0; i < V; i++) {
        var opt = document.createElement('option');
        opt.value = i;
        opt.textContent = VOCAB[i];
        sel.appendChild(opt);
    }
});

document.getElementById('dkl-add-btn').addEventListener('click', function() {
    var s = [parseInt(addW1.value), parseInt(addW2.value), parseInt(addW3.value)];
    // Check if already in support
    for (var i = 0; i < support.length; i++) {
        if (support[i].seq[0] === s[0] && support[i].seq[1] === s[1] && support[i].seq[2] === s[2]) {
            return; // already present
        }
    }
    support.push({ seq: s, p: 1 / (support.length + 1) });
    renormalize();
    rebuildTeacher();
    resetAdam();
    rebuildAllDOM();
});

// ===== Interaction: n-gram order =====
var ngramBtns = document.querySelectorAll('.dkl-ngram-btn');
ngramBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
        ngramOrder = parseInt(this.dataset.n);
        ngramBtns.forEach(function(b) { b.classList.remove('active'); });
        this.classList.add('active');
        initStudent();
        rebuildStatsItems();
    });
});

// ===== Interaction: preset =====
document.getElementById('dkl-preset').addEventListener('change', function() {
    loadPreset(this.value);
});

// ===== Interaction: reset =====
document.getElementById('dkl-reset').addEventListener('click', function() {
    initStudent();
});

// ===== Interaction: speed / LR =====
document.getElementById('dkl-speed').addEventListener('input', function() {
    stepsPerFrame = parseInt(this.value);
});
document.getElementById('dkl-lr').addEventListener('input', function() {
    baseLR = 0.01 * Math.pow(100, this.value / 100);
});

// ===== Animation loop =====
var lastFwdResult = null, lastRevResult = null;

function frame() {
    if (!dragging) {
        for (var i = 0; i < stepsPerFrame; i++) {
            var fwd = computeForwardKL(logitsFwd, ngramOrder);
            adamStep(logitsFwd, fwd.grad, adamFwd, baseLR);

            var rev = computeReverseKL(logitsRev, ngramOrder);
            adamStep(logitsRev, rev.grad, adamRev, baseLR);

            lastFwdResult = fwd;
            lastRevResult = rev;
        }
    } else {
        // Recompute display values even during drag
        lastFwdResult = computeForwardKL(logitsFwd, ngramOrder);
        lastRevResult = computeReverseKL(logitsRev, ngramOrder);
    }

    if (cachedStatsOrder !== ngramOrder) rebuildStatsItems();

    if (lastFwdResult && fwdBarEls.length === support.length) {
        updatePanelBars(fwdBarEls, lastFwdResult.probs, ngramOrder,
                        lastFwdResult.kl, lastFwdResult.wasted, 'fwd');
        var qFwdCounts = ngramCounts(lastFwdResult.probs, ngramOrder, ngramOrder);
        drawStats(statsFwdCanvas, qFwdCounts, 'rgba(220, 50, 50, 0.55)');
    }
    if (lastRevResult && revBarEls.length === support.length) {
        updatePanelBars(revBarEls, lastRevResult.probs, ngramOrder,
                        lastRevResult.kl, lastRevResult.wasted, 'rev');
        var qRevCounts = ngramCounts(lastRevResult.probs, ngramOrder, ngramOrder);
        drawStats(statsRevCanvas, qRevCounts, 'rgba(50, 100, 220, 0.55)');
    }

    requestAnimationFrame(frame);
}

// ===== Init =====
loadPreset('simple');
frame();

})();
</script>
</div>

**What to look for:**

- **n=1 (unigram):** The student can't capture word correlations. Forward KL
  (red, top) spreads mass across all supported sequences&mdash;it covers the
  teacher's support, but wastes mass elsewhere. Reverse KL (blue, bottom)
  concentrates on a few favorites and keeps wasted mass low. Try the
  "Correlated" preset to see this clearly.

- **n=2 (bigram):** Pairwise dependencies are captured, and the gap narrows.
  With "Correlated," bigrams model patterns like "cat&rarr;sat" and
  "dog&rarr;ran" that unigrams miss entirely.

- **n=3 (trigram):** Full expressiveness&mdash;both KL directions converge to
  $\approx 0$, perfectly matching the teacher.

- **Wasted mass:** Forward KL ignores mass outside the teacher's
  support&mdash;the student can freely assign probability to nonsense sequences
  like "small the big." Reverse KL penalizes this heavily, driving wasted mass
  to near zero.

- **Ambiguous preset:** Two equally-weighted clusters of phrases. Reverse KL may
  pick one cluster and abandon the other; forward KL tries to cover both.

- **N-gram counts** (right): Shows the sufficient statistics of the
  model&mdash;pooled n-gram frequencies under teacher (gray) and student
  (colored). Forward KL (MLE) is moment-matching: these bars converge exactly.
  Reverse KL has no such guarantee, so the counts may diverge even at
  convergence. Switch n-gram order to see unigram, bigram, or trigram counts.

#### Connection to real language models

Real language models use the same autoregressive factorization: $p(x_1, \ldots, x_T) = p(\text{eos} \mid x_{\le T}) \prod_{t=1}^{T} p(x_t \mid x_{\lt t})$. A full-context model ($n = T$) is a trigram here. (Our demo fixes sequence length at $T{=}3$, so there is no stopping probability to learn.)

Knowledge distillation trains a smaller student to match a teacher's output
distribution. The choice of KL direction matters: forward KL (standard maximum
likelihood) produces broad, covering students, while reverse KL (used in some
RLHF variants) produces sharper, more focused ones. The n-gram order is a
stand-in for model capacity&mdash;a smaller model simply can't represent all the
teacher's correlations, and *how* it fails depends on the objective.

The n-gram model is an exponential family: each conditional $q(x_t \mid
\text{ctx})$ is a categorical with natural parameters (the logits) and
sufficient statistics (token indicators). At the MLE solution (forward KL), the
expected sufficient statistics under $q$ match the observed statistics under
$p$&mdash;this is the moment-matching property. Reverse KL has no such
guarantee, so the marginals panel on the right shows how the two directions
differ in what statistics they preserve.
