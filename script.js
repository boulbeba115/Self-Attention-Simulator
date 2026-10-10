    /* The training text (CORPUS_TEXT) lives in corpus.js, which is loaded before this file. */

    /* =====================================================================
       MATHS HELPERS
       ===================================================================== */
    const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
    const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
    const f = v => v === -Infinity ? '−∞' : (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2).replace('-', '−'));
    const f3 = v => (Math.abs(v) < 0.0005 ? '0.000' : v.toFixed(3).replace('-', '−'));
    const pct = v => (v * 100).toFixed(1) + '%';
    const tokenize = s => s.toLowerCase().replace(/[’`]/g, "'").replace(/[^\p{L}\p{N}'\s]/gu, ' ').split(/\s+/).filter(Boolean);
    function mulberry32(a) {return function () {a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296}}
    function hashStr(s) {let h = 2166136261; for (const c of s) {h ^= c.codePointAt(0); h = Math.imul(h, 16777619)} return h >>> 0}
    function gauss(r) {let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)}
    const randMat = (r, R, C, std) => Array.from({length: R}, () => Array.from({length: C}, () => gauss(r) * std));
    const matmul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
    const T = A => A[0].map((_, j) => A.map(r => r[j]));
    const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
    const scale = (A, c) => A.map(r => r.map(v => v * c));
    const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
    const cos = (a, b) => dot(a, b) / ((Math.hypot(...a) * Math.hypot(...b)) || 1);
    function softmaxRow(r) {const m = Math.max(...r.filter(isFinite)); const e = r.map(v => isFinite(v) ? Math.exp(v - m) : 0); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s)}
    function rowStats(r) {const mu = r.reduce((a, b) => a + b, 0) / r.length; const sd = Math.sqrt(r.reduce((a, b) => a + (b - mu) ** 2, 0) / r.length + 1e-5); return {mu, sd}}
    const layerNorm = A => A.map(r => {const {mu, sd} = rowStats(r); return r.map(x => (x - mu) / sd)});
    function posEnc(n, d) {return Array.from({length: n}, (_, p) => Array.from({length: d}, (_, j) => {const i = Math.floor(j / 2); const a = p / Math.pow(10000, 2 * i / d); return j % 2 === 0 ? Math.sin(a) : Math.cos(a)}))}

    /* =====================================================================
       "TRAINING": phrase counts + learned word vectors
       ===================================================================== */
    const NMAX = 4, BACKOFF = 0.2;   // longest phrase that is looked up, and how much say the shorter phrases keep
    /* The whole training text as one long list of word numbers, with a full stop (number 0) between sentences,
       plus an index of where every word occurs. Any phrase can then be looked up without storing every phrase. */
    const TRAIN = (() => {
      const sents = CORPUS_TEXT.trim().split('\n').map(tokenize).filter(t => t.length);
      const seen = new Set(); let N = 0;
      for (const t of sents) {for (const w of t) seen.add(w); N += t.length + 1;}
      const vocab = ['.'].concat([...seen].sort()), id = Object.create(null); vocab.forEach((w, i) => id[w] = i);
      const stream = new Int32Array(N + 1); let p = 1;
      for (const t of sents) {for (const w of t) stream[p++] = id[w]; stream[p++] = 0;}
      const V = vocab.length, uni = new Int32Array(V), start = new Int32Array(V + 1);
      for (let i = 1; i < stream.length; i++) uni[stream[i]]++;
      for (let i = 0; i < stream.length; i++) start[stream[i] + 1]++;
      for (let i = 0; i < V; i++) start[i + 1] += start[i];
      const where = new Int32Array(stream.length), fill = start.slice(0, V);
      for (let i = 0; i < stream.length; i++) where[fill[stream[i]]++] = i;
      return {vocab, id, stream, uni, N, start, where, sentences: sents.length};
    })();
    const CORPUS_VOCAB = TRAIN.vocab, CORPUS_ID = TRAIN.id;
    const ctxLabel = c => c === '.' ? '[start]' : c.replace(/^\. /, '[start] ');

    /* How likely is each word after this context? The longest phrase seen in training counts most,
       and every shorter phrase still adds a little (interpolated back-off).
       Returns P, one probability per word of `vocab` (the training words first, then any new words). */
    function ngram(ctxToks, vocab) {
      const {stream, start, where, uni, N, id} = TRAIN, V = vocab.length, len = stream.length;
      const words = ['.'].concat(ctxToks), seq = words.map(t => t in id ? id[t] : -1), den = N + 0.5 * V;
      let P = new Float64Array(V); for (let i = 0; i < V; i++) P[i] = ((i < uni.length ? uni[i] : 0) + 0.5) / den;
      const tally = (ps, off) => {const m = new Map(); for (const p of ps) {const w = stream[p + off]; m.set(w, (m.get(w) || 0) + 1);} return m;};
      const mix = (m, tot) => {const Q = P.map(x => BACKOFF * x); for (const [w, k] of m) Q[w] += (1 - BACKOFF) * k / tot; P = Q;};
      const items = m => [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([w, k]) => [vocab[w], k]);
      const occ = (w, room, ok) => {const ps = []; if (w >= 0) for (let k = start[w]; k < start[w + 1]; k++) {const p = where[k]; if (p + room < len && ok(p)) ps.push(p);} return ps;};
      const tries = [], top = Math.min(NMAX, seq.length), L = seq.length; let match = null;
      let ps = occ(seq[L - 1], 1, () => true);
      for (let n = 1; n <= top; n++) {
        if (n > 1) {const want = seq[L - n]; ps = want < 0 ? [] : ps.filter(p => p - n + 1 >= 0 && stream[p - n + 1] === want);}
        const c = ctxLabel(words.slice(-n).join(' '));
        tries.unshift({c, ok: ps.length > 0});
        if (!ps.length) {for (let m = n + 1; m <= top; m++) tries.unshift({c: ctxLabel(words.slice(-m).join(' ')), ok: false}); break;}
        const m = tally(ps, 1); mix(m, ps.length); match = {c, n, tot: ps.length, items: items(m)};
        if (seq[L - n] === 0) break;   // never read back past the start of the sentence
      }
      if (!match && L >= 2) {   // unknown last word: skip over it and use the words before
        const a = seq[L - 2], found = [];
        let qs = occ(a, 2, p => stream[p + 1] !== 0);
        for (let n = 1; n <= 2 && L > n; n++) {
          if (n === 2) {const b = seq[L - 3]; qs = a === 0 || b < 0 ? [] : qs.filter(p => p >= 1 && stream[p - 1] === b);}
          const c = ctxLabel(words.slice(-1 - n, -1).join(' ')) + ' ___';
          found.unshift({c, ok: qs.length > 0});
          if (!qs.length) break;
          const m = tally(qs, 2); mix(m, qs.length); match = {c, n: 0, tot: qs.length, items: items(m), skip: true};
        }
        tries.push(...found);
      }
      return {P, match, tries};
    }

    /* Word vectors. Count which words come shortly before and after each word, keep the pairs that
       happen more often than chance, then squeeze that big table down to d numbers per word.
       E[w] describes the word itself; U[w] describes what tends to come right before w. */
    const EMB_CACHE = {};
    let PPMI = null;
    function cooccur() {
      if (PPMI) return PPMI;
      const {stream} = TRAIN, V = CORPUS_VOCAB.length, V2 = 2 * V, len = stream.length;
      const cnt = new Map(), rs = new Float64Array(V), cs = new Float64Array(V2); let tot = 0;
      const bump = (r, c, w) => {const k = r * V2 + c; cnt.set(k, (cnt.get(k) || 0) + w); rs[r] += w; cs[c] += w; tot += w;};
      for (let i = 1; i < len; i++) {
        const a = stream[i]; if (a === 0) continue;
        for (let k = 1; k <= 3 && i + k < len; k++) {
          const b = stream[i + k], w = 1 / k;
          bump(a, b, w); bump(b, V + a, w);   // column b: "b comes soon after"; column V + a: "a comes soon before"
          if (b === 0) break;                 // stay inside the sentence
        }
      }
      const nz0 = cnt.size, ri = new Int32Array(nz0), ci = new Int32Array(nz0), xv = new Float64Array(nz0), rn = new Float64Array(V); let nz = 0;
      for (const [k, c] of cnt) {const r = Math.floor(k / V2), col = k % V2, v = Math.log(c * tot / (rs[r] * cs[col])); if (v > 0) {ri[nz] = r; ci[nz] = col; xv[nz] = v; rn[r] += v * v; nz++;} }
      for (let e = 0; e < nz; e++) xv[e] /= Math.sqrt(rn[ri[e]]);   // every word gets the same say, however common it is
      return PPMI = {ri, ci, xv, nz};
    }
    function learnEmb(d) {
      if (EMB_CACHE[d]) return EMB_CACHE[d];
      const W = CORPUS_VOCAB, V = W.length, V2 = 2 * V, {ri, ci, xv, nz} = cooccur();
      const mtv = q => {const y = new Float64Array(V2); for (let e = 0; e < nz; e++) y[ci[e]] += xv[e] * q[ri[e]]; return y;};
      const mv = y => {const z = new Float64Array(V); for (let e = 0; e < nz; e++) z[ri[e]] += xv[e] * y[ci[e]]; return z;};
      const r = mulberry32((d * 31 + 3) >>> 0);
      let Qs = Array.from({length: d}, () => Float64Array.from({length: V}, () => gauss(r)));
      for (let it = 0; it < 14; it++) {
        const Y = Qs.map(q => mv(mtv(q)));
        for (let k = 0; k < d; k++) {
          for (let p = 0; p < k; p++) {let dp = 0; for (let a = 0; a < V; a++) dp += Y[k][a] * Y[p][a]; for (let a = 0; a < V; a++) Y[k][a] -= dp * Y[p][a];}
          let nr = 0; for (let a = 0; a < V; a++) nr += Y[k][a] ** 2; nr = Math.sqrt(nr) || 1; for (let a = 0; a < V; a++) Y[k][a] /= nr;
        }
        Qs = Y;
      }
      const Vs = [], sig = [];
      Qs.forEach((q, k) => {
        const y = mtv(q); let s = 0; for (let c = 0; c < V2; c++) s += y[c] ** 2; s = Math.sqrt(s) || 1;
        let sum = 0; for (let a = 0; a < V; a++) sum += q[a]; const sign = sum < 0 ? -1 : 1;
        for (let a = 0; a < V; a++) q[a] *= sign; for (let c = 0; c < V2; c++) y[c] *= sign / s;
        Vs[k] = y; sig[k] = s;
      });
      const E = Object.create(null), U = Object.create(null), sd = Math.sqrt(d);
      const mean = Qs.map((q, k) => {let s = 0; for (let a = 0; a < V; a++) s += q[a]; return s / V * Math.sqrt(sig[k]);});
      const vr = new Float64Array(d);
      W.forEach((w, a) => {
        const v = Qs.map((q, k) => q[a] * Math.sqrt(sig[k]) - mean[k]), nr = Math.hypot(...v) || 1;
        E[w] = v.map(x => x / nr * sd); E[w].forEach((x, k) => vr[k] += x * x);
        const o = Vs.map((y, k) => y[a] * Math.sqrt(sig[k])), no = Math.hypot(...o) || 1;
        U[w] = o.map(x => x / no * sd);
      });
      const dims = [...Array(d).keys()].sort((a, b) => vr[b] - vr[a]).slice(0, 2);
      return EMB_CACHE[d] = {E, U, dims};
    }

    function buildModel(cfg, toks) {
      const d = cfg.d, base = learnEmb(d);
      const E = Object.create(base.E), U = Object.create(base.U), vid = Object.create(CORPUS_ID);
      const vocab = CORPUS_VOCAB.slice(), unknown = new Set();
      for (const t of toks) if (!(t in E)) {
        unknown.add(t); vid[t] = vocab.length; vocab.push(t);
        const r = mulberry32((hashStr(t) ^ Math.imul(cfg.seed, 2654435761)) >>> 0);
        const v = Array.from({length: d}, () => gauss(r)); const nr = Math.hypot(...v);
        E[t] = v.map(x => x / nr * Math.sqrt(d)); U[t] = v.map(() => 0);
      }
      const r = mulberry32((cfg.seed * 7919 + d * 131 + 17) >>> 0);
      const noisyI = (base, ns) => Array.from({length: d}, (_, i) => Array.from({length: d}, (_, j) => (i === j ? base : 0) + gauss(r) * ns / Math.sqrt(d)));
      const layers = [];
      for (let l = 0; l < cfg.L; l++) layers.push({
        Wq: noisyI(1, .45), Wk: noisyI(1, .45), Wv: noisyI(1, .4), Wo: noisyI(.6, .4),
        W1: randMat(r, d, 2 * d, 1 / Math.sqrt(d)), W2: randMat(r, 2 * d, d, .5 / Math.sqrt(2 * d))
      });
      return {vocab, vid, E, U, layers, V: vocab.length, unknown, dims: base.dims};
    }

    /* =====================================================================
       ONE FORWARD PASS = one predicted word
       ===================================================================== */
    function runIteration(tokens, g, model, cfg) {
      const n = tokens.length, d = cfg.d, sq = Math.sqrt(d);
      const ctx = {
        tokens, n, d, sq, g, model, cfg,
        rl: tokens.map((t, i) => `${i} ${t}`), dl: [...Array(d)].map((_, j) => 'd' + j), hl2: [...Array(2 * d)].map((_, j) => 'h' + j)
      };
      const S = [];
      S.push({key: 'tok', ctx});
      const Emb = tokens.map(t => model.E[t]);
      S.push({key: 'emb', ctx, Emb});
      const PE = posEnc(n, d); let X = add(Emb, PE);
      S.push({key: 'pos', ctx, Emb, PE, X});
      model.layers.forEach((W, l) => {
        const Q = matmul(X, W.Wq), K = matmul(X, W.Wk), V = matmul(X, W.Wv);
        S.push({key: 'qkv', ctx, layer: l, X, W, Q, K, V});
        const Kt = T(K), Sc = scale(matmul(Q, Kt), 1 / sq);
        S.push({key: 'score', ctx, layer: l, Q, Kt, S: Sc});
        const Sm = Sc.map((r, i) => r.map((v, j) => j > i ? -Infinity : v));
        S.push({key: 'mask', ctx, layer: l, S: Sc, Sm});
        const A = Sm.map(softmaxRow);
        S.push({key: 'soft', ctx, layer: l, Sm, A});
        const Z = matmul(A, V);
        S.push({key: 'mix', ctx, layer: l, A, V, Z});
        const O = matmul(Z, W.Wo), H = layerNorm(add(X, O));
        S.push({key: 'norm', ctx, layer: l, X, Z, Wo: W.Wo, O, H});
        const pre = matmul(H, W.W1), Hid = pre.map(r => r.map(v => Math.max(0, v)));
        const F = matmul(Hid, W.W2), Xo = layerNorm(add(H, F));
        S.push({key: 'ffn', ctx, layer: l, H, W1: W.W1, pre, Hid, W2: W.W2, F, Xo});
        X = Xo;
      });
      const h = X[n - 1], kn = ngram(tokens, model.vocab);
      const scored = model.vocab.map((w, i) => {
        const net = dot(h, model.U[w]) / sq, lp = Math.log(Math.max(kn.P[i], 1e-12));
        return {w, net, a: cfg.alpha * net, lp, b: cfg.beta * lp, z: cfg.alpha * net + cfg.beta * lp};
      }).sort((a, b) => b.z - a.z);
      S.push({key: 'logits', ctx, h, scored, kn});
      const zs = scored.map(s => s.z / cfg.temp), mx = Math.max(...zs);
      const ex = zs.map(z => Math.exp(z - mx)), sum = ex.reduce((a, b) => a + b, 0);
      scored.forEach((s, i) => s.p = ex[i] / sum);
      const K = Math.min(cfg.topk, scored.length), kept = scored.slice(0, K);
      const ks = kept.reduce((a, s) => a + s.p, 0); kept.forEach(s => s.q = s.p / ks);
      let chosen = kept[0], r = null;
      if (cfg.decode === 'sample') {
        r = mulberry32((cfg.seed * 31 + g * 977 + 5 + (cfg.draw || 0)) >>> 0)(); let c = 0;
        chosen = kept[K - 1];
        for (const s of kept) {c += s.q; if (r < c) {chosen = s; break;} }
      }
      S.push({key: 'pick', ctx, kept, chosen, r});
      return {stages: S, next: chosen.w, top: kept.slice(0, 3).map(s => ({w: s.w, q: s.q}))};
    }

    /* =====================================================================
       PLAIN-LANGUAGE GLOSSARY (shown in pop-overs)
       ===================================================================== */
    const GLOSS = {
      token: {t: 'Token', d: 'A piece of text the model reads as one unit. Here one word is one token. Big models also split words into pieces, like “play” + “ing”.'},
      vocab: {t: 'Vocabulary', d: 'The list of every token the model knows. Each one has a fixed number, called its ID.'},
      embedding: {t: 'Embedding', d: 'A list of numbers that stands for a word’s meaning. Words used in similar ways get similar lists.', ex: 'cat → [0.8, −1.2, 0.3, …]'},
      vector: {t: 'Vector', d: 'Simply a list of numbers. You can picture it as an arrow pointing somewhere in space.', ex: '[0.2, −1.1, 0.7]'},
      matrix: {t: 'Matrix', d: 'A table of numbers. Here each row is usually one word, and each column is one feature of it.', lesson: 'mm'},
      weights: {t: 'Weights', d: 'Numbers the model learns during training. Multiplying by them transforms a word’s vector into something new.', lesson: 'mm'},
      position: {t: 'Positional encoding', d: 'A unique wave pattern added to each position, so the model can tell the first word from the fifth.'},
      xinput: {t: 'Input matrix X', d: 'One row per word: its meaning (embedding) plus its position. Everything after this works on X.'},
      query: {t: 'Query', d: 'What a word is looking for. “The” at the end might be looking for “what is this about?”', lesson: 'mm'},
      key: {t: 'Key', d: 'What a word offers to others, like a label. Queries are compared against keys.', lesson: 'mm'},
      value: {t: 'Value', d: 'The information a word passes along when another word pays attention to it.', lesson: 'mm'},
      dot: {t: 'Dot product', d: 'Multiply two lists pair by pair, then add it all up. The result is big when the two lists point the same way.', ex: '[1, 2] · [3, 4] = 1×3 + 2×4 = 11', lesson: 'dot'},
      transpose: {t: 'Transpose, Kᵀ', d: 'Flipping a matrix so its rows become columns. This lets every query meet every key in one multiplication.'},
      sqrtd: {t: 'Why divide by √d?', d: 'Dot products grow as vectors get longer. Dividing by the square root of the vector size keeps scores calm, so softmax doesn’t become too extreme.'},
      scores: {t: 'Attention scores', d: 'Row = the word asking, column = the word being looked at. Higher means a better match between query and key.', lesson: 'dot'},
      mask: {t: 'Mask', d: 'Hides future words by setting their score to −∞. After softmax, they get exactly 0% attention.'},
      ninf: {t: 'Minus infinity (−∞)', d: 'A number smaller than any other. e to the power of −∞ is 0, which is why masked words get 0%.'},
      softmax: {t: 'Softmax', d: 'Turns any list of numbers into percentages that add up to 100%. Bigger numbers get a much bigger share.', ex: '[2, 1, 0.1] → [66%, 24%, 10%]', lesson: 'sm'},
      attention: {t: 'Attention', d: 'How much one word focuses on each other word when it updates its own meaning.', lesson: 'sm'},
      blend: {t: 'Blended vectors, Z', d: 'Each word’s new vector: a mix of the value vectors, weighted by its attention percentages.', lesson: 'mm'},
      residual: {t: 'Residual connection', d: 'Adding the input back to the output, so nothing important gets lost along the way.'},
      layernorm: {t: 'LayerNorm', d: 'Rescales each row so its average is 0 and its spread is 1. It keeps numbers from growing out of control.', ex: '[2, 4, 6] → [−1.22, 0, 1.22]'},
      relu: {t: 'ReLU', d: 'The simplest rule in deep learning: keep positive numbers, turn negative ones into 0.', ex: '[1.5, −0.7, 0.2] → [1.5, 0, 0.2]'},
      ffn: {t: 'Feed-forward network', d: 'A tiny neural network with two weight matrices. It processes each word on its own, without looking at the others.'},
      logit: {t: 'Logit', d: 'A raw score for a candidate word, before it becomes a probability. Higher is better.'},
      prob: {t: 'Probability', d: 'A chance from 0 to 1 (0% to 100%). All the candidates’ probabilities add up to 1.'},
      temperature: {t: 'Temperature', d: 'Scores are divided by it before softmax. Low temperature: the favourite almost always wins. High: surprises happen.'},
      topk: {t: 'Top-k', d: 'Only the k highest-scoring words may be chosen. Everything else is dropped before the draw.'},
      ngram: {t: 'Training-text knowledge', d: 'Counts of which word followed which phrase (up to four words long) in the training sentences. It stands in for what a real model learns in training. At 0 the network is on its own.'},
      depth: {t: 'Words to predict', d: 'How many words the model writes before it pauses. Each new word means running all twelve steps again. There is no upper limit: press “Keep writing” at the end for more.'},
      stop: {t: 'Stop at end of sentence', d: 'When the model predicts a full stop, it stops writing, even if “Words to predict” allows more. Turn it off to let it start a new sentence.'},
      decode: {t: 'How to choose', d: 'Random draw picks a word with odds equal to its probability, so each run can differ. Most likely always takes the top word, so temperature and top-k no longer change the result.'},
      vectorsize: {t: 'Vector size', d: 'How many numbers describe each word. Larger means more room for meaning. GPT-3 uses 12,288.'},
      layers: {t: 'Layers', d: 'How many times steps 4 to 10 are repeated. Real models stack dozens of layers.'},
      seed: {t: 'Random seed', d: 'Starting point for the network’s random weights. Change it to get a slightly different network. The random draw itself is fresh on every run.'},
      speed: {t: 'Animation speed', d: 'How fast each step is animated.'},
      hvec: {t: 'Final vector h', d: 'The last word’s vector after all the layers. It now carries information from the whole sentence.'}
    };
    const tm = (key, text) => `<button class="term" data-g="${key}">${text}</button>`;

    /* =====================================================================
       WORKED CALCULATIONS (bottom sheet)
       ===================================================================== */
    function mmCalc(A, B, aId, bId, name, post) {
      return (i, j) => {
        const terms = A[i].map((a, k) => `(${f(a)} × ${f(B[k][j])})`);
        const sum = A[i].reduce((s, a, k) => s + a * B[k][j], 0);
        let html = `<div class="calc-title">Row ${i} of ${aId} meets column ${j} of ${bId}</div><div class="calc-math">Multiply each pair, then add them up:</div><div class="calc-math">${terms.join(' + ')}</div><div class="calc-math">= <b>${f(sum)}</b></div>`;
        if (post) html += post(sum);
        return {html, hl: [{id: aId, row: i}, {id: bId, col: j}], lesson: 'mm'};
      };
    }
    function lnCalc(Xa, Ya, xId, yId, name, Out) {
      return (i, j) => {
        const R = Xa[i].map((v, k) => v + Ya[i][k]), {mu, sd} = rowStats(R);
        return {
          html: `<div class="calc-title">Add, then rescale the row</div>
      <div class="calc-math">1. Add the input back: ${f(Xa[i][j])} + ${f(Ya[i][j])} = ${f(R[j])}</div>
      <div class="calc-math">2. The whole row ${i} is now: ${R.map(f).join(',  ')}</div>
      <div class="calc-math">3. Its average is ${f3(mu)} and its spread is ${f3(sd)}</div>
      <div class="calc-math">4. (${f(R[j])} − ${f3(mu)}) ÷ ${f3(sd)} = <b>${f(Out[i][j])}</b></div>`,
          hl: [{id: xId, row: i, col: j}, {id: yId, row: i, col: j}], lesson: null
        };
      };
    }
    function smCalc(Sm, A, sId, tokens) {
      return (i, j) => {
        const row = Sm[i], mx = Math.max(...row.filter(isFinite));
        const parts = row.map(v => isFinite(v) ? `e^${f(v - mx)} = ${f3(Math.exp(v - mx))}` : 'e^−∞ = 0');
        const sum = row.reduce((s, v) => s + (isFinite(v) ? Math.exp(v - mx) : 0), 0);
        const num = isFinite(row[j]) ? Math.exp(row[j] - mx) : 0;
        return {
          html: `<div class="calc-title">How much “${esc(tokens[i])}” listens to “${esc(tokens[j])}”</div>
      <div class="calc-math">1. Raise e to each score in the row (we subtract the largest, ${f(mx)}, first; the result is the same): ${parts.join(',  ')}</div>
      <div class="calc-math">2. Add them up: ${f3(sum)}</div>
      <div class="calc-math">3. Divide: ${f3(num)} ÷ ${f3(sum)} = <b>${pct(A[i][j])}</b></div>`,
          hl: [{id: sId, row: i}], lesson: 'sm'
        };
      };
    }

    /* =====================================================================
       RENDERING PRIMITIVES
       ===================================================================== */
    let reg = {};
    function mat(id, label, data, rows, cols, o = {}) {return {id, label, data, rows, cols, kind: o.kind || 'div', calc: o.calc || null, pend: !!o.pend, disp: o.disp || null, attr: o.attr || null, g: o.g || null};}
    function cellStyle(v, max, kind) {
      if (!isFinite(v)) return '';
      const a = kind === 'prob' ? Math.min(1, v) * .9 : Math.min(1, Math.abs(v) / max) * .9;
      const rgb = kind === 'prob' ? '45,75,255' : v < 0 ? '0,122,255' : '255,149,0';
      return `background-color:rgba(${rgb},${(0.06 + a).toFixed(3)});${a > .5 ? 'color:#fff' : ''}`;
    }
    function matHTML(m) {
      reg[m.id] = m;
      const D = m.disp || m.data;
      const max = m.kind === 'prob' ? 1 : Math.max(1e-9, ...D.flat().filter(isFinite).map(Math.abs));
      let h = `<div class="mat"><div class="mat-label"><span>${m.label}</span><span class="dim">${m.data.length} × ${m.data[0].length}</span>${m.g ? `<button class="info" data-g="${m.g}" aria-label="What is ${esc(m.label.replace(/<[^>]+>/g, ''))}?">i</button>` : ''}</div><div class="mscroll"><table class="mt"><thead><tr><th></th>${m.cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>`;
      D.forEach((r, i) => {
        h += `<tr><th class="rl">${esc(m.rows[i])}</th>` + r.map((v, j) =>
          `<td data-m="${m.id}" data-i="${i}" data-j="${j}" ${isFinite(v) ? `data-v="${v}"` : ''} ${m.attr ? m.attr(i, j) : ''} class="${m.calc ? 'clk' : ''}${m.pend ? ' pend' : ''}${v === -Infinity ? ' ninf' : ''}" style="${cellStyle(v, max, m.kind)}">${f(v)}</td>`).join('') + '</tr>';
      });
      return h + '</tbody></table></div></div>';
    }
    const eq = (...p) => `<div class="eq">${p.map(x => typeof x === 'string' ? `<span class="op${x.length > 2 ? ' word' : ''}">${x}</span>` : matHTML(x)).join('')}</div>`;
    const tabs = list => `<div class="segc" role="tablist">${list.map((t, k) => `<button class="${k ? '' : 'on'}" data-tab="${t[0]}">${t[1]}</button>`).join('')}</div>` +
      list.map((t, k) => `<div class="pane${k ? '' : ' on'}" data-pane="${t[0]}">${t[2]}</div>`).join('');
    const M = s => `<i>${s}</i>`;

    /* ---------- animation helpers ---------- */
    let SPEED = 1;
    const sleep = ms => new Promise(r => setTimeout(r, ms * SPEED));
    const cells = (id, i, j) => {let s = `#slide td[data-m="${id}"]`; if (i != null) s += `[data-i="${i}"]`; if (j != null) s += `[data-j="${j}"]`; return $$(s);};
    const clearScan = () => $$('#slide td.scan').forEach(c => c.classList.remove('scan'));
    function countUp(c) {
      const t = +c.dataset.v; if (!isFinite(t)) return;
      const t0 = performance.now(), dur = 320 * SPEED;
      const tick = now => {const k = Math.min(1, (now - t0) / dur); c.textContent = f(t * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(tick);};
      requestAnimationFrame(tick);
    }
    const reveal = c => {if (c.classList.contains('pend')) {c.classList.remove('pend'); c.classList.add('show'); countUp(c);} };
    function setTab(name) {$$('#slide .segc button').forEach(t => t.classList.toggle('on', t.dataset.tab === name)); $$('#slide .pane').forEach(p => p.classList.toggle('on', p.dataset.pane === name));}
    const aTab = name => async () => setTab(name);
    const aWait = ms => async () => sleep(ms);
    function aProduct(res, a, b, total = 1700) {
      return async dead => {
        const m = reg[res], R = m.data.length, C = m.data[0].length, perCell = R * C <= 42;
        const units = perCell ? R * C : R, step = Math.max(26, total / units);
        for (let u = 0; u < units; u++) {
          if (dead()) return;
          clearScan();
          const i = perCell ? Math.floor(u / C) : u, j = perCell ? u % C : null;
          cells(a, i).forEach(c => c.classList.add('scan'));
          if (perCell) cells(b, null, j).forEach(c => c.classList.add('scan'));
          cells(res, i, j).forEach(reveal);
          await sleep(step);
        }
        clearScan();
      };
    }
    function aRows(res, srcs, total = 1400) {
      return async dead => {
        const R = reg[res].data.length, step = Math.max(60, total / R);
        for (let i = 0; i < R; i++) {
          if (dead()) return;
          clearScan();
          srcs.forEach(s => cells(s, i).forEach(c => c.classList.add('scan')));
          cells(res, i).forEach(reveal);
          await sleep(step);
        }
        clearScan();
      };
    }

    /* =====================================================================
       STEP METADATA
       ===================================================================== */
    const KEYS = ['tok', 'emb', 'pos', 'qkv', 'score', 'mask', 'soft', 'mix', 'norm', 'ffn', 'logits', 'pick'];
    const NAMES = ['Tokens', 'Embeddings', 'Positions', 'Query, key, value', 'Scores', 'Mask', 'Attention', 'Blend', 'Add & normalize', 'Feed-forward', 'Word scores', 'Choose'];
    const COLORS = ['#ff9500', '#ff2d55', '#af52de', '#5856d6', '#007aff', '#8e8e93', '#30b0c7', '#34c759', '#00c7be', '#ff3b30', '#a2845e', 'ai'];
    const ABOUT = [
      'The sentence is cut into tokens, and each one gets its number.',
      'Every token becomes a list of numbers that stands for its meaning.',
      'A wave pattern is added so the model knows the word order.',
      'Each word asks a question, offers a label and packs a message.',
      'Every question is compared with every label.',
      'Words are not allowed to peek at the words after them.',
      'Scores turn into percentages: who listens to whom.',
      'Each word gathers information from the words it listens to.',
      'The original is added back and the numbers are kept stable.',
      'A small neural network thinks about each word on its own.',
      'Every word in the vocabulary gets a score.',
      'Scores become odds, and one word is drawn.'
    ];
    const IC = {
      tok: '<path d="M3 5h18"/><rect x="3" y="11" width="5" height="8" rx="1.6"/><rect x="10.5" y="11" width="4" height="8" rx="1.6"/><rect x="17" y="11" width="4" height="8" rx="1.6"/>',
      emb: '<path d="M8 3H5v18h3M16 3h3v18h-3"/><g fill="currentColor" stroke="none"><circle cx="12" cy="7.5" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="16.5" r="1.5"/></g>',
      pos: '<path d="M2 9c1.67-4 3.33-4 5 0s3.33 4 5 0 3.33-4 5 0 3.33 4 5 0"/><path d="M3 20h18M6 17v3M12 17v3M18 17v3"/>',
      qkv: '<circle cx="5" cy="12" r="2.2"/><path d="M7.2 12H11c3 0 3-7 6-7M11 12h6M11 12c3 0 3 7 6 7"/><g fill="currentColor" stroke="none"><circle cx="19" cy="5" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="19" cy="19" r="2"/></g>',
      score: '<rect x="3" y="3" width="18" height="18" rx="2.5"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/><rect x="9" y="9" width="6" height="6" fill="currentColor" stroke="none"/>',
      mask: '<path d="M3 3h6v6h6v6h6v6H3z" fill="currentColor" fill-opacity=".3"/><path d="M12 3h9v9" stroke-dasharray="2 3.2"/>',
      soft: '<circle cx="12" cy="12" r="9"/><path d="M12 12V3a9 9 0 0 1 6.36 15.36z" fill="currentColor" fill-opacity=".3"/>',
      mix: '<path d="M3 5c7 0 6 7 13 7M3 19c7 0 6-7 13-7M3 12h17M17 8.5l3.5 3.5-3.5 3.5"/>',
      norm: '<path d="M2 20c5 0 5.5-12 10-12s5 12 10 12"/><path d="M5 3v5M2.5 5.5h5"/>',
      ffn: '<path stroke-width="1.4" d="M5 7l7-3M5 7l7 5M5 7l7 13M5 17l7-13M5 17l7-5M5 17l7 3M12 4l7 8M12 12h7M12 20l7-8"/><g fill="currentColor" stroke="none"><circle cx="5" cy="7" r="2.2"/><circle cx="5" cy="17" r="2.2"/><circle cx="12" cy="4" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="12" cy="20" r="2.2"/><circle cx="19" cy="12" r="2.2"/></g>',
      logits: '<path stroke-width="3" d="M4 6h16M4 12h11M4 18h6"/>',
      pick: '<rect x="3.5" y="3.5" width="17" height="17" rx="4"/><g fill="currentColor" stroke="none"><circle cx="8.5" cy="8.5" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="15.5" cy="15.5" r="1.6"/></g>'
    };
    const svgI = k => `<svg class="icon" viewBox="0 0 24 24">${IC[k]}</svg>`;
    const tile = k => {const i = KEYS.indexOf(k), c = COLORS[i]; return `<div class="tile${c === 'ai' ? ' ai' : ''}" style="--c:${c === 'ai' ? '#151515' : c}">${svgI(k)}</div>`;};

    /* slide shell: centred head, one big card, "Learn more" underneath */
    function shell(st, {title, sub, vis, more, lesson, hint = true, glow = false}) {
      const k = KEYS.indexOf(st.key), L = SIM.cfg.L;
      const li = st.layer != null && L > 1 ? ` · Layer ${st.layer + 1} of ${L}` : '';
      return `<div class="s-head">${tile(st.key)}<div class="s-step">Step ${k + 1} of 12${li}</div><h2>${title}</h2><p class="s-sub">${sub}</p></div>
    <div class="cardwrap${glow ? ' glow' : ''}"><div class="s-card">${vis}</div></div>
    ${hint ? '<div class="hint"><svg class="icon" style="width:16px;height:16px" viewBox="0 0 24 24"><path d="M9 11V5a2 2 0 0 1 4 0v6M13 11V9a2 2 0 0 1 4 0v4a7 7 0 0 1-7 7h-.5A6.5 6.5 0 0 1 4 15.2l-.5-1.7a1.8 1.8 0 0 1 3.2-1.6L9 14"/></svg>Tap any number to see how it was made.</div>' : ''}
    ${more ? `<div class="more"><button class="more-btn"><span class="plus">+</span>Learn more</button><div class="more-body"><div><div class="more-in">${more}${lesson ? `<button class="lesson-link" data-lesson="${lesson[0]}">${lesson[1]} ›</button>` : ''}</div></div></div></div>` : ''}`;
    }

    /* =====================================================================
       THE TWELVE SLIDES
       ===================================================================== */
    const R = {};
    const lab = w => w === '.' ? 'full stop' : w;

    R.tok = st => {
      const {tokens, model, n} = st.ctx, unk = [...model.unknown];
      const toks = tokens.map((t, i) => `<div class="tok" style="--i:${i};--c:${(n - 1) / 2}"><b>${esc(t)}</b><span>#${model.vid[t]}</span>${model.unknown.has(t) ? '<em>new word</em>' : ''}</div>`).join('');
      return {
        html: shell(st, {
          title: 'Words become numbers.',
          sub: `The sentence is cut into ${tm('token', 'tokens')}, and each one is swapped for its number in the ${tm('vocab', 'vocabulary')}.`,
          vis: `<div class="toks">${toks}</div>
      <div class="stats"><div class="stat"><b>${model.V}</b><span>words in the vocabulary</span></div><div class="stat"><b>${n}</b><span>tokens in your sentence</span></div><div class="stat"><b class="grad">1</b><span>word to predict</span></div></div>`,
          more: `<p>A computer can’t do maths with letters, so every token gets a fixed ID number. The vocabulary here holds every word in the model’s training text, plus any new words you typed.</p>
      <p>However long the text gets, this model only looks at the last ${CONTEXT} words. That is its context window; real models have windows of many thousands of tokens.</p>
      <p>Real models like ChatGPT split text into smaller pieces called sub-words, so even rare words can be built from known parts. Their vocabularies hold about 100,000 tokens.</p>
      ${unk.length ? `<p><b>Heads-up:</b> “${unk.map(esc).join('”, “')}” never appears in the training text, so the model has no idea what ${unk.length > 1 ? 'those words mean' : 'it means'}.</p>` : ''}
      <div class="formula">“${esc(tokens.join(' '))}” → [${tokens.map(t => model.vid[t]).join(', ')}]</div>`, hint: false
        }),
        anims: [aWait(1000 + n * 130 + 600)]
      };
    };

    R.emb = st => {
      const {tokens, model, rl, dl, d} = st.ctx, E = model.E, [dx, dy] = model.dims;
      const words = CORPUS_VOCAB.filter(w => w !== '.'), all = words.concat([...model.unknown]);
      const xs = all.map(w => E[w][dx]), ys = all.map(w => E[w][dy]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const px = w => 22 + (E[w][dx] - x0) / ((x1 - x0) || 1) * 456, py = w => 22 + (1 - (E[w][dy] - y0) / ((y1 - y0) || 1)) * 276;
      const mine = [...new Set(tokens)], pal = ['#ff2d55', '#5856d6', '#ff9500', '#34c759', '#007aff', '#af52de', '#30b0c7', '#ff3b30'];
      let svg = '';
      const every = Math.ceil(words.length / 450);
      words.forEach((w, i) => {if (i % every === 0 && !mine.includes(w)) svg += `<circle class="dot" style="animation-delay:${(i % 30) * 20}ms" cx="${px(w).toFixed(1)}" cy="${py(w).toFixed(1)}" r="3" fill="#c7c7cc"/>`;});
      const placed = [];
      mine.forEach((w, i) => {
        const x = px(w), y = py(w), c = pal[i % 8], wd = w.length * 8.4 + 18;
        let lx = Math.min(x + 12, 512 - wd), ly = y - 12, tries = 0;
        const hit = () => placed.some(p => lx < p.x + p.w + 4 && lx + wd + 4 > p.x && ly < p.y + 28 && ly + 28 > p.y);
        while (hit() && tries < 12) {ly += (tries % 2 ? -1 : 1) * 28 * (Math.floor(tries / 2) + 1); ly = Math.max(4, Math.min(292, ly)); tries++;}
        placed.push({x: lx, y: ly, w: wd});
        svg += `<g class="me" style="animation-delay:${500 + i * 150}ms;transform-origin:${x}px ${y}px"><path d="M${x.toFixed(1)},${y.toFixed(1)} L${(lx + 2).toFixed(1)},${(ly + 12).toFixed(1)}" stroke="${c}" stroke-width="1.2" opacity=".6"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="11" fill="${c}" opacity=".18"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${c}"/><rect x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" width="${wd}" height="24" rx="12" fill="#fff" stroke="${c}" stroke-width="1.5"/><text x="${(lx + wd / 2).toFixed(1)}" y="${(ly + 16.5).toFixed(1)}" text-anchor="middle" font-size="13" font-weight="600" fill="#1d1d1f" font-family="-apple-system,Inter,sans-serif">${esc(w)}</text></g>`;
      });
      const nn = mine.map(w => {
        if (model.unknown.has(w)) return `<div class="nnr"><b>${esc(w)}</b><span class="pill">new word, random vector</span></div>`;
        const best = words.filter(x => x !== w).map(x => [x, cos(E[w], E[x])]).sort((a, b) => b[1] - a[1]).slice(0, 3);
        return `<div class="nnr"><b>${esc(w)}</b>≈ ${best.map(([x, c]) => `<span class="pill">${esc(x)}<small>${c.toFixed(2)}</small></span>`).join('')}</div>`;
      }).join('');
      const Emb = mat('Emb', 'Your words as vectors', st.Emb, rl, dl, {pend: true, g: 'embedding', calc: (i, j) => ({html: `<div class="calc-title">Nothing to calculate: it’s a lookup</div><div class="calc-math">Row #${model.vid[tokens[i]]} of the embedding table belongs to “${esc(tokens[i])}”. Its number in column d${j} is <b>${f(st.Emb[i][j])}</b>.</div>`, hl: []})});
      return {
        html: shell(st, {
          title: 'Each word becomes a vector.',
          sub: `Every token is swapped for a list of ${d} numbers, its ${tm('embedding', 'embedding')}. Words used in similar ways get similar numbers.`,
          vis: `<div class="two"><div style="display:flex;flex-direction:column;gap:20px;align-items:center">${eq(Emb)}<div class="nn">${nn}</div></div>
      <div class="box"><h6>Where your words live</h6><svg viewBox="0 0 520 320" role="img" aria-label="Map of word vectors">${svg}</svg></div></div>`,
          more: `<p>These numbers were learned from the training text. We counted which words appear near each other, then squeezed that big table down to ${d} numbers per word (the same idea as GloVe and word2vec).</p>
      <p>The map shows two of the ${d} dimensions. The pills next to each word are its closest neighbours, measured by cosine similarity, where 1.00 means pointing in exactly the same direction.</p>
      <div class="formula">${M('X')}<sub>emb</sub>[${M('i')}] = ${M('E')}[ id of word ${M('i')} ]</div>`
        }),
        anims: [aRows('Emb', [], 1300), aWait(700)]
      };
    };

    R.pos = st => {
      const {tokens, rl, dl, d, n} = st.ctx;
      const W = 640, lane = 34, H = d * lane + 26, xmax = Math.max(n - 1, 1), X = p => 46 + p / xmax * (W - 70);
      const cs = ['#af52de', '#5856d6', '#007aff', '#30b0c7', '#34c759', '#ff9500', '#ff2d55', '#ff3b30'];
      let svg = '';
      for (let j = 0; j < d; j++) {
        const yc = 14 + j * lane + lane / 2, i2 = Math.floor(j / 2), div = Math.pow(10000, 2 * i2 / d);
        let p = '';
        for (let s = 0; s <= 90; s++) {const ps = s / 90 * xmax, v = j % 2 ? Math.cos(ps / div) : Math.sin(ps / div); p += `${s ? 'L' : 'M'}${X(ps).toFixed(1)},${(yc - v * lane * .38).toFixed(1)}`;}
        svg += `<text x="6" y="${yc + 4}" font-size="11" font-weight="600" fill="#86868b" font-family="-apple-system,Inter,sans-serif">d${j}</text><path d="${p}" fill="none" stroke="${cs[j % 8]}" stroke-width="2.4" stroke-linecap="round" stroke-dasharray="1000" stroke-dashoffset="1000"><animate attributeName="stroke-dashoffset" from="1000" to="0" dur="${(1.4 * SPEED).toFixed(2)}s" begin="${(j * .1 * SPEED).toFixed(2)}s" fill="freeze" calcMode="spline" keySplines=".2 .9 .1 1" keyTimes="0;1"/></path>`;
        for (let ps = 0; ps < n; ps++) {const v = j % 2 ? Math.cos(ps / div) : Math.sin(ps / div); svg += `<circle cx="${X(ps).toFixed(1)}" cy="${(yc - v * lane * .38).toFixed(1)}" r="3.4" fill="#fff" stroke="${cs[j % 8]}" stroke-width="2"/>`;}
      }
      for (let ps = 0; ps < n; ps++) svg += `<text x="${X(ps).toFixed(1)}" y="${H}" font-size="13" font-weight="600" text-anchor="middle" fill="#1d1d1f" font-family="-apple-system,Inter,sans-serif">${esc(tokens[ps])}</text>`;
      const pe = (i, j) => {const k = Math.floor(j / 2), a = i / Math.pow(10000, 2 * k / d); return {html: `<div class="calc-title">Position ${i}, wave d${j}</div><div class="calc-math">Angle: ${i} ÷ 10000^(2×${k}÷${d}) = ${i} ÷ ${f3(Math.pow(10000, 2 * k / d))} = ${f3(a)}</div><div class="calc-math">${j % 2 ? 'cos' : 'sin'}(${f3(a)}) = <b>${f(st.PE[i][j])}</b></div>`, hl: []};};
      return {
        html: shell(st, {
          title: 'Order matters.',
          sub: `“Dog bites man” isn’t “man bites dog”. A unique ${tm('position', 'wave pattern')} is added to each word so the model knows who came first.`,
          vis: `<div class="box"><h6>One wave per dimension. The dots are what each word receives.</h6><svg viewBox="0 0 ${W} ${H + 8}" role="img" aria-label="Position waves">${svg}</svg></div>
      ${eq(mat('Emb', 'Meaning', st.Emb, rl, dl, {g: 'embedding'}), '+', mat('PE', 'Position', st.PE, rl, dl, {calc: pe, g: 'position'}), '=',
            mat('X', 'Input X', st.X, rl, dl, {pend: true, g: 'xinput', calc: (i, j) => ({html: `<div class="calc-title">Meaning + position</div><div class="calc-math">${f(st.Emb[i][j])} + ${f(st.PE[i][j])} = <b>${f(st.X[i][j])}</b></div>`, hl: [{id: 'Emb', row: i, col: j}, {id: 'PE', row: i, col: j}]})}))}`,
          more: `<p>Attention compares words with each other but has no built-in sense of order. Adding a position signal fixes that.</p>
      <p>Each dimension gets a wave of a different speed. Fast waves tell neighbouring words apart; slow waves tell distant ones apart, a bit like the hour and minute hands of a clock.</p>
      <div class="formula"><span class="row">PE[pos, 2${M('i')}] = sin(pos ÷ 10000<sup>2${M('i')}/${M('d')}</sup>)</span><span class="row">PE[pos, 2${M('i')}+1] = cos(pos ÷ 10000<sup>2${M('i')}/${M('d')}</sup>)</span></div>`
        }),
        anims: [aWait(1000), aRows('X', ['Emb', 'PE'], 1500)]
      };
    };

    R.qkv = st => {
      const {rl, dl} = st.ctx, {X, W, Q, K, V} = st;
      const pane = (s, Wm, Rm, wl, wn, rlab, g) => eq(mat('X' + s, 'X', X, rl, dl, {g: 'xinput'}), '×', mat('W' + s, wl, Wm, dl, dl, {g: 'weights'}), '=', mat(s.toUpperCase(), rlab, Rm, rl, dl, {pend: true, g, calc: mmCalc(X, Wm, 'X', wn, s.toUpperCase())}));
      return {
        html: shell(st, {
          title: 'Three questions for every word.',
          sub: `Each word makes a ${tm('query', 'query')}, a ${tm('key', 'key')} and a ${tm('value', 'value')} by multiplying with three ${tm('weights', 'weight matrices')}.`,
          vis: tabs([['q', 'Query', pane('q', W.Wq, Q, 'W<sub>Q</sub>', 'W_Q', 'Query Q', 'query')], ['k', 'Key', pane('k', W.Wk, K, 'W<sub>K</sub>', 'W_K', 'Key K', 'key')], ['v', 'Value', pane('v', W.Wv, V, 'W<sub>V</sub>', 'W_V', 'Value V', 'value')]]),
          more: `<p>Think of a library. The <b>query</b> is your question (“what is this sentence about?”), each <b>key</b> is the label on a book’s spine, and each <b>value</b> is what’s written inside the book.</p>
      <p>All three come from the same input X, just multiplied by different weight matrices. In a trained model those weights are learned; here they start close to “keep the input as it is” with a little randomness.</p>
      <div class="formula"><span class="row">${M('Q')} = ${M('X')} · ${M('W')}<sub>Q</sub>&nbsp;&nbsp;&nbsp;${M('K')} = ${M('X')} · ${M('W')}<sub>K</sub>&nbsp;&nbsp;&nbsp;${M('V')} = ${M('X')} · ${M('W')}<sub>V</sub></span></div>`,
          lesson: ['mm', 'Forgot how to multiply matrices? See a 2×2 example']
        }),
        anims: [aTab('q'), aProduct('Q', 'Xq', 'Wq', 1700), aWait(350), aTab('k'), aProduct('K', 'Xk', 'Wk', 1400), aWait(350), aTab('v'), aProduct('V', 'Xv', 'Wv', 1400)]
      };
    };

    R.score = st => {
      const {rl, dl, d, sq} = st.ctx;
      return {
        html: shell(st, {
          title: 'Who matters to whom?',
          sub: `Every query is compared with every key using a ${tm('dot', 'dot product')}. The bigger the number, the stronger the match.`,
          vis: eq(mat('Q', 'Query Q', st.Q, rl, dl, {g: 'query'}), '×', mat('Kt', 'Key, flipped Kᵀ', st.Kt, dl, rl, {g: 'transpose'}), '÷ √' + d,
            mat('S', 'Scores', st.S, rl, rl, {pend: true, g: 'scores', calc: mmCalc(st.Q, st.Kt, 'Q', 'Kᵀ', 'Q Kᵀ', s => `<div class="calc-math">Then divide by √${d}: ${f(s)} ÷ ${f3(sq)} = <b>${f(s / sq)}</b></div>`)})),
          more: `<p>Read the scores table like this: the row is the word asking, the column is the word being looked at. A high number means that word’s key answers this word’s question well.</p>
      <p>We flip K (rows become columns) so a single multiplication compares every pair at once. Then we ${tm('sqrtd', 'divide by √d')} = ${f(sq)} to keep the numbers calm.</p>
      <div class="formula">Scores = ${M('Q')} · ${M('K')}<sup>T</sup> ÷ √${M('d')}</div>`,
          lesson: ['dot', 'What is a dot product? See a tiny example']
        }),
        anims: [aProduct('S', 'Q', 'Kt', 1900)]
      };
    };

    R.mask = st => {
      const {rl, n, tokens} = st.ctx;
      const maskAll = () => $$('#slide td[data-mask]').forEach(c => {c.textContent = '−∞'; c.style.backgroundColor = ''; c.style.color = ''; c.classList.add('ninf');});
      return {
        html: shell(st, {
          title: 'No peeking ahead.',
          sub: `A word may only look at itself and the words before it. Every future position gets a score of ${tm('ninf', '−∞')}.`,
          vis: eq(mat('S', 'Scores', st.S, rl, rl, {g: 'scores'}), 'apply mask',
            mat('Sm', 'Masked scores', st.Sm, rl, rl, {
              disp: st.S, g: 'mask', attr: (i, j) => j > i ? 'data-mask="1"' : '',
              calc: (i, j) => ({html: j > i ? `<div class="calc-title">Hidden</div><div class="calc-math">“${esc(tokens[i])}” sits at position ${i}. “${esc(tokens[j])}” comes later, at position ${j}, so its score becomes <b>−∞</b>.</div>` : `<div class="calc-title">Visible</div><div class="calc-math">Position ${j} is not in the future of position ${i}, so the score stays <b>${f(st.S[i][j])}</b>.</div>`, hl: [{id: 'S', row: i, col: j}]})
            })),
          more: `<p>When the model writes, the next words don’t exist yet. Training must match that situation, so no word is ever allowed to see what comes after it.</p>
      <p>That’s why the top-right triangle is blocked. This single rule is what makes GPT-style models “causal”: they can write one word at a time.</p>
      <div class="formula">Masked[${M('i')}, ${M('j')}] = Scores[${M('i')}, ${M('j')}] if ${M('j')} ≤ ${M('i')}, otherwise −∞</div>`
        }),
        anims: [aWait(400), async dead => {
          for (let k = 1; k < n; k++) {
            if (dead()) return;
            for (let i = 0; i + k < n; i++) {const c = cells('Sm', i, i + k)[0]; c.textContent = '−∞'; c.style.backgroundColor = ''; c.style.color = ''; c.classList.add('ninf', 'zap');}
            await sleep(Math.max(100, 1200 / n));
          }
        }],
        finish: maskAll
      };
    };

    function drawArcs(q, animate) {
      const box = $('#arcs'); if (!box || !CUR || !CUR.st.A) return;
      const A = CUR.st.A, svg = $('#arcsvg'), toks = [...box.querySelectorAll('.atk')];
      const br = box.getBoundingClientRect();
      toks.forEach((t, j) => {
        t.classList.toggle('q', j === q); t.classList.toggle('fut', j > q);
        t.querySelector('small').textContent = j <= q ? pct(A[q][j]) : 'hidden';
        t.style.setProperty('--w', j <= q ? Math.min(1, A[q][j] * 1.6).toFixed(2) : 0);
      });
      const pts = toks.map(t => {const r = t.getBoundingClientRect(); return {x: r.left + r.width / 2 - br.left, y: r.top - br.top};});
      let paths = '';
      for (let j = 0; j <= q; j++) {
        const w = A[q][j], p = pts[q], k = pts[j];
        const d = j === q ? `M${p.x - 12},${p.y} C${p.x - 36},${p.y - 66} ${p.x + 36},${p.y - 66} ${p.x + 12},${p.y}`
          : (() => {const hh = Math.min(150, 36 + Math.abs(p.x - k.x) * .5); return `M${p.x},${p.y} C${p.x},${p.y - hh} ${k.x},${p.y - hh} ${k.x},${k.y}`;})();
        paths += `<path d="${d}" fill="none" stroke="#2d4bff" stroke-linecap="round" stroke-width="${(1.5 + 12 * w).toFixed(1)}" opacity="${(.25 + .75 * w).toFixed(2)}"/>`;
      }
      svg.innerHTML = paths;
      if (animate) svg.querySelectorAll('path').forEach((p, i) => {
        const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L;
        p.getBoundingClientRect();
        p.style.transition = `stroke-dashoffset ${0.9 * SPEED}s cubic-bezier(.2,.9,.1,1) ${i * 0.07 * SPEED}s`; p.style.strokeDashoffset = 0;
      });
    }

    R.soft = st => {
      const {rl, tokens, n} = st.ctx;
      const toks = tokens.map((t, j) => `<button class="atk" data-q="${j}"><span>${esc(t)}</span><small></small></button>`).join('');
      return {
        html: shell(st, {
          title: 'Scores become attention.',
          sub: `${tm('softmax', 'Softmax')} turns each row into percentages that add up to 100%. Tap a word below to see who it listens to.`,
          vis: eq(mat('Sm', 'Masked scores', st.Sm, rl, rl, {g: 'mask'}), 'softmax', mat('A', 'Attention A', st.A, rl, rl, {kind: 'prob', pend: true, g: 'attention', calc: smCalc(st.Sm, st.A, 'Sm', tokens)})) +
            `<div class="arcs" id="arcs"><h6>Who “<span id="qname">${esc(tokens[n - 1])}</span>” listens to</h6><svg id="arcsvg" aria-hidden="true"></svg><div class="arc-toks">${toks}</div></div>`,
          more: `<p>Softmax does two things: it makes every number positive, and it scales them so they add up to 100%. Bigger scores win a much bigger share, and −∞ becomes exactly 0%.</p>
      <p>The last word’s row matters most here: its updated vector is the one that will predict the next word.</p>
      <div class="formula">softmax(${M('x')})<sub>${M('j')}</sub> = e<sup>${M('x')}<sub>${M('j')}</sub></sup> ÷ Σ<sub>${M('k')}</sub> e<sup>${M('x')}<sub>${M('k')}</sub></sup></div>`,
          lesson: ['sm', 'See softmax step by step with three numbers']
        }),
        anims: [aRows('A', ['Sm'], 1500), async () => {drawArcs(n - 1, true); await sleep(1300);}],
        finish: () => drawArcs(n - 1, false)
      };
    };

    R.mix = st => {
      const {rl, dl, tokens, n} = st.ctx, top = st.A[n - 1].map((w, j) => [tokens[j], w, j]).sort((a, b) => b[1] - a[1]).slice(0, 3);
      return {
        html: shell(st, {
          title: 'Words share what they know.',
          sub: `Each word’s new vector is a blend of the ${tm('value', 'values')} of the words it pays attention to. The last “${esc(tokens[n - 1])}” is now ${top.map(([t, w, j]) => `${pct(w)} “${esc(t)}”${j === n - 1 ? ' (itself)' : ''}`).join(', ')}…`,
          vis: eq(mat('A', 'Attention A', st.A, rl, rl, {kind: 'prob', g: 'attention'}), '×', mat('V', 'Value V', st.V, rl, dl, {g: 'value'}), '=', mat('Z', 'Blended Z', st.Z, rl, dl, {pend: true, g: 'blend', calc: mmCalc(st.A, st.V, 'A', 'V', 'Z')})),
          more: `<p>This is the moment information actually moves between words. Multiplying the attention table by the values means each row of the result is a weighted average of the value vectors.</p>
      <p>So if “the” pays 46% attention to “on”, then 46% of its new vector comes from the value of “on”.</p>
      <div class="formula">${M('Z')}[${M('i')}] = Σ<sub>${M('j')}</sub> ${M('A')}[${M('i')}, ${M('j')}] · ${M('V')}[${M('j')}]</div>`,
          lesson: ['mm', 'See how matrix multiplication works']
        }),
        anims: [aProduct('Z', 'A', 'V', 1900)]
      };
    };

    R.norm = st => {
      const {rl, dl} = st.ctx;
      return {
        html: shell(st, {
          title: 'Keep the original. Keep it stable.',
          sub: `The input is added back with a ${tm('residual', 'residual connection')}, then every row is rescaled by ${tm('layernorm', 'LayerNorm')}.`,
          vis: eq(mat('Z', 'Blended Z', st.Z, rl, dl, {g: 'blend'}), '×', mat('Wo', 'W<sub>O</sub>', st.Wo, dl, dl, {g: 'weights'}), '=', mat('O', 'Output O', st.O, rl, dl, {pend: true, calc: mmCalc(st.Z, st.Wo, 'Z', 'W_O', 'O')})) +
            eq(mat('Xr', 'Input X', st.X, rl, dl, {g: 'xinput'}), '+', mat('O2', 'Output O', st.O, rl, dl), 'LayerNorm', mat('H', 'Result H', st.H, rl, dl, {pend: true, g: 'layernorm', calc: lnCalc(st.X, st.O, 'Xr', 'O2', 'H', st.H)})),
          more: `<p>The blended vectors go through one more weight matrix, W<sub>O</sub>. Then the original input X is added back, so whatever the word knew before is never lost.</p>
      <p>LayerNorm finally shifts and scales each row so its average is 0 and its spread is 1. Without it, numbers could grow bigger and bigger from layer to layer.</p>
      <div class="formula"><span class="row">${M('O')} = ${M('Z')} · ${M('W')}<sub>O</sub></span><span class="row">${M('H')} = LayerNorm(${M('X')} + ${M('O')})</span></div>`
        }),
        anims: [aProduct('O', 'Z', 'Wo', 1500), aWait(300), aRows('H', ['Xr', 'O2'], 1300)]
      };
    };

    R.ffn = st => {
      const {rl, dl, hl2, d} = st.ctx;
      const relu = () => $$('#slide td[data-neg]').forEach(c => {c.textContent = '0.00'; c.style.backgroundColor = ''; c.style.color = '';});
      return {
        html: shell(st, {
          title: 'Each word thinks on its own.',
          sub: `A small ${tm('ffn', 'neural network')} works on every word separately: it widens the vector, switches off negatives with ${tm('relu', 'ReLU')}, and narrows it back.`,
          vis: tabs([
            ['e', 'Widen + ReLU', eq(mat('H', 'H', st.H, rl, dl), '×', mat('W1', 'W<sub>1</sub>', st.W1, dl, hl2, {g: 'weights'}), 'ReLU',
              mat('Hid', 'Hidden', st.Hid, rl, hl2, {disp: st.pre, pend: true, g: 'relu', attr: (i, j) => st.pre[i][j] < 0 ? 'data-neg="1"' : '', calc: mmCalc(st.H, st.W1, 'H', 'W₁', 'H W₁', s => `<div class="calc-math">ReLU keeps positives and zeroes negatives: ReLU(${f(s)}) = <b>${f(Math.max(0, s))}</b></div>`)}))],
            ['s', 'Narrow', eq(mat('Hid2', 'Hidden', st.Hid, rl, hl2, {g: 'relu'}), '×', mat('W2', 'W<sub>2</sub>', st.W2, hl2, dl, {g: 'weights'}), '=', mat('F', 'F', st.F, rl, dl, {pend: true, calc: mmCalc(st.Hid, st.W2, 'Hidden', 'W₂', 'F')}))],
            ['n', 'Add + normalize', eq(mat('H2', 'H', st.H, rl, dl), '+', mat('F2', 'F', st.F, rl, dl), 'LayerNorm', mat('Xo', 'Layer output', st.Xo, rl, dl, {pend: true, g: 'layernorm', calc: lnCalc(st.H, st.F, 'H2', 'F2', 'Output', st.Xo)}))]
          ]),
          more: `<p>Attention lets words talk to each other. The feed-forward network is where each word then “thinks” on its own: ${d} numbers are expanded to ${2 * d}, negative values are switched off, and the result is squeezed back to ${d}.</p>
      <p>In large models, about two thirds of all the weights live in these layers, and much of what the model knows is stored here.</p>
      <div class="formula"><span class="row">hidden = ReLU(${M('H')} · ${M('W')}<sub>1</sub>)</span><span class="row">output = LayerNorm(${M('H')} + hidden · ${M('W')}<sub>2</sub>)</span></div>`,
          lesson: ['mm', 'See how matrix multiplication works']
        }),
        anims: [aTab('e'), aProduct('Hid', 'H', 'W1', 1600), aWait(450), async () => {
          $$('#slide td[data-neg]').forEach(c => {c.classList.add('zap'); c.textContent = '0.00'; c.style.backgroundColor = ''; c.style.color = '';}); await sleep(900);
        },
        aTab('s'), aProduct('F', 'Hid2', 'W2', 1400), aWait(300), aTab('n'), aRows('Xo', ['H2', 'F2'], 1300)],
        finish: relu
      };
    };

    R.logits = st => {
      const {tokens, n, d, cfg, model} = st.ctx, kn = st.kn, last = tokens[n - 1];
      const top = st.scored.slice(0, 8), zmin = top[top.length - 1].z - 1, zmax = top[0].z;
      let k = `<h6>What the training text says</h6><div class="tries">${kn.tries.map(t => `<span class="try ${t.ok ? 'ok' : 'no'}">“${esc(t.c)} …”</span>`).join('')}</div>`;
      if (kn.match) k += `<div style="font-size:15px">${kn.match.skip ? `“${esc(last)}” is new to the model, so it skips it and looks at the words before. ` : ''}“<b>${esc(kn.match.c)}</b>” appeared ${kn.match.tot} time${kn.match.tot > 1 ? 's' : ''}. It was followed by:</div><div class="follow">${kn.match.items.slice(0, 8).map(([w, c], i) => `<span class="pill" style="animation-delay:${200 + i * 80}ms">${esc(lab(w))}<small>${c}×</small></span>`).join('')}</div>`;
      else k += `<div style="font-size:15px">“${esc(last)}” never appeared in the training text, so the model falls back on how common each word is. Expect a weak guess.</div>`;
      const rows = top.map((s, i) => `<div class="lb-row${i < cfg.topk ? ' in' : ''}" style="--i:${i}"><span class="n" style="text-align:left">${i + 1}</span><span class="w">${esc(lab(s.w))}</span><span class="lbar"><i data-w="${(6 + 94 * (s.z - zmin) / ((zmax - zmin) || 1)).toFixed(1)}"></i></span><span class="n hx">${f(s.a)}</span><span class="n hx">${f(s.b)}</span><span class="z">${f(s.z)}</span></div>`).join('');
      return {
        html: shell(st, {
          title: 'Every word gets a score.',
          sub: `The last word’s final vector is compared with all ${model.V} words in the vocabulary. What the model learned from its ${tm('ngram', 'training text')} adds its vote. Each result is a ${tm('logit', 'logit')}.`,
          vis: `<div class="kcard">${k}</div>
      ${eq(mat('h', `Final vector of “${esc(last)}”`, [st.h], [`${n - 1} ${last}`], st.ctx.dl, {g: 'hvec'}))}
      <div class="lb"><div class="lb-row head"><span></span><span>Word</span><span>Score</span><span class="n hx">Network</span><span class="n hx">Training text</span><span class="n">Total</span></div>${rows}</div>`,
          more: `<p>Each word’s score has two parts. The <b>network</b> part checks how well the final vector lines up with that word’s output vector (a dot product). The <b>training text</b> part is the log of how likely that word is after this phrase, judged from the counts above. Longer phrases count most.</p>
      <p>A real model keeps all of this inside its trained weights after reading billions of words. This tiny model isn’t trained by gradient descent, so the counted phrases stand in for training. Set “Training-text strength” to 0 to watch the raw network babble.</p>
      <div class="formula">${M('z')}(${M('w')}) = ${cfg.alpha} · ${M('h')}·${M('U')}[${M('w')}] ÷ √${d} + ${cfg.beta} · log ${M('P')}(${M('w')} | “${esc(kn.match ? kn.match.c : last)}”)</div>`,
          lesson: ['dot', 'What is a dot product?'], hint: false
        }),
        anims: [aWait(250), async () => {$$('#slide .lbar i').forEach(b => b.style.width = b.dataset.w + '%'); await sleep(1500);}],
        finish: () => $$('#slide .lbar i').forEach(b => b.style.width = b.dataset.w + '%')
      };
    };

    const PCOL = ['#ff5a3c', '#2d4bff', '#2fc8a0', '#af52de', '#e0457b', '#0f9bb5', '#8a6d3b', '#5856d6'];
    R.pick = st => {
      const {cfg} = st.ctx, kept = st.kept, ch = st.chosen, sample = cfg.decode === 'sample';
      const bars = kept.map((s, i) => `<div class="prow"><span class="w">${esc(lab(s.w))}</span><span class="bar" style="--c:${PCOL[i % 8]}"><i data-w="${(s.q / kept[0].q * 100).toFixed(1)}"></i></span><span class="v">${pct(s.q)}</span></div>`).join('');
      let strip = '', why;
      if (sample) {
        const segs = kept.map((s, i) => `<div class="sseg${s === ch ? ' won' : ''}" style="width:${(s.q * 100).toFixed(2)}%;--c:${PCOL[i % 8]}">${s.q > .1 ? esc(lab(s.w)) : ''}</div>`).join('');
        strip = `<div class="strip-wrap"><div class="knob" id="knob"><b>${st.r.toFixed(3)}</b><i></i></div><div class="strip" id="strip">${segs}</div><div class="ticks"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1</span></div></div>`;
        why = `A random number, ${st.r.toFixed(3)}, landed in the slice that belongs to “${esc(lab(ch.w))}”.`;
      } else why = 'Most likely mode always takes the top word.';
      const rx = sample ? (st.r * 100).toFixed(2) + '%' : '0%';
      const fill = conf => {
        $$('#slide .prow .bar i').forEach(b => b.style.width = b.dataset.w + '%');
        const v = $('#verdict'); if (v) v.classList.add('on');
        const sp = $('#strip'); if (sp) sp.classList.add('decided');
        const card = $('#slide .cardwrap'); if (card) card.classList.add('glowing');
        const slot = $('#slot'); if (slot && !slot.classList.contains('filled')) {slot.innerHTML = `<span>${esc(ch.w)}</span>`; slot.classList.add('filled');}
      };
      return {
        html: shell(st, {
          title: 'And the next word is…',
          sub: `Scores become ${tm('prob', 'probabilities')}. ${tm('temperature', 'Temperature')} (${cfg.temp}) and ${tm('topk', 'top-k')} (${cfg.topk}) shape the odds, then ${sample ? 'one word is drawn' : 'the best word is taken'}.`,
          vis: `<div class="pbars">${bars}</div>${strip}<div class="verdict" id="verdict"><small>Next word</small><div class="big grad">${esc(lab(ch.w))}</div><p>${why}</p></div>`,
          more: `<p>Softmax turns the scores into probabilities. Before that, they are divided by the temperature: a low temperature makes the favourite even stronger, a high one gives other words a chance.</p>
      <p>Only the top ${cfg.topk} words stay in, and their odds are rescaled to add up to 100%. ${sample ? 'Then a random number between 0 and 1 picks a point on the strip, and the word that owns that slice wins.' : ''}</p>
      <p>The chosen word joins the sentence, and all twelve steps run again for the next one. That loop is called autoregressive generation.</p>
      <div class="formula">${M('p')}(${M('w')}) = e<sup>${M('z')}<sub>${M('w')}</sub>/${M('T')}</sup> ÷ Σ e<sup>${M('z')}/${M('T')}</sup></div>`,
          lesson: ['sm', 'See softmax step by step'], hint: false, glow: true
        }),
        anims: [aWait(200), async () => {$$('#slide .prow .bar i').forEach(b => b.style.width = b.dataset.w + '%'); await sleep(1000);},
        async () => {const k = $('#knob'); if (k) {k.animate([{left: '0%'}, {left: '100%'}, {left: '0%'}, {left: '100%'}, {left: rx}], {duration: 2800 * SPEED, easing: 'cubic-bezier(.3,.1,.2,1)', fill: 'forwards'}); await sleep(2900);} else await sleep(400);},
        async () => {fill(true); await sleep(1500);}],
        finish: () => {const k = $('#knob'); if (k) {k.getAnimations().forEach(a => a.cancel()); k.style.left = rx;} fill(false);}
      };
    };

    R.done = () => {
      const it = SIM.iters, all = SIM.full, skip = Math.max(0, it.length - 12);
      const words = all.map((t, i) => `<span class="${i >= SIM.nIn ? 'gen' : ''}${t === '.' ? ' punct' : ''}" style="--i:${Math.min(i, 24)}">${esc(t)}</span>`).join('');
      const rows = (skip ? `<div class="hrow"><span class="n">…</span><span class="ctx">${skip} earlier word${skip > 1 ? 's' : ''}</span><span class="alt"></span></div>` : '') + it.slice(skip).map((x, j) => [x, j + skip]).map(([x, k]) => `<div class="hrow"><span class="n">${k + 1}</span><span class="ctx">…${esc(x.tokens.slice(-3).join(' '))} → <b class="grad">${esc(lab(x.next))}</b></span><span class="alt">${x.top.map(t => `${esc(lab(t.w))} ${pct(t.q)}`).join('  ·  ')}</span></div>`).join('');
      return {
        html: `<div class="s-head"><div class="tile ai">${svgI('pick')}</div><div class="s-step">${it.length} word${it.length > 1 ? 's' : ''} written</div><h2>That’s how it’s done.</h2><p class="s-sub">Run the network, choose one word, add it, repeat. ChatGPT-sized models do exactly this, with about a hundred layers and billions of multiplications for every single word.</p></div>
    <div class="cardwrap"><div class="s-card"><div class="final${all.length > 28 ? ' long' : ''}">${words}</div><div class="hist">${rows}</div><div class="btns"><button class="pillbtn" id="more">Keep writing</button><button class="pillbtn gray" id="replay">Watch again</button><button class="pillbtn gray" id="newOne">Try another sentence</button></div></div></div>`,
        anims: [aWait(500)]
      };
    };

    /* =====================================================================
       MATH REFRESHER (tiny interactive lessons)
       ===================================================================== */
    const LESSONS = {
      dot: {
        caps: ['Two lists of numbers, the same length: <b>a</b> and <b>b</b>.',
          'Multiply the first pair: <b>1 × 4 = 4</b>.',
          'Then the second pair: <b>2 × 5 = 10</b>.',
          'And the third: <b>3 × 6 = 18</b>.',
          'Add all the products: <b>4 + 10 + 18 = 32</b>. That single number is the dot product.',
          'Why it matters: a query and a key are compared exactly like this. A big result means the two words match well.'],
        render(s) {
          const a = [1, 2, 3], b = [4, 5, 6], p = [4, 10, 18], hot = i => s >= 1 && s <= 3 && i === s - 1;
          return `<div class="lrow"><span class="lname">a</span>${a.map((v, i) => `<span class="lc ${hot(i) ? 'hot' : ''}">${v}</span>`).join('')}</div>
        <div class="lrow"><span class="lname">b</span>${b.map((v, i) => `<span class="lc ${hot(i) ? 'hot' : ''}">${v}</span>`).join('')}</div>
        <div class="lrow"><span class="lname">×</span>${p.map((v, i) => s > i ? `<span class="lc res">${v}</span>` : '<span class="lc ghost">?</span>').join('')}</div>
        <div class="lsum ${s >= 4 ? 'on' : ''}">4 + 10 + 18 = <b>32</b></div>`;
        }
      },
      mm: {
        caps: ['To multiply <b>A</b> by <b>B</b>, every cell of the answer pairs <b>one row of A</b> with <b>one column of B</b>.',
          'Top-left: row 1 of A with column 1 of B. <b>1×5 + 2×7 = 19</b>.',
          'Top-right: row 1 with column 2. <b>1×6 + 2×8 = 22</b>.',
          'Bottom-left: row 2 with column 1. <b>3×5 + 4×7 = 43</b>.',
          'Bottom-right: row 2 with column 2. <b>3×6 + 4×8 = 50</b>. Done!',
          ''],
        render(s) {
          const A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], C = [[19, 22], [43, 50]], cell = [[0, 0], [0, 1], [1, 0], [1, 1]];
          const cur = s >= 1 && s <= 4 ? cell[s - 1] : null;
          const g = (Mx, hotFn, showFn) => `<div class="mgrid" style="grid-template-columns:repeat(2,58px)">${Mx.flatMap((r, i) => r.map((v, j) => showFn(i, j) ? `<span class="lc ${hotFn(i, j) ? 'hot' : ''}${Mx === C ? ' res' : ''}">${v}</span>` : '<span class="lc ghost">?</span>')).join('')}</div>`;
          const shown = (i, j) => cell.findIndex(c => c[0] === i && c[1] === j) < s;
          const eqn = cur ? `${A[cur[0]][0]}×${B[0][cur[1]]} + ${A[cur[0]][1]}×${B[1][cur[1]]} = <b>${C[cur[0]][cur[1]]}</b>` : '';
          return `<div class="mm"><div><div class="lbl">A</div>${g(A, (i) => cur && i === cur[0], () => true)}</div><span class="op">×</span><div><div class="lbl">B</div>${g(B, (i, j) => cur && j === cur[1], () => true)}</div><span class="op">=</span><div><div class="lbl">A × B</div>${g(C, (i, j) => cur && i === cur[0] && j === cur[1], shown)}</div></div>
        <div class="lsum ${cur ? 'on' : ''}">${eqn || '&nbsp;'}</div>`;
        }
      },
      sm: {
        caps: ['Three scores. We want percentages that add up to <b>100%</b>.',
          'Step 1: raise <b>e ≈ 2.718</b> to each score. Everything becomes positive, and bigger scores grow much bigger.',
          'Step 2: add them up: <b>7.39 + 2.72 + 1.11 = 11.21</b>.',
          'Step 3: divide each one by the total: <b>66%, 24%, 10%</b>. They add up to 100%.',
          'A score of <b>−∞</b> would give e<sup>−∞</sup> = 0, which is exactly how the mask hides future words.'],
        render(s) {
          const sc = [2.0, 1.0, 0.1], ex = sc.map(Math.exp), tot = ex.reduce((a, b) => a + b, 0);
          return `<div class="smcols">${sc.map((v, i) => `<div class="smcol"><span class="k">score</span><span class="lc wide">${v.toFixed(1)}</span><span class="k">e<sup>score</sup></span>${s >= 1 ? `<span class="lc wide res">${ex[i].toFixed(2)}</span>` : '<span class="lc wide ghost">?</span>'}<div class="smbar"><i style="height:${s >= 3 ? (ex[i] / tot * 100).toFixed(1) : 0}%"></i></div><b style="font-size:20px;opacity:${s >= 3 ? 1 : 0};transition:opacity .5s">${s >= 3 ? Math.round(ex[i] / tot * 100) + '%' : '0%'}</b></div>`).join('')}</div>
        <div class="lsum ${s === 2 ? 'on' : ''}">total = <b>${tot.toFixed(2)}</b></div>`;
        }
      }
    };
    let LES = 'mm', LSTEP = 0;
    function lessonCaps() {
      const c = LESSONS[LES].caps.slice();
      if (LES === 'mm') {const n = SIM ? SIM.iters[0].tokens.length : 5, d = SIM ? SIM.cfg.d : 6; c[5] = `The rule: <b>rows of A</b> meet <b>columns of B</b>, so their sizes must match. In the model, X (${n}×${d}) × W (${d}×${d}) gives ${n}×${d}: one row for each word.`;}
      return c;
    }
    function renderLesson() {
      const caps = lessonCaps();
      $$('#lessonTabs button').forEach(b => b.classList.toggle('on', b.dataset.l === LES));
      $('#lcap').innerHTML = caps[LSTEP];
      $('#lviz').innerHTML = LESSONS[LES].render(LSTEP);
      $('#ldots').innerHTML = caps.map((_, i) => `<i class="${i === LSTEP ? 'on' : ''}"></i>`).join('');
      $('#lBack').disabled = LSTEP === 0; $('#lBack').style.opacity = LSTEP === 0 ? .4 : 1;
      $('#lNext').textContent = LSTEP === caps.length - 1 ? 'Start over' : 'Next';
    }
    function openLesson(k) {hidePop(); LES = k || 'mm'; LSTEP = 0; renderLesson(); $('#lessonBg').classList.add('on'); if (playing) setPlay(false);}
    function closeLesson() {$('#lessonBg').classList.remove('on');}
    $('#lNext').onclick = () => {const n = lessonCaps().length; LSTEP = LSTEP === n - 1 ? 0 : LSTEP + 1; renderLesson();};
    $('#lBack').onclick = () => {if (LSTEP > 0) {LSTEP--; renderLesson();} };
    $$('#lessonTabs button').forEach(b => b.onclick = () => {LES = b.dataset.l; LSTEP = 0; renderLesson();});
    $('#closeLesson').onclick = closeLesson;
    $('#lessonBg').addEventListener('click', e => {if (e.target.id === 'lessonBg') closeLesson();});

    /* =====================================================================
       POP-OVERS
       ===================================================================== */
    const pop = $('#pop'); let popPinned = false, popFor = null, hoverT = null;
    function showPop(el, pin) {
      const g = GLOSS[el.dataset.g]; if (!g) return;
      pop.innerHTML = `<b>${g.t}</b><p>${g.d}</p>${g.ex ? `<div class="ex">${g.ex}</div>` : ''}${g.lesson ? `<button class="poplink" data-lesson="${g.lesson}">See a simple example ›</button>` : ''}`;
      popPinned = pin; popFor = el;
      pop.classList.remove('above'); pop.style.left = '0px'; pop.style.top = '0px'; pop.classList.add('on');
      const r = el.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
      let left = Math.max(10, Math.min(innerWidth - pw - 10, r.left + r.width / 2 - pw / 2));
      let top = r.bottom + 10;
      if (top + ph > innerHeight - 90) {top = r.top - ph - 10; pop.classList.add('above');}
      pop.style.left = left + 'px'; pop.style.top = top + 'px';
      pop.style.setProperty('--ox', (r.left + r.width / 2 - left) + 'px');
    }
    function hidePop() {pop.classList.remove('on'); popPinned = false; popFor = null;}
    document.addEventListener('click', e => {
      const les = e.target.closest('[data-lesson]'); if (les) {e.preventDefault(); openLesson(les.dataset.lesson); return;}
      const t = e.target.closest('.term,.info[data-g]');
      if (t) {e.stopPropagation(); if (popFor === t && popPinned) hidePop(); else showPop(t, true); return;}
      if (!e.target.closest('#pop')) hidePop();
    });
    if (matchMedia('(hover:hover)').matches) {
      document.addEventListener('mouseover', e => {const t = e.target.closest('.term,.info[data-g]'); if (t && !popPinned) {clearTimeout(hoverT); hoverT = setTimeout(() => showPop(t, false), 160);} });
      document.addEventListener('mouseout', e => {const t = e.target.closest('.term,.info[data-g]'); if (t && !popPinned) {clearTimeout(hoverT); hoverT = setTimeout(() => {if (!popPinned) hidePop();}, 120);} });
    }
    addEventListener('scroll', () => {if (!popPinned) hidePop(); removeCoach();}, {passive: true});

    /* =====================================================================
       CONTROLLER
       ===================================================================== */
    const CFG = {depth: 5, stop: true, d: 6, L: 1, temp: 0.9, topk: 3, decode: 'sample', beta: 2, seed: 7, speed: 1};
    let SIM = null, pos = 0, CUR = null, animTok = 0, playing = false, playTimer = null, animDone = false, coachLeft = 2;
    const readCfg = () => Object.assign({}, CFG, {alpha: 0.3});
    const DEFAULTS = Object.assign({}, CFG);
    const CONTEXT = 24;   // the model only looks at this many of the latest words, like a real model’s context window
    const PLAY = '<svg class="icon" viewBox="0 0 24 24" style="fill:currentColor;stroke:none"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg>';
    const PAUSE = '<svg class="icon" viewBox="0 0 24 24" style="fill:currentColor;stroke:none"><rect x="6" y="4" width="4.5" height="16" rx="1.2"/><rect x="13.5" y="4" width="4.5" height="16" rx="1.2"/></svg>';
    $('#play').innerHTML = PLAY;

    function startSim() {
      let toks = tokenize($('#inp').value);
      if (!toks.length) {$('#inp').focus(); return;}
      toks = toks.slice(0, 14);
      $('#wait').classList.add('on');
      setTimeout(() => {
        const cfg = Object.assign(readCfg(), {draw: (Math.random() * 4294967296) >>> 0}), model = buildModel(cfg, toks);
        SPEED = CFG.speed; document.body.style.setProperty('--spd', SPEED);
        SIM = {cfg, model, iters: [], steps: [], full: toks.slice(), nIn: toks.length};
        writeWords(cfg.depth);
        $('#wait').classList.remove('on');
        $('#setup').classList.remove('on'); $('#sim').classList.add('on'); document.body.classList.add('in-sim');
        window.scrollTo(0, 0);
        go(0, 'fwd');
      }, 40);
    }

    /* Write up to `count` more words onto the current run. Called again by “Keep writing”, so a run never has to end. */
    function writeWords(count) {
      const {cfg, model, iters, steps, full} = SIM;
      if (steps.length && steps[steps.length - 1].st.key === 'done') steps.pop();
      for (let k = 0; k < count; k++) {
        const g = iters.length, tokens = full.slice(-CONTEXT), r = runIteration(tokens, g, model, cfg);
        iters.push({tokens, from: full.length - tokens.length, next: r.next, top: r.top});
        r.stages.forEach(st => steps.push({g, st}));
        full.push(r.next);
        if (cfg.stop && r.next === '.') break;
      }
      steps.push({g: iters.length - 1, st: {key: 'done'}});
    }
    function keepWriting(count) {const at = SIM.steps.length - 1; writeWords(count); go(at, 'fwd');}

    function renderTop() {
      const {g, st} = SIM.steps[pos], it = SIM.iters[g], isDone = st.key === 'done';
      const toks = isDone ? it.tokens.concat([it.next]) : it.tokens;
      $('#sentence').innerHTML = (it.from > 0 ? `<span class="older" title="Earlier words are outside the model’s ${CONTEXT}-word context window">…</span>` : '') + toks.map((t, i) => `<span class="${it.from + i >= SIM.nIn ? 'gen' : ''}${t === '.' ? ' punct' : ''}">${esc(t)}</span>`).join('') + (isDone ? '' : '<span class="slot" id="slot">?</span>');
      $('#counter').textContent = isDone ? 'Finished' : `Word ${g + 1} of ${SIM.iters.length}`;
      const k = isDone ? KEYS.length : KEYS.indexOf(st.key);
      $('#prog').innerHTML = KEYS.map((key, i) => `<button class="pseg${i < k ? ' done' : ''}${i === k ? ' cur' : ''}" data-key="${key}" style="--c:${COLORS[i] === 'ai' ? '#151515' : COLORS[i]}" aria-label="Step ${i + 1}: ${NAMES[i]}" title="${NAMES[i]}"><i></i></button>`).join('');
      const li = st.layer != null && SIM.cfg.L > 1 ? ` · Layer ${st.layer + 1} of ${SIM.cfg.L}` : '';
      $('#plabel').textContent = isDone ? 'All twelve steps, done' : `${k + 1}. ${NAMES[k]}${li}`;
      $('#prev').disabled = pos === 0; $('#next').disabled = pos === SIM.steps.length - 1;
      $('#nextTok').disabled = false; $('#nextTok').textContent = isDone ? 'Keep writing' : 'Next word'; $('#skip').disabled = false;
    }

    function go(p, dir = 'fwd') {
      if (!SIM) return;
      p = Math.max(0, Math.min(SIM.steps.length - 1, p));
      pos = p; animTok++; clearTimeout(playTimer); closeDrawer(); hidePop(); removeCoach();
      const {st} = SIM.steps[pos];
      reg = {};
      CUR = Object.assign({st}, R[st.key](st));
      renderTop();
      $('#stage').innerHTML = `<div class="slide ${dir}" id="slide">${CUR.html}</div>`;
      window.scrollTo({top: 0, behavior: 'smooth'});
      bindSlide();
      runAnims();
    }
    async function runAnims() {
      const my = ++animTok, dead = () => my !== animTok; animDone = false;
      await sleep(420); if (dead()) return;
      const cw = $('#slide .cardwrap');
      if (cw) {
        const sb = $('.simbar'), r = cw.getBoundingClientRect(), bar = sb.offsetHeight + (parseFloat(getComputedStyle(sb).top) || 0);
        if (r.bottom > innerHeight - 110 && r.top > bar + 20) {window.scrollTo({top: scrollY + r.top - bar - 14, behavior: 'smooth'}); await sleep(450); if (dead()) return;}
      }
      for (const fn of CUR.anims || []) {if (dead()) return; await fn(dead);}
      if (dead()) return;
      finishSlide(); afterAnims(); maybeCoach();
    }
    function finishSlide() {
      const s = $('#slide'); if (!s) return;
      s.classList.add('done'); $$('#slide td.pend').forEach(c => {c.classList.remove('pend'); c.textContent = f(+c.dataset.v);}); clearScan();
      if (CUR.finish) CUR.finish();
      animDone = true;
    }
    function skipAnim() {if (animDone) {go(pos + 1); return;} animTok++; finishSlide(); afterAnims();}
    function afterAnims() {
      clearTimeout(playTimer);
      if (playing) {if (pos < SIM.steps.length - 1) playTimer = setTimeout(() => go(pos + 1), 1700 * SPEED); else setPlay(false);}
    }
    function setPlay(on) {
      playing = on; const b = $('#play');
      b.innerHTML = on ? PAUSE : PLAY; b.setAttribute('aria-label', on ? 'Pause' : 'Play');
      if (on) {if (pos >= SIM.steps.length - 1) go(0); else if (animDone) afterAnims();}
      else clearTimeout(playTimer);
    }
    function bindSlide() {
      $$('#slide .segc button').forEach(t => t.onclick = () => setTab(t.dataset.tab));
      $$('#slide .atk').forEach(t => t.onclick = () => {drawArcs(+t.dataset.q, true); $('#qname').textContent = t.querySelector('span').textContent;});
      $$('#slide .more-btn').forEach(b => b.onclick = () => b.parentElement.classList.toggle('open'));
      const rp = $('#replay'); if (rp) rp.onclick = () => go(0, 'fwd');
      const mo = $('#more'); if (mo) mo.onclick = () => keepWriting(SIM.cfg.depth);
      const nw = $('#newOne'); if (nw) nw.onclick = exitSim;
    }

    /* first-time hint pointing at a number */
    let coachEl = null, beaconEl = null;
    function maybeCoach() {
      if (coachLeft <= 0 || playing) return;
      const ids = Object.keys(reg).filter(id => reg[id].calc && reg[id].pend);
      if (!ids.length) return;
      const cell = cells(ids[ids.length - 1], 0, 0)[0]; if (!cell || !cell.offsetParent) return;
      const r = cell.getBoundingClientRect(); if (r.top < 100 || r.bottom > innerHeight - 100) return;
      coachLeft--;
      beaconEl = cell; cell.classList.add('beacon');
      coachEl = document.createElement('div'); coachEl.className = 'coach'; coachEl.textContent = 'Tap a number to see how it was made';
      document.body.appendChild(coachEl);
      const w = coachEl.offsetWidth, left = Math.max(10, Math.min(innerWidth - w - 10, r.left + r.width / 2 - w / 2));
      coachEl.style.left = left + 'px'; coachEl.style.top = (r.top - coachEl.offsetHeight - 12) + 'px';
      coachEl.style.setProperty('--ax', (r.left + r.width / 2 - left) + 'px');
    }
    function removeCoach() {if (coachEl) {coachEl.remove(); coachEl = null;} if (beaconEl) {beaconEl.classList.remove('beacon'); beaconEl = null;} }

    $('#stage').addEventListener('click', e => {
      removeCoach();
      const td = e.target.closest('td.clk'); if (!td) return;
      const m = reg[td.dataset.m]; if (!m || !m.calc) return;
      const r = m.calc(+td.dataset.i, +td.dataset.j);
      $$('#slide td.hl, #slide td.self').forEach(c => c.classList.remove('hl', 'self'));
      td.classList.add('self');
      (r.hl || []).forEach(h => cells(h.id, h.row, h.col).forEach(c => c.classList.add('hl')));
      $('#drawerBody').innerHTML = r.html + (r.lesson ? `<button class="lesson-link" data-lesson="${r.lesson}">New to this? Walk through a simple example ›</button>` : '');
      $('#drawer').classList.add('open');
    });
    function closeDrawer() {$('#drawer').classList.remove('open'); $$('#slide td.hl, #slide td.self').forEach(c => c.classList.remove('hl', 'self'));}
    $('#closeDrawer').onclick = closeDrawer;
    $('#prog').addEventListener('click', e => {
      const n = e.target.closest('.pseg'); if (!n) return;
      const {g, st} = SIM.steps[pos], key = n.dataset.key;
      let idx = SIM.steps.findIndex(s => s.g === g && s.st.key === key && (st.layer == null || s.st.layer == null || s.st.layer === st.layer));
      if (idx < 0) idx = SIM.steps.findIndex(s => s.g === g && s.st.key === key);
      if (idx >= 0) {setPlay(false); go(idx, idx < pos ? 'bwd' : 'fwd');}
    });
    function exitSim() {setPlay(false); animTok++; closeDrawer(); removeCoach(); hidePop(); $('#sim').classList.remove('on'); $('#setup').classList.add('on'); document.body.classList.remove('in-sim'); window.scrollTo(0, 0); heroUpdate();}

    $('#prev').onclick = () => {setPlay(false); go(pos - 1, 'bwd');};
    $('#next').onclick = () => {setPlay(false); go(pos + 1, 'fwd');};
    $('#skip').onclick = () => skipAnim();
    $('#play').onclick = () => setPlay(!playing);
    $('#nextTok').onclick = () => {setPlay(false); if (SIM.steps[pos].st.key === 'done') {keepWriting(1); return;} const g = SIM.steps[pos].g; const i = SIM.steps.findIndex((s, k) => k > pos && (s.g > g || s.st.key === 'done')); go(i < 0 ? SIM.steps.length - 1 : i, 'fwd');};
    $('#exit').onclick = exitSim;
    /* the top nav is shared by both screens: its links lead back to the home page */
    $('.nav').addEventListener('click', e => {
      const a = e.target.closest('a[href^="#"]'); if (!a) return;
      const inSim = $('#sim').classList.contains('on'); if (inSim) exitSim();
      if (a.id === 'brand') {e.preventDefault(); if (!inSim) window.scrollTo({top: 0, behavior: 'smooth'});}
    });
    $('#start').onclick = startSim;
    $('#inp').addEventListener('keydown', e => {if (e.key === 'Enter') startSim();});
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {closeDrawer(); hidePop(); closeLesson(); closeSettings(); return;}
      if ($('#setBg').classList.contains('on')) return;
      if ($('#lessonBg').classList.contains('on')) {if (e.key === 'ArrowRight') $('#lNext').click(); if (e.key === 'ArrowLeft') $('#lBack').click(); return;}
      if (!$('#sim').classList.contains('on') || ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (e.key === 'ArrowRight') {setPlay(false); go(pos + 1, 'fwd');}
      else if (e.key === 'ArrowLeft') {setPlay(false); go(pos - 1, 'bwd');}
      else if (e.key === ' ') {e.preventDefault(); setPlay(!playing);}
    });
    window.addEventListener('resize', () => {if (CUR && CUR.st.key === 'soft' && animDone) {const q = $('#slide .atk.q'); drawArcs(q ? +q.dataset.q : CUR.st.ctx.n - 1, false);} });

    /* =====================================================================
       LANDING
       ===================================================================== */
    const TILT = [-1.6, 1.1, -.6, 1.7, -1.2, .7];
    let heroTimer = null, heroA = null, heroToks = [], heroSel = 0, heroSpin = null, heroPrev = '';
    function heroUpdate() {
      const toks = tokenize($('#inp').value).slice(0, 12), box = $('#heroMat');
      settingsLine(); if ($('#setBg').classList.contains('on')) previewUpdate();
      clearInterval(heroSpin);
      if (!toks.length) {heroA = null; box.innerHTML = '<p class="muted">Type a few words and they will show up here.</p>'; $('#heroListen').textContent = ''; $('#heroTop').innerHTML = ''; return;}
      const cfg = Object.assign(readCfg(), {decode: 'greedy'});
      const model = buildModel(cfg, toks), r = runIteration(toks, 0, model, cfg);
      heroA = r.stages.filter(s => s.key === 'soft').pop().A; heroToks = toks; heroSel = toks.length - 1;
      box.innerHTML = '<svg class="harcs" aria-hidden="true"></svg>' + toks.map((t, i) => `<button class="wt" data-i="${i}" style="--r:${TILT[i % 6]}deg">${esc(t)}</button>`).join('') + `<span class="wt guess" id="heroNext">${esc(lab(r.next))}</span>`;
      heroArcs(); showCands($('#heroTop'), r.top);
      /* a new guess flips through the candidates before it settles, like a reel */
      const key = toks.join(' ') + '>' + r.next, g = $('#heroNext'), pool = r.top.map(t => lab(t.w));
      if (key !== heroPrev && pool.length > 1 && !matchMedia('(prefers-reduced-motion:reduce)').matches) {
        let k = 0; heroSpin = setInterval(() => {g.textContent = pool[k % pool.length]; if (++k > 7) {clearInterval(heroSpin); g.textContent = lab(r.next);} }, 75);
      }
      heroPrev = key;
    }
    /* arcs from the chosen word to the earlier words it listens to; thicker means it listens more */
    function heroArcs() {
      const box = $('#heroMat'), svg = box.querySelector('.harcs'); if (!svg || !heroA) return;
      const b = box.getBoundingClientRect(), tiles = [...box.querySelectorAll('.wt[data-i]')], q = Math.min(heroSel, tiles.length - 1);
      const top = el => {const r = el.getBoundingClientRect(); return [r.left - b.left + r.width / 2, r.top - b.top];};
      const [qx, qy] = top(tiles[q]); let h = '';
      for (let j = 0; j < q; j++) {
        const w = heroA[q][j]; if (w < .015) continue;
        const [x, y] = top(tiles[j]), lift = Math.min(88, 24 + Math.abs(qx - x) * .3);
        h += `<path d="M${qx.toFixed(1)} ${qy.toFixed(1)}C${qx.toFixed(1)} ${(qy - lift).toFixed(1)} ${x.toFixed(1)} ${(y - lift).toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)}" stroke-width="${(1.5 + 11 * w).toFixed(1)}" opacity="${(.3 + .7 * w).toFixed(2)}"/>`;
      }
      svg.innerHTML = h;
      tiles.forEach((t, i) => {t.classList.toggle('sel', i === q); t.classList.toggle('fut', i > q); t.style.setProperty('--w', i < q ? heroA[q][i].toFixed(3) : 0);});
      const rank = heroA[q].slice(0, q + 1).map((w, j) => [j, w]).sort((x, y) => y[1] - x[1]).slice(0, 3);
      $('#heroListen').innerHTML = `<b>“${esc(heroToks[q])}”</b> listens to ` + rank.map(([j, w]) => `${j === q ? 'itself' : `“${esc(heroToks[j])}”`} ${Math.round(w * 100)}%`).join(' · ') + (q ? '' : ' (it comes first, so there is nobody else)') + '<span> · point at another word</span>';
    }
    const heroPick = e => {const t = e.target.closest('.wt[data-i]'); if (t && +t.dataset.i !== heroSel) {heroSel = +t.dataset.i; heroArcs();} };
    ['pointerover', 'focusin', 'click'].forEach(ev => $('#heroMat').addEventListener(ev, heroPick));
    $('#heroMat').addEventListener('pointerleave', e => {if (e.pointerType === 'mouse' && heroA && heroSel !== heroToks.length - 1) {heroSel = heroToks.length - 1; heroArcs();} });
    addEventListener('resize', heroArcs);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(heroArcs);
    function showCands(box, list) {
      if (box.children.length !== list.length) box.innerHTML = list.map(() => '<div class="cand"><span></span><div class="cbar"><i></i></div><em></em></div>').join('');
      list.forEach((t, i) => {const c = box.children[i], p = Math.round(t.q * 100); c.querySelector('span').textContent = lab(t.w); c.querySelector('i').style.width = Math.max(2, p) + '%'; c.querySelector('em').textContent = p + '%';});
    }
    /* write a continuation with the given settings, without building slides (previews and the examples list) */
    function quickWrite(toks, cfg, max) {
      const model = buildModel(cfg, toks), words = [], full = toks.slice(); let first = null;
      for (let g = 0; g < max; g++) {
        const r = runIteration(full.slice(-CONTEXT), g, model, cfg); if (!g) first = r;
        words.push(r.next); full.push(r.next);
        if (cfg.stop && r.next === '.') break;
      }
      return {words, first, text: words.join(' ').replace(/ \./g, '.')};
    }
    /* in the settings pop-up: what the current settings write for the sentence in the box, and the odds behind the first word */
    function previewUpdate() {
      const toks = tokenize($('#inp').value).slice(0, 14), out = $('#pvOut');
      if (!toks.length) {out.textContent = 'Type a sentence on the home page to preview it here.'; $('#pvTop').innerHTML = ''; $('#pvNote').textContent = ''; return;}
      const cfg = readCfg(), cap = Math.min(cfg.depth, 30), w = quickWrite(toks, cfg, cap);
      out.innerHTML = `${esc(toks.join(' '))} <b>${esc(w.text)}</b>${w.words.length === cap && cap < cfg.depth ? ' …' : ''}`;
      showCands($('#pvTop'), w.first.stages[w.first.stages.length - 1].kept);
      $('#pvNote').textContent = (w.words.length === cap && cap < cfg.depth ? `First ${cap} of up to ${cfg.depth} words` : `${w.words.length} of up to ${cfg.depth} word${cfg.depth > 1 ? 's' : ''} written`) + (cfg.decode === 'sample' ? ' · random draw, so a real run can differ' : ' · always the most likely word');
    }
    function settingsLine() {
      const c = CFG;
      $('#setLine').textContent = `Up to ${c.depth} word${c.depth > 1 ? 's' : ''} · ${c.decode === 'sample' ? 'random draw' : 'most likely word'} · temperature ${c.temp.toFixed(1)} · top-k ${c.topk}`;
    }
    $('#inp').addEventListener('input', () => {clearTimeout(heroTimer); heroTimer = setTimeout(heroUpdate, 150);});

    /* example sentences: a few next to the box, and a longer list by theme with the model’s own answer in grey */
    const EXAMPLES = ['once upon a time', 'the capital of france is', 'two plus three is', 'bonjour comment'];
    const TRY = [
      ['Stories', ['once upon a time', 'one day a little', 'the bird flew']],
      ['Facts', ['the capital of france is', 'the opposite of hot is', 'the lion lives']],
      ['Numbers and dates', ['two plus three is', 'after monday comes', 'there are seven days in']],
      ['Everyday talk', ['hello', 'good morning', 'can you help']],
      ['About itself', ['the model predicts the', 'attention is', 'a token is']],
      ['Français', ['bonjour comment', 'je suis', 'le chat']]
    ];
    const TRYCOL = ['#ff5a3c', '#2d4bff', '#2fc8a0', '#ffd23f', '#af52de', '#151515'];
    $('#examples').innerHTML = EXAMPLES.map(x => `<button class="tryword" data-try="${x}">${x}</button>`).join(', ');
    $('#tryList').innerHTML = TRY.map(([k, xs], i) => `<div class="trycard" style="--c:${TRYCOL[i % 6]}"><h3>${k}</h3>${xs.map(x => `<button class="trylink" data-try="${x}"><span>${x}</span> <i></i></button>`).join('')}</div>`).join('');
    $('#steps12').innerHTML = KEYS.map((k, i) => `<li><span class="sq" style="--c:${COLORS[i] === 'ai' ? '#151515' : COLORS[i]}">${svgI(k)}</span><small>${String(i + 1).padStart(2, '0')}</small><span>${NAMES[i]}</span></li>`).join('');
    document.addEventListener('click', e => {const t = e.target.closest('[data-try]'); if (t) {$('#inp').value = t.dataset.try; heroUpdate(); startSim();} });
    function fillAnswers() {
      const cfg = Object.assign(readCfg(), DEFAULTS, {alpha: 0.3, decode: 'greedy', stop: true});
      $$('#tryList [data-try]').forEach(b => b.querySelector('i').textContent = quickWrite(tokenize(b.dataset.try), cfg, 8).text);
      $('#nSents').textContent = TRAIN.sentences.toLocaleString('en'); $('#nWords').textContent = (CORPUS_VOCAB.length - 1).toLocaleString('en');
    }

    /* settings live in a pop-up; changing them while stepping restarts the run with the new settings */
    let setSnap = '';
    const simKey = () => JSON.stringify(Object.assign({}, CFG, {speed: 0}));
    function openSettings() {
      hidePop(); if (playing) setPlay(false);
      setSnap = simKey(); $('#setNote').textContent = $('#sim').classList.contains('on') ? 'Changes restart the current run.' : '';
      previewUpdate(); $('#setBg').classList.add('on');
    }
    function closeSettings() {
      if (!$('#setBg').classList.contains('on')) return;
      $('#setBg').classList.remove('on'); hidePop();
      SPEED = CFG.speed; document.body.style.setProperty('--spd', SPEED);
      if ($('#sim').classList.contains('on') && simKey() !== setSnap) startSim();
    }
    document.addEventListener('click', e => {if (e.target.closest('[data-open="settings"]')) openSettings();});
    $('#closeSet').onclick = $('#doneSet').onclick = closeSettings;
    $('#setBg').addEventListener('click', e => {if (e.target.id === 'setBg') closeSettings();});
    $('#setBg .modal').addEventListener('scroll', () => hidePop(), {passive: true});

    /* iOS-style settings */
    const SETS = [
      {
        title: 'Writing', rows: [
          {k: 'depth', label: 'Words to predict', type: 'step', min: 1, max: 200, g: 'depth'},
          {k: 'stop', label: 'Stop at end of sentence', type: 'switch', g: 'stop'},
          {k: 'decode', label: 'How to choose', type: 'seg', opts: [['sample', 'Random draw'], ['greedy', 'Most likely']], g: 'decode'},
          {k: 'temp', label: 'Temperature', type: 'slider', min: .1, max: 2, step: .1, g: 'temperature'},
          {k: 'topk', label: 'Top-k', type: 'step', min: 1, max: 8, g: 'topk'}]
      },
      {
        title: 'Model', rows: [
          {k: 'd', label: 'Vector size', type: 'seg', opts: [[4, '4'], [6, '6'], [8, '8']], g: 'vectorsize'},
          {k: 'L', label: 'Layers', type: 'seg', opts: [[1, '1'], [2, '2'], [3, '3']], g: 'layers'},
          {k: 'beta', label: 'Training-text strength', type: 'slider', min: 0, max: 5, step: .1, g: 'ngram'},
          {k: 'seed', label: 'Random seed', type: 'step', min: 1, max: 999, g: 'seed'},
          {k: 'speed', label: 'Animation speed', type: 'seg', opts: [[1.6, 'Slow'], [1, 'Normal'], [.5, 'Fast']], g: 'speed'}]
      }
    ];
    function ctrlHTML(r) {
      const v = CFG[r.k];
      if (r.type === 'step') return `<div class="stepper"><button data-k="${r.k}" data-d="-1" aria-label="Less">−</button><input id="v-${r.k}" data-k="${r.k}" inputmode="numeric" value="${v}" aria-label="${r.label}"><button data-k="${r.k}" data-d="1" aria-label="More">+</button></div>`;
      if (r.type === 'switch') return `<button class="switch${v ? ' on' : ''}" data-k="${r.k}" role="switch" aria-checked="${v}" aria-label="${r.label}"></button>`;
      if (r.type === 'seg') return `<div class="seg">${r.opts.map(([val, t]) => `<button data-k="${r.k}" data-v="${val}" class="${v == val ? 'on' : ''}">${t}</button>`).join('')}</div>`;
      const p = (v - r.min) / (r.max - r.min) * 100;
      return `<div class="slider"><input type="range" data-k="${r.k}" min="${r.min}" max="${r.max}" step="${r.step}" value="${v}" style="--p:${p}%" aria-label="${r.label}"><span id="v-${r.k}">${(+v).toFixed(1)}</span></div>`;
    }
    const renderSettings = () => $('#groups').innerHTML = SETS.map(gp => `<div class="group"><h5>${gp.title}</h5><div class="glist">${gp.rows.map(r => `<div class="srow"><span class="lbl">${r.label}<button class="info" data-g="${r.g}" aria-label="About ${r.label}">i</button></span>${ctrlHTML(r)}</div>`).join('')}</div></div>`).join('');
    renderSettings();
    $('#resetSet').onclick = () => {Object.assign(CFG, DEFAULTS); renderSettings(); heroUpdate();};
    const ROW = Object.fromEntries(SETS.flatMap(g => g.rows).map(r => [r.k, r]));
    $('#groups').addEventListener('click', e => {
      const b = e.target.closest('button[data-k]'); if (!b) return;
      const r = ROW[b.dataset.k];
      if (r.type === 'step') {CFG[r.k] = Math.max(r.min, Math.min(r.max, CFG[r.k] + +b.dataset.d)); $('#v-' + r.k).value = CFG[r.k];}
      else if (r.type === 'switch') {CFG[r.k] = !CFG[r.k]; b.classList.toggle('on', CFG[r.k]); b.setAttribute('aria-checked', CFG[r.k]);}
      else if (r.type === 'seg') {CFG[r.k] = typeof r.opts[0][0] === 'number' ? +b.dataset.v : b.dataset.v; b.parentElement.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));}
      if (r.k !== 'speed') heroUpdate();
    });
    $('#groups').addEventListener('input', e => {
      const s = e.target.closest('input[type=range]'); if (!s) return;
      const r = ROW[s.dataset.k]; CFG[r.k] = +s.value; $('#v-' + r.k).textContent = (+s.value).toFixed(1);
      s.style.setProperty('--p', ((s.value - r.min) / (r.max - r.min) * 100) + '%');
      clearTimeout(heroTimer); heroTimer = setTimeout(heroUpdate, 60);
    });
    /* the number in a stepper can also be typed */
    $('#groups').addEventListener('change', e => {
      const n = e.target.closest('.stepper input'); if (!n) return;
      const r = ROW[n.dataset.k], v = Math.round(+n.value);
      if (Number.isFinite(v)) CFG[r.k] = Math.max(r.min, Math.min(r.max, v));
      n.value = CFG[r.k]; heroUpdate();
    });
    $('#groups').addEventListener('keydown', e => {if (e.key === 'Enter' && e.target.matches('.stepper input')) e.target.blur();});

    settingsLine();
    setTimeout(() => {heroUpdate(); fillAnswers();}, 0);   // learn the word vectors after the page has had a chance to paint
