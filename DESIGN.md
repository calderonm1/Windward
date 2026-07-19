# Windward: A Reinforcement Learning Sailing Simulator

**Technical Design Document**

*Version 0.1 — Draft*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Success Criteria](#2-project-goals-and-success-criteria)
3. [Portfolio Framing: What Each Component Demonstrates](#3-portfolio-framing-what-each-component-demonstrates)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Procedural World Generation](#6-procedural-world-generation)
7. [Ship Physics Model](#7-ship-physics-model)
8. [The Reinforcement Learning Environment](#8-the-reinforcement-learning-environment)
9. [Training Methodology](#9-training-methodology)
10. [Weight Export and Cross-Language Inference](#10-weight-export-and-cross-language-inference)
11. [The 3D Viewer](#11-the-3d-viewer)
12. [Leaderboard Design](#12-leaderboard-design)
13. [Testing Strategy](#13-testing-strategy)
14. [Roadmap and Milestones](#14-roadmap-and-milestones)
15. [Risk Register](#15-risk-register)
16. [Presentation and README Plan](#16-presentation-and-readme-plan)
17. [Future Extensions](#17-future-extensions)
18. [Appendices](#18-appendices)

---

## 1. Executive Summary

Windward is a 3D sailing simulator in which a neural network agent learns, through reinforcement learning, to navigate procedurally generated archipelagos and collect treasure while avoiding hazards. The project is built as a public portfolio piece with three deliverables: a reproducible training pipeline, a browser-playable demo, and a leaderboard tracking agent performance across training generations alongside human attempts on identical maps.

The core design commitment is **strict separation between simulation, training, and rendering**. The physics run headless in vectorized NumPy at roughly five orders of magnitude faster than real time; PPO trains against that; the resulting policy — a small multilayer perceptron — exports to a JSON file of a few dozen kilobytes and is re-executed in the browser by a hand-written forward pass. This is not an aesthetic preference. It is the difference between a project that converges in an afternoon and one that never converges at all.

The problem domain was chosen deliberately. A sailing vessel cannot travel directly into the wind; reaching an upwind target requires a zig-zag manoeuvre called tacking. This constraint is non-obvious, non-differentiable from the agent's perspective, and must be discovered rather than programmed. It elevates the control problem above the standard "drive a car around a track" exercise and produces visibly interesting emergent behaviour — the moment an agent first tacks upwind is a legible, screenshot-able research result.

The secondary design commitment is that **treasure value scales inversely with distance to the nearest hazard**. This single rule collapses "treacherous" and "lucrative" onto the same axis and forces the agent to learn a risk appetite rather than a pathfinding routine. Without it, the project is a maze solver with boats.

---

## 2. Project Goals and Success Criteria

### 2.1 Primary Goals

| # | Goal | Measurable Criterion |
|---|------|----------------------|
| G1 | Demonstrate procedural content generation | Seeded, deterministic worlds; identical seed produces byte-identical world in Python and JavaScript |
| G2 | Demonstrate reinforcement learning competence | Trained agent significantly outperforms random and scripted baselines on **held-out seeds** |
| G3 | Demonstrate generalization, not memorization | Held-out seed performance within 20% of training seed performance |
| G4 | Produce a clickable artifact | Live demo on GitHub Pages, loads in under 3 seconds, runs at 60fps |
| G5 | Demonstrate engineering discipline | Test suite, CI, reproducible training runs, documented failure modes |

### 2.2 Success Criteria

The project is complete when all of the following hold:

- A visitor with no context can open a URL, watch a boat sail intelligently, and understand what they are looking at within ten seconds.
- The trained agent tacks upwind. Not because tacking was scripted, but because it was discovered.
- `python -m windward.train --config configs/final.yaml` reproduces the published result from a clean checkout, given the same seed.
- The README contains a training curve showing train and held-out performance, and an ablation table with at least four rows.
- A human can play the same eval seeds and lose.

### 2.3 Explicit Non-Goals

Scope discipline is what makes this finishable. The following are **out of scope** and should be actively resisted:

- **Realistic fluid dynamics.** No wave simulation, no hull-water interaction beyond a lookup table. Water is a shader.
- **Multi-agent racing.** Tempting, doubles the difficulty, adds nothing to the core narrative.
- **A backend service.** No accounts, no database, no global leaderboard. Static hosting only. (See §12.4 for the rationale.)
- **Photorealism.** Stylized low-poly is faster to build, faster to load, and ages better.
- **Combat, trading, crew management, weather systems beyond wind.** These are game features. This is a research demo wearing a game costume.
- **Beating a published benchmark.** Nobody is benchmarking sailing agents. The contribution is the engineering, not a SOTA number.

---

## 3. Portfolio Framing: What Each Component Demonstrates

A portfolio project is an argument. Each component should carry a specific claim, and the README should make the claims legible to someone skimming for ninety seconds.

| Component | Claim to a reader | Who cares |
|-----------|-------------------|-----------|
| Seeded PCG with rejection sampling | Understands determinism, hashing, reproducibility | Backend, infra, gamedev |
| fBm + domain warping + curl noise | Knows the standard PCG toolkit and when each applies | Gamedev, graphics |
| Poisson-disk sampling | Knows that "random placement" and "good random placement" differ | Gamedev, simulation |
| Vectorized headless sim | Understands throughput and where time actually goes | ML engineering |
| PPO implementation/integration | Can operate modern RL, not just import it | ML |
| Potential-based reward shaping (with citation) | Knows the *theory*, not just the knobs | ML research-adjacent |
| Domain randomization + held-out eval | Understands overfitting in RL — the field's most common unforced error | ML, anyone senior |
| Cross-language physics parity test | Thinks about correctness at system boundaries | Senior engineering generally |
| Weight export + hand-written JS forward pass | Understands what a neural network actually *is* | ML, and it's a great interview story |
| Three.js viewer | Can ship a frontend | Full-stack, gamedev |
| Documented failure modes | Has actually done this before and is honest about it | Everyone. This is the highest-signal section. |

The last row deserves emphasis. Every RL project produces reward-hacking stories: the agent that circles a treasure forever farming shaping reward without touching it, the agent that discovers running aground at low speed avoids the crash penalty, the agent that learns to sail in circles because the timeout penalty was smaller than the crash penalty. **Write these down as they happen.** They are the most credible evidence that the work is real, and they are the thing a tutorial-follower cannot fabricate.

---

## 4. System Architecture

### 4.1 The Three-Tier Split

```
┌───────────────────────────────────────────────────────────┐
│                    TIER 1: SIMULATION                     │
│                                                           │
│   worldgen.py   physics.py   env.py                       │
│                                                           │
│   Pure NumPy. No graphics. No torch. No I/O.              │
│   Vectorized across N parallel worlds.                    │
│   Target: >100,000 agent-steps/sec on a laptop CPU.       │
└───────────────────────┬───────────────────────────────────┘
                        │  Gymnasium-style API
                        │  reset(seed) -> obs
                        │  step(action) -> obs, reward, done, info
                        │
┌───────────────────────▼───────────────────────────────────┐
│                    TIER 2: TRAINING                       │
│                                                           │
│   train.py   policy.py   callbacks.py                     │
│                                                           │
│   PyTorch + PPO. Reads Tier 1, writes checkpoints.        │
│   Logs to TensorBoard. Never imports Tier 3.              │
└───────────────────────┬───────────────────────────────────┘
                        │  export.py
                        │  weights.json  (~40 KB)
                        │  { "layers": [{"W": [[...]], "b": [...]}, ...] }
                        │
┌───────────────────────▼───────────────────────────────────┐
│                    TIER 3: PRESENTATION                   │
│                                                           │
│   viewer/  (three.js)                                     │
│     ├── worldgen.js   ← PORT of Tier 1 worldgen           │
│     ├── physics.js    ← PORT of Tier 1 physics            │
│     ├── policy.js     ← 40-line MLP forward pass          │
│     └── render.js     ← the only file that knows 3D exists │
│                                                           │
│   Static. Deploys to GitHub Pages. No build server.       │
└───────────────────────────────────────────────────────────┘
```

### 4.2 Why the Split Exists: The Throughput Argument

This is worth stating numerically in the README because it is the design decision that makes everything else possible.

A PPO run on a problem of this complexity needs somewhere between 10 and 50 million environment steps to converge. Consider two architectures:

**Naïve (train inside the renderer):** The simulation is coupled to the render loop at 60Hz. One environment instance. Throughput: 60 steps/sec.

- 10M steps ÷ 60 steps/sec = **46 hours** for a single run.
- Hyperparameter search is impossible. Ablations are impossible. A single bug costs two days.

**Decoupled (headless, vectorized):** No rendering. Physics as NumPy array operations across 64 worlds simultaneously — one `step()` call advances all 64. Throughput: ~150,000 steps/sec.

- 10M steps ÷ 150,000 steps/sec = **67 seconds**.
- The full ablation matrix runs over lunch.

The 2,500× gap is not an optimization. It is the difference between a project that exists and one that doesn't. Every architectural decision downstream — the NumPy-only Tier 1, the ban on `torch` imports in the environment, the polar-table physics — serves this number.

### 4.3 The Cost of the Split

Honesty requires acknowledging the tax: **the physics and worldgen exist twice**, once in Python and once in JavaScript. This is a real maintenance burden and a real source of bugs.

Three mitigations:

1. **Keep Tier 1 tiny and dependency-free.** If `physics.py` is 200 lines of arithmetic with no library calls beyond NumPy, the port is mechanical rather than creative.
2. **Freeze Tier 1 early.** Port to JS only once physics is stable (after Milestone 4). Do not port a moving target.
3. **Test the parity, don't trust it.** See §13.2. A golden-trajectory test that runs the same seed and action sequence through both implementations and asserts agreement to within 1e-4 catches divergence the moment it appears.

The alternative — compiling the Python to WASM via Pyodide, or writing the physics in Rust and compiling to both — trades this duplication for a heavier toolchain, a larger download, and a worse story. For a 200-line physics model, duplication wins.

### 4.4 Repository Layout

```
windward/
├── README.md                 ← the GIF goes here, line 3
├── DESIGN.md                 ← this document
├── LICENSE
├── pyproject.toml
├── .github/workflows/
│   ├── test.yml              ← pytest + vitest on push
│   └── deploy.yml            ← build viewer, push to gh-pages
│
├── windward/                 ← TIER 1 + 2 (Python)
│   ├── worldgen.py           ← noise, islands, treasure, validation
│   ├── noise.py              ← simplex, fBm, curl — no deps
│   ├── rng.py                ← splitmix64, deterministic
│   ├── physics.py            ← polar model, integration, collision
│   ├── env.py                ← gym API, obs/reward/termination
│   ├── vec_env.py            ← N-world vectorization
│   ├── policy.py             ← MLP definition
│   ├── train.py              ← PPO loop
│   ├── evaluate.py           ← deterministic scoring harness
│   └── export.py             ← checkpoint -> weights.json
│
├── configs/
│   ├── debug.yaml            ← tiny, fast, for iteration
│   ├── curriculum.yaml
│   └── final.yaml            ← reproduces the published result
│
├── tests/
│   ├── test_determinism.py
│   ├── test_physics.py
│   ├── test_worldgen.py
│   └── test_parity.py        ← Python vs JS golden trajectories
│
├── viewer/                   ← TIER 3 (JavaScript)
│   ├── index.html
│   ├── src/
│   │   ├── worldgen.js       ← port
│   │   ├── noise.js          ← port
│   │   ├── rng.js            ← port
│   │   ├── physics.js        ← port
│   │   ├── policy.js         ← MLP forward pass
│   │   ├── render.js         ← three.js scene
│   │   ├── hud.js
│   │   └── leaderboard.js
│   ├── public/
│   │   ├── weights/          ← gen_0010.json, gen_0200.json, ...
│   │   └── leaderboard.json  ← agent scores, committed to repo
│   ├── package.json
│   └── vite.config.js
│
├── docs/
│   ├── failures.md           ← the reward-hacking diary
│   ├── ablations.md
│   └── media/
│       ├── hero.gif
│       └── curves.png
│
└── notebooks/
    └── analysis.ipynb        ← plots for the README
```

**Rule enforced by CI:** nothing in `windward/worldgen.py`, `physics.py`, or `env.py` may import `torch`. If it does, the vectorized environment has been contaminated with a dependency that will eventually slow it down or break the port. A one-line grep test in CI is sufficient.

---

## 5. Technology Stack

### 5.1 Chosen Stack

| Layer | Choice | Version target | Rationale |
|-------|--------|----------------|-----------|
| Sim language | Python 3.11+ | — | Ecosystem. 3.11 for the interpreter speedup. |
| Numerics | NumPy | 1.26+ | The only Tier 1 dependency. Vectorization is the whole strategy. |
| RL framework | Stable-Baselines3 (PPO) | 2.x | Battle-tested, correct, well-documented. See §5.2. |
| DL framework | PyTorch | 2.x | SB3 dependency; also the export path. |
| Env API | Gymnasium | 0.29+ | Standard interface; SB3 speaks it natively. |
| Config | Hydra or plain YAML + dataclasses | — | Reproducibility: every run's config is archived with its checkpoint. |
| Logging | TensorBoard | — | Free, local, no account. W&B if you want hosted charts. |
| Testing (Py) | pytest | — | — |
| Renderer | three.js | r160+ | The default for a reason. WebGL2. |
| Build (JS) | Vite | 5.x | Zero-config, fast, static output. |
| Testing (JS) | Vitest | — | Same config as Vite. |
| Hosting | GitHub Pages | — | Free, static, and the URL lives on the repo. |
| CI | GitHub Actions | — | Test on push; deploy viewer on merge to main. |

### 5.2 Key Decision: Stable-Baselines3 vs. Implementing PPO Yourself

This is a genuine fork with a real trade-off.

**Use SB3 if:** the portfolio claim is "I can build environments and train agents." SB3's PPO is correct — yours probably won't be on the first try — and the bugs you'd spend a week finding (advantage normalization placement, value clipping, the `done`/`truncated` distinction in bootstrapping) are bugs SB3 already fixed. Your time goes into the environment, which is the interesting part and the part that's actually yours.

**Implement PPO yourself if:** the portfolio claim is "I understand RL algorithms." A from-scratch PPO in ~300 lines, tested against SB3 on the same environment, is a strong signal — *provided the curves match*. If your implementation underperforms SB3 and you ship it anyway, you've made a negative claim.

**Recommendation: use SB3 for the headline result, and optionally add `windward/ppo_from_scratch.py` later as a documented side-quest with a comparison plot.** This gets both signals in the right order and de-risks the critical path. The environment is where the differentiated work is; PPO is a solved problem and reimplementing it on the critical path is how projects die at 80%.

### 5.3 Alternatives Considered and Rejected

| Alternative | Why it's attractive | Why rejected |
|-------------|---------------------|--------------|
| **Unity + ML-Agents** | Gorgeous out of the box; ML-Agents handles training | Repo becomes binary assets and `.meta` files. Nobody can read your code on GitHub. No clickable demo without a WebGL build (large, slow). The portfolio artifact is worse. |
| **Godot 4 + custom RL** | Open source, readable scenes, good web export | Same headless-throughput problem. You'd end up writing a separate sim anyway — which is this architecture, plus a game engine you didn't need. |
| **All-JavaScript + neuroevolution** | Trains in-browser, no Python, spectacular demo (render all 200 boats of a generation at once) | Loses PyTorch/PPO from the resume. Neuroevolution is less sample-efficient and reads as "couldn't do gradient-based RL." **Viable if targeting gamedev/graphics roles rather than ML roles** — see §17.4. |
| **Pyodide (run the Python sim in-browser)** | Eliminates the double-implementation | Multi-megabyte download, slow cold start, and NumPy-in-WASM performance is not what you want in a render loop. Solves a 200-line problem with a 10MB hammer. |
| **Rust core → PyO3 + WASM** | Genuinely elegant; one implementation, both targets, very fast | Real answer if the physics were complex. For a polar-table model it's a week of toolchain work for a problem you don't have. Note it in the README as "considered, over-engineered for this scale" — that sentence itself is a signal. |
| **MuJoCo / PyBullet** | Real physics, free | You don't want real physics. You want a polar diagram. The rigid-body solver is 100× slower and models nothing you care about. |

### 5.4 Hardware Requirements

Deliberately modest, and this is worth stating in the README:

- **Training:** CPU-only laptop. The environment is the bottleneck, not the network — a 2×64 MLP's forward/backward pass is negligible next to the sim. A GPU provides almost no speedup here and adds a CUDA dependency to the reproduction instructions.
- **Target run:** 10M steps in under 30 minutes on 8 cores.
- **Viewer:** any device with WebGL2, including phones.

"Trains in 20 minutes on a laptop, no GPU" is a stronger claim than "trains in 20 minutes on an A100." It means a reviewer can actually reproduce it.

---

## 6. Procedural World Generation

### 6.1 The Determinism Contract

Everything about a world derives from a single 64-bit integer seed. This is the foundational constraint of the entire project, and it buys four things:

1. **A fair leaderboard.** Two runs on seed 8_675_309 face the identical map. Human and agent scores are comparable.
2. **Tiny storage.** A leaderboard entry is `(seed: u64, score: f32)` — twelve bytes. No map files, no assets, no CDN.
3. **Reproducible bugs.** "The agent crashes on seed 4471 at t=340" is a bug report you can act on.
4. **The parity test.** §13.2 is only possible because the same seed must produce the same world in both languages.

The contract is enforced, not assumed:

```python
# tests/test_determinism.py
def test_world_is_pure_function_of_seed():
    for seed in [0, 1, 42, 2**31, 2**63 - 1]:
        a = generate_world(seed)
        b = generate_world(seed)
        assert hash_world(a) == hash_world(b)

def test_seeds_are_independent():
    # Adjacent seeds must not produce correlated worlds — a classic
    # failure with naive LCGs where low bits barely change.
    worlds = [generate_world(s) for s in range(100)]
    assert len({hash_world(w) for w in worlds}) == 100
```

**Never use `random`, `np.random`'s global state, or `Math.random()`.** Every stochastic call threads an explicit RNG object. The global-state versions cannot be made deterministic across a vectorized environment and will silently ruin the parity test.

### 6.2 The RNG: SplitMix64

We need a PRNG that is (a) trivially portable to JavaScript, (b) well-distributed on adjacent seeds, and (c) splittable so each subsystem gets an independent stream.

SplitMix64 satisfies all three in about ten lines. The critical subtlety is that JavaScript has no native 64-bit integers in its `Number` type, so the port must use `BigInt` — and `BigInt` is slow. Two options:

- **Option A (recommended):** Use SplitMix64 with `BigInt` in JS *only during world generation*, which happens once per episode and is not in the hot loop. Cost: a few milliseconds. Irrelevant.
- **Option B:** Use a 32-bit PRNG (mulberry32 / PCG32) everywhere. Faster in JS, but 32-bit seeds mean a smaller world space and more care around correlation.

Go with A. Worldgen is not the hot path; the physics loop is, and the physics loop uses no randomness at all after `reset()`.

**Stream splitting.** Derive a child RNG per subsystem so that changing the terrain parameters doesn't reshuffle the treasure:

```python
root     = SplitMix64(seed)
rng_terrain  = SplitMix64(root.next())
rng_wind     = SplitMix64(root.next())
rng_current  = SplitMix64(root.next())
rng_treasure = SplitMix64(root.next())
rng_spawn    = SplitMix64(root.next())
```

Without this, tuning one knob invalidates every world you've looked at, and debugging becomes archaeology.

### 6.3 Terrain: Islands and Hazards

**Step 1 — Base field.** Fractal Brownian motion over 2D simplex noise:

```
fbm(p) = Σ_{i=0}^{octaves-1}  amplitude^i · noise(p · lacunarity^i)
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `octaves` | 5 | More = finer detail. Beyond ~6, invisible and costly. |
| `lacunarity` | 2.0 | Frequency multiplier per octave. |
| `gain` | 0.5 | Amplitude decay per octave. <0.5 = smoother; >0.5 = rougher. |
| `base_frequency` | 0.008 | Sets island scale relative to world units. |

**Step 2 — Domain warping.** Raw fBm produces recognizably "Perlin" blobs — soft, isotropic, and immediately identifiable to anyone who has seen procedural terrain before. Domain warping fixes this by offsetting the sample position with another noise field:

```
q = ( fbm(p + offset_a), fbm(p + offset_b) )
height = fbm(p + warp_strength · q)
```

The result is organic, sinuous coastlines with peninsulas and inlets — visually far better, at the cost of two extra fBm evaluations. `warp_strength ≈ 40.0` in world units is a reasonable starting point. This is a two-line change with an outsized visual payoff, and it's specific enough that mentioning it in the README signals you've actually read the PCG literature rather than copying a Perlin tutorial.

**Step 3 — Thresholding.** Land is `height > sea_level`. Sweeping `sea_level` is the primary difficulty knob: high sea level = open water with sparse rocks (easy); low sea level = a dense archipelago of narrow channels (hard). This gives the curriculum (§9.4) a single continuous parameter to anneal.

**Step 4 — Hazard classification.** Not all land is equal:

- `height > sea_level + 0.15` → **island**: large, visible from a distance, rendered with terrain.
- `sea_level < height ≤ sea_level + 0.15` → **rock/shoal**: small, low-lying, *partially submerged*. These are the dangerous ones — hard to see, deadly to hit. They are what makes the water "treacherous" and they're where the valuable treasure lives.

**Representation.** A `(256, 256)` float32 heightmap covering a 2048×2048 world-unit square (8 units per cell). Collision queries bilinearly interpolate the heightmap, which is both fast and continuous — a grid-cell lookup would produce a discontinuous collision surface and give the agent's rangefinders a stair-stepped, aliased view of the world.

### 6.4 The Wind Field

Wind is the actuator and the antagonist. Model:

```
true_wind(x, z, t) = base_direction + gust_field(x, z, t)
```

- `base_direction` — sampled uniformly from [0, 2π) per episode from `rng_wind`. **This must be randomized**; a fixed wind direction lets the agent memorize a compass heading instead of learning to read the wind.
- `base_speed` — sampled from a plausible range, e.g. 6–18 knots in whatever units you adopt.
- `gust_field` — low-frequency 3D noise `fbm(x·f, z·f, t·f_t)` producing slow spatial and temporal variation of ±20° and ±30% speed. Keep `f_t` low (~0.05) so gusts evolve over ~20 seconds, not per-frame. Fast wind noise is unlearnable and just injects variance into your gradients.

**Design note:** wind shifts are what make sailing tactically interesting for humans, but they are also a source of aleatoric uncertainty the agent cannot control. Start with gusts *disabled* (`gust_strength = 0`) and enable them late in the curriculum. Debugging a non-converging agent is much harder when the environment is stochastic.

### 6.5 The Current Field

Currents use **curl noise**, which is worth understanding rather than copying.

A naive vector field — `(noise(p), noise(p + offset))` — has non-zero divergence, meaning fluid appears from nowhere at some points and vanishes at others. It looks subtly wrong: eddies that leak. Curl noise takes the curl of a potential field, and by the vector identity `∇ · (∇ × F) = 0`, the result is divergence-free by construction — incompressible flow, which is what water actually is.

In 2D, with scalar potential ψ:

```
current(x, z) = ( ∂ψ/∂z , −∂ψ/∂x )
```

with the partials evaluated by finite differences on an fBm field. Five lines of code for a field of swirling eddies that never leak.

**Gameplay function:** currents are a *hidden* state. They are not directly observable — the agent cannot see the current, only its integrated effect on position over time. This forces implicit state estimation. If you want to be generous, expose a noisy current estimate in the observation; if you want a harder problem, don't. Recommendation: don't expose it initially; if the agent stalls, that's a diagnostic (§9.6), not a reason to weaken the task.

### 6.6 Treasure Placement: Poisson-Disk Sampling

Uniform random placement clusters. This is the well-known and counter-intuitive property of the Poisson process — random points *look* clumpy, with voids and doublets. For treasure this is bad: clusters trivialize the reward (one detour, three pickups) and voids produce dead map regions.

**Bridson's algorithm** produces blue-noise distributions — every point at least `r` from every other, but not gridded. O(n), about 40 lines, and the visual difference is immediate.

Procedure:

1. Run Bridson over the full world bounds with `r = 80` world units.
2. Reject any candidate on land (`height > sea_level`).
3. Reject any candidate within `min_clearance = 15` units of land — treasure inside a rock is not collectable.
4. Take the first `N = 25` survivors in generation order (deterministic; don't shuffle).

### 6.7 Risk-Weighted Valuation — The Central Mechanic

This is the rule that makes the project's premise honest, and it is one line:

```python
d = distance_to_nearest_hazard(pos)          # world units
value = value_max * clip(hazard_radius / (d + eps), 0.1, 1.0)
```

with `hazard_radius ≈ 30`, `value_max = 100`.

Treasure in open water is worth ~10 points. Treasure wedged in a rock channel 5 units from a shoal is worth 100. **"Treacherous" and "lucrative" are now the same axis**, and the agent must learn a risk appetite: is a 100-point pickup worth a 15% crash probability, given that crashing forfeits everything collected so far?

This is what separates the project from a maze solver. Without it, the optimal policy is "visit all treasure, avoid all rocks" — a travelling-salesman problem with obstacle avoidance, which is a fine exercise but not an interesting decision problem. With it, the optimal policy depends on how much treasure you're already carrying, which means the agent must learn something resembling loss aversion. That behaviour is visible in the demo and it's a genuinely good README paragraph.

**Expected emergent behaviour worth watching for:** an agent that takes big risks early (nothing to lose) and plays conservatively once loaded (a crash now forfeits 600 points). If you observe this, it is the single best result the project can produce. Instrument for it: log `risk_taken` vs. `treasure_held` and plot the correlation.

### 6.8 Validation and Rejection Sampling

Generated worlds can be unusable. Validate before returning:

| Check | Method | Action on failure |
|-------|--------|-------------------|
| Spawn is in water | heightmap lookup | resample spawn (up to 50×) |
| Spawn has sea room | no land within 60 units | resample spawn |
| Treasure is reachable | flood-fill water from spawn on the heightmap grid; assert every treasure is in the reachable component | drop unreachable treasure; if >30% dropped, **reject the seed entirely** |
| Enough treasure survives | `len(treasure) >= 15` | reject the seed |
| Not trivially open | land fraction > 3% | reject the seed |

On rejection, advance to `seed + 1` and retry, up to 10 attempts, then raise. **Log the rejection rate** — if it exceeds ~5%, your generation parameters are wrong and you're papering over it with retries.

The flood-fill matters more than it sounds. Without it, some fraction of episodes contain treasure inside a fully enclosed lagoon. The agent cannot reach it, receives shaping reward for approaching the lagoon wall, and learns to press against rocks. This is a real, subtle bug that presents as "the agent likes crashing" and will cost you a day if you don't rule it out first. It belongs in `docs/failures.md` either way.

### 6.9 World Data Structure

```python
@dataclass(frozen=True)
class World:
    seed: int
    heightmap: np.ndarray      # (256, 256) float32
    sea_level: float
    spawn: np.ndarray          # (2,) float32 — x, z
    spawn_heading: float
    wind_base_dir: float
    wind_base_speed: float
    treasure_pos: np.ndarray   # (N, 2) float32
    treasure_value: np.ndarray # (N,) float32
    sdf: np.ndarray            # (256, 256) float32 — precomputed, see below
```

**The SDF is the key optimization.** A signed distance field to the nearest land, precomputed once per world with `scipy.ndimage.distance_transform_edt`, turns two hot-loop operations into O(1) lookups:

- **Collision:** `sdf(pos) < hull_radius`. One bilinear sample instead of a polygon test.
- **Rangefinders:** sphere-marching along each ray — step by `sdf(p)` (guaranteed safe, since nothing is closer than that), repeat. Converges in ~5 steps instead of ~50 fixed-size steps.

Given 32 rays × 64 worlds × 150k steps/sec, ray casting *is* the hot loop. The SDF is the difference between 150k steps/sec and 15k. Precompute it in `generate_world()` and never think about it again.

---

## 7. Ship Physics Model

### 7.1 Philosophy: The Polar Diagram

**Do not simulate sails from first principles.** Lift coefficients, apparent-wind angles of attack, sail-camber models, hull hydrodynamics — all of it is a month of work producing a model you will then spend another month tuning until it behaves like the lookup table you should have written on day one.

Real sailors don't think in lift coefficients either. They use a **polar diagram**: a chart of achievable boat speed as a function of true wind angle and true wind speed. It's the standard instrument of the sport. It's a 2D lookup table.

```
              0° (in irons — no-go)
                    ╱│╲
                 ╱   │   ╲          ← no-go zone, ±40°
              ╱      │      ╲          boat speed ≈ 0
           ╱         │         ╲
      45°  ●─────────┼─────────●  45°   ← close-hauled: fast
        ╱            │            ╲
      ╱              │              ╲
   90°●              │              ●90° ← beam reach: FASTEST
      ╲              │              ╱
        ╲            │            ╱
          ●──────────┼──────────●       ← broad reach
            ╲        │        ╱
               ╲     │     ╱
                  ╲  │  ╱
                   180° (running — slow-ish)
```

The physically important and non-obvious facts this encodes:

- **You cannot sail into the wind.** Within roughly ±40° of the true wind direction, the sail cannot generate forward lift. This is the *no-go zone* or "in irons."
- **Beam reach (~90°) is fastest**, not downwind. This surprises non-sailors, and it is why the agent's paths will look strange and correct.
- **Directly downwind (180°) is slower than a broad reach (~135°).** Real racers tack downwind for this reason.

**The agent must discover tacking.** To reach a target upwind, the only route is a zig-zag of alternating close-hauled legs. This is not scripted, not rewarded directly, and not hinted at in the observation space. It emerges from the polar curve plus a distance-shaping term, and the first time you see it happen it is worth a GIF.

### 7.2 State and Parameters

```python
# Per-boat state (vectorized: each is shape (N,) or (N, 2))
pos          : (N, 2) float32   # world units
heading      : (N,)   float32   # radians, 0 = +x axis
vel          : (N, 2) float32   # world units / sec
yaw_rate     : (N,)   float32   # rad / sec
sail_trim    : (N,)   float32   # [0, 1] — 0 = sheeted in, 1 = eased out
rudder       : (N,)   float32   # [-1, 1]
treasure     : (N,)   float32   # accumulated value this episode
```

| Constant | Value | Note |
|----------|-------|------|
| `dt` | 0.05 s | 20 Hz sim. Fine for this dynamics; the render interpolates. |
| `hull_radius` | 4.0 | Collision radius. |
| `max_speed` | 12.0 | Units/sec at optimal polar point. |
| `rudder_authority` | 1.2 | rad/s at full deflection, full speed. |
| `yaw_damping` | 2.5 | Higher = less skid. |
| `drag_linear` | 0.4 | |
| `no_go_half_angle` | 40° | The defining constraint. |

### 7.3 The Step Function

```python
def step(state, world, action, dt):
    rudder_cmd, trim_cmd = action[:, 0], action[:, 1]

    # 1. Actuator lag — the agent commands a target, not a value.
    #    Without this, the policy learns bang-bang control that looks
    #    like a seizure and is unwatchable in the demo.
    state.rudder    += clip(rudder_cmd - state.rudder,    -RUDDER_RATE*dt, RUDDER_RATE*dt)
    state.sail_trim += clip(trim_cmd  - state.sail_trim,  -TRIM_RATE*dt,   TRIM_RATE*dt)

    # 2. Apparent wind = true wind − boat velocity. This is the whole game.
    true_wind = world.wind_at(state.pos, t)
    app_wind  = true_wind - state.vel
    awa       = wrap_pi(atan2(app_wind[:,1], app_wind[:,0]) - state.heading)
    aws       = norm(app_wind)

    # 3. Polar lookup: target speed given apparent wind angle & speed.
    target_speed = polar_table(abs(awa), aws)          # (N,)

    # 4. Trim efficiency: correct trim for this angle is a known curve;
    #    error costs speed. Gives the second action dimension a purpose.
    optimal_trim = trim_curve(abs(awa))
    efficiency   = 1.0 - 0.6 * abs(state.sail_trim - optimal_trim)
    target_speed *= efficiency

    # 5. Approach target speed with first-order lag (inertia).
    fwd        = stack([cos(state.heading), sin(state.heading)], -1)
    target_vel = fwd * target_speed[:, None]
    state.vel += (target_vel - state.vel) * (1 - exp(-dt / SPEED_TAU))

    # 6. Current advects the hull. Hidden state — not in the observation.
    state.vel += world.current_at(state.pos) * CURRENT_COUPLING

    # 7. Steering. Rudder authority scales with speed: a stalled boat
    #    cannot steer. This is real, and it produces a great failure mode —
    #    an agent stuck in irons, drifting, unable to turn out.
    speed_factor = clip(norm(state.vel) / 3.0, 0, 1)
    state.yaw_rate += (state.rudder * RUDDER_AUTHORITY * speed_factor
                       - state.yaw_rate * YAW_DAMPING) * dt
    state.heading   = wrap_pi(state.heading + state.yaw_rate * dt)

    # 8. Semi-implicit Euler: integrate velocity, then position.
    state.pos += state.vel * dt

    # 9. Collision & pickup: O(1) each via precomputed SDF / KD-tree.
    crashed = world.sdf_at(state.pos) < HULL_RADIUS
    picked  = pickup_check(state.pos, world.treasure_pos, PICKUP_RADIUS)

    return state, crashed, picked
```

Every line here is a NumPy operation over `N` boats at once. There is no Python loop over boats anywhere. That is what buys the throughput in §4.2.

### 7.4 The Polar Table

A `(37, 5)` array — apparent wind angle in 5° increments from 0° to 180°, wind speed in 5 bins — bilinearly interpolated. Hand-authored from the qualitative shape in §7.1:

| AWA | Relative speed | Sailing term |
|-----|----------------|--------------|
| 0–40° | 0.0 | in irons / no-go |
| 45° | 0.65 | close-hauled |
| 60° | 0.85 | close reach |
| 90° | **1.00** | beam reach (fastest) |
| 135° | 0.90 | broad reach |
| 180° | 0.60 | running |

Smooth the transition at the no-go boundary with a `smoothstep` from 40° to 50°. **A hard discontinuity at 40° is a wall in the value function** and will slow learning measurably. Everything the agent experiences should be differentiable-ish.

### 7.5 Numerical Determinism and Cross-Language Parity

The parity test (§13.2) requires bit-comparable behaviour between NumPy and JavaScript. Sources of divergence:

| Hazard | Mitigation |
|--------|------------|
| float32 (NumPy default) vs. float64 (JS `Number`) | **Use float64 in Python for physics.** Memory is not the constraint; the sim state is a few KB. Match JS. |
| Transcendental functions (`sin`, `exp`, `atan2`) differ in last-bit rounding across libm implementations | Unavoidable. Assert with tolerance `1e-4` over 500 steps, not bit-exactness. Divergence beyond that is a real bug, not rounding. |
| Reduction order in `np.sum` vs. a JS loop | Avoid reductions in physics. There are none in §7.3. |
| `np.clip` vs `Math.min/max` NaN handling | Assert no NaNs each step in debug builds. A single NaN silently poisons an entire training run. |

Set the tolerance and hold the line. If 500 steps diverge by more than 1e-4, something structural is wrong — usually an operator-precedence typo in the port, and the test will localize it to the step where it first exceeds tolerance.


---

## 8. The Reinforcement Learning Environment

### 8.1 Observation Space

**Every observation is ego-relative and normalized to roughly [-1, 1].** This is not stylistic. An agent given world-space coordinates learns "treasure is usually at x=340" and collapses the moment you change the seed. Ego-relative observations are what make generalization possible at all.

| Block | Dims | Encoding | Rationale |
|-------|------|----------|-----------|
| Rangefinders | 32 | `1 - clip(dist/max_range, 0, 1)` per ray, fanned ±120° around the bow | The workhorse. Local obstacle perception, sensor-like, translation- and rotation-invariant. |
| Apparent wind | 3 | `sin(awa)`, `cos(awa)`, `aws / max_wind` | The agent must know where the wind is to sail at all. |
| Velocity | 2 | forward and lateral components in **body frame**, `/ max_speed` | Body frame, not world frame. World frame leaks absolute heading. |
| Yaw rate | 1 | `/ max_yaw_rate` | |
| Actuator state | 2 | current `rudder`, `sail_trim` | The agent commands targets; it needs to know where the actuators actually are. |
| Nearest treasure ×3 | 9 | per target: `sin(bearing)`, `cos(bearing)`, `clip(dist/range, 0, 1)` — bearing in body frame | Three, not one: allows route planning ("grab this one on the way to that one"). |
| Treasure held | 1 | `held / expected_max` | **Essential for risk modulation.** Without this the agent cannot condition on what it has to lose, and the loss-aversion behaviour of §6.7 is unlearnable. |
| **Total** | **50** | | |

**Angles are always `(sin θ, cos θ)`, never raw θ.** A raw angle has a discontinuity at ±π: two nearly identical physical situations produce observations at opposite ends of the input range, and the network must burn capacity learning to stitch that seam. The sin/cos encoding is continuous everywhere and costs one extra dimension. This is the single most common beginner error in robotics RL and getting it right is free.

**On not observing the current:** currents are deliberately absent from the observation (§6.5). The agent must infer drift from the mismatch between commanded and achieved motion. With an MLP and no memory this is only partially possible — the agent will learn a robust, slightly conservative policy rather than an exact one. That is an acceptable and interesting outcome. If you later add an LSTM or frame-stacking and the agent visibly improves in high-current regions, **that is a publishable-quality ablation for a portfolio project** and a much better use of an LSTM than adding one by default.

### 8.2 Action Space

`Box(low=-1, high=1, shape=(2,))`, continuous:

| Index | Meaning | Mapping |
|-------|---------|---------|
| 0 | Rudder target | `[-1, 1]` → full port to full starboard |
| 1 | Sail trim target | `[-1, 1]` → linearly remapped to `[0, 1]` sheeted-in to eased-out |

Continuous over discrete, deliberately: discretizing the rudder produces visible stair-stepping in the demo, and PPO handles Gaussian policies natively. The actuator lag in §7.3 step 1 is what keeps the Gaussian's sampling noise from turning into visible jitter — the boat physically cannot slam the rudder every 50ms.

### 8.3 Reward Function

This is where the project fails if you are careless. The naïve reading of the brief — "reward treasure, penalize crashing" — is a sparse reward with a terminal penalty. The agent will discover that the safest policy is to never move, collect zero, and never crash. It will sit there. Forever. Reward zero is a local optimum with a very wide basin, and random exploration will not escape it in any reasonable number of steps.

The reward has four terms:

```python
r = 0.0

# 1. Terminal treasure — the actual objective.
r += treasure_value_collected_this_step          # ~10 to 100

# 2. Potential-based shaping toward the nearest treasure.
#    Φ(s) = −distance_to_nearest_uncollected_treasure
r += GAMMA * phi(s_next) - phi(s)                # ~±0.05 per step

# 3. Time cost — makes dawdling strictly worse than acting.
r -= 0.01

# 4. Crash — forfeits everything.
if crashed:
    r -= 50.0
```

**Term 2 is the important one and it has a specific mathematical form for a specific reason.**

Ng, Harada & Russell (1999), *"Policy Invariance Under Reward Transformations"*, proves that a shaping term of the form `F(s, s') = γΦ(s') − Φ(s)` — a difference of potentials — leaves the optimal policy **unchanged**. Any other shaping function can, and generally does, change what the agent is optimizing.

This matters concretely. The obvious alternative, "reward the agent a little each step for being close to treasure," is `F(s) = −k·d(s)`, which is *not* potential-based. Its optimal policy is to park next to the highest-value treasure and never collect it, farming proximity reward forever. **This is a real failure mode that you will hit if you use the naïve form, and it produces a fantastic GIF for `docs/failures.md`** — a boat serenely circling a chest it will never touch.

Cite the paper in the README. Knowing *why* the potential-difference form specifically is what separates "tuned reward coefficients until it worked" from "understands reward shaping."

**Recommended parameter sweep (as an ablation, not a guess):**

| `shaping_scale` | Expected outcome |
|-----------------|------------------|
| 0.0 | Never learns. Sparse reward, agent sits still. |
| 0.5 | Learns slowly. |
| 1.0 | Baseline. |
| 3.0 | Fast early learning; may over-prioritize the nearest chest and ignore the value gradient — "greedy nearest-neighbour" behaviour, visibly suboptimal. |

That table is itself a README section.

### 8.4 Termination and Truncation

Gymnasium distinguishes these and **the distinction is load-bearing**:

| Condition | Type | Bootstrap value? |
|-----------|------|------------------|
| Crashed into land | `terminated = True` | **No.** The episode genuinely ended; V(s') = 0. |
| All treasure collected | `terminated = True` | No. |
| Timeout (3000 steps = 150 s) | `truncated = True` | **Yes.** The episode was cut off artificially; V(s') must be bootstrapped or you teach the agent that the world ends at t=3000. |

Getting this backwards is a classic, silent RL bug: bootstrap on a real terminal and the value function learns that crashing has a future; fail to bootstrap on a timeout and the agent learns to panic near the horizon. SB3 handles it correctly *if* you return the flags correctly. Return them correctly.

### 8.5 Vectorization

The environment is written to hold `N` worlds simultaneously. Not `N` copies of a single-world environment behind a wrapper — that reintroduces the Python loop and forfeits the throughput.

```python
class VecWindward:
    def __init__(self, n_envs=64, seed_sampler=None): ...
    def reset(self)          -> obs   # (64, 50)
    def step(self, actions)  -> obs, rewards, terminated, truncated, infos
                                      # (64, 50), (64,), (64,), (64,), list
```

**Auto-reset with a fresh seed.** When world `i` terminates, immediately regenerate it from a new seed and continue. Never let a world idle — an idling world is wasted throughput, and staggered resets keep the batch decorrelated, which is good for the gradient estimate anyway.

**The one Python loop that survives.** `generate_world()` is not vectorizable — the flood-fill, Bridson sampling, and rejection logic are inherently sequential. But it runs only on reset (~once per 500–3000 steps per world). Budget ~5ms per world; at 64 worlds with staggered resets this is a fraction of a percent of wall-clock. Profile it once to confirm, then leave it alone. If resets ever become the bottleneck, the fix is a background thread pre-generating a queue of worlds — but measure before building that.

---

## 9. Training Methodology

### 9.1 Network Architecture

```
obs (50) → Linear(50, 64) → Tanh → Linear(64, 64) → Tanh → ┬→ Linear(64, 2)  [policy mean]
                                                            └→ Linear(64, 1)  [value]

log_std: learnable parameter vector (2,), state-independent, init −0.5
```

Roughly 8,000 parameters. Deliberately small, for four reasons:

1. It is sufficient. This is a 50-dimensional reactive control problem, not ImageNet. Capacity is not the constraint.
2. It trains fast on CPU, keeping the environment as the bottleneck where it belongs.
3. It exports to a ~40KB JSON that a browser parses instantly.
4. **A 40-line JavaScript forward pass can reproduce it exactly.** This is what makes §10 possible, and §10 is one of the better stories in the project.

**Tanh over ReLU**, deliberately: bounded activations produce smoother policies, and smooth policies produce boats that don't twitch. On a demo whose entire purpose is to be watched, this matters.

### 9.2 PPO Hyperparameters

Starting point. These are reasonable defaults for continuous control, not magic numbers — but resist tuning them until the environment is proven, because hyperparameter tuning is a very comfortable way to avoid noticing that your reward function is broken.

| Parameter | Value | Note |
|-----------|-------|------|
| `n_envs` | 64 | Parallel worlds. |
| `n_steps` | 256 | Per env per update → 16,384 sample batch. |
| `batch_size` | 2048 | Minibatch for SGD. |
| `n_epochs` | 10 | Passes per rollout. |
| `gamma` | 0.995 | **High.** Episodes are ~3000 steps; treasure is far. γ=0.99 gives a ~100-step horizon — too myopic to plan a tack. |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `ent_coef` | 0.005 | Some exploration pressure. Anneal to 0 late. |
| `vf_coef` | 0.5 | |
| `max_grad_norm` | 0.5 | |
| `learning_rate` | 3e-4 → 0 | Linear anneal. |
| `total_timesteps` | 30M | ~30–45 min on 8 cores. |

**`gamma = 0.995` is the one to think hardest about.** The effective horizon is roughly `1/(1-γ)` steps: γ=0.99 → 100 steps → 5 seconds of foresight. A tack takes 20+ seconds to pay off. **An agent with γ=0.99 mathematically cannot learn to tack** — the reward is beyond its horizon. This is a beautiful, concrete illustration of what the discount factor *means*, and a γ sweep (0.98 / 0.99 / 0.995 / 0.999) showing the tacking behaviour appear at a threshold is one of the most compelling ablations available to this project. Prioritize running it.

### 9.3 Domain Randomization — The Central Experiment

**Every episode gets a new seed.** This is the difference between a project that demonstrates RL and one that demonstrates a memorized trajectory.

```python
train_seeds = range(0, 1_000_000)         # sampled per episode
eval_seeds  = range(9_000_000, 9_000_100) # 100 held-out, NEVER trained on
```

Randomized per episode: world seed (hence terrain, treasure, wind direction, currents), spawn position and heading, wind base speed, and — late in training — gust strength.

**The plot this produces is the most important figure in the README:** training reward and held-out-seed reward on the same axes. If they track, you have generalization. If held-out flatlines while training climbs, you have memorization, and you have caught the single most common unforced error in applied RL. Either outcome is worth publishing; the second is worth publishing *with* the fix.

For maximum effect, run the memorizing baseline on purpose: train with a fixed seed, show the agent achieving a great score on that map and a terrible one everywhere else. **Two curves on one plot make the argument instantly.** Ship it as `docs/ablations.md` Figure 1.

### 9.4 Curriculum

Sailing with obstacles, currents, gusts, and risk-weighted treasure from step zero is a hard exploration problem, and a randomly initialized agent will crash within 20 steps forever, never seeing a reward, never learning.

Anneal difficulty against a scalar `d ∈ [0, 1]` driven by a performance gate:

| Stage | `d` | `sea_level` | Gusts | Currents | Treasure count | Gate to advance |
|-------|-----|-------------|-------|----------|----------------|-----------------|
| 0 | 0.0 | very high (open water) | off | off | 5, all low-value | mean return > 200 |
| 1 | 0.25 | high (sparse rocks) | off | weak | 10 | mean return > 300 |
| 2 | 0.5 | medium | weak | medium | 15 | mean return > 400 |
| 3 | 0.75 | low (archipelago) | medium | medium | 20 | mean return > 500 |
| 4 | 1.0 | full range, sampled | full | full | 25 | — |

Advance when the 100-episode rolling mean clears the gate. **Never regress the curriculum** on a bad patch; it oscillates and never converges.

**Alternative worth considering:** sample `d ~ Uniform(0, d_max)` where only `d_max` advances. This keeps easy episodes in the mix and prevents catastrophic forgetting of the open-water skills. Slightly more code, meaningfully more robust — and it's the version to write about, because "I hit catastrophic forgetting when the curriculum advanced and fixed it by sampling difficulty rather than stepping it" is a genuine engineering narrative.

### 9.5 Evaluation Protocol

Evaluation must be deterministic or the leaderboard is noise.

```python
def evaluate(policy, seeds=EVAL_SEEDS, deterministic=True):
    # deterministic=True -> use the Gaussian's MEAN, not a sample.
    # Sampling at eval adds variance that has nothing to do with skill.
    scores = []
    for seed in seeds:
        world = generate_world(seed)   # identical world every time
        scores.append(run_episode(policy, world, deterministic=True))
    return {
        "mean":   np.mean(scores),
        "median": np.median(scores),
        "p10":    np.percentile(scores, 10),   # tail risk
        "crash_rate": ...,
        "per_seed": dict(zip(seeds, scores)),  # -> leaderboard.json
    }
```

Report **median and p10, not just mean.** An agent that scores 900 on 80% of maps and crashes instantly on 20% has the same mean as one that reliably scores 700 — and it is a much worse agent. The p10 is where risk appetite shows up, and given that risk appetite is the project's central mechanic (§6.7), it deserves to be measured directly rather than averaged away.

### 9.6 Expected Failure Modes and Diagnostics

Write these in `docs/failures.md` **as they happen**, with the GIF. This section is the highest-signal content in the entire repository.

| Symptom | Likely cause | Diagnostic | Fix |
|---------|--------------|------------|-----|
| Reward flat at ~0, boat motionless | Sparse reward; sitting still is a local optimum | Check `shaping_scale > 0` | Enable potential shaping |
| Boat circles a chest without touching it | Shaping is not potential-based | Inspect the reward formula | Use `γΦ(s') − Φ(s)` |
| Learns, then collapses to zero | Value loss explosion, or curriculum jumped too far | Watch `value_loss`, `clip_fraction` | Lower LR; sample difficulty (§9.4) |
| Never tacks; stalls facing upwind | Horizon too short | Sweep `gamma` | γ ≥ 0.995 |
| Great on train seeds, terrible on eval | Memorization | The two-curve plot (§9.3) | Randomize seed per episode |
| Beaches itself deliberately at low speed | Crash penalty < accumulated time cost | Log return decomposition | Raise crash penalty; lower time cost |
| Refuses all high-value treasure | Crash penalty too high — pathological risk aversion | Log `risk_taken` vs. `treasure_held` | Lower crash penalty |
| Twitchy, unwatchable steering | Actuator lag missing or too fast | Watch the demo | Lower `RUDDER_RATE` |
| NaNs after ~1M steps | Unbounded obs, or a `/0` in the bearing calc | Assert finite each step | Clip obs; add `eps` |
| Throughput collapses | Ray casting without the SDF | `py-spy top` | Precompute the SDF (§6.9) |

`py-spy` is worth calling out. It attaches to a running process without instrumentation and tells you where the time actually goes, which is almost never where you assume. Run it once during Milestone 3; it will likely reveal that 70% of your time is in one line you didn't think about.

---

## 10. Weight Export and Cross-Language Inference

### 10.1 The Export Format

```json
{
  "format_version": 1,
  "obs_dim": 50,
  "act_dim": 2,
  "obs_normalizer": { "mean": [...], "var": [...] },
  "layers": [
    { "W": [[...]], "b": [...], "activation": "tanh" },
    { "W": [[...]], "b": [...], "activation": "tanh" },
    { "W": [[...]], "b": [...], "activation": "identity" }
  ],
  "metadata": {
    "generation": 200,
    "total_timesteps": 30000000,
    "eval_mean": 847.3,
    "eval_median": 890.0,
    "git_sha": "a3f9c21",
    "trained_at": "2026-07-12T14:22:00Z"
  }
}
```

The `metadata` block is not decoration. It is what lets the viewer label the leaderboard, and what lets you answer "which commit produced this?" six months from now when someone asks in an interview.

### 10.2 The JavaScript Forward Pass

```javascript
export function forward(weights, obs) {
  // Observation normalization — MUST match training exactly.
  // If SB3's VecNormalize was used, its running mean/var ship with
  // the weights. Forgetting this is the #1 export bug: the agent
  // works perfectly in Python and sails directly into a rock in JS.
  let x = obs.map((v, i) =>
    (v - weights.obs_normalizer.mean[i]) /
    Math.sqrt(weights.obs_normalizer.var[i] + 1e-8)
  );

  for (const layer of weights.layers) {
    const y = new Float64Array(layer.b.length);
    for (let j = 0; j < layer.b.length; j++) {
      let s = layer.b[j];
      for (let i = 0; i < x.length; i++) s += layer.W[j][i] * x[i];
      y[j] = layer.activation === "tanh" ? Math.tanh(s) : s;
    }
    x = y;
  }
  return x;  // policy mean — deterministic action, no sampling
}
```

Forty lines, no dependencies, no ONNX runtime, no TensorFlow.js, no multi-megabyte WASM blob. **This is a genuinely good thing to have in a portfolio** — it demonstrates that you understand a neural network is a sequence of matrix multiplies and not a magic box you need a framework to open. It is also a fine interview answer to "how would you deploy a model to the browser?"

### 10.3 Export Verification

Non-negotiable, and it belongs in CI:

```python
# tests/test_export.py
def test_python_and_js_agree():
    """Run 1000 random observations through both. Assert agreement."""
    obs = rng.standard_normal((1000, 50))
    py_actions = policy(torch.tensor(obs)).detach().numpy()
    js_actions = run_node("viewer/src/policy.js", weights, obs)
    assert np.allclose(py_actions, js_actions, atol=1e-6)
```

Without this, the classic failure is a viewer where the boat *almost* works — plausible-looking but subtly wrong — and you spend a day assuming the physics port is broken when in fact you forgot the observation normalizer.

---

## 11. The 3D Viewer

### 11.1 Rendering Approach

Stylized low-poly. Not a compromise — a choice. It loads in under three seconds, runs at 60fps on a phone, requires no artist, and doesn't sit in the uncanny valley where realistic-but-not-quite water looks worse than obviously-stylized water.

| Element | Technique |
|---------|-----------|
| Ocean | Single plane, custom shader. Gerstner waves in the vertex shader (3–4 summed) for displacement; fresnel + a sampled sky color for the surface. **The waves are visual only — physics never reads them.** |
| Terrain | Heightmap → `PlaneGeometry` with displaced vertices, flat-shaded. Height-banded colors: sand → rock → grass. |
| Ship | ~200 triangles. Hull, mast, sail. The sail's mesh rotates to the trim angle and billows with the apparent wind — the single most legible visual cue in the entire demo. |
| Wake | Trailing ribbon geometry, alpha-faded over ~2s. |
| Treasure | Emissive icosahedron, bobbing, rotating. **Scale and glow intensity scale with value** — the player sees risk/reward at a glance. |
| Wind | Particle streaks drifting with the true wind field. Answers "why is the boat going *that* way?" without a HUD element. |
| Sky | Gradient skybox. No HDRI download. |

### 11.2 The Rangefinder Visualization

**Build this. It is the best single feature in the viewer.**

A toggle that draws the agent's 32 rangefinder rays as lines from the bow, colored by return distance. The viewer stops being a boat game and becomes *a window into what the network sees*. It makes the observation space (§8.1) concrete for anyone watching, it is invaluable for your own debugging, and it is the frame you should grab for the README GIF.

Pair it with a small overlay showing the live 50-dimensional observation vector as a bar strip, and the two action outputs as gauges. Total cost: an afternoon. Total value: it is the thing people will remember.

### 11.3 Camera and HUD

Camera: smoothed chase, offset behind and above, with a slight speed-dependent FOV widening. Lerp the position, `slerp` the rotation. Free-orbit and top-down tactical views on hotkeys.

HUD: apparent wind indicator (the classic masthead arrow), speed, treasure held, elapsed time, seed. Keep it minimal — the wind indicator is the only one that's essential.

### 11.4 Human Play Mode

Same physics, same worlds, same seeds. Arrow keys → rudder; W/S → trim.

This costs almost nothing once the viewer exists (the physics is already there; you're replacing `forward(weights, obs)` with a keyboard handler) and it delivers three things:

1. **A hook.** "Beat the AI on seed 42" is a reason to stay on the page.
2. **A human baseline** for the leaderboard, which is the most interpretable baseline that exists.
3. **Design validation** — see the roadmap. If *you* can't sail to treasure using only what the agent observes, the agent has no chance either, and you want to learn that on day two rather than day twenty.

---

## 12. Leaderboard Design

### 12.1 What It Actually Is

Not a global high-score table. **A leaderboard of your own agents across training**, auto-generated from checkpoints:

| Rank | Agent | Steps trained | Mean | Median | p10 | Crash rate |
|------|-------|---------------|------|--------|-----|------------|
| 1 | `gen_0400` | 30.0M | 892 | 910 | 610 | 8% |
| 2 | `gen_0200` | 15.0M | 731 | 755 | 380 | 14% |
| 3 | *You* | — | 640 | — | — | — |
| 4 | `gen_0050` | 3.8M | 402 | 380 | 90 | 39% |
| 5 | `gen_0010` | 0.8M | 88 | 40 | 0 | 82% |
| 6 | Scripted greedy | — | 210 | 190 | 0 | 61% |
| 7 | Random | — | 12 | 0 | 0 | 97% |

This is strictly better than a public high-score table:

- **It tells the story of the training run.** The reader watches the agent get good.
- **It cannot be spammed**, needs no backend, and can't be gamed.
- **The human row is the hook** — and the moment a visitor loses to `gen_0400`, they understand the project.
- **The baseline rows are the science.** Random and scripted-greedy establish that the task is non-trivial and that learning did something.

### 12.2 Selectable Opponents

The dropdown of checkpoints is the feature. Watching `gen_0010` (spins in circles, beaches immediately), then `gen_0050` (sails, greedily, into rocks), then `gen_0400` (tacks upwind, threads a channel, declines a 100-point chest it has judged too risky) is a **three-click demonstration of what training does.** No plot communicates this as well as watching it.

Ship 5–6 checkpoints, ~40KB each. 250KB total. Trivial.

### 12.3 Storage

```
viewer/public/leaderboard.json   ← agent scores, committed to the repo,
                                   regenerated by evaluate.py
localStorage                     ← the visitor's own runs, their machine only
```

That's the whole design.

### 12.4 Why There Is No Backend

Stated plainly because the reasoning is itself a signal:

1. **It would break.** A free-tier service left alone for a year is a dead link on your portfolio at the exact moment someone finally clicks it.
2. **It would get spammed.** An unauthenticated score endpoint is spammed within a week. Authentication means accounts, which means a database, which means GDPR, which means this is no longer a sailing project.
3. **It adds no signal.** "I can call a KV store from a serverless function" is not a claim anyone doubts. Every hour spent there is an hour not spent on the thing that's actually differentiated.
4. **Static hosting is free and permanent.** GitHub Pages will outlive several startups.

If asked in an interview why there's no global leaderboard, the answer above is a better answer than a working global leaderboard would have been.

---

## 13. Testing Strategy

### 13.1 Python Test Suite

| Test | Asserts |
|------|---------|
| `test_determinism` | Same seed → identical world, twice, in-process and across processes. |
| `test_seed_independence` | 100 adjacent seeds → 100 distinct worlds. |
| `test_worldgen_validity` | Over 1000 seeds: spawn in water, all treasure reachable, land fraction in range, rejection rate < 5%. |
| `test_physics_sanity` | Boat pointed at 90° AWA accelerates. Boat pointed into the no-go zone does not. Boat with zero rudder holds heading. |
| `test_physics_energy` | No configuration produces speed > `max_speed × 1.1`. Catches integration blowups. |
| `test_no_nans` | 10,000 random actions from random states → all state finite. |
| `test_obs_bounds` | Over 10,000 random states, every observation dim ∈ [-1.1, 1.1]. |
| `test_reward_shaping_is_potential` | Sum of shaping over any closed loop returning to the start state ≈ 0. **This is the actual mathematical property** — if a cycle accumulates shaping reward, the agent can farm it, and this test is what catches the §8.3 circling bug before you ever see it. |
| `test_termination_flags` | Crash → `terminated`; timeout → `truncated`. Never both. |
| `test_vec_matches_single` | 64-world vectorized step ≡ 64 single-world steps. **Catches broadcasting bugs**, which are silent and devastating. |
| `test_no_torch_in_env` | `grep -L torch windward/{env,physics,worldgen}.py`. Architectural boundary as a test. |

`test_reward_shaping_is_potential` deserves emphasis: it converts a theoretical property from a paper into an executable assertion. That is a rare and good thing to have in a repository.

### 13.2 The Parity Test

The most important test in the project, because it guards the one architectural compromise (§4.3).

```python
# tests/test_parity.py
@pytest.mark.parametrize("seed", [0, 42, 1337, 99999])
def test_python_js_trajectories_match(seed):
    actions = SplitMix64(seed).uniform(-1, 1, size=(500, 2))

    py = simulate_python(seed, actions)          # (500, 4): x, z, heading, speed
    js = simulate_node("viewer/src/physics.js", seed, actions)

    np.testing.assert_allclose(py, js, atol=1e-4)
```

Runs in CI on every push. When the port drifts — and it will, the first time you tune a physics constant and forget the JS file — this fails immediately and names the step where divergence exceeded tolerance. Without it, you discover the drift weeks later as "the viewer feels wrong somehow," which is the worst possible bug report to receive from yourself.

### 13.3 CI

```yaml
on: [push, pull_request]
jobs:
  python:  pytest -x -q             # ~30s
  js:      vitest run               # ~10s
  parity:  pytest tests/test_parity.py
  smoke:   python -m windward.train --config configs/debug.yaml --steps 50000
           # asserts mean_return > random baseline. Catches "training is
           # silently broken" without a 30-minute run.
deploy:
  needs: [python, js, parity]
  if: github.ref == 'refs/heads/main'
  run: cd viewer && npm ci && npm run build && gh-pages -d dist
```

The smoke test is the valuable one. A 50k-step run that asserts the agent beats random is a 60-second guard against the most expensive class of bug: the one where training runs fine, produces no error, and learns nothing.

---

## 14. Roadmap and Milestones

Estimates assume roughly 10–15 focused hours per week. **Milestones 4 and 5 are where the time actually goes** — everything before is mechanical, everything after is polish. Plan accordingly and do not be surprised.

### M0 — Headless Sim + PCG *(week 1)*

Deliverables: `rng.py`, `noise.py`, `worldgen.py`, `physics.py`, determinism tests, a matplotlib top-down plot.

**No 3D. No neural network. No JavaScript.** The temptation to open three.js on day one is strong and must be resisted; it is how this project becomes a boat model with no agent.

**Exit:** `plot_world(42)` renders islands, treasure, wind, and current. `plot_trajectory(42, scripted_actions)` shows a boat sailing. Determinism tests green.

### M1 — 3D Viewer + Human Play *(week 2)*

Port physics to JS, build the three.js scene, wire up the keyboard. Parity test green.

**Exit — and this is a real gate, not a checkbox: play it yourself and reach treasure. Then play it again using *only* the information in the observation space of §8.1.** Turn the camera off if you have to; drive from the rangefinder overlay and the wind indicator alone.

If you can't do it, **the agent cannot do it either**, and the observation space is wrong. Fixing that now costs an hour. Discovering it at M4 costs a week of blaming PPO. This is the single highest-leverage step in the roadmap and it is the one everybody skips.

### M2 — Baselines + Scoring Harness *(week 2–3)*

`env.py` (gym API), `evaluate.py`, random and scripted-greedy baselines, eval seed set frozen.

The scripted baseline — "point at the nearest treasure, tack if it's in the no-go zone, turn away from rocks under 30 units" — is worth the two hours. It is a real number for the leaderboard, it proves the task is completable, and if PPO can't beat it, you have a *specific* problem rather than a vague one.

**Exit:** `evaluate(random_policy)` → ~12. `evaluate(scripted)` → ~210. Numbers reproducible across runs.

### M3 — First Learning Signal *(week 3)*

Trivial world: open water, one treasure, no rocks, no gusts, no currents. PPO. Get the number to move.

**Exit:** an agent that reliably sails to a single chest. It will look stupid. It doesn't matter. **The purpose of M3 is to prove the plumbing works** — that gradients flow, that the reward reaches the optimizer, that `terminated`/`truncated` are wired correctly. Debugging plumbing and debugging behaviour at the same time is how weeks disappear.

### M4 — The Real Problem *(weeks 4–5)* ← *the hard part*

Obstacles, rangefinders, shaping, the value gradient, risk. This is where the failures in §9.6 happen. Budget double whatever you think.

**Exit:** an agent that beats the scripted baseline on training seeds.

### M5 — Generalization *(week 6)* ← *the part that matters*

Domain randomization, curriculum, the γ sweep, the held-out evaluation, the two-curve plot.

**Exit:** held-out performance within 20% of training performance. **This is the milestone that makes the project credible.** M4 produces a demo; M5 produces evidence.

### M6 — Integration *(week 7)*

Export, JS forward pass, export verification, checkpoint dropdown, leaderboard JSON.

**Exit:** the live URL runs `gen_0400` and it sails as well as it did in Python.

### M7 — Presentation *(week 8)*

The GIF. The README. `docs/failures.md`. `docs/ablations.md`. The plots. CI badge. Deploy.

**Exit:** a stranger understands the project in ninety seconds.

### Reordering Notes

- **M1 before M3 is deliberate** and is the most commonly violated ordering. Human play validates the observation space before you spend a week debugging an agent that was never given enough information to succeed.
- **M2 before M3** means you have a number to beat before you start optimizing. Otherwise "is 340 good?" is unanswerable.
- **M7 is not optional and not last-if-there's-time.** An excellent project with a bad README is invisible. If the schedule slips, cut scope from M4, not M7.

---

## 15. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | **Agent never learns to tack** | Medium | High — it's the headline | γ ≥ 0.995 (§9.2). Curriculum starts downwind. Worst case: it's still a good project, and "here is why γ=0.99 cannot learn a 20-second manoeuvre" is a *better* README section than a working tack. |
| R2 | **Reward hacking eats a week** | **High** | Medium | Expected, not feared. §9.6 is the playbook. Every instance is content for `docs/failures.md`. |
| R3 | **Physics port diverges** | Medium | Medium | Parity test in CI (§13.2). Freeze physics before porting. |
| R4 | **Scope creep** (multiplayer, weather, combat, "what if two boats race") | **High** | **High** | §2.3 is a contract. Re-read it weekly. Everything on that list is genuinely fun and will kill the project. |
| R5 | Training too slow to iterate | Low | High | §4.2 architecture prevents it. `py-spy` at M3 confirms. |
| R6 | Overfits to training seeds | Medium | High | M5 exists specifically for this. Caught by the two-curve plot. |
| R7 | Demo doesn't run on mobile | Medium | Medium | Low-poly budget from day one. Test on a real phone at M1, not M7. |
| R8 | **Loses motivation at M4** | **High** | **Total** | The real risk, and the honest one. M1 (a playable boat) and M3 (a working agent, however stupid) exist as morale checkpoints. Ship the viewer early so there's something to show. A finished 80% project beats an abandoned 100% one by an infinite margin. |

R8 is the one that actually kills projects like this. The architecture in §4 is partly a hedge against it: because the tiers are decoupled, **M1 alone is already a portfolio piece** (procedural world + playable sailing sim), and every milestone after it is additive rather than all-or-nothing. There is no point at which quitting leaves you with nothing.

---

## 16. Presentation and README Plan

### 16.1 Structure

```markdown
# Windward — a sailing agent that learns to tack

![hero](docs/media/hero.gif)          ← LINE 3. Non-negotiable.

[Live demo](https://you.github.io/windward) · [Design doc](DESIGN.md) · [What went wrong](docs/failures.md)

A neural network learns to sail. It was never told that boats can't sail
into the wind — it worked that out.

## What this is
[3 sentences. Not 3 paragraphs.]

## Results
[Two-curve plot: train vs held-out.]
[Leaderboard table.]
[Sentence: agent beats scripted baseline by 4.2× and beats me.]

## How it works
### Procedural generation  [fBm + domain warping + curl noise + Poisson-disk, 1 GIF]
### Physics               [the polar diagram ASCII chart from §7.1]
### The agent             [obs/action/reward, the shaping citation]
### Training              [PPO, domain randomization, the curriculum]

## The interesting part: risk
[§6.7. The value-scales-with-danger mechanic and the loss-aversion behaviour.]

## What went wrong
[Link to failures.md. Include one GIF inline — the circling boat.]

## Reproduce
    pip install -e .
    python -m windward.train --config configs/final.yaml   # ~30 min, CPU
    python -m windward.evaluate --checkpoint runs/final/gen_0400.zip

## Ablations
[Table. Link to ablations.md.]
```

### 16.2 The GIF

**More effort goes here than any other single artifact.** It is the entire project for 80% of visitors, and most of them will never scroll past it.

Requirements: under 5MB, under 10 seconds, loops seamlessly, and shows — in this order — a boat tacking upwind, a rangefinder overlay flash, a treasure grab in a tight channel. Chase camera. No text overlay; the caption is below it.

Capture at high resolution, downscale, and use `gifski` for the palette — the default ffmpeg GIF palette looks like 1998. Consider an MP4 with `<video autoplay loop muted playsinline>` instead: better quality, smaller file, and GitHub renders it.

### 16.3 The Failures Doc

Structure each entry:

```markdown
### The boat that loved a chest it would never touch

**Symptom:** After 4M steps, mean return plateaued at 340. Watching the
demo, the agent sailed to the nearest treasure and orbited it at ~8 units.
Forever. It never collected anything.

![the circling boat](media/circling.gif)

**Diagnosis:** My shaping term was `r -= 0.1 * distance_to_treasure` —
a reward for *being near*, not for *getting nearer*. Orbiting at radius 8
is the optimal policy for that reward: it pays 0.8/step indefinitely, and
collecting the chest ends the gravy train.

**Fix:** Potential-based shaping, `r += γΦ(s') − Φ(s)` with `Φ = −d`.
Ng, Harada & Russell (1999) prove this form leaves the optimal policy
invariant. Return hit 340 in 400k steps instead of 4M, and kept climbing.

**Lesson:** "Reward proximity" and "reward approach" are different
functions with different optima. The paper is 30 years old and I should
have read it first.
```

Three to five of these. **This is the section that convinces a senior engineer you did the work.** Anyone can produce a working demo by following a tutorial; nobody can fabricate a specific, correctly-diagnosed failure with the right citation attached.

---

## 17. Future Extensions

Listed to be **explicitly deferred**, not built. Their presence in the doc signals that they were considered and scoped out — which is a stronger signal than building them badly.

### 17.1 Recurrent Policy (LSTM)
Currents are unobservable (§6.5); an MLP can't integrate evidence over time. An LSTM should measurably outperform in high-current regions. **This is the best next experiment** because it comes with a built-in hypothesis and a clean ablation.

### 17.2 From-Scratch PPO
`ppo_from_scratch.py` benchmarked against SB3 on identical seeds, with a curve overlay. Strong signal *if the curves match.* Off the critical path (§5.2).

### 17.3 Neuroevolution Comparison
NEAT or CMA-ES on the same environment. Interesting because it should *lose* on sample efficiency and *win* on wall-clock parallelism — a real, quantified trade-off. Excellent blog post; not required.

### 17.4 Full In-Browser Training
Port the trainer to JS, train a population in Web Workers, render all 200 boats of a generation simultaneously. **Spectacular demo** — a fleet of boats, most crashing, a few finding the channel. Loses the PyTorch story. Consider as a *second* project rather than a replacement.

### 17.5 Learned World Model / Dreamer
Interesting, hard, and a different project. Note and move on.

### 17.6 Vision-Based Agent
Replace the rangefinders with a rendered depth buffer and a small CNN. Much harder, much slower, much cooler. This is the natural sequel.

---

## 18. Appendices

### A. Glossary

| Term | Meaning |
|------|---------|
| **AWA** | Apparent Wind Angle — wind angle as felt by the moving boat: `true_wind − boat_velocity`. What the sail actually responds to. |
| **Beam reach** | Sailing ~90° to the wind. Fastest point of sail. |
| **Close-hauled** | Sailing as near the wind as possible, ~45°. |
| **In irons** | Stopped, pointed into the wind, unable to steer. A failure state. |
| **No-go zone** | The ±40° arc around the true wind where no forward lift is possible. |
| **Tacking** | Zig-zagging upwind through the no-go zone. The manoeuvre the agent must discover. |
| **Polar diagram** | Boat speed as a function of wind angle and speed. The physics model. |
| **fBm** | Fractal Brownian motion — summed noise octaves at increasing frequency. |
| **Curl noise** | Divergence-free vector field from the curl of a potential. Realistic flow. |
| **Poisson-disk** | Blue-noise point distribution; random but never clustered. |
| **SDF** | Signed Distance Field — precomputed distance to nearest surface. |
| **PPO** | Proximal Policy Optimization — the RL algorithm. |
| **GAE** | Generalized Advantage Estimation — the advantage estimator PPO uses. |
| **Potential-based shaping** | Reward shaping of the form `γΦ(s') − Φ(s)`; provably policy-invariant. |
| **Domain randomization** | Randomizing environment parameters per episode to force generalization. |

### B. References

1. Ng, Harada & Russell (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML. — **The reward shaping result. §8.3 depends on it.**
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Schulman et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
4. Bridson (2007). *Fast Poisson Disk Sampling in Arbitrary Dimensions.* SIGGRAPH sketches. — §6.6.
5. Bridson, Houriham & Nordenstam (2007). *Curl-Noise for Procedural Fluid Flow.* SIGGRAPH. — §6.5.
6. Quílez, I. *Domain Warping.* iquilezles.org/articles/warp — §6.3.
7. Tobin et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS. — §9.3.
8. Andrychowicz et al. (2020). *What Matters in On-Policy Reinforcement Learning?* arXiv:2006.05990. — The empirical hyperparameter study. Read before tuning anything.
9. Engstrom et al. (2020). *Implementation Matters in Deep RL: A Case Study on PPO and TRPO.* ICLR. — Why §5.2 recommends SB3.

### C. Hyperparameter Quick Reference

```yaml
# configs/final.yaml
world:
  size: 2048
  heightmap_res: 256
  octaves: 5
  lacunarity: 2.0
  gain: 0.5
  base_frequency: 0.008
  warp_strength: 40.0
  treasure_count: 25
  treasure_min_dist: 80.0
  hazard_radius: 30.0
  value_max: 100.0

physics:
  dt: 0.05
  hull_radius: 4.0
  max_speed: 12.0
  no_go_half_angle_deg: 40.0
  rudder_authority: 1.2
  yaw_damping: 2.5
  speed_tau: 1.5

env:
  n_rays: 32
  ray_fov_deg: 240
  ray_max_range: 200.0
  n_treasure_observed: 3
  max_episode_steps: 3000
  crash_penalty: -50.0
  time_cost: -0.01
  shaping_scale: 1.0

ppo:
  n_envs: 64
  n_steps: 256
  batch_size: 2048
  n_epochs: 10
  gamma: 0.995          # ← the one that matters; see §9.2
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005
  vf_coef: 0.5
  max_grad_norm: 0.5
  learning_rate: 3.0e-4
  lr_schedule: linear
  total_timesteps: 30_000_000

eval:
  seeds: [9000000, 9000100]   # range, held out, never trained on
  deterministic: true
  n_episodes_per_seed: 1
```

### D. Naming

"Windward" — the direction from which the wind blows; the hard direction to sail; the direction the agent must learn to reach. Short, available on GitHub, and it means something.

Alternatives: *Tack*, *Leeward*, *Beaufort*, *Regatta*, *Doldrums*.

---

*End of document.*
