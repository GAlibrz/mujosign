# Mujosign

**Mujosign** is a documentation-first (but prototype-friendly) project to discover and store physiologically plausible **muscle activation vectors** that realize a small library of **static hand gestures** (e.g., fist, thumbs-up, V-sign, pinch) in [MyoSuite](https://github.com/facebookresearch/myosuite) (MuJoCo).

---

## Project Goals

- Define gestures in a **deterministic, reproducible way** (JSON specs with joint ranges, tolerances, relations).
- Implement a scoring function that checks **pose accuracy, effort, stability, ROM/tendon safety**.
- Use search methods (IK → static opt, fallback synergies with CMA-ES) to find **muscle activation vectors**.
- Archive each gesture solution in a **library/** folder with provenance, scores, and thumbnails.

---

## Documentation-First but Prototype-Friendly

We use a **doc-first mindset** to ensure clarity and reproducibility, but balance it with **early prototyping** so we can adapt quickly.

### Must-have docs before coding
1. **SPEC_GESTURE_JSON.md** – how gestures are defined in JSON (targets, tolerances, relations).

2. **SPEC_SCORING.md** – deterministic scoring rules (pose error, ROM, effort, stability).

3. **ARCHITECTURE.md** – high-level system components and data flow.

### Flexible docs (can evolve with prototypes)
- **RUNBOOK.md** – refined as you learn how to run batches.

- **ROM tables, joint naming, solver configs** – updated once implementation details are clearer.

- **ROADMAP.md** – iterated as scope grows.

This way, the **core contract is frozen early**, but experimental parts can flex as we prototype.

---

## Repository Layout

```
mujosign/
├─ docs/          # Documentation-first source of truth
├─ gestures/      # Input gesture specs (JSON)
├─ library/       # Output artifacts (per gesture, generated later)
├─ scripts/       # Runner entrypoints (to be implemented)
├─ src/           # Core implementation (to be implemented)
└─ README.md
```

See [`docs/`](./docs/) for the must-have specs and architecture.


---

## Status

🚧 Work in progress.  
- Core docs authored: ✅ SPEC_GESTURE_JSON.md, SPEC_SCORING.md, ARCHITECTURE.md

- Gesture specs drafted: ✅ fist, thumbs_up, v_sign, pinch

- Scoring + solver code: 🔲 not implemented yet

- Batch runner & library generation: 🔲 not implemented yet


---

## How to Get Started

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/mujosign
   cd mujosign
   ```

2. **Review core docs** under `docs/` – they are the source of truth (gesture schema, scoring spec, architecture).  

3. **Inspect gestures** under `gestures/` – each gesture has a JSON definition.  

4. **Prototype**:  
   - Write a minimal loader for gesture JSONs (`src/mujosign/specs.py`).  
   - Implement a scoring stub (`src/mujosign/scoring.py`).  
   - Run `scripts/run_single.py` to load a gesture and print its contents.  

5. **Iterate**:  
   - Adjust tolerances and ROM tables as you learn from MyoSuite.  
   - Expand docs in `docs/` to reflect working reality.  

---

## License

TBD
