# AutoResearch for ChronosArt: SVD-Based Style Personalization

Autonomous ML Research Loop on a Single GPU, following Andrej Karpathy's 3-File Protocol.

---

## 🎯 The Research Mission

You are an autonomous AI research scientist optimizing **Training-Free Style-Personalization via SVD-Based Feature Decomposition** on Visual Autoregressive Models (Infinity-2B).

Your goal is simple: **Maximize `composite_score`** evaluated by the locked judge:
$$\text{composite\_score} = 40.0 \times S_{\text{style}} + 35.0 \times S_{\text{content}} + 25.0 \times D_{\text{diversity}}$$

- $S_{\text{style}}$: CLIP style image cosine fidelity ($\uparrow$)
- $S_{\text{content}}$: CLIP text prompt semantic alignment ($\uparrow$)
- $D_{\text{diversity}}$: Multi-seed micro-brushwork stochastic diversity ($\uparrow$)

---

## 📋 The Rules (Karpathy's Protocol)

| Rule | In Practice |
| :--- | :--- |
| **One file** | You **ONLY edit `auto_research/candidate.py`**. Everything inside is fair game: SVD singular value decay math, rank selection, scale schedules, SAC attention parameters, temperature curves, token jittering. |
| **Fixed budget** | Each evaluation takes **~2.5 - 3.5 minutes** on 1 GPU. If any run exceeds 7 minutes, kill and discard. (~15-20 experiments/hour). |
| **Locked judge** | `auto_research/evaluate.py` is **STRICTLY OFF-LIMITS**. You cannot alter benchmark cases, seeds, or scoring formulas. |
| **Simplicity** | A tiny score gain (+0.05) that adds 20 lines of hacky code $\to$ **DISCARD**. Equal score with simpler/deleted code $\to$ **KEEP**. |
| **No check-ins** | **NEVER STOP**. Never ask: *"Should I keep going?"* or *"Is this a good stopping point?"*. The human is asleep. You run continuously until manually interrupted. |

---

## 🚀 Setup (One-time at the start)

1. **Agree on a run tag**: Propose a tag based on date, e.g. `autoresearch/mar15` or `autoresearch/svd-opt`.
2. **Create the branch**:
   ```bash
   git checkout -b autoresearch/<tag>
   ```
3. **Read in-scope files**:
   - `auto_research/program.md` (this protocol)
   - `auto_research/candidate.py` (your editable canvas)
   - `auto_research/evaluate.py` (the locked judge)
   - `auto_research/results.tsv` (the experiment ledger)
4. **Establish Baseline**:
   Run the initial evaluation without changes to record the baseline:
   ```bash
   python3 auto_research/run_loop.py --baseline
   ```
5. Confirm setup is complete and enter the infinite loop.

---

## 🔄 The Autonomous Research Loop

**LOOP FOREVER:**

### 1. Inspect State & History
- Read the last 5 rows of `auto_research/results.tsv`.
- Note the current `best_score` and what has been tried.

### 2. Formulate 1 Precise Hypothesis
Think like a senior AI researcher. Consider:
- **Decay Functions:** Replace $e^{-\alpha(i-1)}$ with power-law $(1+\alpha i)^{-p}$, cosine decay $\cos(\frac{\pi i}{2r})$, or adaptive cumulative energy cutoff ($\ge 85\%$).
- **Scale-Adaptive Rank:** Coarse scales (k=1..3) need lower rank (structure), fine scales (k=4..7) need higher rank (texture).
- **SAC Injection Modulation:** Soft blending $\mathbf{Q} \leftarrow \lambda \mathbf{Q}^{\text{con}} + (1-\lambda) \mathbf{Q}^{\text{gen}}$ vs hard replacement.
- **Safe Stochastic Schedule:** Cosine annealing vs linear ramping for $\tau_k$, or adaptive token jittering rates (10% vs 20%).

### 3. Implement in `auto_research/candidate.py`
- Make clean, mathematically sound edits in `auto_research/candidate.py`.
- Do not touch any other file.

### 4. Git Commit
```bash
git commit -am "exp: <short hypothesis description>"
```

### 5. Run Fixed-Budget Evaluation
```bash
python3 auto_research/evaluate.py > auto_research/run.log 2>&1
```

### 6. Read Out the Result
```bash
grep "^composite_score:\|^status:\|^clip_style:\|^clip_content:\|^diversity:\|^peak_vram_mb:" auto_research/run.log
```

### 7. Decide (Keep or Discard)
- **If `status: SUCCESS` and `composite_score > best_score`:**
  - **KEEP!** The branch advances.
  - Append row to `auto_research/results.tsv`:
    ```
    <commit_hash>\t<score>\t<clip_style>\t<clip_content>\t<diversity>\t<mem_gb>\tkeep\t<description>
    ```
- **If `status: CRASH`:**
  - Check `tail -n 25 auto_research/run.log`.
  - If trivial typo, fix and retry once.
  - If fundamentally flawed or OOM: `git reset --hard HEAD~1`, log as `crash` in `results.tsv`.
- **If `composite_score <= best_score`:**
  - **DISCARD!**
  - Revert immediately:
    ```bash
    git reset --hard HEAD~1
    ```
  - Append row to `auto_research/results.tsv` with status `discard`.

### 8. Repeat Immediately
Do not wait. Do not ask for user input. Formulate the next hypothesis and iterate.
