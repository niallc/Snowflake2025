# Gumbel AlphaZero Root Search — Detailed Algorithm and Parameter Roles

## 1. Overview

At the root of a Monte Carlo Tree Search (MCTS), the Gumbel AlphaZero algorithm combines
neural network priors with search-derived values to decide which move to play.

The neural network provides, for each legal move \( a \):
- **policy logits** \( z_a \) — unnormalized scores for how promising each move is,
- **a value prediction** \( v_\theta(s) \) — expected game outcome from the current position \( s \).

The root search combines these with Monte Carlo rollouts (simulations) to form
a final move selection rule.

---

## 2. Root ranking score

Each move receives a **score**:

\[
\text{score}(a) = g_a + \ell_a + \sigma (q_a - v_\pi)
\]

| Symbol | Meaning | Source |
|---------|----------|--------|
| \( g_a \) | Gumbel noise, one per root | used to approximate sampling from the policy distribution |
| \( \ell_a \) | Policy term = log p(a) | log-softmax of the raw logits |
| \( q_a \) | Completed Q-value | average of rollout returns for move a |
| \( v_\pi \) | Baseline value | policy-weighted average of Q-values |
| \( \sigma \) | Scale factor | tunes how much value influences ranking |

Moves are ranked by `score(a)` during sequential halving rounds, and the top move after
the final round is selected.

---

## 3. Policy term — log p(a)

The neural net outputs raw **logits** \( z_a \).  
We convert these to log probabilities using a *masked log-softmax*:

\[
\ell_a = \log p_a = z_a - \log \sum_{b \in \text{legal}} e^{z_b}.
\]

This:
- removes arbitrary additive shifts (softmax is shift-invariant),
- keeps the spread proportional to policy confidence,
- matches the theoretical basis of the Gumbel-max trick (`argmax(log p + g)`).

No further normalization is applied.

---

## 4. Value term — advantage form

Each move’s value is centered around a **baseline** \( v_\pi \):

\[
\text{advantage}(a) = q_a - v_\pi.
\]

This keeps the value term numerically stable and comparable across positions.
- \( q_a \): estimated value after exploring move a.
- \( v_\pi \): baseline, roughly the policy-weighted average of all q’s.

Unvisited moves have \( q_a \approx v_\pi \Rightarrow \text{advantage} \approx 0 \).

---

## 5. Sigma (\( \sigma \)) — policy–value trade-off

\( \sigma \) controls the weight of the value term relative to the policy term.

- Small \( \sigma \): policy dominates (trust the network’s prior).
- Large \( \sigma \): value dominates (trust search outcomes).

We keep \( \sigma \) **constant per root** (frozen for all halving rounds).

### Choosing \( \sigma \)

To balance the two signals:
1. Measure over representative positions:
   - \( \text{std}_\ell = \mathrm{std}(\ell_a) \)
   - \( \text{std}_q = \mathrm{std}(q_a - v_\pi) \)
2. Choose \( \sigma \) so that  
   \( \sigma \times \text{std}_q \approx \text{std}_\ell. \)

This ensures the value and policy terms contribute on similar scales.
Practically, we set `c_scale = σ` and tune it by sweeping over several values (e.g., 32–1024)
until the search performs robustly at the desired simulation budget.

---

## 6. Sequential halving procedure

1. Evaluate the network → get logits \( z_a \) and value \( v_\theta(s) \).
2. Mask illegal actions and compute log p.
3. Compute baseline \( v_\pi \).
4. Assign each legal move a fixed Gumbel noise \( g_a \).
5. Repeat:
   - rank candidates by \( g_a + \ell_a + \sigma(q_a - v_\pi) \),
   - keep the top half,
   - allocate simulations to those moves,
   - update their \( q_a \) estimates.
6. When one move remains, or after final round:
   - in evaluation mode, drop the noise term \( g_a = 0 \),
   - choose the move with the highest score.

---

## 7. Relationship to other constants

| Constant | Role |
|-----------|------|
| **c_puct** | inside-tree exploration weight between value and prior |
| **c_scale** | root-level scaling of value vs policy |
| **num_simulations** | affects how accurate Q-values become and thus the effective weight of the value term |
| **σ (sigma)** | same as `c_scale` when fixed per root |

They interact: larger `c_puct` or fewer simulations → noisier Q estimates → sometimes warrant higher σ; but in practice one constant σ works well once tuned.

---

## 8. Optional entropy-based guard rail

Optionally, modulate σ once per root using policy entropy:

\[
H = -\sum_a p_a \log p_a, \quad
H_{max} = \log K_{legal}, \quad
r = H / H_{max}.
\]

Then set:

\[
\sigma_{eff} = \sigma_0 (1 + \alpha r)
\]

with small \( \alpha \) (e.g. 0.5–2.0) and clamp within [0.25σ₀, 4σ₀].
This lets value contribute slightly more when the policy is uncertain.

---

## 9. Practical tuning cycle

1. Run a diagnostic script on 20–50 positions at your target number of simulations.
2. Log:
   ```
   std_logits, std_advantage, sigma * std_advantage
   ```
3. Adjust `c_scale` (σ) so that `sigma * std_advantage ≈ std_logits`.
4. Keep it fixed for training/evaluation.
5. Optionally re-tune every few training checkpoints if network behaviour drifts.

---

## 10. Intuitive summary

- **Policy term (log p)**: “where to look” — encodes network prior and confidence.
- **Value term (q − vπ)**: “what works” — adjusts based on actual outcomes.
- **σ**: “how much to listen” — sets the trust balance between policy and value.
- **Gumbel noise (g)**: adds diversity; ensures candidates explore proportional to policy probability.

When properly scaled, this blend reproduces the behaviour of AlphaZero-style search:
- policy dominates in clear, forced positions,
- value dominates in complex, uncertain ones,
- and both remain numerically stable across simulation budgets.
