# Ideas for Code development

**Last Updated:** 2025-09-12

### 1. Use Q-completion (see e.g. Gumbel Alpa Zero paper) instead of 1-hot policy labels
 - Store Gumbel score / visit counts
 - Check wheher regular MCTS can do this.
 - Have a fallback for games without these scores?

### 2. Fast.ai tricks for model training
 - Try single-cycle very-high-learning rate 
 - AdamW optimizer

### 3. Try reducing the weight of the value head (it is noisier)
 - Can I do this "in flight", using an already trained model?
 - I think so but I need to adjust the optimizer?

### 4. Temperature normalization across models and algorithms
#### Background: Gumbel likes temperatures in the range 0.6-1.0, policy around 0.1, MCTS around 0.1. Different models will be more or less 'peaked'.
 - Approach 1: To comparae performance, findout the best temperature for the algorithm where at least k of n games are unique using self play.
 - Approach 2: Gumbel can't effectively use low temperatures using schemes I've tried. What about network output scaling (e.g. policy preds ^ (1/temp) & renormalize, accounting for P vs. Q disparity, e.g. by doing something similar to value preds)?

### 5. Techniacl debt:
 - There's tonnes of bloat: Duplicated code, verbose code, overly completionish code
 - Too many scripts and MDs in write_ups/

### 6. MCTS tree statistics
 - Return to code to provide detailed tree exploration guidance at low sims
 - Generate less detailed but statistics to describe the search tree at higher sims

### 12. Update on moves in winning games more?
 - Feature request rather than a bug: Do we want the policy head to update more on moves from the winning player?

