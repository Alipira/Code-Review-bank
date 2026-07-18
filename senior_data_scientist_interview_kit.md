# Senior Data Scientist — Interviewer's Question Bank (120 Questions)

> **How to use this kit.** This is a *hiring manager's* bank, not a candidate crib sheet. Each section maps to a round in a modern (2026) senior loop: coding/debugging, SQL, ML depth, evaluation, stats & experimentation, deep learning, GenAI/LLM, system design, MLOps, and leadership. Questions marked **[LOOK FOR]** include what a strong senior answer covers vs. a red flag. You will *not* use all 120 — pick 8–12 per round and follow the candidate's answers with probes.
>
> **The senior filter.** At senior level you are not testing whether they *know* definitions. You are testing (1) diagnostic reasoning under a broken scenario, (2) whether they've shipped and operated systems (not "science projects"), (3) whether they connect the model to a business decision, and (4) judgment on trade-offs. Lead with "here's a broken thing, diagnose it," not "what is X."

---

## Section 1 — Warm-up & Experience Calibration (Q1–8)

1. Walk me through the most impactful model or data system you personally built. What was *your* specific contribution vs. the team's? **[LOOK FOR: crisp ownership boundary, business metric moved, not just "we improved accuracy." Red flag: can't separate self from team, no metric.]**
2. Tell me about a model you shipped that **did not work the way you expected** in production. What happened, and what did you do? **[LOOK FOR: a *specific* model, systematic diagnosis, honesty. Red flag: "everything went fine," or blames data without owning the framing.]**
3. What's a technical decision you made that you'd do differently today, with what you now know?
4. Describe a time the "right" statistical answer was not the answer the business wanted. How did you handle it?
5. How do you decide a problem does *not* need machine learning?
6. Walk me through how you'd spend your first 90 days if you joined this team.
7. What does "done" mean for a modeling project on your teams? Who signs off?
8. What's an area of ML you were confidently wrong about, and what changed your mind?

---

## Section 2 — Code Review & Debugging Challenges (Q9–13)

> Paste each snippet, screen-share style. Ask: *"This was written by a junior. Find correctness bugs and performance issues, then tell me how you'd fix them."* Score the **process**: do they check for leakage/edge cases/complexity systematically, or spot one thing and stop?

### Q9 — Vehicle-routing preprocess (the one you brought)
```python
def preprocess(self, parcel_weights: list[int], parcel_idxs: list[int]):
    removed_customers = []
    max_car_capacity = -1
    if np.where(self.car_availabilities == 1)[0].shape[0] > 0:
        should_continue = True
        initial_max = np.max(self.cars_capacities[np.where(self.car_availabilities == 1)[0]])
        while should_continue:
            to_remove = []
            for pi in range(len(parcel_weights)):
                p_weight = parcel_weights[pi]; p_idx = parcel_idxs[pi]
                max_car_capacity = np.max(self.cars_capacities[np.where(self.car_availabilities == 1)[0]])
                if p_weight > max_car_capacity * self.capacity_violate_threshold:
                    for i, c in enumerate(self.cars_capacities):
                        if c == max_car_capacity and self.car_availabilities[i]:
                            the_car = i; break
                    self.car_availabilities[the_car] = 0
                    removed_customers.append(p_idx)
                    self.assigned_cars.append(the_car)
                    _, d, t = self.compute_total_times([p_idx])
                    self.cluster_total_times.append(t); self.cluster_total_distances.append(d)
                    self.clusters.append([p_idx]); to_remove.append(pi)
                    if self.cars_capacities[np.where(self.car_availabilities == 1)[0]].shape[0] == 0:
                        max_car_capacity = -1; break
                    max_car_capacity = np.max(self.cars_capacities[np.where(self.car_availabilities == 1)[0]])
            to_remove.sort(); to_remove.reverse()
            for t_remove in to_remove:
                parcel_weights.pop(t_remove); parcel_idxs.pop(t_remove)
            if max_car_capacity == -1: should_continue = False
            elif initial_max != max_car_capacity: should_continue = True; initial_max = max_car_capacity
            else: should_continue = False
    return removed_customers
```
**Answer key:**
- **Correctness:** (a) `the_car` can be **unbound** → `NameError`. The inner car search uses **float equality** `c == max_car_capacity`; if capacities are floats, `np.max` should return an exact member, but any upstream float arithmetic makes this fragile — use `np.argmax` over the availability mask instead. (b) **Stale threshold within a pass:** when a car is consumed and `max_car_capacity` drops mid-loop, parcels *already checked earlier in the same pass* were compared against the old, higher capacity; they're only re-caught on the next `while` iteration. Correct but wasteful and easy to get wrong. (c) **Destructive side effect on inputs:** `parcel_weights.pop()` / `parcel_idxs.pop()` mutate the *caller's* lists — the docstring promises a return value, not that it empties your inputs. Operate on copies or return the survivors explicitly. (d) The two lists must stay index-aligned; the descending-pop pattern is the only reason it doesn't desync — this is a landmine for the next editor.
- **Performance:** `np.where(...)` and `np.max(...)` are recomputed **inside the inner loop** → repeated O(m) numpy scans per parcel, O(n·m) overall, plus the O(m) linear car search per removal and O(n) `list.pop` per removal → O(n²). Fix: compute the available mask once, maintain the max available capacity with a **max-heap / sorted structure** updated on removal, use `np.argmax(mask * capacities)` to find the car, and rebuild survivor lists with a single boolean mask instead of popping.
- **Senior tell:** do they notice the input mutation and the O(n²)? Do they ask *why* a threshold is recomputed every parcel rather than per pass?

### Q10 — Softmax
```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
```
**Answer key:** (1) **Numerical overflow** — no max-subtraction; large `x` → `inf/inf` → `nan`. Fix: `x = x - np.max(x, axis=-1, keepdims=True)`. (2) For **batched (2D) input**, `np.sum` reduces over *all* elements, not per row → wrong normalization. Fix: `axis=-1, keepdims=True`.

### Q11 — Feature engineering with leakage
```python
def add_features(df):
    means = df.groupby('category')['target'].mean()
    df['cat_te'] = df['category'].map(means)                     # target encoding
    df['x_norm'] = (df['x'] - df['x'].mean()) / df['x'].std()   # standardize
    df['roll7'] = df['sales'].rolling(7).mean()                 # 7-row rolling mean
    return df
```
**Answer key:** (1) **Target leakage** — target encoding uses each row's *own* target and the full dataset; must be out-of-fold (or leave-one-out) and fit on train only. (2) **Scaling leakage** — mean/std computed on full data leak test statistics; fit on train, transform test. (3) **Rolling leakage + ordering bug** — `rolling(7)` isn't sorted by date, isn't grouped by entity (mixes customers/SKUs), and includes the current row → use it as a *future-safe* lag: `sort_values('date').groupby(entity)['sales'].rolling(7).mean().shift(1)`. (4) Minor: `.std()` uses ddof=1 vs sklearn's ddof=0 — note for consistency.

### Q12 — PyTorch training loop
```python
for epoch in range(epochs):
    for xb, yb in loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
```
**Answer key:** (1) **Missing `optimizer.zero_grad()`** → gradients accumulate across steps (silent, catastrophic). (2) No `model.train()` / `model.eval()` toggle → dropout & BatchNorm behave wrong at eval. (3) Eval/inference needs `torch.no_grad()`. (4) If loss is accumulated for logging, use `.item()` / `.detach()` or you retain the graph → memory blowup.

### Q13 — AUC computed wrong
```python
from sklearn.metrics import roc_auc_score
preds = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
auc = roc_auc_score(y, preds)
```
**Answer key:** **AUC is computed on binarized 0/1 predictions, not scores.** ROC/PR curves need the *ranking* (probabilities/decision scores); binarizing collapses AUC toward balanced accuracy and throws away the threshold sweep. Fix: pass `predict_proba(X)[:, 1]` (or `decision_function`). *(This is one of the most common real bugs in production evaluation code.)*

---

## Section 3 — Python & Algorithms (Q14–23)

> Original prompts framed in DS/logistics terms. Score complexity awareness and whether they reach for the right structure (hash/heap/two-pointer/DP) before coding.

14. Given an array of daily sales, find the length of the **longest contiguous window** whose total stays under budget `B`. (Sliding window, O(n).)
15. Stream of model-inference latencies arriving one at a time — maintain a **running p95** efficiently. (Two heaps / t-digest; discuss exact vs approximate.)
16. Given `(customer_id, timestamp)` events, return the **first customer who repeats**. (Hash set, O(n); ordering matters.)
17. Parcel weights and a single vehicle capacity `C` — maximum-value subset that fits (**0/1 knapsack**). Then: what if `C` is huge but item count is small? (Meet-in-the-middle.)
18. Merge overlapping delivery time windows into the minimal set of intervals. (Sort + sweep, O(n log n).)
19. Find the **minimum vehicle capacity** such that all parcels can be delivered in ≤ `k` trips. (Binary search on the answer + greedy feasibility check.)
20. Deduplicate a 50 GB CSV of transactions that won't fit in memory. (External sort / hashing by chunk / `dask`; discuss the memory boundary explicitly.)
21. Given a co-purchase graph, count the **connected product communities**. (Union-Find or BFS/DFS; compare.)
22. Implement top-k frequent items in a stream with bounded memory. (Count-Min Sketch / Misra–Gries; when is exact infeasible?)
23. You have `O(n²)` pairwise-distance code that's too slow for 5M rows. Walk me through vectorizing it and then when you'd switch to **approximate nearest neighbors** (FAISS/HNSW) instead.

---

## Section 4 — SQL & Data Wrangling (Q24–31)

> Original schemas. Push on window functions and correctness under NULLs/ties.

24. Compute a **running 28-day revenue total** per customer, gaps included. (Window frame + date spine.)
25. For each product category, return the **2nd-highest-grossing** product. (`DENSE_RANK` / correlated subquery; handle ties.)
26. **Month-over-month growth %** per region, with months that have zero sales shown as rows. (LAG + calendar join.)
27. Find each customer's **longest streak of consecutive active days**. (Gaps-and-islands: `date - ROW_NUMBER()`.)
28. Deduplicate rows keeping the **latest** record per key. (`ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ts DESC)`.)
29. Build a **funnel**: of users who viewed, what % added to cart, then purchased, within 24h. (Self-joins / conditional aggregation; watch the time constraint.)
30. Why can `WHERE` vs `HAVING` vs a filter inside a `LEFT JOIN ... ON` give three different results? Give a concrete example.
31. A query with a `LEFT JOIN` is dropping rows you expected to keep. What are the top three causes? (Filtering the right table in `WHERE`, join key type mismatch, fan-out from a 1-to-many join.)

---

## Section 5 — ML Fundamentals & Modeling (Q32–46)

32. **[Scenario]** A fraud model has 94% accuracy and the fraud team is furious. Diagnose. **[LOOK FOR: immediately names class imbalance, pivots to precision/recall/PR-AUC and cost of FN vs FP *before* touching architecture. Red flag: starts debugging the model.]**
33. Explain the bias–variance trade-off *operationally* — how do you tell which one you have and what do you change? (Learning curves; more data vs simpler model vs regularization.)
34. L1 vs L2 regularization — mechanism, effect on weights, when you pick each. Follow-up: what does elastic net buy you?
35. Bagging vs boosting — the core statistical difference and when each wins. Then: how does XGBoost actually build a tree (gradient + Hessian, split gain)?
36. How do you handle severe class imbalance? Rank the options (class weights, threshold tuning, resampling, focal loss) and their failure modes. **[LOOK FOR: threshold tuning + cost-sensitive learning over blind SMOTE.]**
37. What is **data leakage**? Give three sneaky sources you've personally hit.
38. Feature selection: filter vs wrapper vs embedded methods — trade-offs, and why "just use feature importance" is a trap.
39. Multicollinearity — when does it actually matter, and when can you ignore it? (Inference vs prediction.)
40. Cross-validation for **time series** — why is standard k-fold wrong, and what do you do instead?
41. Design a **business-aligned loss function** for a demand-forecasting problem where under-stocking and over-stocking have asymmetric costs. (Newsvendor / quantile / pinball loss.)
42. You have 5,000 customers and want segments. VAE + K-Means vs GMM vs hierarchical vs plain K-Means — argue a choice and how you'd validate segments are *useful*, not just tight.
43. Own-price vs cross-price elasticity — how would you estimate them, and what confounds a naive regression of quantity on price? (Endogeneity; promotions; instruments/causal ML.)
44. When would you prefer a **calibrated logistic regression** over a higher-AUC gradient-boosted model in production?
45. Curse of dimensionality — concretely, what breaks, and what are your go-to mitigations?
46. Cold-start in a recommender — new user *and* new item. Walk me through both.

---

## Section 6 — Model Evaluation & Metrics (Q47–55)

47. When do you report **PR-AUC** instead of ROC-AUC, and why does ROC-AUC mislead on rare positives?
48. Your offline eval used a **50/50 balanced** test set but production prevalence is 2%. What breaks when you deploy, and how do you correct the reported metrics? **[LOOK FOR: precision/PPV collapses at true prevalence; recalibrate thresholds and re-estimate precision at production base rate.]**
49. What is **model calibration**, how do you measure it (reliability diagram, ECE, Brier), and how do you fix a miscalibrated model? (Platt / isotonic.)
50. A model's AUC is stable but the business KPI dropped. Where do you look? (Threshold, calibration drift, population shift, label delay, action layer.)
51. How do you choose an operating threshold when FN and FP have different dollar costs? (Cost curve / expected-value optimization.)
52. Why can accuracy, F1, and AUC all disagree about which of two models is "better"? Which do you trust and when?
53. How do you evaluate a **ranking** model (recommendations/search)? (NDCG, MAP, recall@k, and their offline-vs-online gap.)
54. What's wrong with tuning hyperparameters on your test set, and how does nested CV fix it?
55. How do you know your evaluation set itself isn't leaking or stale? (Temporal split, entity-disjoint split, drift on the eval set.)

---

## Section 7 — Statistics, Probability & Experimentation (Q56–68)

56. What does a **p-value** actually mean, and what does it *not* mean? (Not P(H0 true).)
57. Type I vs Type II error and statistical **power** — how do they trade off, and how do you size a test?
58. Design an A/B test for a new recommendation algorithm. Walk me through metric, MDE, sample size, duration, and randomization unit. **[LOOK FOR: primary + guardrail metrics, power calc, network/interference risk.]**
59. **Peeking** at an A/B test daily and stopping when significant — what's wrong, and what fixes it? (Inflated Type I; sequential testing / always-valid inference.)
60. What is **CUPED** and why would you use it? (Variance reduction using pre-experiment covariates.)
61. **Simpson's paradox** — give a real example and how you'd detect it before it embarrasses you.
62. Confidence interval vs credible interval — the actual interpretive difference.
63. MLE vs MAP — when does the prior matter, and what happens as data grows?
64. When is the **bootstrap** the right tool, and when does it fail? (Dependent data, extremes, small n.)
65. Correlation vs causation — how would you estimate a *causal* effect from observational data? (DAGs, backdoor adjustment, IV, diff-in-diff, matching.)
66. Your experiment shows a big lift week 1 that fades. Novelty effect vs real effect — how do you distinguish?
67. A stakeholder wants to launch on a result with p=0.06, n=80. What do you tell them, and what would change your mind?
68. Central limit theorem — state it precisely and name a case where it doesn't save you.

---

## Section 8 — Deep Learning (Q69–78)

69. Walk me through backprop intuitively, then: what causes **vanishing/exploding gradients** and your fixes? (Init, residuals, normalization, clipping.)
70. Why does **weight initialization** matter, and what's the difference between Xavier/Glorot and He init? When each?
71. **BatchNorm vs LayerNorm** — what each normalizes over, and why BatchNorm misbehaves with small batches / sequence models / at inference. **[LOOK FOR: train/eval running-stats gotcha.]**
72. Adam vs SGD-with-momentum — trade-offs, and why the "best test performance" answer is often SGD.
73. Dropout — what it does statistically, and why you must turn it off at inference.
74. Explain self-attention and why transformers replaced RNNs for sequence modeling.
75. Why use a **VAE** for customer segmentation over a plain autoencoder? What does the KL term buy you, and what does β control? (Reconstruction vs disentanglement trade-off; posterior collapse.)
76. Your deep model overfits a tabular dataset that XGBoost handles fine. Why might trees still beat nets on tabular data?
77. Learning-rate schedules — warmup, cosine decay — what problem does each solve?
78. How do you make a training run **reproducible**? (Seeds across libs *and* dataloader workers, deterministic ops, frozen preprocessing, pinned deps — note the CUDA nondeterminism caveats.)

---

## Section 9 — GenAI / LLM / RAG (Q79–91)

> Now standard in senior loops. The bar (2026): *orchestration, not training.* Reward concrete numbers ("512-token chunks, precision@5 = 0.83") over hand-waving.

79. **RAG vs fine-tuning vs both** — give me your *decision framework*, not definitions. **[LOOK FOR: fresh/changing knowledge & citations → RAG; behavior/style/format → fine-tune; cost, latency, data governance drive it. Red flag: "fine-tune for knowledge."]**
80. Design a production RAG pipeline end-to-end: ingestion → chunking → embedding → indexing → retrieval → rerank → generation → eval. Where does each stage most commonly fail? **[LOOK FOR: retrieval quality named as the #1 lever.]**
81. **Chunking strategy** — fixed vs semantic vs recursive; how do chunk size and overlap trade recall against precision and cost?
82. How do you pick an **embedding model**, and how do you know your embeddings have drifted or don't fit your domain?
83. **Hybrid retrieval** (BM25 + dense) — why combine lexical and semantic, and how do you fuse the scores? (RRF.)
84. What does a **reranker** do that first-stage retrieval can't, and what's the latency/quality trade?
85. How do you **evaluate** a RAG system? Give me a multi-layer strategy. **[LOOK FOR: automated (task metrics + LLM-as-judge on faithfulness/relevance/completeness) + NLI-based groundedness + periodic human eval. Red flag: "BLEU."]**
86. **Detect and mitigate hallucinations.** How? **[LOOK FOR: NLI entailment against retrieved context, self-consistency across samples, citation verification; mitigate via "answer only from context / say I don't know," lower temperature, better retrieval.]**
87. **Graph RAG** vs vanilla RAG — when does a knowledge graph earn its complexity?
88. LoRA/QLoRA/PEFT — what problem do they solve vs full fine-tuning, and what are the quality trade-offs?
89. Where do you put **guardrails** in an LLM app, and why as middleware rather than in the model? (Update independently of the model.)
90. Context leakage / prompt injection in a RAG app — concrete attack and your defenses. (Isolation, access control at retrieval, output filtering.)
91. How do you control **cost and latency** for an LLM feature at scale? (Caching, routing to smaller models, batching, retrieval budget, distillation.)

---

## Section 10 — ML System Design (Q92–99)

> Open-ended. Score the *framing*: do they nail down the objective, the label, the serving constraints, and the feedback loop before drawing boxes?

92. Design a **demand-forecasting** system for a confectionery distributor (thousands of SKUs, seasonality, promotions, cold-start SKUs). Objective, label, features, cadence, and how you'd measure business value.
93. Design a **B2B product-recommendation** system for wholesale buyers. What's different from B2C? (Basket size, repeat orders, business constraints, sparsity.)
94. Design a **fraud / anomaly-detection** pipeline where labels are scarce and adversaries adapt. (Semi/unsupervised, feedback loops, threshold management, human-in-the-loop.)
95. Design the **feature platform**: how do you guarantee the same feature values at train and serve time? **[LOOK FOR: online/offline parity, point-in-time correctness, frozen transforms — train/serve skew is the trap.]**
96. Design a **monitoring** system for models in production: what signals, what alerts, and who gets paged? (Data drift, concept drift, prediction drift, latency, business KPI.)
97. Design an **experimentation platform** so PMs can safely A/B-test model changes. (Assignment, guardrails, sequential stats, rollback.)
98. A batch model needs to become **real-time** (<50 ms p99). What has to change across features, model, and infra?
99. Design the **retraining** loop: how often, triggered by what, validated how, promoted how? (Champion/challenger, shadow, canary.)

---

## Section 11 — MLOps & Production Reliability (Q100–108)

100. **Data drift vs concept drift** — define both, how you detect each, and why they need *different* remedies. **[LOOK FOR: drift → retrain on recent data; concept drift → relabel/re-feature/re-frame.]**
101. A model that performed well for six months silently degrades. Give me your triage order. **[LOOK FOR: boring high-probability checks first — pipeline errors, schema changes, null spikes — before "retrain."]**
102. What do you version, and why *all* of it? (Code + data + model + features + config.)
103. Shadow deployment vs canary vs blue-green vs A/B for model rollout — when each?
104. How do you catch **train/serve skew** before users do?
105. What belongs in a **model card / governance doc** for a regulated use case (e.g., credit scoring)? (Intended use, data, metrics by segment, fairness, limitations.)
106. Your feature pipeline had a silent bug for two weeks. How do you detect it, and how do you prevent the next one? (Data contracts, expectations tests, freshness/volume/schema monitors.)
107. How do you decide **build vs buy** for a model (e.g., an LLM feature)?
108. A model is accurate but too slow/expensive to serve. Your optimization playbook? (Quantization, distillation, pruning, caching, batching, cheaper architecture.)

---

## Section 12 — Behavioral & Leadership (Senior/Staff) (Q109–120)

> STAR answers. At senior level, listen for *influence without authority*, *judgment under ambiguity*, and *impact tied to a business outcome* — not heroics.

109. Tell me about a time you **disagreed with a senior stakeholder** on methodology. How did it resolve? **[LOOK FOR: evidence-based persuasion, disagree-and-commit, no ego.]**
110. Describe leading a project with **ambiguous or shifting requirements**. How did you create clarity?
111. A time you had to say **"the data doesn't support this"** to someone who wanted a specific answer.
112. Explain a **complex model to a non-technical executive** and drive a decision. What analogy did you use?
113. How do you **prioritize** among competing data requests from three VPs, all "urgent"?
114. Tell me about **mentoring a struggling junior** — what did you do, and what changed?
115. A time you traded **technical debt for delivery speed** — how did you decide, and did it come back to bite you?
116. Describe a moment you raised an **ethical or fairness concern** about a model. What happened?
117. A project was going to **miss its deadline**. Walk me through how you handled it and communicated it.
118. Tell me about **building or shaping a data function** — hiring, leveling, standards. (Directly relevant if they've led a team.)
119. How do you give **hard technical feedback** in code review without demoralizing the author?
120. What's a strong opinion you hold about how data science *should* be done that not everyone shares?

---

## Scoring Rubric (per round)

| Signal | Weak (1–2) | Solid (3) | Strong senior (4–5) |
|---|---|---|---|
| **Diagnostic reasoning** | Jumps to a fix; spots one issue and stops | Systematic checklist | Prioritizes by probability & cost; asks clarifying Qs first |
| **Depth vs recall** | Textbook definitions | Explains mechanism | Connects theory → a concrete diagnostic they'd run |
| **Production sense** | "Just retrain it" | Knows drift/monitoring exist | Has operated systems; names the boring failure modes |
| **Business connection** | Optimizes a metric in a vacuum | Mentions impact | Frames every choice by the decision/dollars it affects |
| **Communication** | Rambling or jargon-walls | Clear | Structures the answer, quantifies, adapts to audience |
| **Trade-off judgment** | One "right" answer | Lists options | Argues a choice under stated constraints, notes what would change it |

**Universal red flags:** blames data/tooling without owning the framing; "we" with no "I"; can't name a single failure; reaches for SMOTE/`.fit()` reflexively; evaluates on the test set; says "fine-tune to add knowledge"; treats accuracy as the metric on imbalanced data; no notion of train/serve skew or leakage.

**Universal green flags:** asks what the decision is before modeling; names leakage and skew unprompted; quantifies ("precision@5 = 0.83", "p99 < 50 ms"); leads with trade-offs; distinguishes data drift from concept drift; has a reproducibility/versioning discipline; connects to business KPI.
