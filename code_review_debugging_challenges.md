# Senior Data Scientist — Code Review & Debugging Bank (142 Challenges)

> **Companion to the main interview kit.** This expands *Section 2 (Code Review & Debugging)* into a standalone bank of **142 buggy snippets across 16 areas**. Each entry has a difficulty tag, the buggy code, and an answer key (**Bug** + **Fix**).
>
> **How to run one.** Paste the snippet, screen-share style: *"A junior wrote this. It runs. Find the correctness bugs and performance problems, then fix the hot path."* Score the **hunt**, not the checklist: do they systematically check for leakage, edge cases, dtype, complexity, and train/serve parity — or spot one thing and stop? The best candidates ask *why* the code is shaped this way before rewriting.
>
> **Difficulty:** `E` easy (screen), `M` medium (core round), `H` hard (senior signal).
>
> **Areas:** A. NumPy & numerical stability · B. Pandas & wrangling · C. Python core & complexity · D. Leakage & preprocessing · E. scikit-learn & pipelines · F. Cross-validation & splitting · G. Evaluation & metrics · H. PyTorch & deep learning · I. Statistics & probability · J. Time series · K. NLP / embeddings / LLM / RAG · L. Recommenders · M. Spark / PySpark · N. MLOps & serving · O. SQL · P. Concurrency, perf & memory.

---

## A — NumPy, Vectorization & Numerical Stability (1–10)

**1. In-place normalize mutates caller & breaks int arrays** · `E`
```python
def normalize(x):
    x -= x.mean()
    x /= x.std()
    return x
```
**Bug:** `-=` / `/=` modify the caller's array in place (aliasing side effect); on an int dtype, `/=` truncates or raises.
**Fix:** `return (x - x.mean()) / x.std()`; ensure float input.

**2. Broadcasting the wrong axis** · `M`
```python
row_centered = X - X.mean(axis=1)   # X is (n, d); mean is (n,)
```
**Bug:** `(n,d) - (n,)` broadcasts against the last axis (or errors) — you did *not* subtract each row's mean.
**Fix:** `X - X.mean(axis=1, keepdims=True)` → shape `(n, 1)`.

**3. Float equality mask** · `E`
```python
mask = (predictions == 0.1)
```
**Bug:** `0.1` isn't exactly representable; equality silently misses matches.
**Fix:** `np.isclose(predictions, 0.1)`.

**4. Integer overflow in a sum of squares** · `M`
```python
a = np.array([100_000, 200_000], dtype=np.int32)
energy = (a * a).sum()          # overflows int32
```
**Bug:** `a*a` overflows int32 and wraps to garbage silently.
**Fix:** cast up: `a.astype(np.int64)` or `float64`.

**5. Reseeding the global RNG every call** · `M`
```python
def sample():
    np.random.seed(42)
    return np.random.randn(10)
```
**Bug:** reseeding to a constant returns *identical* "random" draws each call and mutates global RNG state for everyone else.
**Fix:** build one `rng = np.random.default_rng(seed)` and reuse; never reseed per call.

**6. Mean of ratios instead of ratio of means** · `H`
```python
ctr = (clicks / impressions).mean()
```
**Bug:** averaging per-row CTR overweights low-impression rows; the true aggregate CTR is `clicks.sum()/impressions.sum()`. Also divide-by-zero when `impressions == 0`.
**Fix:** pool numerator/denominator; guard zeros.

**7. argmax over a flattened array** · `E`
```python
pred_class = np.argmax(logits)      # logits is (batch, n_classes)
```
**Bug:** no axis → argmax over the *entire* flattened array; returns one scalar.
**Fix:** `np.argmax(logits, axis=1)`.

**8. Growing an array with np.append in a loop** · `M`
```python
result = np.array([])
for row in data:
    result = np.append(result, transform(row))
```
**Bug:** `np.append` reallocates the whole array each iteration → O(n²); plus a Python loop.
**Fix:** append to a list then `np.array(...)`, or vectorize.

**9. Silent NaN in an aggregate** · `E`
```python
score = np.mean(values)
```
**Bug:** a single NaN makes the mean NaN silently; the real question is *why* there are NaNs.
**Fix:** `np.nanmean` (after understanding the NaNs), or handle explicitly.

**10. Explicit matrix inverse for least squares** · `H`
```python
beta = np.linalg.inv(X.T @ X) @ X.T @ y
```
**Bug:** forming the inverse is numerically unstable and slow; blows up under collinearity (near-singular `XᵀX`).
**Fix:** `np.linalg.solve(X.T @ X, X.T @ y)` or `lstsq`/QR; add ridge for collinearity.

---

## B — Pandas & Data Wrangling (11–22)

**11. Chained assignment silently no-ops** · `M`
```python
df[df['age'] > 30]['income'] = 0
```
**Bug:** chained indexing writes to a temporary copy; the original `df` is unchanged (SettingWithCopy).
**Fix:** `df.loc[df['age'] > 30, 'income'] = 0`.

**12. `inplace=True` returns None** · `E`
```python
df = df.dropna(inplace=True)
```
**Bug:** `inplace=True` returns `None`, so `df` becomes `None`.
**Fix:** either `df.dropna(inplace=True)` *or* `df = df.dropna()` — not both.

**13. `iterrows` for a vectorizable op** · `M`
```python
for i, row in df.iterrows():
    df.at[i, 'total'] = row['a'] + row['b']
```
**Bug:** `iterrows` boxes each row into a Series (slow, dtype-coercing) → orders of magnitude slower.
**Fix:** `df['total'] = df['a'] + df['b']`.

**14. Merge fan-out explodes rows** · `H`
```python
result = orders.merge(customers, on='customer_id')
```
**Bug:** if `customers.customer_id` isn't unique, the merge multiplies order rows (row explosion), silently corrupting downstream sums.
**Fix:** enforce uniqueness or `merge(..., validate='many_to_one')`.

**15. `groupby` drops NaN keys** · `M`
```python
counts = df.groupby('category').size()
```
**Bug:** rows with a NaN `category` are dropped by default → undercount.
**Fix:** `df.groupby('category', dropna=False)`.

**16. NaN silently upcasts ints to float** · `M`
```python
df['count'] = df['count'].fillna(0)   # column already float64 because of earlier NaN
```
**Bug:** an earlier NaN forced the column to float, so ids/counts are now `1.0`, breaking joins/exports.
**Fix:** use nullable `Int64`, or cast back to int after filling.

**17. Arithmetic aligns on index → NaNs** · `M`
```python
a = df1['x'].reset_index(drop=True)
combined = a + df2['y']              # df2['y'] index NOT reset
```
**Bug:** pandas aligns on index before adding; mismatched indices produce NaNs.
**Fix:** align indices or use `.values`.

**18. `apply` without `axis=1`** · `E`
```python
df['z'] = df.apply(lambda r: r['a'] * 2)
```
**Bug:** default `axis=0` applies over columns; `r['a']` misbehaves.
**Fix:** add `axis=1` — or better, vectorize `df['a'] * 2`.

**19. Ambiguous datetime parsing** · `M`
```python
df['date'] = pd.to_datetime(df['date'])
```
**Bug:** mixed/locale-ambiguous strings (`01/02/2024`) parse inconsistently and slowly without an explicit format.
**Fix:** `format=...`, set `dayfirst`, `errors='coerce'`, then check nulls.

**20. Feature function mutates the caller's frame** · `M`
```python
def add(df):
    df['new'] = df['x'] ** 2
    return df
train_feat = add(train)             # `train` is now mutated too
```
**Bug:** in-place column write leaks back into the original DataFrame.
**Fix:** `df = df.copy()` at the top.

**21. `concat` with misaligned columns** · `M`
```python
big = pd.concat([df1, df2])
```
**Bug:** extra/misordered columns create NaN blocks; duplicate index values break later `.loc`.
**Fix:** align columns, `ignore_index=True`, validate schema.

**22. Reading a giant CSV all at once** · `M`
```python
df = pd.read_csv('50gb.csv')
```
**Bug:** loads everything into memory and infers object dtypes that balloon usage.
**Fix:** pass `dtype=`, `usecols=`, `chunksize=`, or use Parquet/Dask.

---

## C — Python Core & Complexity (23–32)

**23. Mutable default argument** · `M`
```python
def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket
```
**Bug:** the default list is created once and shared across all calls → state leaks between calls.
**Fix:** `bucket=None`, then `bucket = [] if bucket is None else bucket`.

**24. Mutating a list while iterating it** · `M`
```python
for x in items:
    if x < 0:
        items.remove(x)
```
**Bug:** removing during iteration shifts indices and skips elements.
**Fix:** `items = [x for x in items if x >= 0]`.

**25. Late-binding closure in a loop** · `H`
```python
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])         # [2, 2, 2], not [0, 1, 2]
```
**Bug:** closures capture the variable, not its value; all see the final `i`.
**Fix:** bind per-iteration: `lambda i=i: i`.

**26. `is` for value comparison** · `E`
```python
if x is 1000:
    ...
```
**Bug:** `is` tests identity; only works for small cached ints by accident, fails for large/most values.
**Fix:** `if x == 1000:`.

**27. O(n²) membership via list** · `M`
```python
seen = []
for x in data:
    if x not in seen:
        seen.append(x)
```
**Bug:** `x not in list` is O(n) → O(n²) overall.
**Fix:** use a `set` for membership.

**28. Naive float summation error** · `M`
```python
total = 0.0
for v in values:                    # millions of small floats
    total += v
```
**Bug:** sequential summation accumulates floating-point error.
**Fix:** `math.fsum(values)` or `np.sum` (pairwise/Kahan).

**29. Exponential recursion, no memoization** · `M`
```python
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```
**Bug:** exponential time; deep `n` also overflows the stack.
**Fix:** memoize (`functools.lru_cache`) or iterate.

**30. Shallow copy of nested lists** · `M`
```python
grid2 = list(grid)                  # grid is a list of lists
grid2[0][0] = 9                     # also mutates grid
```
**Bug:** shallow copy shares the inner lists.
**Fix:** `copy.deepcopy(grid)`.

**31. Bare `except: pass`** · `M`
```python
try:
    risky()
except:
    pass
```
**Bug:** swallows *everything* (including `KeyboardInterrupt`/`SystemExit`) and hides real failures.
**Fix:** catch specific exceptions, log, re-raise as appropriate.

**32. Floor division on a ratio** · `E`
```python
accuracy = correct // total
```
**Bug:** integer floor division → `0` whenever `correct < total`.
**Fix:** `correct / total`.

---

## D — Data Leakage & Preprocessing (33–42)

**33. Scaling before the split** · `H`
```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```
**Bug:** the scaler is fit on *all* rows, leaking test mean/variance into training.
**Fix:** split first; `fit` on train, `transform` test (ideally inside a `Pipeline`).

**34. Global-mean imputation before the split** · `H`
```python
X = X.fillna(X.mean())
X_train, X_test = train_test_split(X)
```
**Bug:** the imputation mean includes test rows → leakage.
**Fix:** fit the imputer on train only.

**35. Target encoding uses the row's own label** · `H`
```python
df['cat_te'] = df.groupby('cat')['y'].transform('mean')
```
**Bug:** each encoded value includes its own target and the full dataset → strong leakage, overfit.
**Fix:** out-of-fold / leave-one-out target encoding with smoothing, fit on train only.

**36. A feature computed from the target** · `H`
```python
df['revenue_per_unit'] = df['revenue'] / df['units']   # revenue IS the label
```
**Bug:** the feature encodes the target directly → perfect train metrics, useless in production.
**Fix:** drop any feature derived from the label.

**37. SMOTE before cross-validation** · `H`
```python
X_res, y_res = SMOTE().fit_resample(X, y)
scores = cross_val_score(model, X_res, y_res)
```
**Bug:** synthetic points generated from *all* data leak neighbors of test rows into training folds → optimistic scores.
**Fix:** resample *inside* each CV fold (imblearn `Pipeline`).

**38. Feature selection on the full dataset** · `H`
```python
X_sel = SelectKBest(k=20).fit_transform(X, y)   # then split
```
**Bug:** selection uses test rows' target signal → leakage and overfit feature set.
**Fix:** select inside a pipeline, train-only per fold.

**39. Rolling feature that peeks at the present/future** · `H`
```python
df['roll_mean'] = df['y'].rolling(7).mean()
```
**Bug:** includes the current row, is unsorted, and mixes entities; without `.shift(1)` it leaks the label.
**Fix:** `sort_values('date').groupby(entity)['y'].rolling(7).mean().shift(1)`.

**40. Same entity in both train and test** · `H`
```python
train, test = train_test_split(df)    # many rows per user land on both sides
```
**Bug:** group leakage — the model memorizes users, inflating metrics.
**Fix:** `GroupShuffleSplit`/`GroupKFold` on the entity id.

**41. Duplicates split across train/test** · `M`
```python
train, test = train_test_split(df)    # exact/near-duplicate rows exist
```
**Bug:** identical rows in both splits leak labels.
**Fix:** dedup (and check near-dupes) *before* splitting.

**42. One-hot encoding fit on all data** · `M`
```python
df = pd.get_dummies(df)               # then split
```
**Bug:** leaks category distribution and hides the unseen-category problem at serve time.
**Fix:** fit the encoder on train with `handle_unknown='ignore'`.

---

## E — scikit-learn & Pipelines (43–52)

**43. Fitting the scaler on the test set** · `M`
```python
scaler.fit(X_test)
X_test = scaler.transform(X_test)
```
**Bug:** transforms are fit on test data → leakage.
**Fix:** `fit` on train, `transform` test.

**44. AUC from hard labels** · `M`
```python
roc_auc_score(y_test, model.predict(X_test))
```
**Bug:** `predict` returns 0/1; AUC needs the ranking (scores/probabilities), so it collapses toward accuracy.
**Fix:** `roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])`.

**45. Ignoring class weight on imbalance** · `M`
```python
LogisticRegression().fit(X, y)        # y is ~99:1
```
**Bug:** default weighting predicts the majority class.
**Fix:** `class_weight='balanced'`, plus threshold tuning / resampling.

**46. No `random_state`** · `E`
```python
RandomForestClassifier().fit(X, y)
```
**Bug:** nondeterministic results → irreproducible experiments.
**Fix:** set `random_state=`.

**47. GridSearch scoring on accuracy for imbalance** · `M`
```python
GridSearchCV(clf, params, scoring='accuracy')   # imbalanced target
```
**Bug:** optimizes a metric that rewards predicting the majority.
**Fix:** `scoring='average_precision'` / `'f1'` / `'roc_auc'`.

**48. `LabelEncoder` on features** · `M`
```python
X['cat'] = LabelEncoder().fit_transform(X['cat'])
```
**Bug:** `LabelEncoder` is for *targets*; it imposes a fake ordinal order on nominal categories and can't handle unseen categories at serve.
**Fix:** `OrdinalEncoder`/`OneHotEncoder` with `handle_unknown`.

**49. Printing negated MSE as MSE** · `E`
```python
score = cross_val_score(reg, X, y, scoring='neg_mean_squared_error').mean()
print(f"MSE: {score}")               # prints a negative number
```
**Bug:** sklearn returns *negative* MSE (higher is better); reported value is negative.
**Fix:** report `-score`.

**50. Classification split without `stratify`** · `M`
```python
train_test_split(X, y)               # imbalanced classes
```
**Bug:** class ratios differ across splits; the rare class may be absent from the test set.
**Fix:** `stratify=y`.

**51. Forgetting the transform at inference** · `H`
```python
X_train = scaler.fit_transform(X_train)
model.fit(X_train, y_train)
preds = model.predict(X_new)          # X_new never scaled
```
**Bug:** train/serve skew — inference features are on a different scale.
**Fix:** bundle scaler + model in a `Pipeline` so serving replays the same steps.

**52. Calling `fit` on the whole dataset then "evaluating"** · `M`
```python
model.fit(X, y)
print(accuracy_score(y, model.predict(X)))
```
**Bug:** reports training accuracy — no generalization estimate.
**Fix:** evaluate on a held-out set / CV.

---

## F — Cross-Validation & Splitting (53–60)

**53. k-fold on time series** · `H`
```python
cross_val_score(model, X, y, cv=5)    # temporal data
```
**Bug:** shuffled folds train on the future to predict the past.
**Fix:** `TimeSeriesSplit`.

**54. Selecting hyperparameters on the test set** · `H`
```python
for p in params:
    model.set_params(**p).fit(X_train, y_train)
    best = max(best, model.score(X_test, y_test))
```
**Bug:** tuning against the test set makes it no longer an unbiased estimate.
**Fix:** use a validation set / nested CV.

**55. Non-grouped CV on grouped data** · `H`
```python
cross_val_score(model, X, y, cv=KFold(5))   # multiple rows per user
```
**Bug:** the same entity appears across folds → leakage.
**Fix:** `GroupKFold` on the entity id.

**56. KFold instead of StratifiedKFold on imbalance** · `M`
```python
KFold(n_splits=5, shuffle=True)       # imbalanced classification
```
**Bug:** folds get uneven class ratios; rare class may vanish from a fold.
**Fix:** `StratifiedKFold`.

**57. Default `shuffle=False` on ordered data** · `M`
```python
KFold(n_splits=5)                     # data sorted by label
```
**Bug:** folds become contiguous blocks (e.g., all class A), wrecking the estimate.
**Fix:** `shuffle=True` (unless a deliberate temporal split).

**58. Best-of-many CV score reported as the estimate** · `H`
```python
# try 50 model configs, report the best CV score
```
**Bug:** the maximum over many trials is optimistically biased (multiple-comparisons / snooping).
**Fix:** nested CV, and keep a final untouched test set.

**59. Test set too small to conclude anything** · `M`
```python
train_test_split(X, y, test_size=0.02)
```
**Bug:** a tiny test set gives a high-variance metric; conclusions are noise.
**Fix:** adequate test size or repeated CV.

**60. Feature window straddles the CV boundary** · `H`
```python
# time-based split, but a 7-day feature window overlaps the train/test cut
```
**Bug:** information bleeds across the split boundary (look-ahead at the seam).
**Fix:** purge/embargo around the boundary (purged CV).

---

## G — Evaluation & Metrics (61–70)

**61. Multiclass F1 with the wrong average** · `M`
```python
f1_score(y_true, y_pred)              # multiclass, default average='binary'
```
**Bug:** errors or answers the wrong question; macro/micro/weighted each mean something different.
**Fix:** choose `average=` deliberately (macro for minority-sensitive).

**62. Reporting accuracy on imbalance as success** · `E`
```python
print("Accuracy:", accuracy_score(y, preds))   # 99%, predicts all-majority
```
**Bug:** accuracy is meaningless when the base rate is extreme.
**Fix:** PR-AUC, recall, confusion matrix at the operating point.

**63. Hardcoded 0.5 threshold** · `M`
```python
preds = (proba > 0.5).astype(int)
```
**Bug:** 0.5 is arbitrary; wrong for imbalance or asymmetric costs.
**Fix:** tune the threshold on validation to the business cost curve.

**64. MAPE with zeros in the denominator** · `M`
```python
mape = np.mean(np.abs((y - yhat) / y))
```
**Bug:** divide-by-zero when `y == 0`; explodes for small `y`; asymmetric penalty.
**Fix:** WAPE/sMAPE, or mask zeros.

**65. RMSE on a log target, reported in "units"** · `H`
```python
model.fit(X, np.log1p(y))
pred = model.predict(X)
rmse = mean_squared_error(np.log1p(y), pred, squared=False)   # not original units
```
**Bug:** the error is in log space but interpreted as original units.
**Fix:** `np.expm1` predictions and targets *before* computing the metric.

**66. Micro-average hides the minority class** · `M`
```python
f1_score(y, p, average='micro')       # imbalanced multiclass
```
**Bug:** micro is dominated by the majority; minority failure is invisible.
**Fix:** macro / per-class breakdown.

**67. Wrong positive label** · `M`
```python
precision_score(y, p, pos_label=0)    # the intended positive was 1
```
**Bug:** measures precision for the wrong class.
**Fix:** set the correct `pos_label`.

**68. Pointwise metric for a ranking task** · `H`
```python
accuracy_score(relevant, predicted_relevant)   # a recommender
```
**Bug:** accuracy ignores order/ranking quality.
**Fix:** NDCG@k, MAP, recall@k.

**69. Comparing models on different folds/seeds** · `M`
```python
# model A scored with seed=1, model B with seed=2
```
**Bug:** the comparison isn't apples-to-apples.
**Fix:** identical folds/seed; paired statistical comparison.

**70. Negative R² read as "small but positive"** · `M`
```python
r2 = r2_score(y_true, y_pred)         # returns -0.3
```
**Bug:** R² can be negative (worse than predicting the mean); a negative value means the model is actively bad, not "weak."
**Fix:** interpret the sign correctly; pair with an error metric in units.

---

## H — PyTorch & Deep Learning (71–82)

**71. No `model.eval()` at validation** · `M`
```python
def validate(model, loader):
    for x, y in loader:
        out = model(x)                # dropout & BatchNorm still in train mode
```
**Bug:** dropout stays on and BatchNorm uses batch stats → wrong validation numbers.
**Fix:** `model.eval()` + wrap in `torch.no_grad()`.

**72. Accumulating loss tensors (graph retained → OOM)** · `H`
```python
total_loss = 0
for x, y in loader:
    loss = criterion(model(x), y)
    total_loss += loss               # keeps the autograd graph alive
```
**Bug:** summing live tensors retains every graph → memory blows up.
**Fix:** `total_loss += loss.item()`.

**73. `nll_loss` fed raw logits** · `M`
```python
out = model(x)                        # raw logits
loss = F.nll_loss(out, y)
```
**Bug:** `nll_loss` expects log-probabilities, not logits → wrong loss.
**Fix:** `F.cross_entropy(out, y)` (log_softmax + nll) or apply `log_softmax` first.

**74. Softmax then CrossEntropyLoss** · `M`
```python
out = torch.softmax(model(x), dim=1)
loss = nn.CrossEntropyLoss()(out, y)
```
**Bug:** `CrossEntropyLoss` applies log-softmax internally; feeding probabilities double-applies it and degrades training.
**Fix:** pass raw logits.

**75. Tensors on the wrong device** · `E`
```python
model.cuda()
for x, y in loader:
    out = model(x)                    # x still on CPU
```
**Bug:** device mismatch → runtime error (or silent CPU fallback).
**Fix:** `x = x.to(device); y = y.to(device)`.

**76. BatchNorm with tiny/size-1 batches** · `H`
```python
model.train()
# batch_size = 1 through a BatchNorm network
```
**Bug:** BN can't estimate variance from one sample; and using `train()` at inference uses unstable batch stats.
**Fix:** larger batches, `eval()` at inference, or LayerNorm/GroupNorm.

**77. RNN hidden state not detached (unbounded BPTT)** · `H`
```python
for x_t in seq:
    h = rnn(x_t, h)                   # graph grows across the whole sequence
```
**Bug:** the graph accumulates over all timesteps → OOM and vanishing/exploding gradients.
**Fix:** truncated BPTT: `h = h.detach()` between chunks.

**78. Shuffling val, not shuffling train** · `M`
```python
train_loader = DataLoader(train, shuffle=False)
val_loader   = DataLoader(val,   shuffle=True)
```
**Bug:** unshuffled training introduces order bias; shuffling validation is pointless.
**Fix:** shuffle train, keep val ordered.

**79. Divergent learning rate → NaN loss** · `E`
```python
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
```
**Bug:** an absurd LR makes the loss diverge to NaN.
**Fix:** sane LR, warmup, and gradient clipping.

**80. Optimizer bound to a stale model** · `H`
```python
optimizer = torch.optim.Adam(model.parameters())
model = MyNet()                       # re-created AFTER the optimizer
```
**Bug:** the optimizer still references the old parameters; the new model never updates.
**Fix:** build the optimizer *after* the final model, or re-bind its param groups.

**81. Loss counts padding tokens** · `H`
```python
loss = criterion(logits.view(-1, V), targets.view(-1))   # includes PAD positions
```
**Bug:** padding positions contribute to the loss → skewed gradients on variable-length sequences.
**Fix:** `criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)` or mask.

**82. No seeding / DataLoader workers reseed** · `M`
```python
# no torch.manual_seed, no numpy/random seed, no worker_init_fn
```
**Bug:** irreproducible runs; multi-worker DataLoaders reseed independently.
**Fix:** seed torch/numpy/random, set `worker_init_fn`, and `torch.use_deterministic_algorithms(True)` (mind the perf trade-off).

---

## I — Statistics & Probability (83–90)

**83. Population std for a sample** · `M`
```python
sample_std = np.std(sample)           # ddof=0 by default
```
**Bug:** NumPy's default `ddof=0` gives the biased (population) std for a sample.
**Fix:** `np.std(sample, ddof=1)`.

**84. Uncorrected multiple comparisons** · `H`
```python
for feature in features:              # 100 features
    if ttest(feature, target).pvalue < 0.05:
        significant.append(feature)
```
**Bug:** at α=0.05 you expect ~5 false positives from 100 tests.
**Fix:** Bonferroni or Benjamini–Hochberg (FDR).

**85. Treating a p-value as an effect size** · `M`
```python
if p1 < p2:
    feature1_is_more_important = True
```
**Bug:** p-value magnitude ≠ effect size or importance (it's confounded with n).
**Fix:** report effect sizes with confidence intervals.

**86. Pearson correlation to declare "no relationship"** · `M`
```python
corr = np.corrcoef(x, y)[0, 1]        # ≈ 0 for a U-shaped relation
```
**Bug:** Pearson only captures *linear* association; a strong nonmonotonic relationship reads as ~0.
**Fix:** Spearman / mutual information / actually plot it.

**87. Bootstrap without replacement** · `H`
```python
boot = np.random.choice(data, size=len(data), replace=False)
```
**Bug:** `replace=False` just permutes the data → every "resample" is identical → zero variance.
**Fix:** `replace=True`.

**88. Peeking at an A/B test and stopping early** · `H`
```python
if ttest(a, b).pvalue < 0.05:
    stop_and_ship()
```
**Bug:** repeated peeking inflates the Type I error rate far above 5%.
**Fix:** fixed sample horizon or a sequential/always-valid test.

**89. t-test on tiny, skewed samples** · `M`
```python
ttest_ind(a, b)                       # n = 8 each, heavy skew
```
**Bug:** the t-test's normality assumption is violated at tiny n with skew.
**Fix:** Mann–Whitney U or a bootstrap.

**90. Pooling groups hides Simpson's paradox** · `H`
```python
overall_rate = df['success'].mean()   # ignores a confounding group variable
```
**Bug:** the aggregate can reverse the within-group trend.
**Fix:** stratify by the confounder / adjust.

---

## J — Time Series (91–98)

**91. Random split on temporal data** · `H`
```python
train_test_split(ts_df)
```
**Bug:** shuffling time lets the model see the future → leakage.
**Fix:** chronological split.

**92. Standardizing with whole-series stats** · `H`
```python
ts['z'] = (ts['y'] - ts['y'].mean()) / ts['y'].std()
```
**Bug:** uses future values' mean/std → leakage.
**Fix:** expanding/rolling stats, or fit on the train window only.

**93. "Lag" feature that isn't lagged** · `M`
```python
ts['lag_target'] = ts['y']            # named a lag, but not shifted
```
**Bug:** uses the current value → direct target leakage.
**Fix:** `ts['y'].shift(1)`.

**94. Rolling over an unsorted, multi-series frame** · `H`
```python
ts['roll'] = ts['y'].rolling(7).mean()   # rows unsorted, several series stacked
```
**Bug:** unordered windows + contamination across different series.
**Fix:** sort by time, `groupby(series_id)` before rolling, then `shift`.

**95. Forecasting on differences, forgetting to invert** · `H`
```python
model.fit(diff_series)
forecast = model.predict(...)         # reported as a level
```
**Bug:** the forecast is in differenced space; reporting it as a level is wrong.
**Fix:** cumulatively invert the differencing.

**96. Backfilling leaks the future** · `M`
```python
ts['y'] = ts['y'].fillna(method='bfill')
```
**Bug:** `bfill` fills past gaps with *future* values → leakage.
**Fix:** `ffill` (or model-based imputation), and handle edge rows.

**97. Resample window overlaps the label** · `H`
```python
weekly = ts.resample('W').mean()      # features and target share the same week
```
**Bug:** the aggregation window overlaps the prediction target → leakage.
**Fix:** define feature and label windows with a gap between them.

**98. CV horizon shorter than the seasonal cycle** · `M`
```python
TimeSeriesSplit(n_splits=5)           # tiny test horizon vs yearly seasonality
```
**Bug:** a test horizon shorter than the seasonal period can't capture seasonality → misleading CV.
**Fix:** horizon ≥ one seasonal cycle; blocked/seasonal CV.

---

## K — NLP / Embeddings / LLM / RAG (99–108)

**99. TF-IDF fit on the full corpus** · `H`
```python
X = TfidfVectorizer().fit_transform(all_texts)   # then split
```
**Bug:** vocabulary and IDF weights are learned from test documents → leakage.
**Fix:** fit the vectorizer on train only.

**100. Dot product mistaken for cosine similarity** · `M`
```python
sim = a @ b                           # a, b are unnormalized embeddings
```
**Bug:** dot product ≠ cosine unless the vectors are unit-normalized; magnitude dominates ranking.
**Fix:** normalize first, or use a cosine function.

**101. Silent truncation dropping context** · `H`
```python
inputs = tokenizer(text, truncation=True, max_length=128)   # long docs
```
**Bug:** long documents are silently cut, losing the content you need for retrieval/classification.
**Fix:** chunk the document (or raise max length) and *log* when truncation happens.

**102. Query and documents embedded by different models** · `H`
```python
doc_emb   = model_A.encode(docs)
query_emb = model_B.encode(query)     # different embedding space
```
**Bug:** query and corpus live in different vector spaces → retrieval is garbage.
**Fix:** use the identical model (and version) for both.

**103. Fixed-character chunking with no overlap** · `M`
```python
chunks = [text[i:i+500] for i in range(0, len(text), 500)]
```
**Bug:** character windows split words/sentences and have no overlap → broken, context-less retrieval units.
**Fix:** token-aware chunking on sentence/semantic boundaries with overlap.

**104. Context assembled without dedup or a token budget** · `H`
```python
context = "\n".join(retrieved)        # may exceed the context window
```
**Bug:** duplicate chunks, no reranking, and silent truncation when the prompt overflows the window.
**Fix:** rerank, dedup, and enforce an explicit token budget.

**105. Evaluating generation with exact match** · `M`
```python
score = (generated == reference)
```
**Bug:** string equality is near-useless for free-form generation.
**Fix:** semantic/faithfulness metrics, groundedness (NLI vs. context), LLM-as-judge, periodic human eval.

**106. High temperature for a factual/extraction task** · `E`
```python
llm(prompt, temperature=1.2)          # structured extraction
```
**Bug:** high temperature makes structured/factual outputs unstable and hallucination-prone.
**Fix:** temperature ≈ 0 for deterministic tasks.

**107. Untrusted content concatenated into the prompt** · `H`
```python
prompt = system_instructions + user_uploaded_doc + question
```
**Bug:** retrieved/user text can override your instructions (prompt injection).
**Fix:** separate roles, delimit untrusted content, add instruction-defense and output filtering, restrict tool access.

**108. Stale index after an embedding-model upgrade** · `H`
```python
# upgraded the embedding model but did NOT re-embed the corpus
```
**Bug:** new query vectors are compared against an index built by the old model → retrieval quality collapses.
**Fix:** re-embed the whole corpus on any model change; version the index alongside the model.

---

## L — Recommender Systems (109–114)

**109. Random split for a temporal recommender** · `H`
```python
train_test_split(interactions)
```
**Bug:** future interactions predict past ones → leakage; offline metrics won't hold online.
**Fix:** time-based split (e.g., leave-last-interaction-out).

**110. Crediting already-seen items as hits** · `M`
```python
# recommend items the user already interacted with in train, count them as correct
```
**Bug:** recommending known items inflates hit-rate/recall.
**Fix:** exclude previously-seen items from candidates and evaluation.

**111. Cold-start KeyError at serve** · `M`
```python
user_vec = embeddings[user_id]        # KeyError for a brand-new user
```
**Bug:** no handling for unseen users/items in production.
**Fix:** fallback (popularity/content-based) and a reserved UNK embedding.

**112. Negative sampling that hits true positives** · `M`
```python
neg = random.choice(all_items)        # might be an item the user actually likes
```
**Bug:** sampled "negatives" can be real positives → noisy labels.
**Fix:** exclude the user's known positives when sampling.

**113. Metric over the full catalog, not top-k** · `M`
```python
recall = hits / total_relevant        # across the whole catalog
```
**Bug:** not a top-k metric; misrepresents ranking quality.
**Fix:** recall@k / NDCG@k / MAP@k.

**114. Implicit feedback treated as explicit ratings** · `H`
```python
model.fit(clicks_as_ratings)          # click = 1, but no-click ≠ dislike
```
**Bug:** absence of interaction isn't a negative; treating implicit signals as explicit ratings mismodels preference.
**Fix:** implicit MF with confidence weighting and proper negative handling.

---

## M — Spark / PySpark & Big Data (115–120)

**115. `.collect()` on a huge DataFrame** · `M`
```python
data = df.collect()                   # pulls every row to the driver
```
**Bug:** driver OOM.
**Fix:** aggregate/`take(n)`/write to storage; keep work distributed.

**116. Python UDF where a built-in exists** · `M`
```python
@udf
def add_one(x): return x + 1
df = df.withColumn('y', add_one('x'))
```
**Bug:** row-by-row Python UDFs serialize to/from the JVM and kill performance.
**Fix:** native `col('x') + 1`, or a vectorized `pandas_udf`.

**117. Repeated actions recompute the lineage** · `M`
```python
for i in range(10):
    print(df.filter(cond).count())    # recomputed from scratch each time
```
**Bug:** no caching → the whole DAG re-executes every action.
**Fix:** `.cache()`/`.persist()` when a DataFrame is reused.

**118. Skewed groupBy with a hot key** · `H`
```python
df.groupBy('key').agg(...)            # one key holds 90% of rows
```
**Bug:** data skew creates a straggler task that dominates runtime.
**Fix:** key salting / enable Adaptive Query Execution.

**119. `toPandas()` on the whole distributed frame** · `M`
```python
pdf = df.toPandas()                   # entire dataset into driver memory
```
**Bug:** OOM on the driver.
**Fix:** sample or aggregate first.

**120. `monotonically_increasing_id` used as a stable key** · `H`
```python
df = df.withColumn('id', monotonically_increasing_id())   # later used as a join key
```
**Bug:** the ids aren't contiguous and aren't stable across recomputation/partitioning — unsafe as a durable key.
**Fix:** hash of business keys, or `row_number()` over a defined window.

---

## N — MLOps, Serving & Reproducibility (121–128)

**121. Divergent preprocessing between train and serve** · `H`
```python
# training: df.fillna(df.mean()); serving code omits it entirely
```
**Bug:** features differ at inference (train/serve skew).
**Fix:** one shared transform artifact (Pipeline / feature store).

**122. No seeds anywhere in the training artifact** · `M`
```python
model = train()                       # nothing seeded
```
**Bug:** the trained model isn't reproducible.
**Fix:** seed all libraries, pin dependencies, log versions and data hashes.

**123. Pickling a model without pinning the environment** · `M`
```python
pickle.dump(model, f)                 # loaded later under a different sklearn version
```
**Bug:** version mismatches cause load errors or silent behavior changes.
**Fix:** record/pin library versions; consider a portable format (ONNX).

**124. Reloading the model on every request** · `H`
```python
def predict(x):
    model = load('model.pkl')         # per-request disk load
    return model.predict(x)
```
**Bug:** cold-loading on each call adds huge latency.
**Fix:** load once at startup and cache the object.

**125. No input validation at the serving boundary** · `H`
```python
def predict(payload):
    return model.predict(np.array(payload['features']))
```
**Bug:** missing/extra features, wrong order, or NaNs produce garbage or crashes; schema drift goes unnoticed.
**Fix:** validate schema, ranges, and null-handling before inference.

**126. Logging raw payloads with PII** · `M`
```python
logger.info(f"request: {payload}")    # payload contains PII/secrets
```
**Bug:** sensitive data leaks into logs.
**Fix:** redact/allow-list fields before logging.

**127. Retraining on the model's own predictions** · `H`
```python
# label fresh data using the current model, then retrain on those labels
```
**Bug:** a self-reinforcing feedback loop entrenches the model's own bias (label leakage over time).
**Fix:** train on ground-truth labels; monitor and break feedback loops.

**128. Deploy-and-forget: no drift monitoring** · `H`
```python
# ship the model; no checks on inputs, predictions, or KPI afterward
```
**Bug:** silent degradation as the world shifts.
**Fix:** monitor input distributions, prediction distribution, and the business KPI, with alerts.

---

## O — SQL (129–136)

**129. `COUNT(col)` skips NULLs** · `E`
```sql
SELECT COUNT(commission) FROM sales;   -- expecting a row count
```
**Bug:** `COUNT(column)` ignores NULLs, undercounting rows.
**Fix:** `COUNT(*)` for rows.

**130. `NOT IN` with a NULL in the subquery** · `H`
```sql
SELECT * FROM a WHERE id NOT IN (SELECT id FROM b);   -- b.id contains NULL
```
**Bug:** a single NULL makes `NOT IN` return no rows at all.
**Fix:** `NOT EXISTS`, or exclude NULLs from the subquery.

**131. Filtering the right table of a LEFT JOIN in WHERE** · `H`
```sql
SELECT * FROM a LEFT JOIN b ON a.id = b.id WHERE b.status = 'active';
```
**Bug:** a `WHERE` predicate on the right table silently converts the LEFT JOIN into an INNER JOIN (drops unmatched rows).
**Fix:** move the condition into the `ON` clause.

**132. Aggregate ignores NULL / integer division** · `M`
```sql
SELECT AVG(score) FROM t;              -- NULLs silently excluded
```
**Bug:** `AVG` drops NULLs (maybe unintended); manual `SUM()/COUNT()` can also do integer division in some engines.
**Fix:** `COALESCE` to decide NULL handling; cast to avoid integer division.

**133. Non-aggregated column not in GROUP BY** · `M`
```sql
SELECT dept, name, MAX(salary) FROM emp GROUP BY dept;
```
**Bug:** `name` is neither grouped nor aggregated → nondeterministic (MySQL) or an error (standard SQL).
**Fix:** use a window function (`ROW_NUMBER` over `salary DESC`) to pick the row.

**134. Join fan-out double-counts a SUM** · `H`
```sql
SELECT SUM(o.amount) FROM orders o JOIN items i ON o.id = i.order_id;
```
**Bug:** a 1-to-many join repeats `o.amount` once per item → inflated total.
**Fix:** aggregate items first, or sum a de-duplicated order grain.

**135. `BETWEEN` on timestamps misses the last day** · `M`
```sql
WHERE ts BETWEEN '2024-01-01' AND '2024-01-31';
```
**Bug:** `'2024-01-31'` becomes midnight, so any time later that day is excluded.
**Fix:** `ts >= '2024-01-01' AND ts < '2024-02-01'`.

**136. `DISTINCT` used to hide a join bug** · `M`
```sql
SELECT DISTINCT customer_id FROM ... ;   -- papering over a fan-out
```
**Bug:** `DISTINCT` masks real duplication (a bad join) instead of fixing it — and hides genuine data issues.
**Fix:** find the root cause of the duplication.

---

## P — Concurrency, Performance & Memory (137–142)

**137. Unbounded global cache in a server** · `H`
```python
cache = {}
def handle(req):
    cache[req.id] = compute(req)      # grows forever; races under threads
```
**Bug:** unbounded memory growth plus data races on shared mutable state.
**Fix:** bounded LRU cache; thread-safe structures / locks.

**138. Reading a whole file into memory** · `E`
```python
lines = open('huge.log').read().splitlines()
```
**Bug:** loads the entire file at once.
**Fix:** iterate line by line (`for line in f:`).

**139. Recomputing an expensive step every call** · `M`
```python
def score(x):
    embed = expensive_embed(x)        # recomputed for the same x each time
    ...
```
**Bug:** no memoization of a deterministic, costly computation.
**Fix:** cache embeddings (dict / LRU / precompute).

**140. N+1 queries (and SQL injection)** · `H`
```python
for uid in user_ids:
    user = db.query(f"SELECT * FROM users WHERE id = {uid}")
```
**Bug:** N round-trips instead of one; the f-string is also a SQL-injection hole.
**Fix:** a single parameterized `... WHERE id IN (...)` / batch query.

**141. String concatenation in a loop** · `M`
```python
s = ""
for chunk in chunks:
    s += chunk                        # repeated re-allocation / temporaries
```
**Bug:** builds many intermediate strings; can degrade badly.
**Fix:** `s = "".join(chunks)`.

**142. Leaked file handles / connections** · `M`
```python
f = open('data.csv')
data = f.read()                       # never closed; in a loop → handle exhaustion
```
**Bug:** resources aren't released; repeated in a loop it exhausts file handles/connections.
**Fix:** use a `with` context manager.

---

## Scoring these challenges

Score the **process**, then the findings:

| Signal | Weak (1–2) | Strong senior (4–5) |
|---|---|---|
| **Search pattern** | Spots one bug, declares done | Sweeps systematically: correctness → edge cases → dtype/NaN → complexity → train/serve parity |
| **Leakage instinct** | Misses it | Flags leakage/skew unprompted and explains the mechanism |
| **Complexity** | Ignores it | States Big-O, identifies the hot path, proposes the right structure |
| **Fix quality** | "Rewrite it" | Minimal, correct fix + how they'd test it (unit test / assertion / property) |
| **Curiosity** | Accepts the code as-is | Asks *why* it's shaped this way, what the inputs look like, what breaks at scale |

**Tiering by round:** use `E` items on a phone screen (can they read code at all?), `M` items in the core coding round, and cluster `H` items — especially the leakage (D, F), PyTorch (H), and RAG (K) ones — to separate seniors from mid-levels. Two or three `H` snippets tell you more than twenty `E` ones.

**Universal green flags:** narrates a repeatable checklist; names leakage/skew without prompting; quantifies complexity; writes a minimal fix *and* a test; distinguishes "bug" from "smell"; asks about inputs and scale before rewriting.

**Universal red flags:** fixates on style over correctness; "just retrain / just rewrite"; reaches for `.fit()` on the whole dataset; can't estimate Big-O; misses that the code mutates its inputs; treats a passing run as proof of correctness.
