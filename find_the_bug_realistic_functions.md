# Senior DS — "Find the Bug" Bank: Realistic Multi-Bug Functions (124 Challenges)

> **What this is.** A third companion file, but a *different style* from the atomic-snippet bank. Every challenge here is a **nontrivial function** — 15–35 lines — with **several interacting bugs** (correctness + performance) deliberately planted, in the spirit of the `preprocess` vehicle-routing example. These are full code-review exercises, not one-liners.
>
> **How to run one.** *"A junior wrote this and it runs on the happy path. Find every correctness bug and performance problem, then rewrite the hot path."* Let them work top-to-bottom out loud. Score whether they sweep **systematically** (correctness → edge cases → dtype/NaN → complexity → leakage → train/serve parity) or spot one bug and stop. Each answer key lists the planted issues; strong candidates find most, great ones find the **subtle** one flagged as the *senior tell*.
>
> **Difficulty:** `M` core round · `H` senior signal. **Domains:** 1 Routing/OR/clustering · 2 Data pipelines · 3 Classical ML train/eval · 4 PyTorch/DL · 5 Recommenders · 6 Time series · 7 NLP/RAG/LLM · 8 Stats/A-B/causal · 9 MLOps/serving/concurrency · 10 SQL/Spark · 11 Numerical/algorithms.

---

## Domain 1 — Routing, Optimization & Clustering (1–12)

**1. Greedy parcel-to-vehicle assignment** · `H`
```python
def assign_parcels_to_vehicles(self, parcels, capacities):
    """Assign each parcel to the first vehicle that can hold it."""
    assignments = {}
    remaining = capacities                      # remaining capacity per vehicle
    for i in range(len(parcels)):
        p = parcels[i]
        for v in range(len(capacities)):
            if remaining[v] >= p['weight']:
                assignments[p['id']] = v
                parcels.pop(i)                  # "remove" the assigned parcel
                break
    return assignments
```
**Bugs:** (1) `remaining` never **decrements** after an assignment → a single vehicle is assigned unlimited parcels; capacity is not actually enforced. (2) `remaining = capacities` **aliases** the caller's list — any later decrement would mutate the input. (3) `parcels.pop(i)` **while iterating `range(len(parcels))`** shifts indices → parcels are skipped and the fixed range over-runs. (4) First-fit with no sorting packs poorly. (5) assumes `p['weight']` always present.
**Fix:** copy `remaining`, decrement it on assignment, iterate a copy (or build a survivors list), first-fit-**decreasing** (sort weights desc), validate keys.
**Senior tell:** catching that capacity is never decremented — the loop *looks* right.

**2. K-Means (assignment + update)** · `H`
```python
def kmeans(X, k, n_iter=100):
    centroids = X[:k].copy()
    for _ in range(n_iter):
        labels = []
        for x in X:
            dists = [np.sqrt(np.sum((x - c) ** 2)) for c in centroids]
            labels.append(np.argmin(dists))
        labels = np.array(labels)
        for j in range(k):
            centroids[j] = X[labels == j].mean(axis=0)
    return centroids, labels
```
**Bugs:** (1) **Empty cluster** → `X[labels==j].mean()` is `NaN`, which then poisons all future distances. (2) `X[:k]` init is deterministic and terrible if `X` is sorted — no k-means++/random. (3) `sqrt` inside an **O(n·k·d) Python double loop** — sqrt is unnecessary for `argmin` and the whole thing should be vectorized (`cdist`/broadcasting). (4) **No convergence check** — always runs all `n_iter`. (5) if `X` is int dtype, centroid assignment truncates.
**Fix:** k-means++ init, vectorize distances, drop sqrt, reinit empty clusters, tolerance-based early stop, float dtype.

**3. Haversine distance matrix** · `M`
```python
def distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            D[i, j] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return D
```
**Bugs:** (1) **Degrees not converted to radians** — `np.sin/cos` expect radians, so every distance is wrong. (2) **O(n²) Python loop** computing a **symmetric** matrix twice (plus the zero diagonal). (3) Floating error can push `a` slightly above 1 → `arcsin` returns `NaN`; clip to `[0,1]`.
**Fix:** `np.radians(...)`, vectorize, fill only the upper triangle then mirror, `np.clip(a, 0, 1)`.

**4. Nearest-neighbor TSP route** · `H`
```python
def nearest_neighbor_route(D, start=0):
    n = D.shape[0]
    visited = [start]
    route = [start]
    current = start
    while len(route) < n:
        row = D[current]
        nxt = np.argmin(row)                    # nearest overall
        route.append(nxt)
        visited.append(nxt)
        current = nxt
    return route
```
**Bugs:** (1) `np.argmin(row)` includes the **current node (distance 0 to itself)** and **already-visited** nodes → it picks self/revisits and the `while` loop may spin forever or fill with duplicates. `visited` is tracked but **never used to mask**. (2) No return-to-depot leg (open vs closed route ambiguity). (3) list-based masking would be O(n) per step.
**Fix:** set visited entries to `inf` before `argmin`, argmin over unvisited only, use a boolean mask, decide open/closed.

**5. Capacity-constrained cluster merge** · `H`
```python
def merge_small_clusters(clusters, capacities, max_capacity):
    for cid in clusters:
        if capacities[cid] < 0.5 * max_capacity:
            target = min(clusters, key=lambda c: distance(clusters[cid], clusters[c])
                         if c != cid else np.inf)
            clusters[target] += clusters[cid]
            capacities[target] += capacities[cid]
            del clusters[cid]
    return clusters
```
**Bugs:** (1) **`del clusters[cid]` while iterating `for cid in clusters`** → `RuntimeError: dictionary changed size during iteration`. (2) The merge **never checks** `capacities[target] + capacities[cid] <= max_capacity` → it creates over-capacity clusters, violating the constraint. (3) Centroid/geometry isn't recomputed after a merge → stale distances for later merges. (4) `distance` recomputed every pass → O(k²). (5) if a `target` was itself deleted earlier → `KeyError`.
**Fix:** iterate `list(clusters)`, check capacity before merging, recompute the merged centroid, clean `capacities`, guard deleted targets.

**6. Sweep algorithm (angular clustering)** · `M`
```python
def sweep_clusters(coords, demands, vehicle_cap, depot):
    angles = []
    for (x, y) in coords:
        angles.append(np.arctan2(x - depot[0], y - depot[1]))
    order = np.argsort(angles)
    routes, load, cur = [], 0, []
    for idx in order:
        if load + demands[idx] > vehicle_cap:
            routes.append(cur)
            cur, load = [], 0
        cur.append(idx)
        load += demands[idx]
    return routes
```
**Bugs:** (1) `np.arctan2(dx, dy)` — the signature is `arctan2(y, x)`; the args are **swapped**, so the sweep starts from the wrong axis / rotates the wrong way. (2) The **final `cur` is never appended** to `routes` after the loop → the last route is silently dropped (off-by-one). (3) A single `demand > vehicle_cap` is never detected → it forms an over-capacity route silently.
**Fix:** `arctan2(dy, dx)`, append the trailing route, reject/flag oversize demands.

**7. Divisive split by capacity (recursive)** · `H`
```python
def split_by_capacity(self, points, weights, cap):
    left, right = [], []
    left_w = right_w = 0
    for i in range(len(points)):
        if left_w <= right_w:
            left.append(points[i]);  left_w += weights[i]
        else:
            right.append(points[i]); right_w += weights[i]
    if left_w > cap:
        self.split_by_capacity(left, weights, cap)
    if right_w > cap:
        self.split_by_capacity(right, weights, cap)
    return left, right
```
**Bugs:** (1) The recursive calls pass the **full `weights` list** instead of the subset for `left`/`right` → `weights[i]` now indexes the wrong points — a **parallel-list desync** (the same failure class as the `preprocess` example). (2) The recursive **return values are discarded** → recursion has no effect; only the first-level split is returned. (3) No base case for a single point whose weight already exceeds `cap` → infinite recursion. (4) Splits by weight only, ignoring geometry → poor spatial clusters.
**Fix:** carry indices so weights stay aligned, use the recursive results, add a base case, incorporate spatial proximity.

**8. Feasibility check** · `M`
```python
def is_feasible(demands, capacities):
    total_demand = sum(demands)
    total_capacity = max(capacities)            # aggregate capacity
    if total_demand > total_capacity:
        return False
    for d in demands:
        if d > max(capacities):
            return False
    return True
```
**Bugs:** (1) `total_capacity = max(capacities)` should be **`sum`** for aggregate feasibility → wildly wrong verdicts. (2) `max(capacities)` **recomputed inside the loop**. (3) Even with the sum fixed, aggregate feasibility is **necessary but not sufficient** (bin-packing is NP-hard) — the check can't guarantee a valid packing.
**Fix:** `sum(capacities)`, hoist the max out, and document that this is a necessary-only screen.

**9. Clarke-Wright savings** · `H`
```python
def clarke_wright(D, demands, cap):
    n = len(demands)
    savings = []
    for i in range(1, n):
        for j in range(1, n):
            s = D[0][i] + D[0][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort()
    routes = {i: [i] for i in range(1, n)}
    for s, i, j in savings:
        ri, rj = routes.get(i), routes.get(j)
        if ri is not None and rj is not None and ri is not rj:
            merged = ri + rj
            if sum(demands[k] for k in merged) <= cap:
                for k in merged:
                    routes[k] = merged
    return routes
```
**Bugs:** (1) `savings.sort()` is **ascending**; Clarke-Wright merges in **descending** savings order → it merges the worst pairs first. (2) The inner loop `for j in range(1, n)` includes `i == j` and both `(i,j)` and `(j,i)` → duplicate/degenerate pairs; should be `j > i`. (3) Merges are only valid when `i` and `j` are **route endpoints**, not interior — this merges regardless of position, producing invalid routes. (4) `routes[k] = merged` shares mutable lists; identity tracking gets fragile after merges.
**Fix:** `savings.sort(reverse=True)`, `j > i`, enforce the endpoint condition, track route membership cleanly.

**10. Delivery time-window check** · `M`
```python
def within_time_window(arrival, window_start, window_end):
    # all are 'HH:MM' strings
    return window_start < arrival < window_end
```
**Bugs:** (1) **Lexicographic string comparison** of `'HH:MM'` only works if strings are zero-padded and identical width — `'9:00'` sorts after `'10:00'` because `'9' > '1'`. (2) Strict `<` **excludes exact-boundary** arrivals (should be `<=`). (3) Strings can't support arithmetic (wait time, service duration) downstream.
**Fix:** parse to `datetime.time`/minutes, use `<=`, enforce a consistent zero-padded format.

**11. DBSCAN region query** · `M`
```python
def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        dist = np.sum((X[point_idx] - X[i]) ** 2)   # squared distance
        if dist <= eps:                              # eps is NOT squared
            neighbors.append(i)
    return neighbors
```
**Bugs:** (1) `dist` is **squared** Euclidean but compared to a raw `eps` → the neighborhood is far too small (unit mismatch). Compare to `eps**2` or take the sqrt. (2) O(n²) per query → O(n³) for full DBSCAN; use a KD-/ball-tree. (3) Includes the point itself (usually fine for core-count, but worth noting).
**Fix:** compare squared distance to `eps**2`, use a spatial index.

**12. Route cost** · `M`
```python
def route_cost(route, D):
    cost = 0
    for i in range(len(route)):
        cost += D[route[i]][route[i+1]]
    return cost
```
**Bugs:** (1) `route[i+1]` on the last iteration → **IndexError**; the loop must run to `len(route)-1`. (2) **No return-to-depot leg** (`D[route[-1]][depot]`) → closed-route cost is undercounted. (3) Empty/single-node routes crash or return 0 without a guard.
**Fix:** iterate to `len-1`, add the closing leg if the route is a cycle, guard short routes.

---

## Domain 2 — Data Pipelines, Pandas & Feature Engineering (13–24)

**13. Build a training table from joins** · `H`
```python
def build_features(orders, customers, products):
    df = orders.merge(customers, on='customer_id').merge(products, on='product_id')
    df['customer_ltv'] = df.groupby('customer_id')['amount'].transform('sum')
    df['product_avg_price'] = df.groupby('product_id')['price'].transform('mean')
    df['label'] = (df['amount'] > df['amount'].mean()).astype(int)
    return df
```
**Bugs:** (1) **Merge fan-out** — duplicate keys in `customers`/`products` explode rows; the inner join silently drops unmatched orders. (2) **Leakage everywhere:** `customer_ltv`/`product_avg_price` aggregate over *all* rows including the future; the `label` is built from `amount` while `amount`-derived features remain in the frame → target leakage. (3) `amount > global mean` bakes the full-data distribution into the label. (4) no copy; heavy memory.
**Fix:** validate merge cardinality, compute aggregates **point-in-time** (only past rows), define the label from a **separate future window**, fit stats on train only.

**14. CSV ingestion + cleaning** · `M`
```python
def clean(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df[df['qty'] < 0]['qty'] = 0
    df.dropna(inplace=True)
    df['customer_id'] = df['customer_id'].fillna(-1)
    return df
```
**Bugs:** (1) `df[df['qty']<0]['qty'] = 0` is **chained assignment** → writes to a copy, original unchanged (silent no-op). (2) `dropna(inplace=True)` runs **before** the `customer_id` fill → it deletes the very rows you meant to keep and fill. (3) `to_datetime` with no `format` → ambiguous/slow parsing. (4) filling `customer_id` with `-1` on a NaN-float column yields `-1.0` and collides with a sentinel.
**Fix:** `.loc[...]` assignment, fill before/targeted dropna, explicit date format, nullable `Int64` ids.

**15. Category summary** · `M`
```python
def category_summary(df):
    return df.groupby('category').agg(
        total=('amount', 'sum'),
        avg=('amount', 'mean'),
        n=('amount', 'count'),
    ).sort_values('total', ascending=False)
```
**Bugs:** (1) `groupby('category')` **drops NaN-category rows** silently → undercount. (2) `'count'` on `amount` counts **non-null amounts**, not rows — inconsistent with `avg` if some amounts are NaN, and misleading if you wanted a row count (use `size`). (3) an unused-level Categorical can create empty groups on older pandas.
**Fix:** `groupby(..., dropna=False, observed=True)`, choose `size` vs `count` deliberately.

**16. Per-entity rolling features** · `H`
```python
def add_rolling(df):
    df['sales_7d']  = df['sales'].rolling(7).mean()
    df['sales_30d'] = df['sales'].rolling(30).mean()
    df['trend']     = df['sales_7d'] / df['sales_30d']
    return df
```
**Bugs:** (1) not **sorted by date** and not **grouped by entity** → the window crosses SKU/customer boundaries and time is unordered. (2) includes the current row and no `.shift(1)` → **leakage** if predicting current `sales`. (3) `trend` divides by a 30-day mean that can be **0** → inf/NaN. (4) leading rows are NaN.
**Fix:** `sort_values(['entity','date']).groupby('entity')['sales'].rolling(...).shift(1)`, guard zero denominators.

**17. Manual one-hot + scaling** · `H`
```python
def preprocess(train, test):
    train = pd.get_dummies(train, columns=['region'])
    test  = pd.get_dummies(test,  columns=['region'])
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])
    return train, test
```
**Bugs:** (1) `get_dummies` is applied **independently** to train and test → **column mismatch** (regions present in one but not the other), so the model sees a different feature space at serve time. (2) no `handle_unknown` → a brand-new region in production breaks everything. (3) the scaler is correct (train-only) but the dummy columns aren't aligned/reindexed.
**Fix:** `ColumnTransformer` + `OneHotEncoder(handle_unknown='ignore')` fit on train, reindex test to train's columns.

**18. Deduplicate keeping the latest** · `M`
```python
def dedup_latest(df):
    df = df.sort_values('updated_at')
    df = df.drop_duplicates(subset='id')
    return df
```
**Bugs:** (1) ascending sort + default `keep='first'` → keeps the **oldest** record, not the latest. (2) `NaT` in `updated_at` sorts unpredictably. (3) index left with gaps.
**Fix:** sort descending (or `keep='last'`), handle `NaT`, `reset_index(drop=True)`.

**19. Split for a time-ordered dataset** · `M`
```python
def split(df):
    return train_test_split(df, test_size=0.2, random_state=42)
```
**Bugs:** (1) a **random split on time-ordered data** puts the future in train → temporal leakage; offline metrics won't hold. (2) no embargo/gap between train and validation → windowed features leak across the seam.
**Fix:** sort by time, cut by a date boundary, purge the seam.

**20. Impute missing values** · `M`
```python
def impute(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df
```
**Bugs:** (1) categorical columns filled with the **int `0`** → mixed types in an object column (should be `'missing'`). (2) numeric fill uses the **global mean** (leakage if before split; fit on train). (3) iterating **all columns** imputes the **target** too if present. (4) datetime columns are neither `object` nor numeric-mean-appropriate → skipped/wrong.
**Fix:** fit imputers on train, per-type strategy, exclude the target, handle datetimes.

**21. Sessionize events** · `H`
```python
def sessionize(events, gap_minutes=30):
    events['gap'] = events['ts'].diff().dt.total_seconds() / 60
    events['new_session'] = events['gap'] > gap_minutes
    events['session_id'] = events['new_session'].cumsum()
    return events
```
**Bugs:** (1) not **grouped by user** → `diff()` crosses users, merging their sessions. (2) not **sorted by `ts`** → diffs are meaningless. (3) the first row's `gap` is `NaN` (`NaN > 30` → False), so the first session boundary is off by one. (4) a global `cumsum` mixes users' session ids.
**Fix:** `sort_values(['user','ts'])`, group by user for both `diff` and `cumsum`, seed the first session.

**22. User-item matrix** · `M`
```python
def user_item_matrix(df):
    return df.pivot(index='user', columns='item', values='rating').fillna(0)
```
**Bugs:** (1) `pivot` (not `pivot_table`) **raises on duplicate `(user, item)`** pairs → crashes on real data. (2) `fillna(0)` conflates "no rating" with "rated 0" — wrong for implicit/explicit feedback. (3) the dense pivot **densifies a sparse matrix** → memory blowup for large catalogs.
**Fix:** `pivot_table(aggfunc='mean')`, keep it sparse, don't fill 0 blindly.

**23. Target encoding helper** · `H`
```python
def target_encode(df, col, target):
    means = df.groupby(col)[target].mean()
    df[col + '_te'] = df[col].map(means)
    return df
```
**Bugs:** (1) **Leakage** — each encoded value includes its own row's target and the full dataset. (2) **No smoothing** → rare categories get extreme, noisy values. (3) **Unseen categories** at serve → `NaN`, no global-mean fallback.
**Fix:** out-of-fold / leave-one-out encoding with Bayesian smoothing, fit on train, fall back to the global mean for unseen levels.

**24. Memory downcasting** · `M`
```python
def optimize_memory(df):
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```
**Bugs:** (1) downcasting **integer id columns to float32** loses precision above ~2²⁴ → **id collisions** (silent, catastrophic for joins). (2) `downcast='float'` needlessly converts integer columns to float. (3) money as float32 accumulates rounding error.
**Fix:** downcast ints as `'integer'` and floats as `'float'` separately, never touch id keys, keep money as integer cents / decimal.

---

## Domain 3 — Classical ML Training & Evaluation (25–36)

**25. End-to-end train/eval routine** · `H`
```python
def train_eval(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    best = None
    for C in [0.01, 0.1, 1, 10]:
        m = LogisticRegression(C=C).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        if best is None or acc > best[0]:
            best = (acc, m)
    auc = roc_auc_score(y_te, best[1].predict(X_te))
    return best[1], auc
```
**Bugs:** (1) scaler `fit_transform` on **all of X before the split** → leakage. (2) **hyperparameter `C` selected on the test set** and then AUC reported on the same test → no unbiased estimate. (3) `accuracy` used for model selection on (likely) imbalanced data. (4) **AUC computed on `predict` (hard labels)** instead of `predict_proba`. (5) no `stratify`, no `random_state`, no `class_weight`.
**Fix:** split → `Pipeline(scaler, model)` → validation/CV for `C` → AUC on `predict_proba[:,1]`; add stratify, seed, class weights.

**26. Cross-val with preprocessing** · `H`
```python
def cv_score(X, y):
    X = SelectKBest(k=10).fit_transform(X, y)
    X, y = SMOTE().fit_resample(X, y)
    return cross_val_score(LogisticRegression(), X, y, cv=5).mean()
```
**Bugs:** (1) **feature selection on the full dataset** uses `y` over all rows → leakage. (2) **SMOTE before CV** synthesizes points from all data, leaking validation neighbors into train folds → optimistic scores. (3) resampling changes prevalence so the CV number doesn't reflect production.
**Fix:** put `SelectKBest` + `SMOTE` + model inside an imblearn `Pipeline` and pass *that* to `cross_val_score`.

**27. Regression model selection** · `M`
```python
def select_regressor(X, y):
    models = {'ridge': Ridge(), 'rf': RandomForestRegressor()}
    scores = {}
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
        scores[name] = s.mean()
    best = max(scores, key=scores.get)
    print(f"Best MSE: {scores[best]}")
    return best
```
**Bugs:** (1) `neg_mean_squared_error` is **negative**; `max` correctly picks the least-negative, but the printout labels a **negative number "MSE"** → confusing/incorrect reporting. (2) **Ridge needs scaling** and RF doesn't — comparing on unscaled features handicaps Ridge unfairly (scale it inside a pipeline). (3) no `random_state`; unshuffled KFold biases folds on ordered data.
**Fix:** negate for display, scale Ridge in a pipeline, seed and shuffle CV.

**28. Imbalanced fraud model** · `M`
```python
def fraud_model(X_tr, y_tr, X_te):
    m = RandomForestClassifier().fit(X_tr, y_tr)   # y is ~99.5% negative
    preds = m.predict(X_te)
    return preds
```
**Bugs:** (1) default **0.5 threshold on extreme imbalance** → predicts almost all negative, misses fraud. (2) no `class_weight='balanced'`. (3) returns **hard labels** — a review queue needs ranked **scores**; and no probability **calibration**.
**Fix:** class weights, threshold tuned to the cost curve on validation, return calibrated `predict_proba`.

**29. Overfitting check** · `M`
```python
def check_fit(model, X, y):
    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"Accuracy: {train_acc}")
    return train_acc
```
**Bugs:** (1) reports **training accuracy** — says nothing about generalization; an overfit model scores highest here. (2) no validation/holdout, no train-vs-val comparison. (3) `score` = accuracy → wrong for imbalance/regression.
**Fix:** held-out or CV, compare train vs validation (learning curves), metric appropriate to the problem.

**30. Feature-importance selection** · `M`
```python
def top_features(model, X):
    imp = model.feature_importances_
    order = np.argsort(imp)
    return X.columns[order[:10]]
```
**Bugs:** (1) `argsort` is ascending + `[:10]` → returns the **10 least important** features. (2) impurity importances are **biased toward high-cardinality/continuous** features and unreliable under correlation → prefer permutation importance/SHAP. (3) selecting on the same data then refitting → mild selection overfit.
**Fix:** sort descending (`[-10:]`), use permutation importance, do selection inside CV.

**31. Probability calibration** · `M`
```python
def calibrate(model, X_tr, y_tr):
    cal = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    cal.fit(X_tr, y_tr)
    return cal
```
**Bugs:** (1) `cv='prefit'` + fitting on the **same data the model was trained on** → overfit, meaningless calibration; needs a **separate** calibration set. (2) `sigmoid` (Platt) assumes a shape — `isotonic` may fit better with enough data.
**Fix:** hold out a calibration set (or `cv=k` refitting), pick the method by data size.

**32. Multiclass evaluation** · `M`
```python
def evaluate_multiclass(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return p, r
```
**Bugs:** (1) default `average='binary'` on **multiclass** → raises or silently needs `pos_label`. (2) macro vs weighted vs micro answer different questions; on imbalance, macro exposes minority failure that micro hides.
**Fix:** set `average=` explicitly, report per-class and a confusion matrix.

**33. Save / load model** · `M`
```python
def save(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load():
    return pickle.load(open('model.pkl', 'rb'))
```
**Bugs:** (1) only the **model** is persisted — the fitted scaler/encoder isn't → serving can't reproduce features (train/serve skew). (2) no **version pinning** → sklearn version mismatch on load = error or silent behavior change. (3) `load()` **leaks a file handle** (no `with`). (4) hardcoded path.
**Fix:** persist the whole pipeline, record library/data versions, use `with`, parametrize the path.

**34. "Is the model good?" gate** · `M`
```python
def is_model_good(model, X, y):
    model.fit(X, y)
    score = model.score(X, y)
    return score > 0.8
```
**Bugs:** (1) arbitrary `0.8` bar on the **training** score. (2) no **baseline** (majority/DummyClassifier) — 0.8 can be *worse* than predicting the majority on imbalance. (3) no holdout.
**Fix:** compare against a baseline on held-out data with a prevalence-appropriate metric.

**35. Ensemble averaging** · `M`
```python
def ensemble_predict(models, X):
    preds = [m.predict(X) for m in models]
    return np.round(np.mean(preds, axis=0))
```
**Bugs:** (1) averages **hard 0/1 labels** then rounds → a majority vote that throws away confidence; for AUC/ranking you want averaged **probabilities**. (2) members with different **calibration** shouldn't be probability-averaged naively → calibrate first. (3) `np.round` uses round-half-to-even → inconsistent ties.
**Fix:** average `predict_proba`, calibrate members, define tie-breaking.

**36. Log-target regression** · `H`
```python
def train_log_target(X, y):
    ylog = np.log(y)
    m = LinearRegression().fit(X, ylog)
    pred = m.predict(X)
    rmse = np.sqrt(mean_squared_error(ylog, pred))
    return m, rmse
```
**Bugs:** (1) `np.log(y)` is **−inf/NaN** when `y` has zeros/negatives (demand can be 0) → use `log1p` and guarantee non-negativity. (2) **RMSE is in log space** but reported as if original units → misleading; `expm1` before the metric. (3) predictions are in log space (caller expects units). (4) back-transforming a log prediction is **biased** (Jensen) → smearing correction.
**Fix:** `log1p`, metric in original units, `expm1` predictions, apply a smearing/duan correction.

---

## Domain 4 — PyTorch & Deep Learning (37–48)

**37. Training loop** · `H`
```python
def train(model, loader, val_loader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0
        for x, y in loader:
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total += loss
        val_acc = evaluate(model, val_loader)
    return model
```
**Bugs:** (1) **missing `opt.zero_grad()`** → gradients accumulate across steps. (2) `total += loss` keeps the autograd graph alive → memory grows; use `.item()`. (3) no `model.train()`/`model.eval()` toggling around train/validation. (4) data/model **device** not managed.
**Fix:** `opt.zero_grad()` each step, `total += loss.item()`, set train/eval modes, `.to(device)`.

**38. Custom Dataset** · `M`
```python
class SalesDataset(Dataset):
    def __init__(self, df):
        self.X = StandardScaler().fit_transform(df[features])
        self.y = df['label'].values
    def __len__(self):
        return len(self.X) - 1
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```
**Bugs:** (1) the scaler is **fit inside the dataset on all rows** → leakage if the same df spans splits; scaling should be train-only and shared. (2) `__len__` returns `len - 1` → **drops the last sample** (off-by-one). (3) `__getitem__` returns **numpy** with default dtypes → `X` should be float tensors and `y` a `long` tensor for `CrossEntropyLoss`.
**Fix:** fit the scaler outside (train only), `__len__ = len(self.X)`, return `torch.tensor` with correct dtypes.

**39. VAE loss** · `H`
```python
def vae_loss(recon_x, x, mu, logvar):
    recon = F.mse_loss(recon_x, x, reduction='mean')
    kl = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```
**Bugs:** (1) **KL sign is wrong** — the divergence term is `-0.5 * sum(1 + logvar - mu² - exp(logvar))`; written positive, training **maximizes** KL and pushes the posterior *away* from the prior. (2) **reduction mismatch** — `mse_loss` uses `mean` (over all elements) while KL uses `sum` (over the batch) → the two terms live on wildly different scales, so one dominates. (3) no **β / KL annealing** → posterior collapse risk. (4) Gaussian-MSE reconstruction may be wrong for bounded/count features.
**Fix:** negate the KL, make reductions consistent (sum over latent dims, mean over batch), add β/annealing, match the reconstruction likelihood to the data.

**40. Inference loop** · `M`
```python
def predict(model, loader):
    preds = []
    for x, _ in loader:
        out = model(x)
        p = torch.softmax(out, dim=1)
        preds.append(torch.argmax(p))
    return preds
```
**Bugs:** (1) no `model.eval()` (dropout/BN wrong) and no `torch.no_grad()` (memory/speed). (2) `torch.argmax(p)` **without `dim`** → argmax over the whole `(batch, classes)` tensor → one scalar per batch instead of per-sample. (3) softmax before argmax is redundant. (4) tensors kept on GPU → `.cpu()` needed.
**Fix:** `model.eval()` + `no_grad`, `argmax(dim=1)`, drop the redundant softmax, move results to CPU.

**41. Scheduler + "early stopping"** · `M`
```python
def train(model, loader, val_loader, epochs):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    best = float('inf')
    for e in range(epochs):
        loss = train_one_epoch(model, loader, opt)
        sched.step()
        if loss < best:
            best = loss
            torch.save(model.state_dict(), 'best.pt')
```
**Bugs:** (1) checkpointing monitors **training loss**, not validation → saves overfit models. (2) despite the name, there's **no early stopping** (no patience counter / break). (3) the "best" model is saved but **never loaded/returned**.
**Fix:** monitor a validation metric, add patience + break, load the best checkpoint at the end.

**42. Gradient clipping** · `M`
```python
for x, y in loader:
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
**Bugs:** (1) `clip_grad_norm_` is called **after `opt.step()`** → the gradients were already applied, so clipping does nothing. It must run **between `backward()` and `step()`**.
**Fix:** move the clip to after `loss.backward()` and before `opt.step()`.

**43. Sequence loss with padding** · `H`
```python
def seq_loss(logits, targets, pad_id=0):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```
**Bugs:** (1) the loss includes **padding positions** → skewed gradients on variable-length sequences; use `ignore_index=pad_id` or an explicit mask. (2) for RNNs, not using `pack_padded_sequence` wastes compute on pads.
**Fix:** `F.cross_entropy(..., ignore_index=pad_id)`, mask, and pack sequences.

**44. Reproducibility setup** · `M`
```python
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
```
**Bugs:** (1) missing `random.seed`, `torch.cuda.manual_seed_all`, DataLoader `worker_init_fn` (**workers reseed independently**), and `torch.use_deterministic_algorithms(True)` / cuDNN flags → runs still differ. (2) a global `np.random.seed` doesn't cover multi-worker loaders or some CUDA ops.
**Fix:** seed all RNGs, set `worker_init_fn` + a `Generator`, enable deterministic algorithms (note the perf cost), pin data ordering.

**45. Single-sample inference** · `M`
```python
def predict_one(model, x):
    model.train()
    return model(x.unsqueeze(0))
```
**Bugs:** (1) `model.train()` at inference → **BatchNorm uses the statistics of a single sample** (unstable/garbage) and dropout stays active. (2) batch size 1 can't produce a meaningful BN variance.
**Fix:** `model.eval()` + `torch.no_grad()`; consider LayerNorm/GroupNorm for tiny batches.

**46. Weight initialization** · `M`
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
```
**Bugs:** (1) **zero-initializing weights** makes every neuron in a layer identical with identical gradients → the network **never breaks symmetry** and can't learn. (2) init should be He (ReLU) or Xavier (tanh), not constant.
**Fix:** `kaiming_normal_` / `xavier_uniform_` by activation; keep biases at 0.

**47. Half-precision training** · `M`
```python
model = model.half()
for x, y in loader:
    out = model(x.half())
    loss = crit(out, y)
    loss.backward()
    opt.step()
```
**Bugs:** (1) naive `.half()` training **without `GradScaler`** → gradient underflow/overflow and NaN loss; use `torch.cuda.amp.autocast` + `GradScaler`. (2) BN/loss in fp16 are unstable. (3) `zero_grad()` missing.
**Fix:** use AMP (`autocast` + `GradScaler`), keep sensitive ops in fp32, zero grads.

**48. Categorical embedding** · `M`
```python
class CatModel(nn.Module):
    def __init__(self, n_categories):
        super().__init__()
        self.emb = nn.Embedding(n_categories, 16)
    def forward(self, cat_idx):
        return self.emb(cat_idx)
```
**Bugs:** (1) a **new category id ≥ `n_categories`** at serve → `IndexError` / CUDA assert; reserve an **UNK slot** and map unseen → UNK. (2) no `padding_idx` for padded sequences. (3) non-contiguous / non-zero-based ids misalign embedding rows.
**Fix:** add UNK (and pad) slots, map unseen ids to UNK, ensure contiguous 0-based indexing.

---

## Domain 5 — Recommender Systems (49–60)

**49. Matrix-factorization SGD** · `H`
```python
def train_mf(interactions, n_users, n_items, k=32, epochs=10):
    U = np.random.randn(n_users, k)
    V = np.random.randn(n_items, k)
    for _ in range(epochs):
        for u, i, r in interactions:
            pred = U[u] @ V[i]
            err = r - pred
            U[u] += 0.01 * err * V[i]
            V[i] += 0.01 * err * U[u]
    return U, V
```
**Bugs:** (1) `V[i] += ... * U[u]` uses the **already-updated `U[u]`** from the line above → an inconsistent SGD step (should use the pre-update value). (2) **no regularization** → embeddings blow up / overfit. (3) treats `r` as **explicit** feedback even for implicit signals. (4) interactions are **not shuffled** per epoch; fixed LR, no decay. (5) `randn` init scale is too large → slow/divergent.
**Fix:** cache old `U[u]` before updating, add L2 reg, shuffle each epoch, tune/decay LR, scale init small (e.g. `randn * 0.1`).

**50. Top-N recommendation** · `H`
```python
def recommend(U, V, user_id, seen, n=10):
    scores = U[user_id] @ V.T
    top = np.argsort(scores)[:n]
    return top
```
**Bugs:** (1) `argsort` is ascending + `[:n]` → returns the **lowest-scored** items. (2) does **not exclude `seen`** items → recommends already-purchased products, inflating offline hits. (3) no cold-start fallback for an unseen `user_id`.
**Fix:** `np.argsort(-scores)`, mask `seen` (set to `-inf`) before ranking, fall back to popularity for new users.

**51. Implicit-feedback labels** · `M`
```python
def build_labels(clicks_df, all_items, all_users):
    labels = np.zeros((len(all_users), len(all_items)))
    for _, row in clicks_df.iterrows():
        labels[row['user'], row['item']] = 1
    return labels
```
**Bugs:** (1) missing entries become a **hard 0 (dislike)** — but absence ≠ negative in implicit feedback. (2) a **dense** users×items matrix blows up memory for real catalogs → use sparse. (3) `iterrows` is slow.
**Fix:** sparse implicit MF (ALS/BPR) with confidence weighting; vectorize construction.

**52. Content-based similarity** · `M`
```python
def similar_items(item_vectors, item_id, n=5):
    sims = item_vectors @ item_vectors[item_id]
    top = np.argsort(sims)[-n:]
    return top[::-1]
```
**Bugs:** (1) a raw **dot product isn't cosine** unless vectors are normalized → long/popular vectors dominate. (2) `argsort[-n:]` **includes the item itself** (max self-similarity) → wastes a slot. (3) if the vectors come from a TF-IDF fit on the full corpus incl. test → upstream leakage.
**Fix:** normalize (or use cosine), exclude `item_id`, fit TF-IDF on train.

**53. Negative sampling** · `M`
```python
def sample_negatives(user, pos_items, n_items, k=5):
    return [random.randint(0, n_items - 1) for _ in range(k)]
```
**Bugs:** (1) sampled "negatives" can land inside `pos_items` (**true positives labeled negative**) → noisy training. (2) uniform sampling ignores popularity → popular items under-penalized.
**Fix:** reject items in `pos_items`; consider popularity-aware sampling.

**54. Temporal split for recsys** · `M`
```python
def split_interactions(df):
    return train_test_split(df, test_size=0.2, random_state=1)
```
**Bugs:** (1) a random split lets the model **see future interactions** → leakage; offline metrics won't hold online. (2) should be a global-timestamp cut or per-user **leave-last-out**.
**Fix:** time-based split; leave-last-interaction-out per user for ranking eval.

**55. Popularity baseline** · `M`
```python
def popularity_recommend(df, n=10):
    pop = df['item'].value_counts().index[:n]
    return list(pop)
```
**Bugs:** (1) popularity computed over the **whole df incl. test** → leakage; use train only. (2) the same list is returned to **every user** without removing items they've already seen. (3) ignores recency (stale-popularity).
**Fix:** train-only popularity, exclude seen per user, optionally time-decay.

**56. recall@k** · `H`
```python
def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(recommended)
```
**Bugs:** (1) the denominator is **`len(recommended)`** — that's a precision-ish quantity mislabeled as recall; recall divides by `len(relevant)`. (2) **division by zero** when `relevant` is empty. (3) duplicates in `recommended` aren't handled.
**Fix:** `hits / len(relevant)` guarding the empty case; dedupe recommendations.

**57. Session next-item sequences** · `M`
```python
def make_sequences(sessions):
    X, y = [], []
    for s in sessions:
        X.append(s)
        y.append(s[-1])
    return X, y
```
**Bugs:** (1) the target `s[-1]` is **also inside the input `s`** → trivial leakage; the input should be `s[:-1]`. (2) length-1 sessions produce an empty input; no minimum-length filter.
**Fix:** input `s[:-1]`, target `s[-1]`, drop sessions shorter than 2.

**58. Cold-start serving** · `M`
```python
def get_recommendations(user_id, model):
    user_emb = model.user_embeddings[user_id]
    scores = user_emb @ model.item_embeddings.T
    return np.argsort(-scores)[:10]
```
**Bugs:** (1) a new `user_id` isn't in `user_embeddings` → **KeyError** at serve; needs a fallback. (2) same risk for new items. (3) doesn't exclude items the user has already seen.
**Fix:** try/except → popularity/content fallback (or an UNK embedding), mask seen items.

**59. Business-rule re-ranking** · `M`
```python
def rerank(candidates, out_of_stock, n=10):
    top = candidates[:n]
    return [c for c in top if c not in out_of_stock]
```
**Bugs:** (1) it **truncates to `n` before filtering** → after removing out-of-stock items you return fewer than `n`; filter first, then take `n`. (2) `c not in out_of_stock` on a **list** is O(m) → O(n·m).
**Fix:** filter over the full candidate list then take top-`n`; use a `set` for the lookup.

**60. Diversity re-ranking (MMR)** · `M`
```python
def diversify(items, sim_matrix, k, lam=0.5):
    selected = [items[0]]
    while len(selected) < k:
        best, best_score = None, -np.inf
        for it in items:
            if it in selected:
                continue
            div = max(sim_matrix[it][s] for s in selected)
            score = lam * relevance[it] + (1 - lam) * div
            if score > best_score:
                best_score, best = score, it
        selected.append(best)
    return selected
```
**Bugs:** (1) MMR should **subtract** the max similarity to already-selected items (penalize redundancy) — this **adds** it, rewarding similarity → produces the *least* diverse set (sign error). (2) `relevance` is referenced but never passed in. (3) O(k·n·|selected|) recompute — cache each candidate's max-sim. (4) `it in selected` list membership is O(n).
**Fix:** `score = lam*rel - (1-lam)*div`, pass `relevance` in, cache max-sim, use a set.

---

## Domain 6 — Time Series & Forecasting (61–72)

**61. Forecasting split + scaling** · `M`
```python
def prepare(df):
    scaler = MinMaxScaler()
    df['y'] = scaler.fit_transform(df[['y']])
    train, test = train_test_split(df, test_size=0.2, shuffle=True)
    return train, test, scaler
```
**Bugs:** (1) `shuffle=True` on time series → **future leaks into train**. (2) the scaler is fit on **all data** → leakage; fit on the train window only. (3) scaling the target in place means you must **inverse-transform** for metrics.
**Fix:** chronological split, fit scaler on train, invert before scoring.

**62. Lag features** · `M`
```python
def add_lags(df, lags=[1, 7, 28]):
    for L in lags:
        df[f'lag_{L}'] = df['y'].shift(-L)
    return df.fillna(method='bfill')
```
**Bugs:** (1) `shift(-L)` pulls **future** values into the features (a negative shift) → leakage; it should be `shift(+L)`. (2) not grouped by series → lags cross entity boundaries. (3) `bfill` fills with **future** values → more leakage.
**Fix:** positive shift, group by series, avoid `bfill` (drop leading rows or ffill from the past).

**63. Rolling z-score** · `H`
```python
def rolling_feats(df):
    df['roll_mean_7'] = df['y'].rolling(7).mean()
    df['roll_std_7']  = df['y'].rolling(7).std()
    df['zscore'] = (df['y'] - df['roll_mean_7']) / df['roll_std_7']
    return df
```
**Bugs:** (1) the rolling window **includes the current row**, so the z-score uses `y` in both the value and its own mean/std → **leakage** when predicting `y`; shift the rolling stats by 1. (2) not sorted / not grouped by series. (3) `roll_std_7` is **0** on a constant window → division by zero → inf.
**Fix:** `.shift(1)` the rolling features, sort + group, guard zero std.

**64. Differencing + forecast** · `M`
```python
def difference_and_forecast(series, model):
    diff = series.diff().dropna()
    model.fit(diff)
    forecast_diff = model.predict(steps=10)
    return forecast_diff
```
**Bugs:** (1) the forecast is in **differenced space** but returned as if it were levels → must cumulatively add back the last observed level. (2) `dropna` shifts alignment; the base level for inversion must be tracked.
**Fix:** invert differencing (`last_level + cumsum(forecast_diff)`), keep the base value.

**65. Seasonal decomposition** · `M`
```python
def deseasonalize(series):
    result = seasonal_decompose(series, period=7)
    return series - result.seasonal
```
**Bugs:** (1) **additive** removal is assumed; if seasonality scales with level (multiplicative), subtraction is wrong → use `model='multiplicative'` or log first. (2) `period=7` is hardcoded — wrong for non-daily data. (3) decomposition over the **entire series** (incl. future) leaks in a forecasting pipeline.
**Fix:** pick additive/multiplicative from the data, set the correct period, estimate seasonality on train only.

**66. Walk-forward backtest** · `M`
```python
def backtest(series, model, horizon=7):
    errors = []
    for i in range(len(series) - horizon):
        train = series[:i]
        model.fit(train)
        pred = model.predict(horizon)
        actual = series[i:i+horizon]
        errors.append(mean_squared_error(actual, pred))
    return np.mean(errors)
```
**Bugs:** (1) `series[:i]` is **empty for small `i`** → the fit crashes; start at a minimum training size. (2) refits from scratch every step → O(n) fits (expensive). (3) `pred` length vs `actual` length can **mismatch at the tail** (last window shorter than `horizon`). (4) no gap between train end and forecast start if features use windows → seam leakage.
**Fix:** set a minimum train size, align lengths, consider incremental refit cadence, purge the seam.

**67. Fill missing timestamps** · `M`
```python
def fill_gaps(df):
    df = df.set_index('date')
    df = df.resample('D').mean()
    return df.fillna(0)
```
**Bugs:** (1) filling gaps with **0** conflates "no data" with "zero demand" → biases models; use interpolation/ffill or a missing flag. (2) `resample('D').mean()` mixes multiple stacked series. (3) original NaNs vs newly created gaps become indistinguishable after the fill.
**Fix:** interpolate/ffill or add a missing indicator, group by series, distinguish structural zeros from missing.

**68. Recursive multi-step forecast** · `M`
```python
def recursive_forecast(model, last_window, steps):
    preds = []
    window = list(last_window)
    for _ in range(steps):
        p = model.predict([window])[0]
        preds.append(p)
        window = window[1:] + [p]
    return preds
```
**Bugs:** (1) feeding predictions back is standard, but the code ignores **error compounding** and never widens uncertainty → overconfident far-horizon forecasts (report intervals or use direct/seq2seq). (2) exogenous features aren't updated per step. (3) `window[1:] + [p]` assumes a lag-1 window only; higher-order features aren't maintained.
**Fix:** acknowledge compounding (prediction intervals), update exogenous inputs, or use direct multi-output forecasting.

**69. Residual anomaly detection** · `M`
```python
def detect_anomalies(series, model):
    fitted = model.predict(series)
    resid = series - fitted
    threshold = 3 * resid.std()
    return series[np.abs(resid) > threshold]
```
**Bugs:** (1) `resid.std()` is computed **including the anomalies themselves** → an inflated threshold that masks them; use a robust scale (MAD) or clean/train residuals. (2) a global std ignores seasonal/heteroscedastic residual variance. (3) fitting and detecting on the same series → in-sample.
**Fix:** robust scale (MAD), rolling/seasonal residual std, out-of-sample residuals.

**70. Event/holiday features** · `M`
```python
def add_events(df, events):
    return df.merge(events, on='date', how='left')
```
**Bugs:** (1) if `events` carries information **not known at prediction time** (e.g., realized promo uplift), that's leakage — only use forward-knowable flags. (2) duplicate dates in `events` cause **row fan-out**. (3) non-event days get NaN → downstream fill needed.
**Fix:** only forward-known event flags, enforce event-table uniqueness, fill non-event indicators.

**71. Global scaling across series** · `M`
```python
def scale_all(df):
    scaler = StandardScaler()
    df['y_scaled'] = scaler.fit_transform(df[['y']])
    return df
```
**Bugs:** (1) a **single global scaler** across many SKUs with different magnitudes distorts small-volume series; scale per series/group or use relative features. (2) fit over all rows incl. future → leakage.
**Fix:** per-series scaling fit on that series' train window (or use log/ratios).

**72. Forecast error metric** · `M`
```python
def forecast_error(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100
```
**Bugs:** (1) **divide-by-zero** when `actual == 0` (intermittent demand); explodes for small actuals; asymmetric (penalizes over- vs under-forecast unevenly). (2) ignores **bias** (systematic over/under).
**Fix:** WAPE/sMAPE/MASE, mask zeros, and also report bias/ME.

---

## Domain 7 — NLP / Embeddings / RAG / LLM (73–84)

**73. TF-IDF text classifier** · `M`
```python
def train_text_clf(texts, labels):
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    X_tr, X_te, y_tr, y_te = train_test_split(X, labels)
    clf = LogisticRegression().fit(X_tr, y_tr)
    return clf, vec
```
**Bugs:** (1) the vectorizer is **fit on all texts before the split** → the vocabulary and IDF weights leak from test. (2) no `random_state`, no `stratify`. (3) default preprocessing (stopwords/ngrams) may not suit the domain.
**Fix:** split first, fit the vectorizer on train, transform test; stratify + seed; wrap in a `Pipeline`.

**74. Embedding semantic search** · `M`
```python
def search(query, docs, doc_embeddings, model):
    q_emb = model.encode(query)
    scores = doc_embeddings @ q_emb
    return np.argsort(scores)[-5:]
```
**Bugs:** (1) the dot product assumes **normalized** embeddings; unnormalized → magnitude bias. (2) `argsort[-5:]` returns results in **ascending** order (worst-to-best within the top 5) → reverse for ranking. (3) assumes `doc_embeddings` came from the **same model** as `model.encode` — a mismatch silently ruins retrieval.
**Fix:** normalize (or cosine), `[::-1]`, guarantee the same embedding model/version.

**75. Document chunking** · `M`
```python
def chunk(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]
```
**Bugs:** (1) **character** windows split words/sentences mid-way with **no overlap** → context is lost across boundaries → poor retrieval units. (2) no token counting → chunks can exceed the embedding model's token limit. (3) ignores document structure (headers/paragraphs).
**Fix:** token-aware chunking on sentence/paragraph boundaries with overlap; respect token limits.

**76. RAG prompt assembly** · `H`
```python
def build_prompt(query, retrieved):
    context = "\n\n".join(d['text'] for d in retrieved)
    return f"Answer using the context.\nContext: {context}\nQuestion: {query}"
```
**Bugs:** (1) **no token budget** → the concatenated context can exceed the model's window and get silently truncated (often losing the most relevant chunk if unranked). (2) **no dedup** of near-duplicate chunks → wasted budget. (3) **prompt injection** — retrieved (possibly user-controlled) text is concatenated into the instructions and can override "answer using the context." (4) no "**say you don't know** if it's not in the context" → hallucination. (5) no citations/provenance.
**Fix:** rerank + dedup + enforce a token budget, delimit and defend untrusted context, add an abstention instruction, include source ids.

**77. RAG retrieval** · `M`
```python
def retrieve(query, index, embed_model, k=3):
    q = embed_model.encode([query])[0]
    return index.search(q, k)
```
**Bugs:** (1) `k=3` may be too small for recall and there's **no reranking** — first-stage dense retrieval alone misses lexical matches (consider hybrid BM25 + dense). (2) if the `index` was built with a **different embedding model**, the spaces don't match. (3) no normalization if the index expects cosine.
**Fix:** retrieve a larger `k` then rerank; hybrid retrieval; consistent model + normalization.

**78. LLM generation params** · `M`
```python
def extract_fields(llm, prompt):
    return llm(prompt, temperature=0.9, max_tokens=None)
```
**Bugs:** (1) `temperature=0.9` for **structured extraction** → unstable, hallucinated fields; use ~0. (2) `max_tokens=None` → unbounded generation (cost/latency, possible truncation of the wrapper). (3) no stop sequences / schema enforcement → parsing failures.
**Fix:** temperature ≈ 0, set `max_tokens`, enforce a JSON schema, add stops + validation.

**79. RAG evaluation** · `H`
```python
def evaluate_rag(system, qa_pairs):
    correct = 0
    for q, gold in qa_pairs:
        ans = system.answer(q)
        if ans == gold:
            correct += 1
    return correct / len(qa_pairs)
```
**Bugs:** (1) **exact string match** on free-form answers → near-zero and meaningless. (2) it doesn't separately measure **retrieval quality** (recall@k) and **faithfulness/groundedness** (is the answer supported by the retrieved context?). (3) a single number hides *where* the pipeline fails (retrieval vs generation).
**Fix:** multi-layer eval — retrieval recall@k, groundedness via NLI-against-context, answer relevance via LLM-as-judge, plus periodic human eval; decompose retrieval vs generation errors.

**80. Hallucination mitigation** · `M`
```python
def answer(llm, query, context):
    prompt = f"Context: {context}\nQ: {query}\nA:"
    return llm(prompt, temperature=1.0)
```
**Bugs:** (1) no instruction to **abstain** when the context lacks the answer → confident fabrication. (2) high temperature raises hallucination. (3) no **citation/verification** step to check claims against the context.
**Fix:** instruct "answer only from the context, otherwise say you don't know," lower temperature, verify claims against retrieved passages (NLI/self-consistency).

**81. Tokenizer batch encoding** · `M`
```python
def encode_batch(tokenizer, texts):
    return tokenizer(texts, truncation=True, max_length=128, return_tensors='pt')
```
**Bugs:** (1) `truncation=True, max_length=128` **silently drops** long inputs → lost content; measure/log the truncation rate and chunk long docs. (2) **no `padding=True`** → variable lengths can't batch (or error); you need padding and an **attention mask**. (3) padding side / special tokens are model-specific.
**Fix:** add padding, use the returned attention mask, monitor truncation, chunk long documents.

**82. Fine-tuning data prep** · `M`
```python
def make_finetune_data(df):
    examples = [{'prompt': r['text'], 'completion': r['label']} for _, r in df.iterrows()]
    random.shuffle(examples)
    train = examples[:int(0.9*len(examples))]
    val = examples
    return train, val
```
**Bugs:** (1) `val = examples` includes the **training set** → the validation split overlaps train → inflated eval; it should be the held-out 10%. (2) no separator/EOS token → generation won't stop; completion formatting matters. (3) no class-balance / dedup / leakage check.
**Fix:** `val = examples[int(0.9*len(examples)):]`, add separators/EOS, dedupe, check balance.

**83. Entity postprocessing** · `M`
```python
def extract_entities(tokens, tags):
    entities = []
    for tok, tag in zip(tokens, tags):
        if tag != 'O':
            entities.append(tok)
    return entities
```
**Bugs:** (1) **subword tokens** (`##ing`, `▁word`) are appended raw → fragmented entities; merge subwords using offset mapping / `word_ids`. (2) the **BIO scheme is ignored** → adjacent entities get merged or split incorrectly. (3) no span boundaries returned.
**Fix:** reconstruct words via word-piece offsets, respect B-/I- tags, return spans.

**84. Near-duplicate document dedup** · `M`
```python
def dedup_docs(embeddings, threshold=0.9):
    keep = []
    for i in range(len(embeddings)):
        dup = False
        for j in keep:
            if embeddings[i] @ embeddings[j] > threshold:
                dup = True; break
        if not dup:
            keep.append(i)
    return keep
```
**Bugs:** (1) an **unnormalized dot** compared to a cosine-like `0.9` is magnitude-dependent and inconsistent → normalize. (2) **O(n²)** exact all-pairs → slow for large corpora; use ANN/LSH/MinHash. (3) greedy first-seen keep doesn't handle near-duplicate transitivity.
**Fix:** normalize (cosine), approximate near-dup detection (LSH), define a dedup policy.

---

## Domain 8 — Statistics, A/B Testing & Causal (85–94)

**85. A/B test analysis** · `H`
```python
def analyze_ab(control, treatment):
    from scipy import stats
    t, p = stats.ttest_ind(control, treatment)
    if p < 0.05:
        return "significant"
    return "not significant"
```
**Bugs:** (1) reports significance with **no effect size or CI** — a trivial effect is "significant" at large n. (2) if called repeatedly during the test (**peeking**), Type I error inflates far above 5%. (3) `ttest_ind` defaults to **equal variance** (`equal_var=True`) → use Welch's. (4) for **ratio metrics** (CTR per user) or non-normal data, a raw t-test is wrong → delta method / bootstrap. (5) no sample-ratio-mismatch (SRM) check.
**Fix:** report effect + CI, fix the horizon or use a sequential test, Welch's t-test, the right method for ratio metrics, an SRM check.

**86. Sample-size calc** · `M`
```python
def sample_size(mde, alpha=0.05):
    z = 1.96
    return int((z / mde) ** 2)
```
**Bugs:** (1) the formula omits **variance/baseline rate** and the **power (β)** term → it drastically underestimates n. Correct form: `n ≈ 2σ²(z_{α/2}+z_β)² / mde²`. (2) `z=1.96` is hardcoded, ignoring `alpha` and one- vs two-sidedness. (3) no adjustment for multiple variants.
**Fix:** include variance and the power term, compute z from `alpha` (and sidedness), adjust for variants.

**87. Segment scan** · `M`
```python
def find_significant_segments(df, segments):
    sig = []
    for seg in segments:
        sub = df[df['segment'] == seg]
        _, p = stats.ttest_ind(sub['control'], sub['treatment'])
        if p < 0.05:
            sig.append(seg)
    return sig
```
**Bugs:** (1) many tests at α=0.05 → **no multiple-comparison correction** (expect false positives). (2) post-hoc segment slicing is **data dredging** — pre-register or treat as exploratory. (3) small segments → low power / assumption violations.
**Fix:** apply Bonferroni/BH-FDR, pre-specify segments, guard small n.

**88. CI for a mean** · `M`
```python
def mean_ci(data):
    m = np.mean(data)
    s = np.std(data)
    return (m - 1.96 * s, m + 1.96 * s)
```
**Bugs:** (1) a CI for the **mean** uses the **standard error** `s/√n`, not the raw SD → this interval is far too wide. (2) `np.std` defaults to `ddof=0` (biased for a sample) → use `ddof=1`. (3) `1.96` assumes large n / known variance → use t for small n.
**Fix:** `SE = np.std(data, ddof=1)/np.sqrt(len(data))`, use the t-distribution for small samples.

**89. Correlation → causation** · `M`
```python
def feature_target_relationship(df, feature, target):
    corr = df[feature].corr(df[target])
    if abs(corr) > 0.5:
        return f"{feature} drives {target}"
    return "no relationship"
```
**Bugs:** (1) Pearson captures only **linear** association → a strong nonlinear/monotonic relationship reads as "no relationship." (2) "**drives**" asserts **causation** from correlation. (3) ignores confounders → possibly spurious.
**Fix:** use Spearman/MI, drop causal language, control for confounders.

**90. Bootstrap CI** · `M`
```python
def bootstrap_mean_ci(data, n_boot=50):
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, len(data), replace=False)
        means.append(np.mean(sample))
    return np.percentile(means, [2.5, 97.5])
```
**Bugs:** (1) `replace=False` makes each "resample" a **permutation** of the same data → every mean is identical → a zero-width CI. It must be `replace=True`. (2) `n_boot=50` is far too few for stable tail percentiles (use ≥1000–10000).
**Fix:** `replace=True`, raise `n_boot`.

**91. Chi-square independence** · `M`
```python
def independence_test(observed):
    from scipy.stats import chi2_contingency
    chi2, p, dof, exp = chi2_contingency(observed)
    return p < 0.05
```
**Bugs:** (1) the chi-square approximation needs **expected cell counts ≥ 5**; with sparse cells it's invalid — the code never checks `exp` (use Fisher's exact or combine categories). (2) it requires **raw counts**, not proportions/percentages. (3) returns only significance, no effect size (Cramér's V).
**Fix:** check expected counts, fall back to Fisher's exact when sparse, ensure raw counts, report effect size.

**92. Naive treatment effect** · `H`
```python
def treatment_effect(df):
    treated = df[df['treated'] == 1]['outcome'].mean()
    control = df[df['treated'] == 0]['outcome'].mean()
    return treated - control
```
**Bugs:** (1) an observational difference-in-means with **no confounder adjustment** is biased — treated and control differ systematically; this isn't a causal effect. (2) selection bias if treatment correlates with outcome drivers. (3) no standard error / CI.
**Fix:** adjust for confounders (regression/matching/propensity/IV/DiD), state the identification assumptions, report uncertainty.

**93. Concluding from p** · `M`
```python
def conclude(p_value):
    if p_value > 0.05:
        return "No effect exists"
    return "Effect exists"
```
**Bugs:** (1) "**No effect exists**" from p > 0.05 confuses *absence of evidence* with *evidence of absence* — the test may be underpowered. (2) a binary verdict ignores effect size and the CI (a wide CI including 0 is *inconclusive*, not null).
**Fix:** report the CI; call a wide one inconclusive; consider power; don't assert the null.

**94. Pooled conversion rate** · `H`
```python
def conversion_rate(df):
    return df['converted'].mean()
```
**Bugs:** (1) pooling across a confounder (e.g., traffic channel or segment) can **reverse** the within-group comparison — **Simpson's paradox** — so the aggregate misleads. (2) no stratification / weighting.
**Fix:** compute stratified rates, aggregate with appropriate weighting, watch for sign reversals.

---

## Domain 9 — MLOps, Serving, Reproducibility & Concurrency (95–106)

**95. Serving endpoint** · `H`
```python
def predict_endpoint(request):
    model = joblib.load('model.pkl')
    features = np.array(request['features'])
    return {'prediction': model.predict([features])[0]}
```
**Bugs:** (1) the model is **loaded on every request** → huge latency; load once at startup. (2) **no input validation** (missing/extra features, wrong order, NaN) → garbage or crash. (3) the fitted **preprocessing isn't applied** → train/serve skew. (4) no error handling / timeout / redacted logging.
**Fix:** load the model+pipeline once (module-level/cached), validate the schema, apply the fitted pipeline, handle errors.

**96. Feature parity** · `H`
```python
# training
train_df['ratio'] = train_df['a'] / train_df['b']
# serving
def serve_features(a, b):
    return {'ratio': a / (b + 1)}
```
**Bugs:** (1) the serving formula (`a/(b+1)`) **differs from training** (`a/b`) → systematic train/serve skew. (2) the two paths also handle `b == 0` differently (training → inf/NaN; serving → hidden).
**Fix:** a single shared feature definition (feature store / shared function), consistent zero-handling, parity tests.

**97. "Reproducible" experiment** · `M`
```python
def run_experiment(config):
    model = train(config)
    joblib.dump(model, 'model.pkl')
    print(f"Accuracy: {evaluate(model)}")
```
**Bugs:** (1) no seeding and no logging of **config / data version / git SHA / library versions** → not reproducible. (2) a fixed output path **overwrites** previous models — no versioning. (3) prints the metric instead of logging it to a tracker → no history.
**Fix:** seed everything, log config + versions + data hash, version artifacts with a unique run id, use an experiment tracker.

**98. Batch scoring** · `M`
```python
def score_all(df, model):
    df['score'] = model.predict(df[features])
    df.to_csv('scores.csv')
    return df
```
**Bugs:** (1) scores the **entire dataset in memory** → OOM at scale; chunk it. (2) no **idempotency/checkpointing** → a mid-job failure loses everything and re-runs duplicate. (3) `to_csv` isn't atomic → a partial file on crash. (4) `df[features]` may contain NaN/unseen categories → silent bad scores.
**Fix:** chunked scoring, checkpoint/upsert by key, atomic writes, input validation.

**99. Drift monitoring** · `M`
```python
def check_drift(reference, current):
    return abs(reference.mean() - current.mean()) > 0.1
```
**Bugs:** (1) comparing **means only** misses variance/shape shifts and per-feature drift → use KS/PSI per feature. (2) the threshold `0.1` is arbitrary and scale-dependent. (3) only input drift — no **prediction drift** or (label-based) **concept drift** monitoring.
**Fix:** per-feature PSI/KS with calibrated thresholds, monitor the prediction distribution and delayed label-based performance, add alerting.

**100. Concurrent prediction cache** · `H`
```python
cache = {}
def get_prediction(key, model, x):
    if key not in cache:
        cache[key] = model.predict(x)
    return cache[key]
```
**Bugs:** (1) the check-then-set is **not atomic** → under concurrency, duplicate computation and inconsistent writes (race condition); needs a lock or a thread-safe cache. (2) **unbounded growth** → memory leak; use a bounded LRU. (3) no TTL → stale predictions after a model update.
**Fix:** a thread-safe bounded cache (`cachetools.LRUCache` + lock or `functools.lru_cache`), with TTL/invalidation.

**101. Logging & secrets** · `M`
```python
API_KEY = "sk-live-abc123"
def handle(payload):
    logging.info(f"Received: {payload}")
    ...
```
**Bugs:** (1) a **hardcoded secret** in source leaks through version control → use env/secret manager. (2) logging the **raw payload** exposes PII → redact. (3) verbose logging can leak internals.
**Fix:** load secrets from a secret store, redact/allow-list logged fields, scrub PII.

**102. Deploy & rollback** · `M`
```python
def deploy(model):
    joblib.dump(model, '/models/production.pkl')
    reload_server()
```
**Bugs:** (1) it **overwrites the live model in place** with no version/rollback → you can't revert a bad deploy; no gradual rollout (canary/shadow). (2) `reload_server()` mid-traffic drops requests; no **validation gate/health check** before promotion. (3) no record of which model version served which prediction.
**Fix:** versioned artifacts + a registry, canary/shadow rollout, a validation gate, instant rollback, log the model version per prediction.

**103. Retraining trigger** · `H`
```python
def maybe_retrain(new_data, model):
    new_data['label'] = model.predict(new_data[features])
    return train(pd.concat([old_data, new_data]))
```
**Bugs:** (1) it trains on the **model's own predictions as labels** → a self-reinforcing feedback loop that entrenches bias and drifts; use ground-truth labels. (2) retrains on any new data with **no trigger/validation gate** → can deploy a worse model. (3) ignores label delay.
**Fix:** use verified labels, champion/challenger validation before promotion, drift + performance triggers, account for label latency.

**104. Config via global** · `M`
```python
CONFIG = {}
def set_config(**kwargs):
    CONFIG.update(kwargs)
def get_lr():
    return CONFIG['lr']
```
**Bugs:** (1) a **mutable global** config is hidden state — hard to test, prone to races. (2) no defaults/validation → `KeyError` if unset; env vars arrive as **strings** (`'0.01'`) → type errors downstream. (3) no schema.
**Fix:** an immutable, typed config (dataclass/pydantic) with defaults and validation, passed explicitly rather than global.

**105. Parallel processing** · `M`
```python
results = []
def worker(chunk):
    results.append(process(chunk))

with ThreadPoolExecutor() as ex:
    ex.map(worker, chunks)
```
**Bugs:** (1) appending to a **shared list from multiple threads** → race conditions / lost updates, and the order is nondeterministic. (2) the `ex.map` return value is ignored, so worker **exceptions are swallowed** until the iterator is consumed. (3) if `process` is CPU-bound, threads don't help (GIL) → use processes.
**Fix:** collect via the `ex.map` return (ordered) or a thread-safe queue, surface worker exceptions, use processes for CPU-bound work.

**106. Validation gate** · `M`
```python
def validate(df):
    assert len(df) > 0
    return df
```
**Bugs:** (1) **`assert` is stripped by `python -O`** → validation silently vanishes in production; raise a real exception instead. (2) it only checks non-empty — no **schema/dtype/range/null** validation. (3) returns the df regardless of subtler issues.
**Fix:** explicit checks that raise (or a schema lib like pandera/Great Expectations) covering dtypes, ranges, nulls, and cardinality.

---

## Domain 10 — SQL & Big Data / Spark (107–116)

**107. Revenue with joins** · `H` *(SQL)*
```sql
SELECT c.region, SUM(o.amount) AS revenue
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
JOIN order_items i ON o.id = i.order_id
WHERE o.status = 'completed'
GROUP BY c.region;
```
**Bugs:** (1) `WHERE o.status='completed'` on the LEFT-joined `orders` **converts the LEFT JOIN into an INNER JOIN** — regions with no completed orders disappear (move it to `ON` to keep them). (2) joining `order_items` **fans out and multiplies `o.amount`** once per item → inflated revenue. (3) no `COALESCE` on the sum → NULL for regions with no orders.
**Fix:** filter in `ON`, avoid fan-out (aggregate items first or sum at the order grain), `COALESCE(SUM(...), 0)`.

**108. Deduplication** · `M` *(SQL)*
```sql
SELECT DISTINCT customer_id, name, email
FROM customers;
```
**Bugs:** (1) `DISTINCT` over **multiple columns** dedupes only exact triples — the same `customer_id` with differing name/email casing or whitespace still yields duplicates. (2) it papers over the root cause of duplication.
**Fix:** `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY updated_at DESC)` and keep `rn=1`; normalize before comparing.

**109. Running total** · `M` *(SQL)*
```sql
SELECT date, amount,
       SUM(amount) OVER (ORDER BY date) AS running_total
FROM sales;
```
**Bugs:** (1) with **duplicate dates**, the default `RANGE` frame sums **all peer rows sharing a date** together rather than row-by-row → unexpected jumps. (2) no partition if a per-customer total was intended.
**Fix:** `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` + a unique tiebreaker in `ORDER BY`; `PARTITION BY` as needed.

**110. Top-N per group** · `M` *(SQL)*
```sql
SELECT category, product, sales
FROM products
WHERE sales IN (SELECT MAX(sales) FROM products GROUP BY category);
```
**Bugs:** (1) `sales IN (SELECT MAX(sales) ... GROUP BY category)` matches a product whose sales equals **any** category's max → cross-category false matches (a product in category A with sales equal to B's max sneaks in). (2) ties return multiple rows.
**Fix:** `ROW_NUMBER()/RANK() OVER (PARTITION BY category ORDER BY sales DESC)` and filter `rn=1`; decide the tie policy.

**111. Date range filter** · `M` *(SQL)*
```sql
SELECT * FROM events
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
```
**Bugs:** (1) on timestamps, `'2024-01-31'` is **midnight**, so most of Jan 31 is excluded. (2) timezone mismatch if `created_at` is UTC and the boundaries are local.
**Fix:** half-open range `>= '2024-01-01' AND < '2024-02-01'`, explicit timezone handling.

**112. NULL comparison** · `M` *(SQL)*
```sql
SELECT * FROM orders
WHERE discount_code != 'SUMMER';
```
**Bugs:** (1) rows where `discount_code IS NULL` are **excluded** — `NULL != 'SUMMER'` evaluates to `NULL`, not true — so NULL-code orders you probably want silently vanish.
**Fix:** `WHERE discount_code IS DISTINCT FROM 'SUMMER'` (or add `OR discount_code IS NULL`).

**113. Spark top customers** · `M` *(PySpark)*
```python
def get_top_customers(df):
    rows = df.orderBy(F.desc('revenue')).collect()
    return rows[:100]
```
**Bugs:** (1) `collect()` **pulls the entire DataFrame to the driver** before slicing → OOM; only 100 rows are needed. (2) `orderBy` triggers a full shuffle — fine, but combine with a limit push-down.
**Fix:** `df.orderBy(F.desc('revenue')).limit(100).collect()` (or `take(100)`).

**114. Spark UDF** · `M` *(PySpark)*
```python
@F.udf('double')
def to_usd(amount, rate):
    return amount * rate
df = df.withColumn('usd', to_usd('amount', 'rate'))
```
**Bugs:** (1) a **Python UDF** serializes every row to Python and back → kills performance and disables Catalyst optimization; this is plain arithmetic → use native columns. (2) no null handling.
**Fix:** `df.withColumn('usd', F.col('amount') * F.col('rate'))` (or a `pandas_udf` if a real Python lib is required); handle nulls.

**115. Spark skewed join** · `H` *(PySpark)*
```python
result = large_df.join(skewed_df, 'key')
```
**Bugs:** (1) **data skew** — if one `key` holds most rows, a single reducer task straggles → the job hangs or OOMs. (2) a large-to-large join without broadcast when one side is actually small enough to broadcast.
**Fix:** `F.broadcast` the small side, or **salt** the hot key / enable Adaptive Query Execution (skew-join handling).

**116. Spark caching** · `M` *(PySpark)*
```python
df2 = expensive_transform(df)
count = df2.count()
df2.write.parquet('out/')
```
**Bugs:** (1) `df2` is **recomputed** for both the `count` and the `write` (the lazy DAG re-executes) → double work; cache/persist before reusing. (2) no `unpersist()` afterward → memory held.
**Fix:** `df2.persist()` before the first action, `unpersist()` when done.

---

## Domain 11 — Numerical Methods & Algorithms (117–124)

**117. Pairwise distances** · `H`
```python
def pairwise_distances(X):
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i] - X[j])
    return D
```
**Bugs:** (1) a double Python loop with per-pair `norm` → **O(n²·d)** when it should be vectorized (`scipy.spatial.distance.cdist` or the `‖a‖²+‖b‖²−2a·b` trick). (2) computes both `D[i,j]` and `D[j,i]` (symmetric) → 2× work plus the zero diagonal. (3) the full `n×n` matrix may not fit in memory for large n.
**Fix:** `cdist` / the dot-product identity, exploit symmetry, chunk (or use ANN) for large n.

**118. Softmax** · `M`
```python
def softmax(logits):
    e = np.exp(logits)
    return e / e.sum()
```
**Bugs:** (1) **no max-subtraction** → `exp` overflows for large logits → inf/NaN. (2) for a 2D batch, `e.sum()` reduces over **all** elements, not per row → wrong normalization.
**Fix:** subtract `logits.max(axis=-1, keepdims=True)`, then sum over the class axis with `keepdims=True`.

**119. Manual log-loss** · `M`
```python
def log_loss_manual(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```
**Bugs:** (1) `np.log(p)` / `np.log(1-p)` are **−inf/NaN** when `p` is exactly 0 or 1 → clip `p` to `[eps, 1-eps]`. (2) no shape/label validation (`y` must be 0/1).
**Fix:** `p = np.clip(p, 1e-15, 1 - 1e-15)`; validate inputs.

**120. Row normalization** · `M`
```python
def normalize_rows(X):
    return X / X.sum(axis=1)
```
**Bugs:** (1) `X.sum(axis=1)` has shape `(n,)`; dividing `(n,d)/(n,)` **broadcasts wrong or errors** — you need `keepdims=True` → `(n,1)`. (2) rows summing to 0 → division by zero → inf/NaN.
**Fix:** `X / X.sum(axis=1, keepdims=True)`; guard zero rows (add eps or handle separately).

**121. Moving average** · `M`
```python
def moving_average(x, w):
    result = []
    for i in range(len(x)):
        result.append(np.mean(x[i:i+w]))
    return result
```
**Bugs:** (1) tail windows `x[i:i+w]` **shrink** near the end (fewer than `w` elements) → inconsistent averaging; decide padding/validity. (2) `np.mean` over slices → **O(n·w)** when a cumulative-sum sliding window is O(n). (3) it's a **forward** window (uses future) — if used as a feature, it leaks.
**Fix:** cumulative-sum sliding window (O(n)), explicit edge handling, use a trailing window for features.

**122. Percentage helper** · `M`
```python
def compute_percentage(part, whole):
    return part / whole * 100
```
**Bugs:** (1) `whole == 0` → **ZeroDivisionError** for ints (floats give inf). (2) fragile if reused with `//` in another context. (3) no rounding/format control.
**Fix:** guard `whole == 0` (return 0/NaN per policy), ensure float, control rounding.

**123. Sort by score** · `M`
```python
def sort_by_score(items):
    for i in range(len(items)):
        for j in range(len(items) - 1):
            if items[j]['score'] < items[j+1]['score']:
                items[j], items[j+1] = items[j+1], items[j]
    return items
```
**Bugs:** (1) a hand-rolled **bubble sort → O(n²)** when the built-in `sorted(..., reverse=True)` is O(n log n) and stable. (2) it **mutates the input** list in place (side effect). (3) tie ordering isn't the stable order callers usually expect.
**Fix:** `sorted(items, key=lambda d: d['score'], reverse=True)`; don't mutate the input.

**124. Variance** · `M`
```python
def variance(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n
```
**Bugs:** (1) divides by `n` (**population variance, ddof=0**) — a sample estimate uses `n-1`. (2) two passes over `data`; if `data` is a **generator**, the second pass is empty → wrong/zero result (the iterator is consumed). (3) `n == 0`/`n == 1` → division by zero / undefined. (4) for large streams, prefer **Welford's** one-pass algorithm (stability + single pass).
**Fix:** `n-1` for a sample, materialize or use Welford's, guard small n.

---

## How to score these

Score the **process first, findings second** — a candidate who narrates a repeatable sweep and misses one bug is stronger than one who spots two by luck and stops.

| Signal | Weak (1–2) | Strong senior (4–5) |
|---|---|---|
| **Search pattern** | Finds one bug, declares done | Sweeps in order: correctness → edge cases (empty/NaN/zero) → dtype/overflow → complexity → **leakage** → **train/serve parity** |
| **Leakage instinct** | Doesn't consider it | Flags scale/impute/encode-before-split, target leakage, temporal/group leakage unprompted, and explains the mechanism |
| **Complexity** | Ignores it | States Big-O, spots the O(n²)/recompute-in-loop, proposes the right structure (heap/set/vectorize/cache) |
| **Correctness depth** | Surface bugs only | Catches the *subtle* one — desynced parallel lists, KL sign, clip-after-step, empty-cluster NaN, fan-out |
| **Fix quality** | "Rewrite it" | Minimal, correct fix **plus** how they'd test it (unit test / assertion / property / a tiny repro) |
| **Curiosity** | Accepts the code as-is | Asks what the inputs look like, what breaks at scale, why it was written this way before rewriting |

**Recommended use.** Pick 3–4 across different domains for a 45-minute round. Lead with one `M` from the candidate's home turf to build rapport, then one `H` from an adjacent area (a forecasting person gets a PyTorch or RAG one) to see how they reason on unfamiliar code. The **planted-bug counts** make these easy to grade consistently: tell them roughly how many issues to find, or leave it open and see how deep they dig.

**The tells that separate levels.** Mid-levels find the crash and the obvious logic error. Seniors find the **silent** ones — the leakage that inflates the metric, the input mutation that corrupts a caller, the fan-out that doubles revenue, the recompute that melts the cluster — and they say how they'd catch it in CI so it never ships again.
