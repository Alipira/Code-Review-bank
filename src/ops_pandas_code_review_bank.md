# Senior DS — Ops & Pandas Code Review Bank (124 Functions)

> **What this is.** A code-review bank in the exact style of the `reassign_clusters_vehicles` review: realistic business/ops functions a junior would ship — DataFrame-heavy, docstrings included — each hiding **several interacting bugs**: logic errors, pandas antipatterns, stale state, contract lies, and performance disasters. The candidate reviews for correctness first, performance second, and must say how they'd *prove* each bug.
>
> **How to run one.** Paste the function: "This is in production. Review it — correctness bugs, then performance, then rewrite the hot path." Every listed bug is actually present in the code. Strong candidates trace the data flow before judging style; the best ones name the test that would have caught each bug.
>
> **Difficulty:** `E` screen · `M` core round · `H` senior signal.

---

## A — Assignment & Swap Heuristics (1–14)

**1. The rebalancer (anchor — sibling of the seed)** · `H`
```python
async def rebalance_loads(fleet_info: list[dict], factor: float = 1.05):
    """Swap vehicles between clusters until no cluster exceeds capacity.
    Does not modify the input. Returns the balanced DataFrame."""
    df = pd.DataFrame(fleet_info)
    improved = True
    while improved:
        improved = False
        bad = df[df.cluster_weight > df.vehicle_capacity * factor]
        ok = df[df.cluster_weight <= df.vehicle_capacity * factor]
        for i in bad.index:
            for j in ok.index:
                ci, cj = df.loc[i, 'cluster_weight'], df.loc[j, 'cluster_weight']
                vi, vj = df.loc[i, 'vehicle_capacity'], df.loc[j, 'vehicle_capacity']
                if vj * factor >= ci and vi * factor >= cj and vi != vj:
                    df.loc[i, 'vehicle_capacity'] = vj
                    df.loc[j, 'vehicle_capacity'] = vi
                    improved = True
                    break
    return df
```
**Bugs:**
1. `bad`/`ok` are snapshots taken once per outer pass but `df` mutates inside — later iterations test *stale* memberships (a row already fixed keeps being treated as violated, a formerly-ok row can now be violated and is never re-examined this pass).
2. Only `vehicle_capacity` is swapped: `car_id`, driver, and any other vehicle-linked column stay behind — the frame silently desynchronizes (the seed's one-sided-swap bug generalized).
3. `vi != vj` is a float inequality used as a "different vehicles" test — two distinct vehicles with equal capacity can never swap, and float noise makes it unreliable anyway; compare identities, not weights.
4. `async def` with zero `await`: callers who forget `await` get a coroutine object (truthy!) and no work happens; there's no async I/O here at all.
5. `break` exits only the inner loop; the outer `for i` keeps scanning with stale `bad` — combined with (1) this can loop the `while` far longer than needed, and with unlucky data oscillate two rows forever (no tabu/no-progress guard).
**Fix:** Drop `async`; recompute violation masks after every accepted swap (or work on plain arrays and restart the scan); swap the *vehicle identity* (all vehicle columns, or a vehicle_id then re-join attributes); track swapped pairs to prevent oscillation; guard with a max-iterations safety valve.
**Senior tell:** Spotting that the docstring lies twice — it *does* return a new frame, but nothing prevents oscillation, and "does not modify input" is only true by accident of `pd.DataFrame(list_of_dicts)` copying.

**2. Greedy truck packing** · `M`
```python
def pack_orders(orders: list[dict], trucks: list[dict]):
    """Assign each order to the first truck with room (first-fit-decreasing)."""
    orders.sort(key=lambda o: o['weight'], reverse=True)
    for o in orders:
        placed = False
        while not placed:
            for t in trucks:
                if t['load'] + o['weight'] < t['capacity']:
                    t['load'] += o['weight']
                    t.setdefault('orders', []).append(o['id'])
                    placed = True
                    break
    return trucks
```
**Bugs:**
1. An order heavier than every truck's remaining capacity never places → `while not placed` spins forever (no fallback, no error).
2. `< t['capacity']` rejects an exact fit; first-fit-decreasing conventionally allows `<=` — trucks ship with phantom slack.
3. `orders.sort(...)` mutates the caller's list in place — the function's signature suggests read-only input.
4. The `while` wrapper is dead weight even when placement succeeds — the `for` already scans all trucks; the loop only exists to hang.
**Fix:** Sort a copy; use `<=`; after the truck scan, handle the unplaceable case explicitly (raise, or open an overflow truck); delete the `while`.

**3. Two-opt that prefers worse** · `H`
```python
def improve_route(route: list[int], dist: np.ndarray):
    """2-opt: reverse segments while it shortens the route."""
    best = route_length(route, dist)
    improved = True
    while improved:
        improved = False
        for a in range(1, len(route) - 2):
            for b in range(a + 1, len(route) - 1):
                cand = route[:a] + route[a:b+1][::-1] + route[b+1:]
                delta = route_length(cand, dist) - best
                if delta > 0:
                    route, improved = cand, True
        # keep going with the shorter route
    return route
```
**Bugs:**
1. `delta > 0` accepts *longer* routes — the sign is flipped; the loop hill-climbs uphill until no worsening move exists (a local *maximum*).
2. `best` is never updated after accepting a candidate — even with the sign fixed, every later delta is measured against the original length, accepting moves that are worse than the current route.
3. Recomputing `route_length` per candidate is O(n) inside O(n²) pairs → O(n³) per pass; 2-opt's point is the O(1) delta from the four affected edges.
4. Mutating `route` mid-scan while `a`/`b` ranges were sized for the old list is safe only because lengths match — but the scan continues on a mix of old and new geometry within one pass (order-dependent results).
**Fix:** `delta < 0`; update `best += delta` on acceptance (or recompute once); compute delta from the two removed and two added edges; restart or cleanly continue the scan after acceptance.
**Senior tell:** Naming the O(1) edge-delta trick — anyone who has actually written 2-opt reaches for it immediately.

**4. Round-robin that isn't** · `M`
```python
def assign_drivers(deliveries: pd.DataFrame, drivers: list[str],
                   unavailable: set[str]):
    """Distribute deliveries evenly across available drivers, round-robin."""
    out = deliveries.copy()
    for i, idx in enumerate(deliveries.index):
        d = drivers[i % len(drivers)]
        if d in unavailable:
            continue
        out.loc[idx, 'driver'] = d
    return out
```
**Bugs:**
1. Unavailable drivers' turns are *skipped, not reassigned*: those deliveries keep `driver = NaN` silently — the docstring promises even distribution, the function delivers holes.
2. Modulo runs over the full `drivers` list, so available drivers don't absorb the skipped slots — distribution is uneven by construction (driver after an unavailable one gets no extra load).
3. `enumerate(deliveries.index)` couples assignment to current row order; same data sorted differently → different assignment (fine if intended, but nothing sorts deterministically first).
**Fix:** Filter to `avail = [d for d in drivers if d not in unavailable]` and cycle over that (`itertools.cycle` or `i % len(avail)`); sort deliveries by an explicit key first; assert no NaN drivers on exit.

**5. The oscillating stabilizer** · `H`
```python
def stabilize(assignments: list[dict]):
    """Swap pairs until total violation stops improving."""
    while True:
        swapped = False
        for a in assignments:
            for b in assignments:
                if a is b:
                    continue
                if violation(a) + violation(b) >= violation_after_swap(a, b):
                    a['cluster'], b['cluster'] = b['cluster'], a['cluster']
                    swapped = True
        if not swapped:
            break
    return assignments
```
**Bugs:**
1. `>=` accepts *zero-improvement* swaps — two rows with equal violation swap every pass, `swapped` stays True, the loop never terminates (the classic tie oscillation).
2. Every pair is visited twice ((a,b) and (b,a)); with `>=`, the second visit undoes the first — even non-tie swaps can ping-pong within a single pass.
3. Swaps mutate `a`/`b` mid-scan, so later `violation()` calls in the same pass evaluate a moving target; results depend on iteration order.
4. No max-pass guard: on pathological input this is an outage, not a slow function.
**Fix:** Strict `>` (require actual improvement, or an epsilon), iterate unordered pairs once (`itertools.combinations`), and cap passes; better, compute all improving swaps first, apply the best, repeat.

**6. Inconsistent violation tests** · `M`
```python
def fixable_violations(df: pd.DataFrame, factor: float):
    """Return violated clusters that could be fixed by an available bigger truck."""
    violated = df[df.cluster_weight >= df.capacity * factor]
    candidates = []
    for i in violated.index:
        need = df.loc[i, 'cluster_weight']
        bigger = df[(df.capacity * factor > need) & (~df.index.isin(violated.index))]
        if len(bigger):
            candidates.append(i)
    return candidates
```
**Bugs:**
1. Violation uses `>=` but fixability uses strict `>` — a truck whose factored capacity *equals* the need is rejected, though by the violation rule equality is acceptable. The two thresholds disagree at the boundary, so some fixable rows are reported unfixable.
2. `bigger` excludes violated rows as donors — but a violated row with a *larger* truck can be part of a fixing swap; the search space is silently smaller than the business rule.
3. `df[...]` recomputed inside the loop over every violated row: O(v·n) for what one vectorized comparison against the max non-violated capacity answers.
**Fix:** Pick one boundary convention (`weight > cap*factor` is a violation; `cap*factor >= weight` fixes it) and use it in both places; decide donor eligibility from the business rule, not convenience; precompute the donor capacity array once.
**Senior tell:** Boundary-consistency between paired conditions — the `<` vs `<=` pair — is checked by table, not by intuition.

**7. Worst-first that's best-first** · `M`
```python
def reassign_priority(df: pd.DataFrame, factor: float = 1.0):
    """Fix the WORST violations first; one swap per pass."""
    while True:
        df['excess'] = df.cluster_weight - df.capacity * factor
        bad = df[df.excess > 0].sort_values('excess')
        if bad.empty:
            return df
        i = bad.index[0]
        j = df[df.excess <= 0].capacity.idxmax()
        df.loc[[i, j], 'capacity'] = df.loc[[j, i], 'capacity'].values
```
**Bugs:**
1. `sort_values('excess')` ascending: `bad.index[0]` is the *smallest* violation — the docstring's "worst first" is implemented as best-first.
2. One swap per `while` pass, with the full mask/sort recomputed every pass: O(passes · n log n) where a single pass over sorted violations would do.
3. The swap gives row `i` the largest *free* capacity without checking it actually covers `excess[i]` — if it doesn't, row `i` remains violated with the same excess ranking and gets picked again forever: infinite loop on any unfixable worst row.
4. Helper column `'excess'` is written into the caller's frame (mutation as a side effect of "read-only" analysis) and shipped in the return.
**Fix:** `ascending=False` (or `idxmax`); verify the donor covers the need, else mark the row unfixable and exclude it; batch the pass; compute `excess` as a local Series, not a column.

**8. Two frames, one swap** · `H`
```python
def swap_pair(vehicles: pd.DataFrame, clusters: pd.DataFrame, i, j):
    """Swap vehicles between clusters i and j, keeping BOTH tables in sync."""
    vi = vehicles.loc[vehicles.cluster == i, 'vehicle_id']
    vj = vehicles.loc[vehicles.cluster == j, 'vehicle_id']
    if vi.empty or vj.empty:
        return vehicles, clusters
    vehicles.loc[vehicles.cluster == i, 'cluster'] = j
    vehicles.loc[vehicles.cluster == j, 'cluster'] = i
    clusters.loc[i, 'vehicle_id'] = vj.iloc[0]
    clusters.loc[j, 'vehicle_id'] = vi.iloc[0]
    return vehicles, clusters
```
**Bugs:**
1. Sequential relabel: the first write turns every cluster-`i` row into `j`; the second write then flips **all** of them (old `j` rows *and* the just-relabeled ones) back to `i` — net effect: cluster `j`'s vehicles move to `i` and cluster `i`'s vehicles… also end up at `i`. The swap is a merge.
2. `vi`/`vj` were snapshotted before the writes — using `vj.iloc[0]` afterwards happens to survive here, but only because of (1)'s ordering; the code's correctness depends on a bug canceling a race.
3. Multi-vehicle clusters: only `.iloc[0]` is written into `clusters`, silently dropping the rest — the two tables now disagree about membership (the desync the docstring promises to prevent).
**Fix:** Relabel via a temporary value or a vectorized map (`vehicles.cluster.map({i: j, j: i}).fillna(vehicles.cluster)`); store lists (or explode to a link table) rather than a single `vehicle_id` per cluster; assert both tables agree after the operation — that assertion is the missing test.

**9. Min-regret with tied minima** · `H`
```python
def min_regret_order(cost: np.ndarray):
    """Process rows in order of regret = (2nd best - best) assignment cost."""
    best = cost.min(axis=1)
    second = np.sort(cost, axis=1)[:, 1]
    regret = second - best
    order = np.argsort(regret)          # smallest regret first
    return order
```
**Bugs:**
1. Regret heuristics process **largest** regret first (the row that loses most if postponed); `argsort` ascending is exactly backwards.
2. With duplicated minima in a row (two equal best cells), `np.sort(...)[:, 1]` equals the best → regret 0, shoving genuinely-constrained rows to the *end* under the (already wrong) ordering; ties need the second *distinct* value or explicit tie policy.
3. No tie-break for equal regrets: `argsort` is not stable across dtypes/platforms by default (`kind='stable'` unspecified), so run-to-run order can differ — nondeterministic assignments downstream.
**Fix:** `np.argsort(-regret, kind='stable')` (or sort by `(-regret, row_id)`); define regret against the second distinct cost or accept 0 but document it; pin the tie-break.

**10. Waitlist promotion** · `M`
```python
def promote(waitlist: list[dict], slots: int):
    """Promote up to `slots` customers whose status is 'ready'."""
    promoted = []
    for cust in waitlist:
        if cust['status'] is 'ready' and slots > 0:
            promoted.append(cust)
            waitlist.remove(cust)
            slots -= 1
    return promoted
```
**Bugs:**
1. `list.remove` while iterating the same list skips the element after every removal — consecutive 'ready' customers are half-processed; some never get seen this call.
2. `is 'ready'` tests identity, not equality — works only when CPython happens to intern the string; statuses read from a file/DB compare `False` even when equal (`SyntaxWarning` on modern Python, silent misbehavior on older).
3. Mutating the caller's `waitlist` as a side effect while also returning the promoted subset splits the state change across two places; callers that ignore the mutation double-promote next call.
**Fix:** `== 'ready'`; build `promoted = [c for c in waitlist if ...][:slots]` and return both the promoted list and the remaining list (or mutate explicitly and document it); never `remove` inside `for` over the same list.

**11. Annealing at constant temperature** · `H`
```python
def anneal(state, T=100, cooling=0.95, steps=10_000):
    """Simulated annealing: accept worse moves with prob exp(delta/T)."""
    best = state
    for step in range(steps):
        cand = neighbor(state)
        delta = cost(cand) - cost(state)
        if delta < 0 or random.random() < math.exp(delta / T):
            state = cand
            if cost(state) < cost(best):
                best = state
        T = T * cooling if step % 100 == 0 else T
    return best
```
**Bugs:**
1. Acceptance probability uses `exp(delta / T)` with `delta > 0` → value **> 1**: every worsening move is accepted with certainty. The Metropolis form is `exp(-delta / T)`; as written this is a random walk, not annealing.
2. Cooling only fires when `step % 100 == 0` — that's fine — but with `steps=10_000` that's 100 cooldowns of 0.95 → T ends ≈ 100·0.95¹⁰⁰ ≈ 0.6, i.e., barely annealed; combined with (1) the schedule is irrelevant anyway. The interplay (broken acceptance masking a lazy schedule) is the point.
3. `cost(state)` and `cost(cand)` are each recomputed up to three times per step — for expensive objectives this triples runtime; cache the current cost and update by delta.
4. `best = state` binds a reference; if `neighbor` mutates in place (common), `best` silently tracks `state` and the "best" is whatever the walk ended on.
**Fix:** `math.exp(-delta / T)`; cache `cur = cost(state)`; deep-copy on `best = copy(state)` or make `neighbor` pure; choose a schedule that actually reaches low T within `steps`.

**12. The self-relaxing constraint** · `M`
```python
def rebalance_relaxed(df: pd.DataFrame, factor: float = 1.02):
    """Try swaps; if a full pass fixes nothing, relax the factor 1% and retry."""
    while has_violations(df, factor):
        fixed = try_all_swaps(df, factor)
        if not fixed:
            factor = factor * 1.01
    return df, factor
```
**Bugs:**
1. Termination is *guaranteed* — by inflating `factor` until nothing counts as a violation. The function can't fail; it just quietly redefines success. Returning the final `factor` is honest, but nothing bounds it or logs the relaxation count — a 40%-over-capacity plan exits as "balanced."
2. `factor * 1.01` compounds geometrically, and each relaxation triggers a full `try_all_swaps` pass: on infeasible input this is many wasted O(n²) passes before the factor balloons past every violation.
3. No distinction between "no *improving* swap" and "no *legal* swap" — the relax branch fires even when improvement exists but `try_all_swaps` has a bug (this design hides bugs in its own subroutine).
**Fix:** Cap the relaxation (`factor <= factor_max`), log/return each relaxation step, and prefer failing loudly (return the residual violations) over redefining the constraint; separate "converged" from "gave up."
**Senior tell:** Recognizing the pattern: a solver that loosens its own acceptance test until it passes is the optimization version of deleting a failing assert.

**13. Splitting a cluster by alias** · `M`
```python
def split_oversize(cluster: dict, cap: float):
    """Split items exceeding cap into a new cluster. Returns (old, new)."""
    new_cluster = {'id': cluster['id'] + '_b', 'items': cluster['items']}
    while sum(i['w'] for i in cluster['items']) > cap:
        item = max(cluster['items'], key=lambda i: i['w'])
        cluster['items'].remove(item)
        new_cluster['items'].append(item)
    return cluster, new_cluster
```
**Bugs:**
1. `new_cluster['items']` *is* `cluster['items']` — same list object. Every `remove` + `append` removes and re-adds to the **same list**: the loop condition never decreases (sum unchanged) → infinite loop; and even conceptually, both clusters "contain" everything.
2. `remove(item)` matches by equality — with duplicate-weight dicts it removes the *first* equal item, not necessarily the max one found (harmless here only if dicts are identical; a landmine once items carry ids).
3. Repeated `sum(...)` and `max(...)` per iteration: O(k²) for k items; a single sort or a running total does it in O(k log k).
**Fix:** `new_cluster['items'] = []`; move items by index (`pop(idx)`); maintain a running weight; verify post-condition `old_sum <= cap` and that the two item sets partition the original.

**14. Hashing clusters by their repr** · `M`
```python
def dedupe_assignments(assignments: list[dict]):
    """Drop duplicate (vehicle, cluster-items) assignments."""
    seen = {}
    for a in assignments:
        key = str(a['items'])
        if key not in seen:
            seen[key] = a
    return list(seen.values())
```
**Bugs:**
1. `str(list)` as identity: order-sensitive (`[1,2]` ≠ `[2,1]` though the cluster is the same set), format-fragile (dict item order, float repr `0.1` vs `0.1000000000000001`), and collision-prone across types (`'1'` vs `1` stringify differently, but `[1, 2]` vs `"[1, 2]"` collide).
2. The vehicle is in the docstring's identity but not in the key — two different vehicles carrying identical item lists dedupe into one assignment, silently dropping a vehicle.
3. First-wins policy is undocumented; if assignments carry timestamps or quality scores, keeping the first is a hidden business decision.
**Fix:** Canonical key: `(a['vehicle_id'], frozenset(item_ids))` (or sorted tuple of ids); state the keep-policy explicitly (`max` by score); never `str()` a mutable structure as a key.

---

## B — Indexing & Alignment Traps (15–28)

**15. Positional loop over a labeled subset** · `E`
```python
def flag_heavy(df: pd.DataFrame, thresh: float):
    """Flag rows above thresh (on the filtered subset)."""
    heavy = df[df.weight > thresh]
    for i in range(len(heavy)):
        heavy.loc[i, 'flag'] = True
    return heavy
```
**Bugs:**
1. `heavy` keeps the original labels; `range(len(heavy))` produces positions. `.loc[i]` is label-based: it hits the wrong rows when labels ≠ 0..n-1, *creates new rows by enlargement* for labels not present, and only "works" by coincidence on a fresh RangeIndex.
2. Writing into `heavy` (a filtered copy) triggers `SettingWithCopyWarning` ambiguity — and the caller's `df` never gets the flags if that was the intent.
3. A loop for a constant assignment: `heavy['flag'] = True` (or `df.loc[mask, 'flag'] = True` on the original) is the whole function.
**Fix:** Decide the target (subset copy vs original), then one vectorized assignment; if iterating labeled data ever, iterate `heavy.index`, never `range(len(...))`.

**16. The NaN block nobody ordered** · `M`
```python
def add_share(df: pd.DataFrame):
    """Add each row's share of its region's total."""
    big = df[df.amount > 0]
    shares = big.amount / big.groupby(big.region).amount.transform('sum')
    df['share'] = shares.reset_index(drop=True)
    return df
```
**Bugs:**
1. `reset_index(drop=True)` renumbers `shares` 0..k-1, then assignment to `df['share']` aligns those *new* labels against `df`'s original index: values land on the wrong rows wherever filtering removed anything, and rows beyond k get NaN. The author added `reset_index` to "fix" alignment and thereby broke it — the correct move was to keep labels and let alignment place values (non-positive rows would be NaN, correctly).
2. Non-positive rows silently get NaN with no policy — should they be 0, excluded, or an error?
3. `big.groupby(big.region)` passes a Series where the column name does — harmless but a tell that alignment mechanics aren't understood.
**Fix:** `df['share'] = df.amount / df.groupby('region').amount.transform('sum')` computed on the full frame with an explicit mask policy (`df.loc[df.amount <= 0, 'share'] = 0` if that's the rule).
**Senior tell:** Explaining *why* the reset breaks it — index alignment is the assignment's semantics, not an obstacle.

**17. The write that never lands** · `E`
```python
def zero_small(df: pd.DataFrame, cutoff: float):
    """Zero-out amounts below cutoff, in place."""
    df[df.amount < cutoff]['amount'] = 0
```
**Bugs:**
1. Chained indexing: `df[mask]` may return a copy; the assignment mutates the temporary and vanishes — the function is a no-op that emits (at most) a `SettingWithCopyWarning`.
2. Returns `None` implicitly while claiming in-place semantics — callers writing `df = zero_small(df, c)` end with `None`; callers not assigning may or may not see changes depending on pandas version/copy-on-write. Every calling convention loses somewhere.
**Fix:** `df.loc[df.amount < cutoff, 'amount'] = 0; return df` (and say whether it mutates); under pandas Copy-on-Write the chained form is *guaranteed* dead.

**18. Sorting by the ghost column** · `M`
```python
def top_then_rest(df: pd.DataFrame, k: int):
    """Return df with the top-k amounts first, original order preserved after."""
    ranked = df.sort_values('amount', ascending=False).reset_index()
    top = ranked.head(k)
    rest = ranked.tail(len(ranked) - k).sort_values('index')
    return pd.concat([top, rest]).drop(columns='index')
```
**Bugs:**
1. Works only because `reset_index()` (no `drop=True`) exports the old index as a column named `'index'` — but if `df` already has a column `'index'`, or a non-default index (e.g., order ids), `sort_values('index')` sorts by something else entirely or collides. The design leans on an accident of naming.
2. If `df`'s index isn't the original row order (post-filter, post-merge frames rarely are), "original order" is *not* what `'index'` restores — the docstring promise is index-order, not arrival-order.
3. `tail(len(ranked) - k)` breaks for `k > len(df)` (negative n means "all but first |n|" → wrong rows, silently) — `iloc[k:]` has the sane edge behavior.
**Fix:** Capture order explicitly: `df.assign(_ord=range(len(df)))`, sort/split, restore by `_ord`, drop it; slice with `iloc[k:]`; validate `k`.

**19. One label, many rows** · `M`
```python
def apply_adjustment(df: pd.DataFrame, adj: pd.DataFrame):
    """adj has index = order_id, column 'delta'. Add delta to matching orders."""
    for oid in adj.index:
        row = df.loc[oid]
        df.loc[oid, 'amount'] = row['amount'] + adj.loc[oid, 'delta']
    return df
```
**Bugs:**
1. If `df`'s index has duplicate `order_id`s (post-concat frames often do), `df.loc[oid]` returns a *DataFrame*, `row['amount']` a Series, and the addition broadcasts — the write then assigns a Series into multiple rows; results range from wrong values to `ValueError`, depending on shapes. The code assumes uniqueness it never checks.
2. Same for `adj`: a duplicated adjustment id silently applies only its last/first value depending on internals.
3. Row-by-row `.loc` writes for what is one aligned addition: `df['amount'] = df['amount'].add(adj['delta'], fill_value=0)` (after both indexes verified unique).
**Fix:** `assert df.index.is_unique and adj.index.is_unique` (or `verify_integrity` on construction), then a single aligned `add`; if duplicates are legal, define the aggregation explicitly.

**20. iloc with someone else's positions** · `M`
```python
def mark_selected(df: pd.DataFrame, selected: pd.DataFrame):
    """selected is a filtered view of df; mark those rows in df."""
    positions = [df.index.get_loc(i) for i in selected.index]
    df['sel'] = False
    for p in positions:
        df.iloc[p, df.columns.get_loc('sel')] = True
    return df
```
**Bugs:**
1. The whole positional round-trip exists to avoid the one-liner `df.loc[selected.index, 'sel'] = True` — and it breaks the moment `df` has duplicate labels (`get_loc` returns a slice/bool array, the loop explodes) or `selected` came from a *differently indexed* copy (KeyError).
2. `df['sel'] = False` resets the column on every call — if the function is meant to *accumulate* marks across calls (plural docstring "those rows"), previous marks are wiped; if not, fine, but it also inserts the column even when `selected` is empty, changing schema as a side effect.
3. Per-cell `.iloc` writes in a loop: slow and noisy vs one vectorized label-based write.
**Fix:** `df.loc[selected.index, 'sel'] = True` after `df['sel'] = df.get('sel', False)` if accumulation is intended; require/assert unique indexes.

**21. Appending to the same label forever** · `M`
```python
def collect_rows(frames: list[pd.DataFrame]):
    """Stack selected rows from many frames into one result frame."""
    out = pd.DataFrame(columns=frames[0].columns)
    for f in frames:
        for _, row in f[f.status == 'ok'].iterrows():
            out.loc[len(out)] = row
    return out
```
**Bugs:**
1. `out.loc[len(out)] = row` grows by *label* — correct only while the index is exactly 0..n-1. One duplicate label (or any future refactor that filters `out`) makes `len(out)` collide with an existing label and **overwrite** instead of append; the pattern is a time bomb even when today's run works.
2. Row-wise growth is O(n²) (each enlargement copies), plus `iterrows` boxing — the classic quadratic appender.
3. `pd.DataFrame(columns=...)` starts all-object dtype; appended rows keep it — numeric columns come out as `object`, breaking downstream math quietly.
4. Empty `frames` → IndexError on `frames[0]`.
**Fix:** Collect masks/frames and `pd.concat([f[f.status=='ok'] for f in frames], ignore_index=True)` once; dtypes and complexity both fixed for free.

**22. `.at` fed positions** · `E`
```python
def bump_first_k(df: pd.DataFrame, k: int, delta: float):
    """Increase 'score' of the first k rows."""
    for i in range(k):
        df.at[i, 'score'] += delta
    return df
```
**Bugs:**
1. `.at` is label-based; `range(k)` are positions. On any frame whose index isn't 0..n-1 (filtered, sorted, keyed by id) this hits wrong rows, raises KeyError, or *creates* rows 0..k-1 by enlargement with NaN elsewhere.
2. "First k rows" is order-dependent and no order is established — first by what? The function's meaning changes with upstream sorts.
3. `+=` on a possibly-NaN cell propagates NaN silently (a fresh enlarged row from bug 1 then gets NaN + delta = NaN).
**Fix:** `df.iloc[:k, df.columns.get_loc('score')] += delta` after an explicit `sort_values` defines "first"; or operate on labels: `df.loc[df.index[:k], 'score'] += delta`.

**23. MultiIndex sliced blind** · `H`
```python
def region_window(df: pd.DataFrame, region: str, start, end):
    """df indexed by (region, date). Return the region's rows in [start, end]."""
    return df.loc[(region, start):(region, end)]
```
**Bugs:**
1. Label slicing on a MultiIndex requires the index to be **lexsorted**; on an unsorted index this raises `UnsortedIndexError` — or worse, on partially sorted data older pandas would return incomplete slices. Nothing here sorts or checks `df.index.is_monotonic_increasing`.
2. The slice is *positional between two label endpoints*: if `(region, start)` doesn't exist exactly, behavior depends on sortedness; with `start`/`end` as strings vs Timestamps the lookup can silently miss (dtype mismatch on level values).
3. The tuple-slice form also breaks if a third index level exists (needs `pd.IndexSlice[region, start:end, :]`).
**Fix:** `df = df.sort_index()` once upstream; slice with `idx = pd.IndexSlice; df.loc[idx[region, start:end], :]`; normalize the date level's dtype; assert monotonicity in the function if callers can't be trusted.

**24. A mask from another index** · `M`
```python
def keep_valid(df: pd.DataFrame):
    """Keep rows whose recomputed check passes."""
    checked = df.reset_index(drop=True)
    mask = checked.apply(is_valid, axis=1)
    return df[mask]
```
**Bugs:**
1. `mask` carries the reset 0..n-1 index; `df[mask]` aligns by label against `df`'s original index. When they differ (filtered/sorted upstream), rows are selected by *coincidental label overlap* — some rows evaluated True are dropped, others never evaluated are kept where labels happen to match; missing labels align to NaN→False. Silent, order-dependent wrongness.
2. `apply(axis=1)` for a row predicate is the slow path if `is_valid` is expressible on columns.
**Fix:** Compute the mask on `df` itself (`df.apply(is_valid, axis=1)`), or align explicitly (`mask.set_axis(df.index)`), or filter positionally (`df.iloc[mask.to_numpy().nonzero()[0]]` — with the numpy escape documented). Same index in, same index out is the rule for boolean filtering.

**25. Half-landed updates** · `M`
```python
def apply_discounts(df: pd.DataFrame, promo_ids: set):
    """Apply 10% discount to promo rows, 5% to loyal customers."""
    df.loc[df.order_id.isin(promo_ids), 'price'] *= 0.90
    df[df.loyalty == 'gold']['price'] *= 0.95
    return df
```
**Bugs:**
1. Line 1 is a real in-place write; line 2 is chained indexing that (usually) mutates a temporary — the gold discount silently doesn't apply. The function's *tests* pass if they only cover promos: a half-working function is worse than a broken one because it survives review.
2. Business logic: rows that are both promo and gold get… whichever lines happen to land — stacking policy is undefined; with both writes working they'd compound to 0.855, which nobody specified.
3. In-place mutation of the caller's frame plus `return df` (aliasing) — callers diff "before/after" copies and find the before mutated too.
**Fix:** One explicit multiplier column built from the policy (`np.select` over conditions with a defined stacking rule), applied once with `.loc`; operate on a copy or document mutation.

**26. The rename that didn't** · `E`
```python
def standardize(df: pd.DataFrame):
    """Rename legacy columns to the standard schema."""
    df = df.rename(columns={'clus_wt': 'cluster_weight', 'veh_cap': 'vehicle_capcty'})
    df['utilization'] = df.cluster_weight / df.vehicle_capacity
    return df
```
**Bugs:**
1. The mapping itself has a typo (`vehicle_capcty`), so the very next line's `df.vehicle_capacity` raises `AttributeError` — *if* the legacy column existed. If it didn't, `rename` silently ignores missing keys (default `errors='ignore'`), and the error surfaces later or never; either way the "standardize" step can't be trusted to have standardized.
2. No verification that the target schema now holds — a one-line `assert {'cluster_weight','vehicle_capacity'} <= set(df.columns)` converts both failure modes into a clear error at the boundary.
3. Attribute access (`df.cluster_weight`) as the access style compounds it: it also collides with methods/spaces and hides typos as attribute errors far from the rename.
**Fix:** Fix the typo; pass `errors='raise'` in `rename` (or assert the post-schema); prefer `df['col']` access.

**27. Truthiness of an alignment** · `M`
```python
def same_order(a: pd.DataFrame, b: pd.DataFrame):
    """True if both frames list the same order_ids in the same order."""
    if a.order_id == b.order_id:
        return True
    return False
```
**Bugs:**
1. Series `==` is elementwise; `if` on a Series raises "truth value is ambiguous" — the function *never* returns True; it either raises or (branch never taken) returns False.
2. Even fixed with `.all()`, elementwise `==` on different-length Series raises, and on equal-length but differently-*indexed* Series pandas aligns by label first — comparing values at matching labels, not matching positions, which silently answers a different question than "same order."
3. NaN order_ids compare unequal to themselves — two identical frames with a NaN both report not-same.
**Fix:** `a.order_id.reset_index(drop=True).equals(b.order_id.reset_index(drop=True))` — `.equals` handles length, position (after the reset), and NaN equality in one call; or compare `.tolist()`.

**28. Sorted head, unsorted tail (anchor)** · `H`
```python
def cap_outliers_keep_rest(df: pd.DataFrame, k: int):
    """Winsorize: cap the k largest amounts at the k+1-th value, keep all rows."""
    s = df.sort_values('amount', ascending=False)
    cap = s.iloc[k].amount
    head = s.head(k)
    head['amount'] = cap
    return pd.concat([head, df.iloc[k:]])
```
**Bugs:**
1. `df.iloc[k:]` slices the **unsorted** frame by position while `head` came from the **sorted** one: the concat contains duplicates (rows in both slices) and omissions (rows in neither) — the output isn't a permutation of the input. The function's row set is wrong, not just its order.
2. `head['amount'] = cap` writes into a slice of a sort-copy — SettingWithCopy territory; whether it sticks depends on internals, and under Copy-on-Write it targets a fresh copy (fine) while pre-CoW it may warn or alias.
3. `s.iloc[k]` raises for `k >= len(df)`; `k == len(df)` should mean "cap nothing," not crash.
4. Ties at the boundary: rows equal to `cap` are arbitrarily split between capped and uncapped depending on sort stability — the winsorization boundary is nondeterministic.
**Fix:** Skip the reshuffle entirely: `cap = df.amount.nlargest(k+1).iloc[-1]; df.assign(amount=df.amount.clip(upper=cap))` — one clip, no concat, ties handled by value not by position, original order preserved.
**Senior tell:** Reaching for `clip` — the whole sorted-head/unsorted-tail surgery is a workaround for not knowing the vectorized idiom.

---

## C — Mutation, Copies & State (29–40)

**29. The double normalizer** · `M`
```python
def normalize_weights(df: pd.DataFrame):
    """Return df with weights normalized to sum to 1."""
    df['weight'] /= df['weight'].sum()
    return df
```
**Bugs:**
1. Mutates the caller's frame while the docstring says "return" — callers holding the original now hold changed data.
2. Not idempotent in spirit but *is* in math — the real trap: two different callers normalizing "their own copy" are normalizing the same object; interleaved with other mutations, order matters.
3. `sum()` = 0 (all zeros) → division yields inf/NaN silently; negative weights "normalize" to something that sums to 1 but means nothing.
**Fix:** `out = df.copy(); out['weight'] = out.weight / total` after validating `total > 0`; or document mutation loudly and return `None` (pick one contract).

**30. The view that sometimes writes** · `M`
```python
def apply_tier_discount(df: pd.DataFrame):
    top = df[df.tier == 'A']
    top['price'] = top['price'] * 0.9      # works on my machine
    return df
```
**Bugs:**
1. `top` is the result of a boolean mask — a copy in modern pandas — so the discount lands on a temporary and `df` returns unchanged; on older pandas/edge layouts it *sometimes* wrote through, which is why "it worked in the notebook."
2. The function returns `df` as if the operation happened — the bug ships because the return shape is right and only values are wrong.
3. Under Copy-on-Write (pandas 2.x default direction) the behavior is finally deterministic: never writes — code relying on the old ambiguity breaks on upgrade.
**Fix:** `df.loc[df.tier == 'A', 'price'] *= 0.9` (on a copy if the contract is non-mutating); enable CoW in CI to flush these out.

**31. Shallow copy, shared guts** · `M`
```python
def snapshot(state: dict):
    """Freeze current assignment state for rollback."""
    return dict(state)          # state = {'clusters': [[ids...], ...], 'ts': ...}

backup = snapshot(state)
mutate(state)                    # oops, need rollback
state = backup
```
**Bugs:**
1. `dict(state)` copies one level; `state['clusters']` lists (and their inner lists) are shared — `mutate` edits them through both names, so the "rollback" restores a dict whose contents already mutated. The backup is a mirror, not a snapshot.
2. Rollback by rebinding `state = backup` only affects this scope's name; other references to the old dict elsewhere keep the mutated object.
**Fix:** `copy.deepcopy` for genuine snapshots (or persist to an immutable form: tuples/frozen dataclasses/serialized bytes); design mutation APIs to return new state instead of editing shared structures.

**32. Caching by object identity** · `H`
```python
_cache = {}

def expensive_summary(df: pd.DataFrame):
    key = id(df)
    if key not in _cache:
        _cache[key] = df.groupby('region').amount.sum()
    return _cache[key]
```
**Bugs:**
1. `id()` is only unique *while the object lives*: after a df is garbage-collected, a new frame can reuse the address → cache returns another dataset's summary. Rare, unreproducible, catastrophic.
2. Mutating `df` after first call returns the stale summary forever — identity says "same object," not "same contents."
3. Module-level `_cache` grows unboundedly and pins every summarized frame's result across the process lifetime; in a server this is a leak plus cross-request data exposure.
**Fix:** Cache on a *content* key (explicit dataset version/hash passed by the caller), bound it (LRU), and never key caches on `id()` of mutable objects.
**Senior tell:** Naming the id-reuse-after-gc failure — it separates "knows the API" from "knows the object model."

**33. The accumulating default** · `E`
```python
def filter_orders(df: pd.DataFrame, exclude: list = []):
    exclude.append('CANCELLED')
    return df[~df.status.isin(exclude)]
```
**Bugs:**
1. Mutable default: the list is created once at def-time; every call appends another `'CANCELLED'` — harmless-looking growth, but the same object also serves callers who passed nothing, and…
2. …callers who *do* pass a list get it mutated (their `exclude` now contains `'CANCELLED'` after the call) — action at a distance in both directions.
**Fix:** `exclude: list | None = None`, then `exclude = {'CANCELLED', *(exclude or [])}`; never mutate parameters.

**34. Dropped twice** · `M`
```python
def strip_internal(df: pd.DataFrame, cols=('cost_basis', 'margin')):
    df.drop(columns=list(cols), inplace=True)
    ...
    return df.drop(columns=[c for c in cols if c in df.columns])
```
**Bugs:**
1. Two drop sites for the same columns: the first mutates in place, the second returns a copy dropping *again* — the guard (`if c in df.columns`) hides the double-drop instead of removing the redundancy; the reader can't tell which drop is load-bearing.
2. First drop has no such guard: missing columns raise `KeyError` — behavior differs by which code path runs first after refactors.
3. `inplace=True` mutation + returning a new frame mixes both contracts; callers can't know if their input changed.
**Fix:** One drop, one contract: `return df.drop(columns=[c for c in cols if c in df.columns])` on a non-mutating path (or `errors='ignore'`), delete the in-place line.

**35. Editing the dict you're walking** · `M`
```python
def add_weekly(frames: dict[str, pd.DataFrame]):
    """For each daily frame, add a weekly resample under key+'_w'."""
    for name, f in frames.items():
        frames[name + '_w'] = f.resample('W').sum()
    return frames
```
**Bugs:**
1. Inserting into a dict during iteration raises `RuntimeError: dictionary changed size during iteration` — and if it *didn't* (older Python/luck), the new `_w` entries would be iterated too, producing `_w_w` frames.
2. Re-running the function on its own output (common in notebooks) compounds keys (`sales_w_w`) — no idempotence guard (`if name.endswith('_w'): continue` or build into a fresh dict).
**Fix:** Iterate a snapshot (`list(frames.items())`) or build a new dict; make the transform idempotent by skipping derived keys.

**36. One dict, many rows** · `E`
```python
def build_rows(ids, base: dict):
    rows = []
    for i in ids:
        base['id'] = i
        base['ts'] = now()
        rows.append(base)
    return pd.DataFrame(rows)
```
**Bugs:**
1. Every element of `rows` is the *same* dict; each loop overwrites it — the final frame has n identical rows (all carrying the last id/ts).
2. The caller's `base` is also mutated (now has the last id stamped into it).
**Fix:** `rows.append({**base, 'id': i, 'ts': now()})` — fresh dict per row, caller's template untouched.

**37. Frames captured by reference** · `M`
```python
def split_batches(df: pd.DataFrame, size: int):
    batches, buf = [], df.iloc[0:0]
    for start in range(0, len(df), size):
        buf = df.iloc[start:start+size]
        buf['batch'] = start // size
        batches.append(buf)
    return batches
```
**Bugs:**
1. `buf['batch'] = ...` writes into a positional slice — SettingWithCopy ambiguity again, and depending on pandas internals the write may or may not stick; under CoW each `buf` is safely a copy, pre-CoW some slices can alias `df` and the loop *mutates the source frame's* rows batch by batch.
2. Even in the safe regime, the pattern (assign into a slice you just took) trains reviewers to accept the warning — the team-level bug.
3. Cosmetic but real: last batch's `'batch'` numbering is fine, but `df.iloc[0:0]` initializer is dead code.
**Fix:** `buf = df.iloc[start:start+size].copy(); buf['batch'] = ...` (explicit copy = explicit intent), or `df.assign(batch=np.arange(len(df)) // size).groupby('batch')`.

**38. Order-dependent methods** · `H`
```python
class Planner:
    def __init__(self, df):
        self.df = df                      # shared reference, not a copy
    def worst_first(self):
        self.df.sort_values('excess', ascending=False, inplace=True)
        return self.df.head(10)
    def utilization(self):
        return (self.df.cluster_weight / self.df.capacity).iloc[:100].mean()
```
**Bugs:**
1. `__init__` stores the caller's frame by reference; `worst_first` then sorts it **in place** — the caller's own frame reorders as a side effect of a "read" method.
2. `utilization` depends on row order (`iloc[:100]`): its value silently changes depending on whether `worst_first` ran first — methods that look independent are coupled through hidden state.
3. `inplace=True` inside a class is the worst place for it: every method call becomes a potential mutation of shared state.
**Fix:** Copy on ingestion (`self.df = df.copy()`), never sort in place for a read (`nlargest(10, 'excess')`), and make order-dependent computations take explicit orderings.

**39. Writing through `.values`** · `M`
```python
def cap_matrix(df: pd.DataFrame, cap: float):
    """Clip all numeric cells at cap, in place."""
    m = df.values
    m[m > cap] = cap
```
**Bugs:**
1. With mixed dtypes, `.values` materializes an object-dtype **copy** — the clip mutates the copy and `df` is untouched; with homogeneous float columns it can be a view and *does* write through. Same code, dtype-dependent semantics: the scariest kind.
2. `m > cap` on an object array compares element-wise via Python → slow and raises on non-numerics (strings, timestamps) that "numeric cells" was supposed to exclude.
**Fix:** `num = df.select_dtypes('number').columns; df[num] = df[num].clip(upper=cap)` — explicit column scope, deterministic semantics, no `.values` writeback superstition.

**40. Config leak between calls** · `H`
```python
DEFAULTS = {'factor': 1.02, 'max_iter': 50}

def plan(df, overrides: dict | None = None):
    cfg = DEFAULTS
    if overrides:
        cfg.update(overrides)
    return solve(df, **cfg)
```
**Bugs:**
1. `cfg = DEFAULTS` binds the module-level dict; `cfg.update(overrides)` **edits DEFAULTS itself** — one caller's `{'factor': 1.4}` becomes every later caller's default, process-wide. In a service this is cross-request contamination with no error, ever.
2. The corruption is order-dependent and invisible in unit tests that construct fresh processes — it only bites in long-lived workers.
**Fix:** `cfg = {**DEFAULTS, **(overrides or {})}`; freeze module config (`MappingProxyType`) so accidental mutation raises.
**Senior tell:** Immediately asking "who else reads DEFAULTS?" — shared-mutable-config is diagnosed by blast radius, not by the local diff.

---

## D — Loops That Shouldn't Exist (41–54)

**41. iterrows arithmetic** · `E`
```python
def add_utilization(df: pd.DataFrame):
    for i, row in df.iterrows():
        df.loc[i, 'util'] = row['cluster_weight'] / row['capacity']
    return df
```
**Bugs:**
1. `iterrows` boxes each row into a Series (dtype-coercing, slow); with a `.loc` write per row this is ~100–1000× slower than the one-liner.
2. Division by zero capacity raises mid-loop, leaving the frame **half-written** — partial mutation on failure.
3. Mixed dtypes: `row['capacity']` may arrive as object/str after boxing → TypeError on data that vectorized division would have surfaced cleanly.
**Fix:** `df['util'] = df.cluster_weight / df.capacity` (decide a zero-capacity policy first: `replace(0, np.nan)` or mask).

**42. The quadratic join** · `M`
```python
def customer_totals(customers: pd.DataFrame, orders: pd.DataFrame):
    totals = []
    for _, c in customers.iterrows():
        t = 0
        for _, o in orders.iterrows():
            if o.customer_id == c.customer_id:
                t += o.amount
        totals.append(t)
    customers['total'] = totals
    return customers
```
**Bugs:**
1. O(n·m) nested scan re-reading all orders per customer — this is a groupby-merge wearing a for-loop costume; at 10k×1M it's hours vs milliseconds.
2. Mutates `customers` in place *and* returns it.
3. Positional trust: `totals` aligns to `customers` only by loop order — any early `continue` added later desyncs the list from the frame silently.
**Fix:** `sums = orders.groupby('customer_id').amount.sum(); customers.assign(total=customers.customer_id.map(sums).fillna(0))`.

**43. Concat in a loop** · `E`
```python
result = pd.DataFrame()
for f in daily_files:
    day = pd.read_csv(f)
    result = pd.concat([result, day])
```
**Bugs:**
1. Each `concat` copies everything accumulated so far → O(n²) total copying; the standard fix (collect then concat once) is O(n).
2. Index duplication across days (`ignore_index` not set) — later `.loc` label lookups hit multiple rows.
3. No dtype/schema check between files: one file with a stray column or string-typed amount silently upcasts the whole result.
**Fix:** `result = pd.concat([pd.read_csv(f) for f in daily_files], ignore_index=True)` with a schema assertion per file.

**44. Compiling regex per row** · `E`
```python
def flag_urgent(df: pd.DataFrame):
    df['urgent'] = df.note.apply(
        lambda s: bool(re.compile(r'(?i)urgent|asap').search(str(s))))
    return df
```
**Bugs:**
1. `re.compile` inside the lambda recompiles the pattern per row (Python's regex cache saves you *by accident*, until the pattern is dynamic) — hoist it, or better:
2. `str.contains` does this vectorized: `df.note.str.contains(r'urgent|asap', case=False, na=False)` — also handling the real bug: `str(NaN)` = `'nan'`, so missing notes are scanned as the literal text "nan" (and would match a pattern containing "nan").
**Fix:** The vectorized `str.contains(..., na=False)` line; reserve `apply` for logic pandas can't express.

**45. Scalar `.loc` in a double loop (the seed's hot path)** · `H`
```python
def pair_costs(df: pd.DataFrame):
    n = len(df)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = abs(df.iloc[i]['weight'] - df.iloc[j]['weight']) \
                         * df.iloc[i]['rate']
    return cost
```
**Bugs:**
1. Each `df.iloc[i]` builds a boxed row Series; done 2n² + n² times, the *indexing* dominates the arithmetic by orders of magnitude — the exact pathology behind the 300–800× rebalancer speedup.
2. The i==j diagonal and both triangles are computed though the matrix is (rate-scaled) antisymmetric-ish in structure — even the numpy version can halve the work if symmetry is real for the use.
3. Ignores NaN weights: they propagate into whole rows/columns of the cost matrix silently.
**Fix:** Pull arrays once: `w = df.weight.to_numpy(); r = df.rate.to_numpy(); cost = np.abs(w[:, None] - w[None, :]) * r[:, None]` — with an upfront NaN check.

**46. groupby inside the group loop** · `M`
```python
def store_reports(df: pd.DataFrame):
    out = {}
    for store in df.store_id.unique():
        g = df.groupby('store_id').get_group(store)
        out[store] = g.amount.describe()
    return out
```
**Bugs:**
1. `df.groupby(...)` re-executes for every store (the whole hash/split each iteration) — O(stores·n); build it once and iterate it, or skip the loop entirely.
2. `unique()` + `get_group` also loses groupby's own guarantee of covering exactly the groups — NaN store_ids vanish from `unique`-driven paths differently than from groupby paths, a subtle count mismatch.
**Fix:** `out = {s: g.amount.describe() for s, g in df.groupby('store_id')}` — one pass; decide NaN-key policy via `dropna=`.

**47. String report, one += at a time** · `E`
```python
report = ""
for _, r in df.iterrows():
    report += f"{r.order_id}: {r.amount:.2f}\n"
```
**Bugs:**
1. Repeated string `+=` builds a new string each time (quadratic in CPython worst case; the in-place optimization is an implementation accident, not a contract) — at 1M rows this stalls.
2. `iterrows` again for pure formatting.
**Fix:** `report = "\n".join(f"{o}: {a:.2f}" for o, a in zip(df.order_id, df.amount))` — or `df.to_csv`/`to_string` if it's literally a table dump.

**48. Lookup by full scan** · `M`
```python
def enrich(events: pd.DataFrame, vehicles: pd.DataFrame):
    caps = []
    for vid in events.vehicle_id:
        caps.append(vehicles[vehicles.vehicle_id == vid].capacity.iloc[0])
    events['capacity'] = caps
    return events
```
**Bugs:**
1. Boolean full-scan of `vehicles` per event: O(events·vehicles) for what a dict/`map` does in O(n+m).
2. `.iloc[0]` throws `IndexError` on the first unknown vehicle — no missing policy — and silently takes the *first* row if vehicle_id ever duplicates.
3. Building a Python list and assigning trusts loop order (same fragility as #42).
**Fix:** `events['capacity'] = events.vehicle_id.map(vehicles.set_index('vehicle_id').capacity)` after asserting the index is unique; decide `fillna`/error for unknown ids.

**49. Distance matrix, both triangles, with sqrt** · `M`
```python
def dist_matrix(pts: list[tuple]):
    n = len(pts)
    d = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d[i][j] = math.sqrt((pts[i][0]-pts[j][0])**2 + (pts[i][1]-pts[j][1])**2)
    return d
```
**Bugs:**
1. Python-loop O(n²) with per-cell attribute lookups — numpy broadcasting or `scipy.spatial.distance.cdist` is 100×+ at n in the thousands.
2. Computes both triangles and the zero diagonal — half the work is thrown away for a symmetric metric.
3. If the consumer only *ranks* neighbors, the `sqrt` is pure cost — squared distances preserve order (state it, then keep or drop deliberately).
**Fix:** `p = np.asarray(pts); diff = p[:, None, :] - p[None, :, :]; d = np.sqrt((diff**2).sum(-1))` — or `cdist(p, p)`; exploit symmetry when memory matters.

**50. list.pop(0) as a queue** · `E`
```python
while jobs:
    job = jobs.pop(0)
    process(job)
    if needs_retry(job):
        jobs.append(job)
```
**Bugs:**
1. `pop(0)` shifts the entire list every call → O(n²) queue; `collections.deque.popleft()` is O(1).
2. Unbounded retry: a permanently failing job re-enqueues forever — no attempt counter, no backoff; the loop never ends and the queue never drains.
**Fix:** `deque` + retry budget on the job (`job['attempts'] += 1; if < MAX: append`), dead-letter the rest.

**51. Recomputing the sum you already know** · `M`
```python
def drain_until(df: pd.DataFrame, target: float):
    """Remove smallest orders until remaining total <= target."""
    df = df.sort_values('amount')
    while df.amount.sum() > target and len(df):
        df = df.iloc[1:]
    return df
```
**Bugs:**
1. `df.amount.sum()` re-scans all remaining rows every iteration and `iloc[1:]` copies the frame each time — O(n²) time and allocation for a running-total problem.
2. Removes the **smallest** orders to reduce the total — the slowest possible route to the target (maximum removals); if the goal is "keep total under target while dropping least value," removing largest-first or a cumsum cut is the intent — the code's policy contradicts any sensible reading of the docstring.
**Fix:** Decide the policy, then one pass: `df = df.sort_values('amount', ascending=False); keep = df[df.amount.cumsum() <= target]` (or the complement) — cumsum replaces the loop entirely.

**52. Five columns, one row at a time** · `M`
```python
def features(df: pd.DataFrame):
    def per_row(r):
        return pd.Series({
            'util': r.cluster_weight / r.capacity,
            'slack': r.capacity - r.cluster_weight,
            'is_over': r.cluster_weight > r.capacity,
            'log_w': math.log(r.cluster_weight),
            'bucket': int(r.cluster_weight // 100)})
    return df.join(df.apply(per_row, axis=1))
```
**Bugs:**
1. `apply(axis=1)` + constructing a `pd.Series` **per row** is among the slowest idioms in pandas (~1000× vs columnar) — every one of these five is a vectorized one-liner.
2. `math.log` raises on zero/negative weights mid-apply (partial failure); `np.log` would at least propagate NaN consistently — but the real fix is deciding the domain policy up front.
3. `int(...//100)` per row: floor-div is vectorized too; and `is_over` uses strict `>` while the rest of the codebase (see #1, #6) uses factored capacity — boundary drift across functions.
**Fix:** Five assignments on columns (`np.log` with a mask, `//100` cast via `.astype(int)`), one shared violation predicate imported everywhere.

**53. Parsing dates per row, in a filter, in a loop** · `M`
```python
def recent(df: pd.DataFrame, days: int):
    keep = []
    for _, r in df.iterrows():
        if (pd.Timestamp.now() - pd.to_datetime(r['ts'])).days <= days:
            keep.append(r)
    return pd.DataFrame(keep)
```
**Bugs:**
1. `pd.to_datetime` per row (string parsing is the expensive part) and `Timestamp.now()` per row — the cutoff *moves during the loop*; rows near the boundary pass or fail depending on scan position. Nondeterministic filter.
2. Rebuilding a frame from a list of Series loses dtypes (everything object) and the index.
3. `.days` truncates toward zero — a row 3 days 23 hours old counts as 3 days: boundary semantics differ from the "last N days" a human means; define the cutoff as a timestamp comparison.
**Fix:** `cutoff = pd.Timestamp.now() - pd.Timedelta(days=days); df[pd.to_datetime(df.ts) >= cutoff]` — one parse, one frozen cutoff, dtype-preserving.

**54. Running median by re-sorting** · `H`
```python
def rolling_median(x: list[float], w: int):
    out = []
    for i in range(len(x)):
        window = x[max(0, i-w+1):i+1]
        out.append(sorted(window)[len(window)//2])
    return out
```
**Bugs:**
1. Sorting every window: O(n·w log w); two-heaps or an order-statistics structure gives O(n log w) — at streaming scale the difference is the feature existing or not. (`pd.Series(x).rolling(w).median()` if pandas is already in play.)
2. `sorted(window)[len//2]` is the *upper* middle for even windows — a max-biased "median" that systematically overstates on trends; even-window medians need the midpoint average (or a declared convention).
3. Warm-up windows (i < w−1) silently use shorter windows — sometimes wanted, but the caller can't tell these apart from full-window values; pandas' `min_periods` makes the choice explicit.
**Fix:** State the even-window and warm-up conventions, then use `rolling(w, min_periods=...).median()` or a heap-based implementation; never ship an unlabeled biased median.

---

## E — Merges, Joins & Keys (55–66)

**55. Revenue, doubled by join** · `M`
```python
def revenue_by_region(orders: pd.DataFrame, payments: pd.DataFrame):
    m = orders.merge(payments, on='order_id')
    return m.groupby('region').order_amount.sum()
```
**Bugs:**
1. Multiple payments per order (installments, retries) fan the join out — `order_amount` repeats once per payment and the sum inflates; nothing validates the relationship (`validate='one_to_one'` would have raised).
2. Inner join silently drops orders with no payment row — the sum is simultaneously inflated (dupes) and deflated (drops); wrong in both directions is the worst place to be, because totals can accidentally look plausible.
**Fix:** Aggregate payments to order grain first (or sum `orders` alone if payment data isn't needed); merge with `how` chosen deliberately and `validate=` always on for money paths; reconcile the result to `orders.order_amount.sum()`.

**56. '007' ≠ 7** · `M`
```python
def attach_customer(orders: pd.DataFrame, customers: pd.DataFrame):
    orders['customer_id'] = orders.customer_id.astype(str)
    m = orders.merge(customers, on='customer_id', how='left')
    m['segment'] = m.segment.fillna('unknown')
    return m
```
**Bugs:**
1. Only `orders`' key is cast; if `customers.customer_id` is int (or zero-padded strings from a legacy export: `'007'` vs `'7'`), matches silently fail — and the `fillna('unknown')` then *launders* the join failure into a plausible business category. A 40% "unknown segment" spike after this ships is the fingerprint.
2. Casting `orders.customer_id` in place mutates the caller's frame as a side effect of an enrichment call.
3. No match-rate check: one line (`m.segment.isna().mean()`) before the fillna would convert silent corruption into an alert.
**Fix:** Normalize *both* keys to one canonical form (type + strip + zero-pad policy), measure and log the unmatched rate against a threshold, and only then fill.
**Senior tell:** Treating `fillna` after a left join as a masking operation that needs a measurement in front of it.

**57. The selective fillna** · `M`
```python
def orders_with_promo(orders: pd.DataFrame, promos: pd.DataFrame):
    m = orders.merge(promos, on='order_id', how='left')
    m = m[m.discount.notna()]           # keep rows where promo applied
    m['final'] = m.amount * (1 - m.discount)
    return m
```
**Bugs:**
1. The comment says "where promo applied," but `notna` after a left join can't distinguish "no promo" from "promo row exists with NaN discount" (bad promo data) — two different populations collapse into one filter.
2. If the intent was *all* orders with promo-adjusted finals, `notna` filtering turned the left join into an inner join — the SQL `WHERE`-on-right-table classic in pandas costume; non-promo orders vanish from the output entirely.
3. No fan-out guard: an order with two overlapping promos duplicates and gets two different `final` values.
**Fix:** Decide the population first: all orders → `m['final'] = m.amount * (1 - m.discount.fillna(0))`; promo-only analysis → filter on a promo-table indicator (`indicator=True`, `_merge == 'both'`), not on a value column; `validate='one_to_one'` or pre-aggregate promos.

**58. Trusting `_x`** · `E`
```python
m = clusters.merge(vehicles, on='cluster_id')
m['util'] = m.weight_x / m.weight_y     # cluster weight over vehicle weight... right?
```
**Bugs:**
1. `_x`/`_y` are positional accidents of the merge call — a reordered merge or an upstream rename flips numerator and denominator with no error; utilization > 1 everywhere is the only symptom, if anyone looks.
2. Both tables having a column named `weight` is the underlying smell — same name, different meanings, colliding at every join.
**Fix:** `suffixes=('_cluster', '_vehicle')` or rename before merging (`vehicles.rename(columns={'weight': 'vehicle_capacity'})`); treat default suffixes in committed code as a review blocker.

**59. Joining on floats** · `H`
```python
def attach_zone(stops: pd.DataFrame, zones: pd.DataFrame):
    stops['lat_r'] = stops.lat.round(4)
    stops['lon_r'] = stops.lon.round(4)
    return stops.merge(zones, left_on=['lat_r', 'lon_r'],
                       right_on=['lat', 'lon'], how='left')
```
**Bugs:**
1. Equality-joining on rounded floats: coordinates that differ by float noise straddle rounding boundaries (`12.34495` vs `12.34505`) and miss; the match rate depends on the 4th decimal's luck. Float equality is not a join predicate.
2. Only the left side is rounded — `zones.lat/lon` at full precision almost never equal a rounded value except by construction; the join may match ~0% and a `how='left'` hides it as NaNs.
3. Geometric intent (point-in-zone or nearest-zone) is being faked with equality — even with both sides rounded, boundary points assign to the wrong zone systematically.
**Fix:** Do the real operation: spatial join (`geopandas.sjoin`) or nearest-neighbor (KDTree) with a distance tolerance; if grid-bucketing is genuinely intended, compute the *same* grid cell id on both sides (floor to grid, integer keys) — integers join, floats don't.

**60. Filtering on the indicator, almost** · `E`
```python
m = a.merge(b, on='id', how='outer', indicator=True)
only_a = m[m._merge == 'left']
```
**Bugs:**
1. The indicator's categories are `'left_only'`, `'right_only'`, `'both'` — `'left'` matches nothing, `only_a` is silently empty, and downstream "no unmatched rows 🎉" is a comparison against a typo. Categorical equality with a nonexistent category doesn't warn.
2. Attribute access `m._merge` works but collides with the private-attribute convention; `m['_merge']` at least reads as the column it is.
**Fix:** `m[m['_merge'] == 'left_only']` — and when the point is an audit, assert the counts: `m['_merge'].value_counts()` logged, not eyeballed.

**61. Normalized in a parallel universe** · `M`
```python
def join_by_name(a: pd.DataFrame, b: pd.DataFrame):
    a_norm = a.copy()
    a_norm['name'] = a_norm.name.str.strip().str.lower()
    b['name'].str.strip().str.lower()
    return a_norm.merge(b, on='name', how='inner')
```
**Bugs:**
1. Line 4 computes the normalized `b` names and throws the result away (no assignment) — `b` joins with raw names against `a`'s normalized ones: near-guaranteed under-matching, invisible because inner join just returns fewer rows.
2. Even fully normalized, joining on free-text names is a fuzzy-matching problem wearing an equality costume — `'ACME Corp'` vs `'Acme Corporation'` needs a real entity-resolution decision, not `.lower()`.
3. `a_norm` copies but the (intended) `b` write would have mutated the caller — inconsistent mutation posture across the two inputs.
**Fix:** Assign the normalization on a copy of both; measure the unmatched rate; if names are the only key, escalate to explicit matching rules or ids — don't let an inner join hide the problem.

**62. Previous row from a self-merge** · `M`
```python
def add_prev_amount(df: pd.DataFrame):
    df['seq'] = range(len(df))
    prev = df[['seq', 'customer_id', 'amount']].copy()
    prev['seq'] += 1
    return df.merge(prev, on=['seq', 'customer_id'],
                    how='left', suffixes=('', '_prev'))
```
**Bugs:**
1. `seq = range(len(df))` numbers rows in *current frame order*, which nothing guarantees is per-customer chronological — "previous" means previous in arbitrary order; a sort by `(customer_id, ts)` is the missing precondition.
2. The self-merge machinery reimplements `groupby('customer_id').amount.shift(1)` — five lines and a copied frame for a built-in, and every line is a fresh place to be wrong.
3. Mutates `df` by adding `seq` (side-effect column in the caller's frame, also returned in the output schema).
**Fix:** `df = df.sort_values(['customer_id', 'ts']); df['amount_prev'] = df.groupby('customer_id').amount.shift(1)`.

**63. asof, leaking forward** · `H`
```python
def price_at_order(orders: pd.DataFrame, prices: pd.DataFrame):
    """Attach the price list valid at each order's timestamp."""
    return pd.merge_asof(orders, prices, on='ts',
                         by='sku', direction='forward')
```
**Bugs:**
1. `direction='forward'` matches each order to the **next** price change at-or-after the order — future information; the "valid at order time" semantics is `direction='backward'` (most recent at-or-before). In a training pipeline this is temporal leakage; in billing it's charging tomorrow's price.
2. `merge_asof` requires both frames sorted by `ts` (within the `by` groups) — nothing sorts; unsorted input raises, but *nearly*-sorted data from a union of sources is the dangerous case that only fails sometimes.
3. Orders before the first price record get NaN silently — a policy (error? first known price?) is needed, not an accident.
**Fix:** Sort both by `ts`, `direction='backward'`, and decide the pre-history policy; add one test with a price change between two orders — it catches the direction bug instantly.

**64. Unique per store, joined globally** · `M`
```python
def attach_details(events: pd.DataFrame, orders: pd.DataFrame):
    # order_id restarts from 1 for each store
    return events.merge(orders[['order_id', 'items', 'total']],
                        on='order_id', how='left')
```
**Bugs:**
1. The comment states the composite key (`store_id, order_id`) and the merge ignores it: order 1017 of store A matches order 1017 of *every* store — fan-out plus cross-store contamination; totals and items attach to the wrong stores' events.
2. `how='left'` + fan-out means `len(result) > len(events)` — the single cheapest post-condition (`assert len(m) == len(events)`) is missing.
**Fix:** Merge `on=['store_id', 'order_id']`, `validate='many_to_one'`, assert the row count is preserved.

**65. combine_first, backwards** · `M`
```python
def apply_corrections(base: pd.DataFrame, fixes: pd.DataFrame):
    """fixes contains corrected values for some (id, column) cells."""
    merged = base.set_index('id').combine_first(fixes.set_index('id'))
    return merged.reset_index()
```
**Bugs:**
1. `a.combine_first(b)` keeps **a's** values wherever a is non-null — corrections in `fixes` only land where `base` was NaN; actual corrections of existing values are silently discarded. The call is inverted: `fixes.set_index('id').combine_first(base...)` — or better, `base.update(fixes)` semantics stated explicitly.
2. Ids present in `fixes` but not `base` get *inserted* as new rows by combine_first — "apply corrections" quietly becomes "append rows," a schema-level surprise.
3. No report of how many cells changed — correction jobs should emit an audit count; zero-changed is the alarm that catches bug 1.
**Fix:** `b = base.set_index('id'); f = fixes.set_index('id'); b.update(f)` (update overwrites with non-NaN from f, never inserts), then log `(before != b).sum()` per column.

**66. Three joins, one reconciliation short (anchor)** · `H`
```python
def customer_summary(cust, orders, payments, tickets):
    m = cust.merge(orders, on='customer_id', how='left') \
            .merge(payments, on='order_id', how='left') \
            .merge(tickets, on='customer_id', how='left')
    return m.groupby('customer_id').agg(
        total_orders=('order_id', 'nunique'),
        total_paid=('paid_amount', 'sum'),
        n_tickets=('ticket_id', 'count'))
```
**Bugs:**
1. Chained many-to-many fan-out: customers × orders × payments × tickets — a customer with 3 orders, 2 payments each, and 4 tickets yields 24 rows; `nunique` on order_id survives it, but `n_tickets` counts each ticket **6×** (once per order-payment combo) and `total_paid` sums each payment across every ticket replica. Two of the three aggregates are inflated by cross-products of unrelated tables.
2. Joining tickets (customer-grain) onto an order-grain frame is the structural mistake: aggregates of different grains must be computed at their own grain and *then* joined.
3. No `validate=`, no row-count checks, no tie-out of `total_paid` against `payments.paid_amount.sum()` — the one comparison that instantly exposes the inflation.
**Fix:** Aggregate each table to customer grain independently (`orders.groupby(...).order_id.nunique()`, `payments→orders→customer` summed at payment grain, `tickets.groupby(...).size()`), then join the three summaries; reconcile each total to its source table.
**Senior tell:** "Aggregate at grain, then join" stated as a rule — plus reaching for the tie-out before reading a single number.

---

## F — Groupby & Aggregation Logic (67–78)

**67. agg where transform was meant** · `E`
```python
def add_region_avg(df: pd.DataFrame):
    df['region_avg'] = df.groupby('region').amount.mean()
    return df
```
**Bugs:**
1. `groupby.mean()` returns one row per region indexed by region name; assigning it to a column aligns region-labels against `df`'s row index — nearly everything lands NaN (or, with integer regions colliding with row labels, lands on *arbitrary wrong rows*). The broadcast-back verb is `transform('mean')`.
2. Silent NaN column ships because nothing validates non-null output.
**Fix:** `df['region_avg'] = df.groupby('region').amount.transform('mean')`; add `assert df.region_avg.notna().all()` if regions can't be null.

**68. Mean of ratios per group** · `M`
```python
def region_utilization(df: pd.DataFrame):
    df['util'] = df.cluster_weight / df.capacity
    return df.groupby('region').util.mean()
```
**Bugs:**
1. Averaging per-vehicle utilization weights a scooter equal to a 10-ton truck — the region's *fleet* utilization is `sum(weight)/sum(capacity)`, and the two disagree exactly when it matters (mixed fleets). Which one the business wants is a decision; the code silently picked the fragile one.
2. Zero/NaN capacity rows poison the per-row ratio before the groupby (inf/NaN into `mean` → NaN region), whereas the ratio-of-sums formulation degrades gracefully.
**Fix:** `g = df.groupby('region'); g.cluster_weight.sum() / g.capacity.sum()` — or keep both metrics, explicitly named (`mean_vehicle_util` vs `fleet_util`).

**69. apply, then .values the order away** · `H`
```python
def top2_share(df: pd.DataFrame):
    """Share of each region's amount held by its top-2 orders."""
    def f(g):
        return g.nlargest(2, 'amount').amount.sum() / g.amount.sum()
    shares = df.groupby('region').apply(f)
    df['top2_share'] = df.region.map(shares).values
    return df
```
**Bugs:**
1. Not the bug you'd expect — `map(shares)` is fine; the `.values` is the tell of a previous alignment fight, and it *becomes* a bug the moment someone "simplifies" `map` away and assigns `shares` (region-indexed) directly. As written it survives, but the `.values` is a scar documenting a misunderstanding: leave alignment on unless you can say why not.
2. Real bug: `groupby.apply(f)` with a scalar-returning f is fine, but if any region has <2 rows `nlargest(2)` quietly returns 1 row — share of "top-2" is silently top-1; and a region with `amount.sum() == 0` → division by zero → inf mapped onto every row of that region.
3. `apply` for this is also the slow path: `nlargest` per group via apply vs a sorted+`groupby.head(2)` aggregation.
**Fix:** `top2 = df.sort_values('amount', ascending=False).groupby('region').head(2).groupby('region').amount.sum(); shares = (top2 / df.groupby('region').amount.sum()).fillna(0)`; document the <2-rows semantics.

**70. The vanished NaN region** · `M`
```python
def region_totals(df: pd.DataFrame):
    t = df.groupby('region').amount.sum()
    assert abs(t.sum() - df.amount.sum()) < 1e-6, "totals must reconcile"
    return t
```
**Bugs:**
1. `groupby` drops NaN keys by default: every order with a missing region silently vanishes from `t` — and the author *wrote the reconciliation assert*, which is exactly right, so this function **fails loudly**… until someone "fixes" the assert instead of the data. The pair (dropped NaN keys + tolerance assert) is the interview: which side do you change?
2. `1e-6` absolute tolerance on money at scale: at billions, float accumulation legitimately exceeds it (false alarms), while at pennies it's too loose — relative tolerance or integer cents.
**Fix:** `groupby('region', dropna=False)` so NaN becomes a visible bucket (then decide its business meaning); keep the reconciliation, with `math.isclose(..., rel_tol=)` or cents-as-ints.

**71. first() means "whatever order we're in"** · `M`
```python
def latest_status(events: pd.DataFrame):
    """One row per order: its most recent status."""
    return events.groupby('order_id').status.first()
```
**Bugs:**
1. `first()` returns the first row *in current frame order* — the docstring's "most recent" requires a sort by timestamp that never happens; after any upstream shuffle (merge, concat of sources) this returns arbitrary statuses per order.
2. Even sorted ascending, `first()` would be the *oldest*; the pairing is sort-desc+first or sort-asc+last — an off-by-direction that tests fine on single-event orders (most of them), the worst kind of coverage illusion.
**Fix:** `events.sort_values('ts').groupby('order_id').status.last()` — or `events.loc[events.groupby('order_id').ts.idxmax(), ['order_id','status']]` when the whole row is needed.

**72. The agg that silently vanished** · `E`
```python
stats = df.groupby('store').agg({'amount': 'sum', 'amout_share': 'mean'})
```
**Bugs:**
1. Typo'd column `'amout_share'` in an agg dict raises KeyError — good — *unless* an upstream rename/version made it optional and someone wrapped this in try/except; the deeper pattern: agg specs written as string dicts get no IDE/static checking, and misspellings live until runtime on the one code path that hits them.
2. Result columns keep bare names (`amount`, `amout_share`) — downstream code can't tell a sum from a mean by name.
**Fix:** Named aggregation: `df.groupby('store').agg(total_amount=('amount','sum'), avg_share=('amount_share','mean'))` — self-documenting output names, and typos in the tuple still fail fast at the callsite you're looking at.

**73. count() as the denominator** · `M`
```python
def fill_rate(df: pd.DataFrame):
    g = df.groupby('warehouse')
    return g.filled_qty.sum() / g.filled_qty.count()
```
**Bugs:**
1. `count()` excludes NaN — rows where `filled_qty` is missing (unfilled lines, exactly the failures) drop out of the denominator: the fill *rate* is computed over successes only and flatters every warehouse. `size()` counts rows; the choice between them **is** the metric definition.
2. If NaN actually means zero-filled, the numerator is also silently right-only-by-luck (`sum` skips NaN); the pair of skips can cancel or compound depending on data — either way the metric is undefined until NaN semantics are stated.
**Fix:** Decide NaN's meaning first (`filled_qty = filled_qty.fillna(0)` if NaN = unfilled), then `g.filled_qty.sum() / g.size()` — and name the metric's denominator in the docstring.

**74. Global cumsum, per-group intent** · `E`
```python
def running_totals(df: pd.DataFrame):
    df = df.sort_values(['customer_id', 'ts'])
    df['cum_amount'] = df.amount.cumsum()
    return df
```
**Bugs:**
1. The sort sets up per-customer sequences and then `cumsum` runs **across** customers — each customer's running total starts from the previous customer's grand total. Correct-looking within the first customer, garbage after.
2. Ties in `ts` within a customer: sort stability makes cumulative values at tied timestamps order-dependent on input order — declare a tiebreak column if downstream compares at-timestamp totals.
**Fix:** `df['cum_amount'] = df.groupby('customer_id').amount.cumsum()` (after the same sort); add `kind='stable'` semantics via an explicit secondary sort key.

**75. rank() with the default nobody wanted** · `M`
```python
def commission_tier(df: pd.DataFrame):
    df['rank'] = df.groupby('region').sales.rank(ascending=False)
    df['tier'] = np.where(df['rank'] <= 3, 'A', 'B')
    return df
```
**Bugs:**
1. Default `method='average'`: two reps tied at 3rd get rank 3.5 — **neither** passes `<= 3`; a tie at the cutoff silently shrinks tier A. Ties at rank 1–2 (ranks 1.5) sneak both in. The tier boundary behaves differently depending on *where* the tie lands.
2. The business rule ("top 3 per region") needs a declared tie policy: `method='min'` (ties share the better rank, tier A can exceed 3), `'first'` (arbitrary but fixed order — then the sort must be deterministic), or `'dense'`. The code chose by omission.
3. Column named `rank` shadows the method on later `df.rank` attribute access — minor, but it's the naming reflex check.
**Fix:** Pick and document the tie policy (`method='min'` for inclusive top-3 is the common business intent); name the column `sales_rank`.

**76. Bucketing on round()** · `H`
```python
def weight_bucket_stats(df: pd.DataFrame):
    df['bucket'] = (df.weight / 100).round() * 100
    return df.groupby('bucket').amount.agg(['count', 'mean'])
```
**Bugs:**
1. `round()` is banker's rounding (half-to-even): 150→200 but 250→200 — boundary weights alternate buckets by parity, so bucket populations are systematically distorted exactly at the human-round-number boundaries where data heaps (see EDA #28's heaping). Counts per bucket look plausible and are wrong.
2. Rounding buckets *center* on multiples of 100 (50–149 → 100); if the intent was floors (0–99 → 0), every bucket label is shifted half a bin — two readers can interpret the same output differently. Binning needs explicit edges, not arithmetic.
3. Float bucket labels (`100.0`) as group keys invite later float-equality joins against integer bucket definitions.
**Fix:** `pd.cut(df.weight, bins=range(0, top, 100), right=False)` — explicit edges, explicit closed-side, categorical labels; or integer floor-div `(df.weight // 100 * 100).astype(int)` with the convention documented.

**77. Percent-of-total with a moving denominator** · `M`
```python
def region_share(df: pd.DataFrame, active_only: bool = True):
    if active_only:
        df = df[df.status == 'active']
    totals = df.groupby('region').amount.sum()
    return totals / df.amount.sum() * 100
```
**Bugs:**
1. Numerator and denominator both use the *filtered* frame — internally consistent — but the function's consumers compare these shares to dashboards computed on **all** orders: same metric name, different universes. The bug is at the contract level: `region_share(active_only=True)` and the company's "region share" are different numbers and nothing in the output says which universe it is.
2. `df = df[...]` rebinding inside the function is fine, but the parameter default silently makes the *filtered* version the canonical one — defaults are policy.
3. Regions with only inactive orders vanish from the index entirely (rather than showing 0%) — downstream reindex/joins misalign.
**Fix:** Return a labeled result (`.rename('share_active_pct')` or a small frame with the universe stated); `reindex(all_regions, fill_value=0)`; make the universe an explicit required argument if both are in use.

**78. Resample per group, by hand (anchor)** · `H`
```python
def weekly_by_store(df: pd.DataFrame):
    out = []
    for store in df.store_id.unique():
        g = df[df.store_id == store].set_index('ts')
        w = g.amount.resample('W').mean()
        w = w.reset_index()
        w['store_id'] = store
        out.append(w)
    return pd.concat(out)
```
**Bugs:**
1. Weekly **mean of daily rows** is not weekly volume — and worse, weeks with missing days average over present days only, while a `sum` would undercount them: either way, gaps interact with the agg choice and nobody chose. For volume, `sum` over a complete calendar (with explicit zero-fill policy) is the semantics; `mean` answers "average order size on days we happened to have," which no one asked.
2. Per-store loop with filter+resample: O(stores·n), and each store's weekly index spans only *that store's* date range — stores with no early sales are missing early weeks entirely, so the concatenated frame has ragged, incomparable week coverage (a pivot downstream fills with NaN unevenly).
3. `resample('W')` default anchors weeks on Sunday (`W-SUN`) — if the business week is Monday-based (common outside the US), every weekly number blends two business weeks. Timezone-naive `ts` also shifts late-evening orders across week boundaries if the data is UTC.
4. Loses the datetime index and rebuilds via reset/concat — `df.groupby(['store_id', pd.Grouper(key='ts', freq='W-MON')]).amount.sum()` is the whole function.
**Fix:** One `pd.Grouper` groupby with the anchor and agg chosen deliberately; then `unstack`/`reindex` to a common weekly calendar with an explicit fill policy; state timezone handling.

---

## G — Datetime & Ordering (79–90)

**79. Sorted, but not really** · `E`
```python
def add_gap_days(df: pd.DataFrame):
    """Days between consecutive deliveries."""
    df.sort_values('delivery_ts')
    df['gap'] = df.delivery_ts.diff().dt.days
    return df
```
**Bugs:**
1. `sort_values` without `inplace=True` (or reassignment) returns a sorted copy that's discarded — `diff()` runs on the original order, producing negative and nonsense gaps that *look* plausible in aggregate.
2. Even sorted, `diff()` across the whole frame mixes entities if this is multi-customer data — gap #1 of customer B is measured from customer A's last delivery (the shift-without-groupby cousin).
**Fix:** `df = df.sort_values(['customer_id', 'delivery_ts']); df['gap'] = df.groupby('customer_id').delivery_ts.diff().dt.days` — and a nonnegativity assert on `gap`.

**80. Yesterday's neighbor is another customer** · `M`
```python
def flag_rapid_reorder(df: pd.DataFrame):
    df = df.sort_values('ts')
    df['prev_ts'] = df.ts.shift(1)
    df['rapid'] = (df.ts - df.prev_ts) < pd.Timedelta('1D')
    return df
```
**Bugs:**
1. Global `shift(1)` on a time-sorted multi-customer frame: each order's "previous" is the chronologically previous order of **whoever** — `rapid` flags bursts of *platform* activity, not customer reorders. High-traffic hours mark everyone rapid.
2. First row's NaT propagates: `NaT < Timedelta` is False — silently "not rapid," which happens to be right, by accident; make the boundary explicit rather than luck-dependent.
**Fix:** `df = df.sort_values(['customer_id','ts']); df['rapid'] = df.groupby('customer_id').ts.diff() < pd.Timedelta('1D')` (diff already handles the shift; first-in-group NaT → False stated in a comment or filled explicitly).

**81. Lexicographic dates** · `E`
```python
def q1_orders(df: pd.DataFrame):
    return df[(df.date >= '2026-1-1') & (df.date <= '2026-3-31')]
```
**Bugs:**
1. If `date` is a *string* column, comparison is lexicographic: `'2026-1-1'` vs `'2026-10-05'` — unpadded months make `'2026-1-…'` sort after `'2026-03-…'`?? Actually `'2026-1'` < `'2026-0'` is False — the point: mixed zero-padding makes string comparison scramble ranges unpredictably; some Q4 dates pass a "Q1" filter. Nothing here guarantees the dtype.
2. Even with proper datetimes, `<= '2026-3-31'` on a timestamp column excludes anything after midnight of the 31st *only if* times are all midnight — with real times, the 31st itself is included fully only when dates are pure dates. The inclusive-end habit needs `< '2026-04-01'` to be robust either way.
**Fix:** `df['date'] = pd.to_datetime(df.date)` at the boundary (with `format=`), then `df[(df.date >= '2026-01-01') & (df.date < '2026-04-01')]` — half-open intervals, always.

**82. .dt.date, timezone amputated** · `H`
```python
def daily_volume(df: pd.DataFrame):
    # ts is tz-aware UTC; business runs in Asia/Tehran
    df['day'] = df.ts.dt.date
    return df.groupby('day').size()
```
**Bugs:**
1. `.dt.date` on UTC timestamps buckets by **UTC** calendar day — Tehran evenings (UTC+3:30) land on the *previous* business day for every order after 20:30 local; daily volumes are systematically shifted and month boundaries misreport. The comment states the requirement the code ignores.
2. `.dt.date` yields Python `date` objects (object dtype) — slower groupby and lost datetime ops downstream; `dt.normalize()` or `dt.floor('D')` keeps datetime64.
3. Half-hour offset zones (+3:30) are exactly where "just add 3 hours" folk fixes also fail — conversion must be by zone name, DST rules included.
**Fix:** `df['day'] = df.ts.dt.tz_convert('Asia/Tehran').dt.normalize()` then group — and standardize: store UTC, convert at the reporting boundary, never strip tz silently.

**83. Weeks that start on the wrong day** · `M`
```python
def weekly_kpi(df: pd.DataFrame):
    return df.set_index('ts').amount.resample('W').sum()
```
**Bugs:**
1. `'W'` = `'W-SUN'`: weeks *end* Sunday — a Monday-based business week (most non-US retail) gets every KPI computed over Tue→Mon-shifted… precisely: bins are (Mon..Sun] labeled by Sunday; if the business defines Mon–Sun weeks labeled by Monday, every number is one day of misalignment plus a label off by six. Comparisons to finance's weekly report never tie out, by a small amount, forever.
2. The trailing partial week is included as a full-looking bar (the incomplete-period artifact) — the newest point is always an apparent crash.
3. Requires sorted unique-ish index; duplicate timestamps are fine for `sum` but `set_index` without sort on unsorted data plus later slicing invites the unsorted-slice trap.
**Fix:** `resample('W-MON', label='left', closed='left')` matched to the stated business definition (write the definition down!); drop or flag the trailing incomplete period.

**84. Thirty days ≠ a month** · `E`
```python
def next_renewal(df: pd.DataFrame):
    df['renewal'] = df.signup_ts + pd.Timedelta(days=30)
    return df
```
**Bugs:**
1. Monthly renewal by +30 days drifts: Jan 31 → Mar 2 after two cycles; over a year, renewal dates walk backward through the month — billing dates disagree with the contract's "monthly" and with finance's month-end close.
2. The corresponding month-end questions (what's Jan 31 + 1 month?) are a *policy*: `pd.DateOffset(months=1)` clamps to Feb 28/29 — which is usually the contract's intent, but must be chosen, not defaulted into via 30-day arithmetic.
**Fix:** `df.signup_ts + pd.DateOffset(months=1)` (or `MonthEnd`), with the clamping behavior documented against the billing contract.

**85. dayfirst, half the file** · `H`
```python
def parse_dates(df: pd.DataFrame):
    df['order_date'] = pd.to_datetime(df.order_date_str, errors='coerce')
    df = df.dropna(subset=['order_date'])
    return df
```
**Bugs:**
1. EU-style `DD/MM/YYYY` strings parsed with default `dayfirst=False`: every day ≤ 12 parses **as the wrong date** (day/month swapped, no error), while days ≥ 13 either parse correctly (pandas infers per-element in mixed mode) or coerce to NaT — the file ends up a *mixture* of swapped-but-valid and correct dates, and…
2. …the `dropna` then deletes the honest failures and keeps the silent swaps: the cleaning step selects for corruption. Row counts look fine (small drop), dates look like dates, and roughly half of days 1–12 are transposed.
3. `errors='coerce'` without *counting* the coercions is the enabling habit — the NaT rate is the smoke detector this function unplugs.
**Fix:** `pd.to_datetime(df.order_date_str, format='%d/%m/%Y')` — explicit format fails loudly on anything else; log the failure count; the max-observed-day ≤ 12 check is the EDA tripwire that catches an already-swapped column.
**Senior tell:** Recognizing that dropna-after-coerce is *survivorship bias applied to your own parsing*.

**86. Months sorted alphabetically** · `E`
```python
report = df.groupby(df.ts.dt.month_name()).amount.sum().sort_index()
```
**Bugs:**
1. `sort_index()` on month *names*: April, August, December, February… — the report's x-axis is alphabetical; trend reading is destroyed while every number is individually correct (the hardest kind to notice in review).
2. Grouping by name also merges the same month across years — if the frame spans >12 months, "March" pools 2025 and 2026.
**Fix:** Group by `df.ts.dt.to_period('M')` (sorts chronologically, keeps year); render names only at display time (`.strftime('%b %Y')`), or use an ordered Categorical of month names when a single year is guaranteed.

**87. SLA in calendar days** · `M`
```python
def sla_breach(df: pd.DataFrame, sla_days: int = 2):
    df['days'] = (df.delivered_ts - df.ordered_ts).dt.days
    df['breach'] = df.days > sla_days
    return df
```
**Bugs:**
1. `.dt.days` truncates: 2 days 23 hours → 2 → not a breach; the SLA silently gains a free day at every boundary. Compare timedeltas (`> pd.Timedelta(days=2)`), not truncated integers.
2. Calendar days vs *business* days: a Friday-evening order delivered Monday is 3 calendar days — breach — though it's 1 business day; whether weekends count is the SLA's definition, and `np.busday_count` (with the right week mask and holiday calendar — regional!) implements it. The code silently picked calendar.
3. Undelivered rows (NaT) → NaT comparison → False: open orders are counted as non-breaches, flattering the metric exactly where risk lives.
**Fix:** Timedelta comparison; explicit business-day policy with holiday calendar; open orders either excluded with a count or evaluated against `now()` as "breaching-so-far."

**88. ffill before sort** · `M`
```python
def fill_prices(prices: pd.DataFrame):
    prices = prices.set_index('ts')
    prices['price'] = prices.price.ffill()
    return prices
```
**Bugs:**
1. `ffill` fills from the *previous row in current order* — nothing sorts by `ts` first; on a frame assembled from concatenated sources, "forward" fill copies prices sideways in time, including **backwards** (a later timestamp's price filling an earlier row that sits below it). Time-semantics applied to arbitrary order.
2. Multi-SKU frames: fills bleed across SKU boundaries (SKU B's first NaN takes SKU A's last price) — the groupby-before-fill requirement.
3. Leading NaNs (before any price exists) stay NaN silently — policy needed (bfill? drop? error?), and whichever is chosen has *leakage* implications if this feeds a model (bfill = future info).
**Fix:** `prices.sort_values(['sku','ts']).groupby('sku').price.ffill()` — sort, group, fill, and state the leading-gap policy (for modeling: never bfill).

**89. One label, a slice of rows** · `M`
```python
def price_at(prices: pd.DataFrame, ts):
    p = prices.set_index('ts')
    return p.loc[ts, 'price']
```
**Bugs:**
1. Duplicate timestamps (two updates same second, or per-SKU rows) make `.loc[ts]` return a **Series/slice**, not a scalar — callers doing arithmetic get broadcasting or ValueError depending on shape; single-row data in tests, multi-row in prod.
2. Exact-match lookup for a *point-in-time price* is wrong even when unique: the price at 14:37 is the last update **at or before** 14:37 — this is an asof/`searchsorted` question; `.loc[ts]` raises KeyError between updates.
3. Repeated calls re-run `set_index` (copy) per lookup.
**Fix:** Build once: `p = prices.sort_values('ts').set_index('ts').price`; lookup via `p.asof(ts)` (documented: last valid at-or-before); define duplicate-timestamp resolution (keep last) at index construction.

**90. Rolling over chaos** · `H`
```python
def smooth_demand(df: pd.DataFrame):
    df['smooth'] = df.demand.rolling(7).mean().fillna(0)
    return df
```
**Bugs:**
1. Row-count window ("7 rows") on data that nothing sorted and that may have gaps/duplicates per day — 7 rows can span 3 days or 3 weeks; the window's *meaning* varies row to row. Time-based windows (`rolling('7D')` on a datetime index) or an asserted regular calendar are the two honest options.
2. Multi-entity frame: the window blends entities at every boundary (the recurring groupby omission).
3. `fillna(0)` on the warm-up NaNs fabricates a demand collapse at each series' start — models and eyeballs both read a fake ramp-up; `min_periods=1` (partial means) or leaving NaN are the deliberate choices.
**Fix:** Sort by (entity, ts); `groupby(entity).demand.rolling('7D' or 7, min_periods=...)` with the warm-up policy stated; never fillna(0) a rolling statistic's warm-up.

---

## H — Numeric, NaN & Dtype Logic (91–102)

**91. Dedupe that can't see NaN** · `M`
```python
def drop_dupe_pairs(df: pd.DataFrame):
    keep = [0]
    for i in range(1, len(df)):
        if not (df.iloc[i].amount == df.iloc[i-1].amount and
                df.iloc[i].region == df.iloc[i-1].region):
            keep.append(i)
    return df.iloc[keep]
```
**Bugs:**
1. `NaN == NaN` is False: consecutive rows with missing amounts are *never* equal, so NaN-duplicates all survive — the dedupe has a blind spot exactly on incomplete rows (often the duplicated ones, e.g., retried loads).
2. Only removes *adjacent* duplicates and nothing sorts first — same pair separated by one row survives; the function's contract ("dupe pairs") is order-dependent and unstated.
3. Row-wise `.iloc` loop for what `df.drop_duplicates(subset=['amount','region'])` (NaN-aware: pandas treats NaNs as equal in duplicated()) or `.shift()` comparison does vectorized.
**Fix:** `df.drop_duplicates(subset=['amount','region'])` if global dedupe is meant; if strictly-adjacent, `mask = ~((df.amount.eq(df.amount.shift()) | (df.amount.isna() & df.amount.shift().isna())) & df.region.eq(df.region.shift()))` — or `.ne()` with `fillna` sentinel; state the adjacency contract.

**92. Truthiness of a discount** · `M`
```python
def final_price(row):
    if row['discount']:
        return row['price'] * (1 - row['discount'])
    return row['price']
```
**Bugs:**
1. `0.0` is falsy: an explicit zero discount takes the no-discount path — same result *here*, but the branches diverge the moment logging/counting happens per path; worse, `discount = 0` vs "no discount recorded" are different facts collapsed by truthiness.
2. **NaN is truthy**: missing discounts enter the discount branch and `price * (1 - NaN)` = NaN — missing data silently NaNs the final price, then flows into sums (which skip it) — revenue undercounts with no error anywhere.
3. Negative discounts (data error) pass truthiness and *raise* prices.
**Fix:** Explicit domain check: `d = row['discount']; if pd.notna(d) and 0 < d <= 1: ...` — or vectorized: `df.price * (1 - df.discount.fillna(0).clip(0, 1))` with a logged count of out-of-range values.

**93. Money in floats, compared with !=** · `H`
```python
def reconcile(ledger: pd.DataFrame, bank: pd.DataFrame):
    m = ledger.merge(bank, on='txn_id')
    m['mismatch'] = m.amount_ledger != m.amount_bank
    return m[m.mismatch]
```
**Bugs:**
1. Both amounts arrived through float pipelines (CSV → float64, VAT multiplications): `41.30` computed as `41.299999999999997` flags a mismatch on **equal** money — the reconciliation report fills with false positives, gets a tolerance slapped on ad hoc, and then true 1-cent frauds hide under the tolerance. Float `!=` on money is the seed's `v1 != v2` bug wearing a finance suit.
2. No tolerance is *also* wrong once floats exist — the only clean regime is exact arithmetic: integer minor units (cents/rials) or `Decimal` end-to-end; a tolerance is an apology for the dtype.
3. Inner merge drops txns present on one side only — the *most important* reconciliation findings (missing transactions) are excluded from a function named `reconcile`; outer join + `_merge` indicator is the actual job.
**Fix:** Convert to integer cents at ingestion (`(s * 100).round().astype('int64')` with a pre-check), outer-merge with indicator, report three classes: value mismatches (exact), ledger-only, bank-only.

**94. The int column that went float** · `M`
```python
def attach_flags(orders: pd.DataFrame, flags: pd.DataFrame):
    m = orders.merge(flags, on='order_id', how='left')
    m['is_priority'] = m.is_priority.fillna(False).astype(bool)
    m.to_parquet('enriched.parquet')
    return m
```
**Bugs:**
1. The left join introduces NaN into `is_priority` — which silently upcasts a bool/int column to **object or float** *before* the fillna; the same happens to any int id column from `flags`: `vehicle_id` becomes `1042.0`, and every downstream string-format, join, or `== 1042` comparison inherits float semantics (large ids lose precision at 2⁵³).
2. `.astype(bool)` after fillna is fine for the flag, but nothing rescues the *other* upcast columns — the bug is systemic to the join, the fix is applied to one column.
3. Parquet then *persists* the accidental dtypes — future readers inherit float ids forever; the file format faithfully preserves the mistake.
**Fix:** Use nullable dtypes through the join (`Int64`, `boolean`) or fillna+downcast every affected column deliberately; assert the schema (`m.dtypes`) against a declared contract before any `to_parquet`.

**95. Floor-divided averages, infinite means** · `M`
```python
def avg_units(df: pd.DataFrame):
    df['aov'] = df.revenue // df.orders
    df['aov'] = df.aov.replace([np.inf, -np.inf], np.nan).fillna(df.aov.mean())
    return df
```
**Bugs:**
1. `//` floors: average order value 41.9 becomes 41 — a systematic downward bias masquerading as rounding; `revenue / orders` was meant.
2. Integer `//` by zero raises; float `//` by zero yields inf — so the inf-replace line implies floats, meaning zero-order rows produced inf, get replaced by NaN, then filled with… `df.aov.mean()` computed **while the infs/NaNs are still influencing nothing but after floor bias** — order-of-operations: mean() skips NaN but the column still carries floor bias, so gaps get filled with a biased mean. Layered patches on an arithmetic mistake.
3. Filling zero-order entities with the global mean invents revenue efficiency for entities with no orders — a per-row imputation *policy* smuggled in as cleanup.
**Fix:** `df['aov'] = df.revenue / df.orders.replace(0, np.nan)` (true division; zero-orders honestly NaN); leave NaN or impute with a stated policy — never auto-fill a KPI with its own mean.

**96. np.where evaluates both worlds** · `M`
```python
def log_ratio(df: pd.DataFrame):
    df['lr'] = np.where(df.demand > 0, np.log(df.demand / df.forecast), 0.0)
    return df
```
**Bugs:**
1. `np.where` evaluates **both** branches eagerly: `np.log` runs on the zero/negative-demand rows too — RuntimeWarnings, and the computed NaN/-inf are then *discarded* by the mask, so the code "works" while spraying warnings that teams learn to ignore (until a real one drowns).
2. The guard checks `demand > 0` but the log's domain also needs `forecast > 0` (and non-NaN): zero forecasts yield inf **inside the selected branch** — the mask is guarding the wrong (or half the) condition.
3. `0.0` as the fill conflates "undefined ratio" with "ratio exactly 1" (log 1 = 0) — downstream means can't tell perfect forecasts from broken rows.
**Fix:** Mask first, compute on the valid slice: `valid = (df.demand > 0) & (df.forecast > 0); df['lr'] = np.nan; df.loc[valid, 'lr'] = np.log(df.demand[valid] / df.forecast[valid])` — NaN (not 0) for undefined, warnings gone because the domain is actually respected.

**97. Counting with a float** · `E`
```python
n_over = (df.cluster_weight > df.capacity).sum()
share = n_over / len(df)
worst = df.iloc[n_over]          # first non-violated row, right?
```
**Bugs:**
1. If either column contains NaN, the comparison is False there (NaN never violates — a policy nobody chose), and with nullable dtypes the sum can be a *float* or masked value.
2. `df.iloc[n_over]` uses a **count as a position** — meaningless unless the frame is sorted with all violated rows first, which nothing did; and if `n_over` came out float (older pandas paths / after a mean somewhere), `iloc` raises TypeError far from the cause.
3. `len(df)` denominator counts NaN rows as "not over" — share is diluted by unknowns; decide whether unknowns are excluded or a third category.
**Fix:** Make NaN policy explicit (`mask = (w > c); mask[w.isna() | c.isna()] = pd.NA`), `int(mask.sum())`, and get "worst" by value: `df.loc[(df.cluster_weight - df.capacity).idxmax()]`.

**98. Rounding half the cents away** · `M`
```python
def add_vat(df: pd.DataFrame, rate: float = 0.09):
    df['vat'] = (df.amount * rate).round(2)
    df['total'] = df.amount + df.vat
    return df
```
**Bugs:**
1. Python/numpy `round` is half-to-even: VAT of `.125` rounds to `.12` while `.135` → `.14` — individually defensible, but finance systems typically mandate half-up; per-line differences of a cent, summed over millions of lines, produce a total that *never reconciles* with the invoicing system, by a drifting amount that looks like a bug hunt and is a rounding-convention mismatch.
2. Rounding per line then summing vs summing then rounding: the two disagree; which one the tax authority requires is a rule to look up, not to default.
3. All of it in binary floats: `0.1 * 3` artifacts feed the rounder inputs that are already off by 1 ulp — the round can flip on the float error, not the true value.
**Fix:** Integer minor units or Decimal with an explicit rounding mode (`ROUND_HALF_UP`) matching the finance spec; document line-level vs total-level rounding; reconcile in the same units as the system of record.

**99. pct_change over the gaps** · `M`
```python
def growth(df: pd.DataFrame):
    s = df.set_index('month').revenue
    return s.pct_change().fillna(0)
```
**Bugs:**
1. Missing months aren't rows — `pct_change` compares each month to the *previous existing row*: after a 3-month gap, the "monthly growth" is actually quarterly growth wearing a monthly label; with no calendar reindex the metric's period varies row to row.
2. Legacy `pct_change` default forward-filled internal NaNs (`fill_method='pad'`) — masking missing revenue as flat months (deprecated for exactly this reason); relying on version-dependent defaults for a KPI.
3. `fillna(0)` writes 0% growth over both the first month (fine) *and* any genuinely incomputable comparisons — plus a previous month of 0 revenue yields inf growth, untouched by fillna and poisoning any mean.
**Fix:** Reindex to a complete monthly calendar first (`asfreq('MS')` with explicit NaN policy), `pct_change(fill_method=None)`, handle inf (`replace`) and warm-up NaN separately and visibly.

**100. Wrap-around revenue** · `H`
```python
def totals(df: pd.DataFrame):
    # qty and price_cents are int32 from the parquet schema
    df['line_cents'] = df.qty * df.price_cents
    return df.groupby('store').line_cents.sum()
```
**Bugs:**
1. int32 × int32 stays int32 in numpy: 50,000 units × 50,000 cents = 2.5e9 > 2³¹−1 → silent **wraparound to negative** — no exception, no warning; a handful of bulk lines poison store totals with negative garbage that partially *cancels* real revenue (worse than crashing: plausible wrong totals).
2. The sum then accumulates in the same narrow dtype on some paths — even valid lines can overflow in aggregate.
3. Python-int intuition ("ints don't overflow") is exactly wrong inside numpy/pandas columns — the bug is invisible to anyone who tested with Python scalars.
**Fix:** Upcast at the operation: `df.qty.astype('int64') * df.price_cents.astype('int64')`; add a range assertion on line_cents ≥ 0; audit parquet schemas for narrow ints at ingestion.
**Senior tell:** Knowing numpy wraps silently (C semantics) while Python ints are unbounded — and testing with column dtypes, not scalars.

**101. Clipped before it mattered** · `M`
```python
def shortage_report(df: pd.DataFrame):
    df['supply'] = df.supply.clip(lower=0)
    df['demand'] = df.demand.clip(lower=0)
    df['shortage'] = (df.demand - df.supply).clip(lower=0)
    return df.groupby('region').shortage.sum()
```
**Bugs:**
1. Clipping inputs first destroys information the *output* clip needed: a negative supply (returns exceeding stock — a real signal, or a data error worth surfacing) becomes 0, understating shortage; the correct order is compute-then-clip (line 3 alone), or better, investigate why supply is negative before any clip.
2. Three silent clips with no counts: how many rows were negative, where, since when — the data-quality signal is amputated at the same moment the number is "fixed."
3. Mutates the caller's `supply`/`demand` columns as a side effect of producing a report.
**Fix:** Compute `shortage = (demand - supply).clip(lower=0)` on locals; log negative-input counts as data-quality metrics; leave inputs untouched.

**102. == np.nan** · `E`
```python
missing = df[df.capacity == np.nan]
```
**Bugs:**
1. `NaN == NaN` is False — this selects **zero rows always**; every downstream "no missing capacities ✅" is a tautology of IEEE semantics, not a fact about the data.
2. The same reflex breaks in `np.where(x == np.nan, ...)`, `replace` chains, and SQL (`= NULL`) — it's one misconception with many costumes.
**Fix:** `df[df.capacity.isna()]` — and grep the codebase for `== np.nan` / `!= np.nan`; each hit is a dead branch.

---

## I — Control Flow & API Contract Lies (103–114)

**103. The coroutine that never ran (the seed's async)** · `H`
```python
async def refresh_assignments(db, factor: float = 1.02):
    """Recompute all cluster assignments and write them back."""
    df = pd.DataFrame(db.fetch_assignments())
    df = rebalance(df, factor)
    db.write_assignments(df.to_dict('records'))
    return True

# caller, elsewhere:
ok = refresh_assignments(db)
if ok:
    log.info("assignments refreshed")
```
**Bugs:**
1. `async def` with zero awaits — pure CPU/blocking-IO code wearing an async costume. There is no reason for this function to be a coroutine.
2. The caller invokes it **without `await`**: nothing executes; `ok` is a coroutine object, which is *truthy*, so the log says "assignments refreshed" while the database was never touched. A `RuntimeWarning: coroutine was never awaited` scrolls by in stderr, unread. This is the exact failure mode the seed's `async def reassign_clusters_vehicles` invites.
3. Even if awaited, blocking calls (`db.fetch`, pandas work) inside a coroutine stall the event loop — async here is worse than useless in a real async service.
**Fix:** Remove `async` (or, if it must live in an async service, run the blocking work via `run_in_executor`); grep the codebase for calls to async functions without `await`; treat the never-awaited RuntimeWarning as an error in CI (`-W error::RuntimeWarning` in tests).
**Senior tell:** Asking "who calls this, and do they await it?" before reviewing a single line of body.

**104. The docstring is the spec, the code disagrees** · `M`
```python
def sorted_copy(df: pd.DataFrame, by: str = 'priority'):
    """Return a NEW DataFrame sorted by `by`. Never modifies the input."""
    if by not in df.columns:
        return df
    df.sort_values(by, inplace=True)
    return df
```
**Bugs:**
1. `inplace=True` mutates the caller's frame — the docstring's core promise ("never modifies") is false on the main path.
2. The early path returns the *same object* unsorted — so "return a NEW DataFrame" is false on both paths, differently: callers who mutate the result corrupt the input either way.
3. Missing-column handling by silently returning unsorted data converts a caller's typo into wrong ordering downstream; the honest behaviors are raise or an explicit sentinel — never a silent no-op that looks like success.
**Fix:** `if by not in df.columns: raise KeyError(...); return df.sort_values(by)` — copy semantics real on every path; docstrings are contracts, test them (a one-line test asserting `input.equals(before)` catches bug 1 forever).

**105. id, list, and sum walk into a function** · `E`
```python
def summarize(assignments):
    sum = 0
    list = []
    for a in assignments:
        id = a['vehicle_id']
        sum += a['cluster_weight']
        list.append(id)
    return {'total': sum, 'vehicles': list, 'n': len(list)}
```
**Bugs:**
1. Shadows three builtins (`sum`, `list`, `id`) — the seed's `id = used_cars_df.loc[...]` habit generalized. Within this function it "works"; the blast radius is any later line (or pasted-in code) calling `sum(...)`/`list(...)`/`id(...)`: `TypeError: 'int' object is not callable`, far from the assignment that caused it.
2. The whole loop is `sum(a['cluster_weight'] for a in assignments)` and a list comprehension — the shadowing exists only because the code re-implements the builtins it shadows.
**Fix:** Rename (`total`, `vehicle_ids`, `vid`); lint with the shadowing rule on (flake8 A001/builtins plugin) so this class dies in CI.

**106. The condition that is always True** · `E`
```python
def needs_review(row):
    if row['status'] == 'violated' or 'pending':
        return True
    return False
```
**Bugs:**
1. `x == 'violated' or 'pending'` parses as `(x == 'violated') or ('pending')` — a non-empty string is truthy, so the function returns True for **every row**. All statuses "need review"; the queue floods, and because flooding looks like caution, nobody debugs it.
2. The `if cond: return True / return False` ladder is itself a smell hiding the bug — written as `return row['status'] in ('violated', 'pending')`, the mistake is impossible to type.
**Fix:** `return row['status'] in {'violated', 'pending'}`; grep for `== ... or '` / `== ... or "` — each hit is this bug.

**107. Return, one indent too deep** · `M`
```python
def region_summaries(df: pd.DataFrame):
    """Summarize every region."""
    out = {}
    for region, g in df.groupby('region'):
        out[region] = {
            'total': g.amount.sum(),
            'n': len(g),
            'top_vehicle': g.loc[g.cluster_weight.idxmax(), 'vehicle_id'],
        }
        return out
```
**Bugs:**
1. `return` sits inside the loop: the function returns after the **first** region — with dict-of-one output that type-checks and passes any test that only asserts "region X is present." Groupby's ordering (sorted keys by default) decides *which* region survives, so the bug even looks deterministic.
2. `idxmax` on an all-NaN `cluster_weight` group raises; and with duplicate index labels, `g.loc[idx, 'vehicle_id']` can return a Series (the #19 trap resurfacing inside an agg).
**Fix:** Dedent the return; cover with a two-region test asserting `len(out) == df.region.nunique()` — the cheapest possible detector for early-return truncation.

**108. except Exception: continue** · `H`
```python
def load_daily(files: list[str]):
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['ts'])
            df['dow'] = df.ts.dt.dayofweek
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True)
```
**Bugs:**
1. The blanket `except ... continue` was written for occasional corrupt files, but it also swallows *programming* errors: rename `ts` upstream and every file raises KeyError → every file is skipped → `pd.concat([])` raises "No objects to concatenate" **at the concat**, pointing away from the cause; or worse, one old-schema file loads and the pipeline "succeeds" on 1/300 of the data.
2. Nothing is logged or counted — the skip rate is invisible; a pipeline that can silently process 0–100% of its input has no defined behavior.
3. `except Exception` also eats `KeyboardInterrupt`'s cousins? (No — KeyboardInterrupt/SystemExit are BaseException — but `MemoryError` *is* caught: an OOM on one file becomes a silent skip.)
**Fix:** Catch the narrow, expected errors (`ParserError`, `EmptyDataError`), log each skip with the filename, count skips and **fail if skip_rate > threshold**; let programming errors crash. Silent-skip loops need a success-rate contract.
**Senior tell:** "What percentage of files does this *have* to load for the output to be valid?" — turning error handling into a stated SLO.

**109. The flag reset in the wrong place** · `M`
```python
def settle(df: pd.DataFrame):
    """Keep applying single-step fixes until stable."""
    while True:
        for i in df.index:
            improved = False
            if fixable(df, i):
                apply_fix(df, i)
                improved = True
        if not improved:
            break
    return df
```
**Bugs:**
1. `improved` is reset **inside the for loop**: after the loop, it reflects only the *last* row's outcome. If the final row wasn't fixable, the while exits even though earlier rows changed this pass (premature stop, unstable output); if the final row happens to be perpetually "fixable," the loop never exits. Both failure directions from one misplaced line — the seed's `improved` flag, misindented.
2. `NameError` lurking: with an empty index, `improved` is never bound and `if not improved` raises.
3. Fixes mutate `df` while iterating its index — safe for value edits, a landmine the day `apply_fix` adds/drops rows.
**Fix:** Initialize `improved = False` before the for; or count fixes per pass (`n_fixed = sum(...)`) and loop `while n_fixed:` — counting is self-documenting and immune to placement bugs; add a max-pass guard.

**110. Cleanup in the for-else** · `M`
```python
def find_and_lock(vehicles: list[dict], min_cap: float):
    lock_fleet()
    for v in vehicles:
        if v['capacity'] >= min_cap and not v['locked']:
            v['locked'] = True
            return v
    else:
        unlock_fleet()
    return None
```
**Bugs:**
1. `for/else` runs the else only when the loop *doesn't* break/return — so on the **success** path (return inside the loop), `unlock_fleet()` never runs: the fleet-wide lock leaks on every successful call, and the system deadlocks after the first success. The else-clause placement reads like "cleanup," behaves like "cleanup only on failure."
2. Even the failure path leaks if `v['capacity']` raises (missing key): no try/finally around a resource acquisition.
3. Function both mutates shared state (`locked`) and holds a global lock across an unbounded scan — locking granularity and duration are design smells past the syntax bug.
**Fix:** `try: ... finally: unlock_fleet()` — resource release must be unconditional; drop the for-else (its semantics surprise most readers; if used, comment it); lock per-vehicle or use a shorter critical section.

**111. Retry forever, politely** · `M`
```python
def fetch_with_retry(client, url, attempts=3):
    try:
        return client.get(url)
    except TimeoutError:
        time.sleep(1)
        return fetch_with_retry(client, url, attempts)
```
**Bugs:**
1. `attempts` is passed through **undecremented** — the parameter is decoration; the function retries forever on a persistent timeout (recursion depth grows ~1/sec until RecursionError hours later, the slowest possible crash).
2. Recursion for retries also stacks a frame per attempt and turns the eventual failure into `RecursionError` instead of the timeout that caused it — the loop version keeps the real exception.
3. Fixed 1s sleep, no backoff/jitter: N workers retrying in sync hammer the recovering service in lockstep (thundering herd).
**Fix:** Iterative loop: `for i in range(attempts): try/except: sleep(base * 2**i + jitter)`, re-raise the last error after exhaustion; log attempt counts.

**112. Sometimes a frame, sometimes None** · `M`
```python
def violations_report(df: pd.DataFrame, factor: float):
    bad = df[df.cluster_weight > df.capacity * factor]
    if len(bad) == 0:
        print("no violations")
    else:
        return bad.groupby('region').size().reset_index(name='n')
```
**Bugs:**
1. The happy path (no violations) **returns None** implicitly — callers doing `report.merge(...)` or `len(report)` crash precisely on the good days; the crash correlates with success, which makes for maximally confusing incident timing.
2. `print` as the reporting channel: invisible in services, uncapturable in tests, and it *replaces* the return value instead of accompanying one.
3. Inconsistent return types (None vs DataFrame) make every caller grow `if report is not None` scar tissue — the empty case should be an **empty frame with the same schema**.
**Fix:** Always return the same shape: `return bad.groupby('region').size().reset_index(name='n')` — which is naturally empty when `bad` is; log via `logging`, not print. "Empty result ≠ absent result" as the contract rule.

**113. Validation by assert** · `H`
```python
def apply_capacity_update(df: pd.DataFrame, updates: list[dict]):
    assert all(u['capacity'] > 0 for u in updates), "bad capacity"
    assert df.vehicle_id.is_unique
    for u in updates:
        assert df.vehicle_id.eq(u['vehicle_id']).any(), f"unknown {u['vehicle_id']}"
        df.loc[df.vehicle_id == u['vehicle_id'], 'capacity'] = u['capacity']
    return df
```
**Bugs:**
1. `python -O` strips every assert — under optimized bytecode this function applies negative capacities to unknown vehicles on a duplicate-id frame without a murmur. Input validation guarding *production data writes* must be real raises; asserts are for internal invariants in dev.
2. The unknown-vehicle assert is O(n) per update inside the loop — O(n·m) validation for what one `isin` computes; and the check + write scan the frame twice per update.
3. Partial application: the update loop mutates `df` as it goes, so a failed assert at update #7 leaves 1–6 applied — no atomicity; callers can't tell what state they hold after the exception.
**Fix:** Explicit `raise ValueError` validations up front (all-or-nothing: validate the whole batch, then apply via a single indexed assignment `df.set_index('vehicle_id').capacity.update(...)`-style); keep asserts for invariants that can't be caused by inputs.
**Senior tell:** Knowing `-O` deletes asserts — and framing the loop as a missing transaction, not a missing check.

**114. The stringly-typed dry run** · `M`
```python
def sync_assignments(df: pd.DataFrame, dry_run=True):
    dry_run = os.environ.get('SYNC_DRY_RUN', dry_run)
    if dry_run == False:
        push_to_db(df)
    elif not dry_run:
        push_to_db(df)      # legacy path, keep in sync
    else:
        log.info("dry run: %d rows", len(df))
```
**Bugs:**
1. `os.environ.get` returns a **string**: `'False'`, `'0'`, `'no'` are all truthy — once the env var exists, the function is *permanently* in dry-run regardless of value; ops sets `SYNC_DRY_RUN=False` to enable the sync and disables it instead. Config parsing is type coercion, not assignment.
2. `dry_run == False` and `not dry_run` are not the same predicate once dry_run can be a string (`'False' == False` → False; `not 'False'` → False) — the two "kept in sync" branches diverge exactly when bug 1 fires, and the duplicate push line is a merge-conflict fossil waiting to double-write if the predicates ever both pass.
3. `== False` instead of `is False`/`not` — style, but it's the reason branch 1 and 2 could coexist unnoticed.
**Fix:** Parse explicitly: `dry_run = env_bool('SYNC_DRY_RUN', default=dry_run)` with a real str→bool mapping that raises on junk; one predicate, one push site; boolean flags never compared with `==`.

---

## J — Concurrency, IO & Scale (115–124)

**115. N+1 with a side of injection** · `H`
```python
def enrich_with_driver(df: pd.DataFrame, conn):
    names = []
    for vid in df.vehicle_id:
        row = conn.execute(
            f"SELECT driver_name FROM vehicles WHERE id = {vid}").fetchone()
        names.append(row[0] if row else None)
    df['driver'] = names
    return df
```
**Bugs:**
1. One query per row (N+1): thousands of round-trips for what a single `WHERE id IN (...)` — or one indexed join — returns; latency scales with rows, and the DB connection is held hot for the whole scan.
2. f-string SQL: `vehicle_id` sourced from data is interpolated raw — injection if ids are ever strings, and even with ints, a NaN renders as `id = nan` → SQL error mid-loop → half-enriched frame (partial mutation again).
3. `row[0] if row else None` silently maps *missing vehicles* to None — the unmatched rate goes unmeasured (the #56 laundering pattern, database edition).
**Fix:** One parameterized batch: `SELECT id, driver_name FROM vehicles WHERE id = ANY(%s)` (or chunked `IN` lists), then `df.vehicle_id.map(dict(rows))`; count and log unmatched; parameters always, f-strings never.

**116. Append mode, header every time** · `M`
```python
def export_batches(batches: list[pd.DataFrame], path: str):
    for b in batches:
        f = open(path, 'a')
        b.to_csv(f, index=True)
        # f.close()  # slows things down, GC handles it
    return path
```
**Bugs:**
1. `to_csv` with defaults writes the **header per batch** — the output file has a header row between every chunk; naive readers ingest them as data rows (string contamination in numeric columns), and `pd.read_csv` only survives by luck of dtype coercion.
2. `index=True` writes the frame index as an unnamed first column — per batch, restarting from 0: a phantom column of repeating 0..n that downstream code will eventually treat as an id.
3. Unclosed handles: CPython's refcounting usually saves you *here*, but buffered writes can be lost on interpreter crash and on other runtimes (PyPy) the file may not flush for a long time; the commented-out close with a performance excuse is the anti-pattern preserved as documentation.
4. Append mode + reruns: a retried job **doubles** the file; no truncate/temp-file-rename atomicity.
**Fix:** `with open(path, 'w') as f:` once; `b.to_csv(f, header=(i == 0), index=False)`; write to a temp file and atomically rename; or skip the loop: `pd.concat(batches).to_csv(...)` when memory allows.

**117. Threads sharing a frame** · `H`
```python
counts = pd.DataFrame({'region': regions, 'n': 0}).set_index('region')

def worker(events_chunk):
    for r in events_chunk.region:
        counts.loc[r, 'n'] = counts.loc[r, 'n'] + 1

with ThreadPoolExecutor(8) as ex:
    ex.map(worker, chunks)
```
**Bugs:**
1. `counts.loc[r,'n'] = counts.loc[r,'n'] + 1` is a read-modify-write: two threads read the same value and both write value+1 — lost updates. The GIL serializes *bytecodes*, not this three-step sequence; pandas ops release the GIL unpredictably, and the final counts undercount nondeterministically (worse: they're *plausible*).
2. Threads for CPU-bound counting buy nothing under the GIL anyway — all of the race, none of the speedup.
3. `ex.map` results are never consumed — exceptions in workers are **silently swallowed** until the executor exits (and with map's lazy iterator in some patterns, not even then): a worker can die on row 1 and the job "succeeds."
**Fix:** Shard-and-reduce: each worker returns its own `value_counts()`, main thread sums them (`sum(results)`), no shared mutable state; consume `ex.map` results (or use `submit` + `result()`) so exceptions propagate; for real parallel CPU work, processes or vectorize (a single `df.region.value_counts()` beats all of it).

**118. Caching the uncacheable** · `M`
```python
@lru_cache(maxsize=128)
def region_stats(df: pd.DataFrame, region: str):
    g = df[df.region == region]
    return g.amount.sum(), g.amount.mean()
```
**Bugs:**
1. `lru_cache` hashes its arguments; DataFrames are unhashable → **TypeError on the first call**. The decorator ships because the module imports fine and the function was never unit-tested.
2. The tempting "fix" (`key = str(df)` wrappers, or hashing `df.values.tobytes()`) either explodes memory/CPU (serializing the frame per call costs more than the groupby) or silently collides — and any content-keying is invalidated the moment `df` mutates.
3. Design smell underneath: caching per-(frame, region) fights the grain — compute *all* regions once (`df.groupby('region').amount.agg(['sum','mean'])`) and look up; the cache is compensating for a loop-shaped access pattern.
**Fix:** Cache on stable scalar keys only (dataset version id + region), or restructure to one groupby result passed around; if memoization of frame-derived results is truly needed, key on an explicit `dataset_version` the caller owns.

**119. Read everything, keep 2%** · `E`
```python
def store_history(path: str, store_id: int):
    df = pd.read_parquet(path)              # 40 GB, 500 stores
    df = df[df.store_id == store_id]
    return df[['ts', 'amount']]
```
**Bugs:**
1. Loads all columns of all stores into memory to keep one store's two columns — parquet supports **column projection** (`columns=['ts','amount','store_id']`) and, on partitioned/rowgroup-statistics data, **predicate pushdown** (`filters=[('store_id','==',store_id)]`); the code pays 40 GB of IO+RAM for ~80 MB of answer.
2. If this function is called per store in a loop (it will be), the same 40 GB is re-read 500 times — the layer above compounds the layer below.
**Fix:** `pd.read_parquet(path, columns=['ts','amount'], filters=[('store_id','==',store_id)])`; partition the dataset by `store_id` if this access pattern is common; for many stores, one filtered scan grouped afterwards.

**120. Mean of chunk means** · `H`
```python
def big_csv_mean(path: str):
    means = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        means.append(chunk.amount.mean())
    return sum(means) / len(means)
```
**Bugs:**
1. The last chunk is almost never full-size: averaging chunk means weights its rows more than everyone else's — a biased global mean whose error depends on file length modulo chunksize (changes when the file grows: a metric that drifts with row count, not with data).
2. `chunk.amount.mean()` skips NaN per chunk, and the unweighted combine then mixes different effective ns — a second, independent weighting error.
3. Any fully-empty/all-NaN chunk contributes NaN and poisons the total.
**Fix:** Accumulate sufficient statistics: `total += chunk.amount.sum(); n += chunk.amount.count()`, return `total / n` — exact, NaN-aware, chunk-size-independent. Same principle for variance (Welford / sum of squares), which is the follow-up question.
**Senior tell:** "Combine sufficient statistics, not statistics" as the stated rule for any chunked/distributed aggregate.

**121. Pickling the DataFrame into every worker** · `M`
```python
def score_all(df: pd.DataFrame, ids: list[int]):
    def score_one(i):
        sub = df[df.entity_id == i]
        return i, model_score(sub)
    with multiprocessing.Pool(12) as p:
        return dict(p.map(score_one, ids))
```
**Bugs:**
1. `score_one` is a closure over `df`: multiprocessing pickles the function *and its captured frame* to every worker — 12 copies of the full DataFrame in RAM, plus serialization time that can exceed the scoring work. (And locally-defined closures aren't picklable by the default pickler at all → `AttributeError: Can't pickle local object` on many platforms — the function may not even run.)
2. Each task then filters the *whole* frame per id — the N+1 scan pattern inside each process.
3. Un-chunked `p.map` over thousands of ids has per-task IPC overhead; `chunksize` unset.
**Fix:** Pre-split by group once (`groups = dict(tuple(df.groupby('entity_id')))`) and map over `(i, groups[i])` pairs; or use shared memory / initializer-loaded data per worker; module-level worker function; set `chunksize`. First question though: is `model_score` heavy enough to beat a vectorized batch call?

**122. print() as the bottleneck** · `M`
```python
def process(df: pd.DataFrame):
    out = []
    for i, row in enumerate(df.itertuples()):
        out.append(transform(row))
        print(f"processed {i+1}/{len(df)}: {row.order_id}")
    return pd.DataFrame(out)
```
**Bugs:**
1. A per-row `print` flushes to stdout each iteration — on containerized/log-shipped stdout this is IO-bound and can dominate runtime by 10× for cheap `transform`s; the progress reporting *is* the workload.
2. Logging row-level PII-ish identifiers (order ids) at INFO firehose volume — log storage costs and compliance both flag it.
3. `len(df)` per iteration is O(1) (fine) but the format work per row isn't; progress belongs at intervals (`if i % 10_000 == 0`) or to `tqdm`, and results-building may vectorize away the whole loop depending on `transform`.
**Fix:** Interval logging or `tqdm(total=len(df))`; `logging` with levels instead of print; then ask whether `transform` vectorizes (usually the real fix).

**123. Retrying everything, including Ctrl-C** · `M`
```python
def robust_call(fn, *args, retries=5):
    for i in range(retries):
        try:
            return fn(*args)
        except:
            time.sleep(2)
    raise RuntimeError("all retries failed")
```
**Bugs:**
1. Bare `except:` catches `BaseException` — including `KeyboardInterrupt` and `SystemExit`: the process ignores Ctrl-C/orchestrator shutdown for up to 5×2s per call site, and retries work that was being *cancelled*. Retry wrappers must catch narrow, retryable errors (`ConnectionError`, `Timeout`, 5xx), never bare.
2. Retrying non-idempotent `fn` blindly: a call that timed out **after** the server committed (payment, insert) executes twice — retry safety is a property of the operation, not the wrapper; the wrapper's API should demand it (`assume_idempotent=` or idempotency keys).
3. The final `RuntimeError` discards the last real exception (`raise ... from e` missing) — the postmortem loses the actual cause; constant 2s sleep, no jitter (thundering herd, again).
**Fix:** `except (TimeoutError, ConnectionError) as e:` with exponential backoff+jitter, `raise RuntimeError(...) from e`, and an explicit idempotency contract in the signature/docs.

**124. nightly_rebalance.py (the finale — triage it)** · `H`
```python
async def nightly_rebalance(conn):
    """Nightly job: rebalance all regions and persist. Safe to re-run."""
    cfg = DEFAULTS
    cfg.update(json.loads(os.environ.get('REBALANCE_CFG', '{}')))
    df = pd.DataFrame(conn.execute("SELECT * FROM assignments").fetchall())
    bad = df[df.cluster_weight > df.capacity * cfg['factor']]
    ok = df[df.cluster_weight <= df.capacity * cfg['factor']]
    for i in bad.index:
        for j in ok.index:
            try:
                if swap_improves(df, i, j):
                    do_swap(df, i, j)
                    conn.execute(f"UPDATE assignments SET vehicle_id = "
                                 f"{df.loc[j, 'vehicle_id']} WHERE id = {j}")
            except:
                continue
    refresh_cache(df)          # async helper from utils
    return True
```
**Bugs (triage: which three do you fix before dawn?):**
1. **Stale snapshots + per-swap DB writes:** `bad`/`ok` never refresh while `df` mutates (the seed bug), and each swap UPDATEs only row `j` — row `i`'s new vehicle is never persisted: the database ends the night **desynchronized from the frame that "fixed" it**. Data-corrupting, top priority.
2. **`except: continue` around the swap+write:** any error (including the SQL errors the f-string interpolation will eventually produce) silently skips — combined with (1), partial writes with no record. Also catches BaseException.
3. **`refresh_cache(df)` is async and never awaited** inside an `async def` — the cache refresh silently doesn't happen; tomorrow serves yesterday's assignments (the #103 failure, embedded).
4. `cfg = DEFAULTS; cfg.update(...)` corrupts module config process-wide (#40) — "safe to re-run" is false: the second run inherits the first run's env overrides even without the env var.
5. f-string SQL with frame values (#115), `SELECT *` schema fragility, no transaction around the swap pair (each UPDATE autocommits half a swap), and no row-count/reconciliation check before `return True` — the function cannot fail, which means it cannot be trusted to have succeeded.
**Fix (the shape of the rewrite):** Pure function computes the full swap plan on arrays (recomputing masks per accepted swap); one **transaction** applies the plan with parameterized batched UPDATEs to *both* rows per swap; verify (re-select and reconcile violations count) before commit; `await refresh_cache(df)`; config via `{**DEFAULTS, **overrides}`; narrow exceptions, structured logging, and a return value that reports what changed.
**Senior tell:** Triaging by blast radius (data corruption → silent skip → stale cache) before touching style — and saying out loud that a job which can't fail can't be trusted.

---

## Scoring guide

| Signal | Weak (1–2) | Strong senior (4–5) |
|---|---|---|
| **Data-flow tracing** | Reads top to bottom, comments on style | Traces what each frame/index/dtype *is* at every line; finds the stale snapshot and the desync |
| **Boundary discipline** | Misses `<` vs `<=`, tie, and NaN cases | Checks every paired condition for consistency; states NaN/zero/empty policy per function |
| **Contract reading** | Ignores docstrings | Diffs docstring vs behavior (mutation, return type, async) and treats mismatches as bugs |
| **Performance model** | "Use vectorization" generically | Names the complexity, the dominating cost (indexing vs arithmetic vs IO), and the specific idiom |
| **Failure semantics** | Happy-path review | Asks what state remains after a mid-loop exception; demands atomicity/reconciliation on writes |
| **Proof orientation** | Asserts bugs from memory | Proposes the one-line test/experiment that demonstrates each bug (the empirical reflex) |

**Usage.** One anchor (`1`, `28`, `66`, `78`, `124`) plus 3–4 short items per 45-minute round. The anchors are triage exercises — score *prioritization* (data corruption before style) as heavily as detection. Highest-signal single items: **1** (stale snapshots — the seed's disease), **63** (asof direction = temporal leakage), **93** (float money), **103** (the unawaited coroutine), **120** (sufficient statistics). A candidate who finds bugs *and* names the missing test for each is operating at the level this bank is built to detect.
