# Senior Data Scientist — EDA Question Bank (124 Questions)

> **What this is.** Exploratory Data Analysis as it's actually tested in senior interviews: not "make a histogram," but *what do you check, what does the finding mean, and what do you do about it*. Each question has a difficulty tag and an answer key with the discriminating points an interviewer listens for.
>
> **How to run one.** Pose the scenario, let the candidate think aloud. Seniors show a *repeatable interrogation process*, connect findings to modeling consequences, and know which checks leak and which don't. Juniors recite tool names.
>
> **Difficulty:** `E` screen · `M` core round · `H` senior signal.

---

## A — First Contact With a Dataset (1–10)

**1. You receive an unknown CSV. What are your first five commands, and what is each one for?** · `E`
**A:** Shape + dtypes (`df.info()`) — size, memory, parsing sanity; `df.sample(10)` (not just `head`) — files are often sorted, so head hides junk; `df.describe(include='all')` — ranges, sentinels, cardinality; missingness summary (`df.isna().mean()`); duplicate check on the presumed key. The senior tell is *why* each: every command tests a hypothesis about how the data can be broken.

**2. `describe()` shows different counts per column. What does that mean and what do you do?** · `E`
**A:** `count` excludes NaN, so differing counts = per-column missingness. Quantify it, then ask *why* those columns and whether missingness co-occurs (block patterns = upstream join or source system). It's the first thread to pull, not a nuisance to impute away.

**3. The `age` column has dtype `object`. Likely causes and the fix?** · `E`
**A:** Mixed content: strings like "unknown"/"N/A", thousands separators, stray whitespace, or a corrupted row shifting columns. Inspect `df['age'][pd.to_numeric(df['age'], errors='coerce').isna() & df['age'].notna()].unique()` to see the offenders, then decide per case (map, coerce, fix parsing). Blind `astype(float)` destroys the evidence.

**4. Why is it dangerous that `customer_id` parsed as int64?** · `M`
**A:** Leading zeros are lost (id "00423" ≠ 423), very long ids overflow or lose precision via float round-trips, and numeric ids invite accidental use as model features (magnitude is meaningless; also a memorization/leakage vector). Cast ids to string on ingest and exclude from features.

**5. `head()` looks clean but `sample(20)` shows garbage rows. What happened and what's the lesson?** · `M`
**A:** The file is ordered (by time or source), so the top rows are the oldest/cleanest; mid-file you find repeated header rows, footer summaries, or a schema change. Lesson: always inspect a *random* sample and the *tail*; also compare parsed row count to `wc -l` — silent row drops from bad quoting/encoding are common.

**6. The DataFrame uses 10× the file's size in memory. Why, and what do you do?** · `M`
**A:** Object-dtype strings dominate (each Python string carries overhead), plus float64/int64 defaults. Fix: `dtype=` on read, convert low-cardinality strings to `category`, downcast numerics, load only needed columns, or move to Parquet. Knowing `category` dtype is the practical tell.

**7. `describe()` shows max values of 999, −1, or 9999 in several columns. Interpretation?** · `E`
**A:** Sentinel codes for missing/unknown masquerading as numbers — legacy-system convention. Left in, they poison means, correlations, and models. Confirm against a data dictionary or by the spike in the histogram, convert to NaN, and record the mapping.

**8. How do you verify you received *all* the data?** · `M`
**A:** Reconcile against the source: row counts vs the system of record, sums of a money column vs a finance report (tie-out), date coverage vs expectation, per-partition counts. Parsing can silently drop rows (quoting, encoding); ETL can lose partitions. EDA starts with "is this the dataset I think it is."

**9. What does one row represent — and how do you *test* your answer?** · `M`
**A:** State the assumed grain (one row = one order? one order-line? one customer-day?) and test key uniqueness: `df.groupby(key).size().max() == 1`. Grain confusion is the root of double-counting, fan-out joins, and wrong denominators; it must be established before any aggregate is trusted.

**10. First checks on the date columns?** · `M`
**A:** Range (future dates? epoch defaults like 1970/1900?), timezone handling, parsing correctness — a classic tell: if the max "day" component is ≤ 12, day/month may be swapped; check gaps and volumes by date for missing partitions; verify `created_at` vs `updated_at` semantics (which one is event time?). Dates are where silent corruption concentrates.

---

## B — Missing Data (11–22)

**11. Explain MCAR / MAR / MNAR with a business example of each, and why the distinction matters.** · `M`
**A:** MCAR: missing independent of everything (random logger crashes). MAR: missingness depends on *observed* variables (older customers skip the app field — explained by channel). MNAR: depends on the *unobserved value itself* (high earners hide income). It matters because mean/model imputation is defensible under MCAR/MAR but biased under MNAR, where the missingness itself is signal to model.

**12. How do you *test* whether missingness is informative?** · `H`
**A:** Create a missing-indicator and check its association with the target and with other features (group comparison, chi-square/logit); compare distributions of other variables between missing and non-missing rows; Little's MCAR test exists but the pragmatic evidence is predictive: if `is_missing` has signal, it's not MCAR — keep the indicator as a feature (if it doesn't leak).

**13. A column is 40% missing. Drop or keep — what's your decision framework?** · `M`
**A:** Depends on: mechanism (informative? keep at least the indicator), signal in the observed 60% (quick univariate/model check), availability at serving time, and whether missingness is stable across time/segments. "Drop above X%" as a blanket rule is the junior answer; a 40%-missing column with MNAR structure can be a top feature via its indicator.

**14. The missingness matrix shows blocks of columns missing together. What does that pattern mean?** · `M`
**A:** Co-missing blocks = structural origin: a failed join (all columns from one table null for unmatched rows), an optional form section, or a source added later in time. Treat the block as one decision (fix the join, add a "section present" flag), not as N independent imputation problems.

**15. How would you detect missing values encoded as 0?** · `M`
**A:** Domain impossibility (weight = 0, price = 0) and a distributional spike at exactly 0 inconsistent with the rest (e.g., a smooth positive distribution plus a point mass). Cross-check: do zero rows also have other fields absent? Compare zero rate across sources/dates — a jump when a schema changed is the confession.

**16. Mean vs median vs model-based imputation — what does each get wrong?** · `M`
**A:** Mean: sensitive to skew/outliers and *shrinks variance*, attenuating correlations and overstating confidence. Median: robust to skew but still collapses variance and distorts multivariate structure. Model-based (kNN/iterative): preserves relationships but risks leaking the target (never include y), adds variance you should propagate, and can invent precision. All three are wrong under MNAR without an indicator.

**17. Where does imputation belong relative to the train/test split?** · `M`
**A:** Fit imputation on train only, apply to test/serving — imputing on pooled data leaks test statistics. Practically: put the imputer inside the pipeline so CV re-fits it per fold. The EDA twist: *exploring* missingness patterns on all data is fine; *estimating* imputation parameters is not.

**18. When do missing-indicator features help, and when are they themselves leakage?** · `H`
**A:** They help under MAR/MNAR where the fact of missingness carries signal (income hidden → risk). They leak when missingness is determined *after* the outcome — e.g., `resolution_notes` is empty until a case closes, so "is_missing" encodes the label's timing. Test availability at prediction time, not just correlation.

**19. Time-series gaps: forward-fill, interpolate, or leave as NaN?** · `M`
**A:** Depends on the mechanism: sensor-down (value existed, unobserved) → interpolation may be defensible; no-event (no sales that day) → the truth is 0, not the last value; irregular reporting → aggregation to a coarser grain may dissolve it. ffill fabricates flat plateaus that inflate autocorrelation; whatever you choose, carry an `imputed` flag and never let filled spans train anomaly detectors.

**20. Missing values in a categorical: impute the mode or create an "Unknown" level?** · `E`
**A:** Almost always "Unknown" as an explicit level: it preserves the information that data was absent, handles MNAR, and avoids distorting the mode's frequency. Mode imputation silently inflates the majority class and erases a potentially predictive signal.

**21. Some rows are missing the *target*. What are your options?** · `M`
**A:** Never impute the target for supervised training — you'd train on your own guesses. Options: drop for the supervised task; investigate *why* (label pipeline lag → right-censoring, a bias if recent rows are systematically unlabeled); optionally exploit unlabeled rows via semi-supervised learning or pretraining. The why-check matters most: missing labels are rarely random in time.

**22. A column's missingness jumps from 2% to 60% after a specific date. What happened?** · `M`
**A:** A schema/source change: field renamed, form redesigned, integration dropped, or a new source lacking the field was merged. Confirm by source/system column if present. Modeling consequence: the feature has a validity window — either restrict training to it, engineer a replacement, or interact the feature with an era flag. Also: alert whoever owns the pipeline.

---

## C — Distributions & Univariate Analysis (23–34)

**23. Mean is 3× the median. What does that tell you and what follows?** · `E`
**A:** Strong right skew (a few large values dominate): report medians/quantiles for "typical" statements; expect MSE models to chase the tail; consider log/GLM approaches; and check whether the tail is genuine (whales) or data error. One number to add: the top-1% share of the total.

**24. Two analysts show histograms of the same column that tell opposite stories. How?** · `M`
**A:** Bin width/anchoring: coarse bins hide bimodality and spikes; shifted bin edges move mass across visual boundaries. Antidotes: vary bin width deliberately, overlay a KDE (with its own bandwidth caveat), or use the ECDF, which has no binning choices at all.

**25. Why prefer an ECDF over a histogram for serious comparisons?** · `M`
**A:** No bins, no bandwidth — every point shown; quantiles read directly; multiple groups overlay cleanly without transparency games; tails visible. Histograms are for shape intuition; ECDFs are for defensible comparison.

**26. When do you log-transform, and what are the traps?** · `M`
**A:** Multiplicative/right-skewed positive data (revenue, latency) where relative changes matter. Traps: log(0) undefined → `log1p` or model zeros separately (zero-inflation is a structure, not a nuisance); interpretation changes to percentage terms; and back-transforming means introduces retransformation bias (Jensen) — correct it or use a log-link GLM instead.

**27. A key metric is clearly bimodal. What's your next move?** · `M`
**A:** Bimodality = mixed populations. Hunt the separating variable: split the histogram by candidate segments (B2B/B2C, platform, region, source system) until the modes separate; if none separates it, suspect a units/pipeline mix (two sources measuring differently). Never model or summarize a mixture with a single mean.

**28. The distribution has spikes at exact values (100, 500, 1000). Meaning?** · `E`
**A:** Heaping: human-entered rounded values, price points, or system defaults. It's a fact about the *measurement process* — evaluation should tolerate rounding, models shouldn't overfit spike boundaries, and any "movement between buckets" analysis is partly measuring rounding behavior.

**29. Kurtosis is huge. Practical consequences?** · `M`
**A:** Heavy tails: MSE dominated by rare extremes; z-score outlier rules mislabel; sample means converge slowly and CIs based on normality undercover; consider robust losses/statistics (median, MAD, Huber) and model the tail explicitly if extremes matter to the business.

**30. 55% of the target values are exactly zero. How does this reshape everything downstream?** · `M`
**A:** Zero-inflation = two processes (whether anything happens × how much). Consequences: Gaussian/MSE mean-collapses; MAPE undefined; consider hurdle/two-stage or Tweedie models; evaluate the zero/non-zero decision and the size separately; EDA question: are zeros "no demand" or "couldn't observe" (censoring)?

**31. How do truncation and censoring show up in a distribution's shape, and why care?** · `H`
**A:** Censoring: a point mass exactly at a boundary (values clipped to a sensor max, "30+ days" coded as 30). Truncation: the distribution just *stops* — values beyond the limit were never recorded (orders above credit limit rejected). Both bias naive means and any model treating the boundary as real data; they demand censored/truncated likelihoods or explicit modeling of the mechanism.

**32. When is Benford's law a legitimate EDA tool?** · `H`
**A:** For naturally occurring, multi-magnitude positive numbers (transaction amounts, populations): first-digit frequencies should follow log₁₀(1+1/d). Deviations flag fabrication, thresholding behavior (amounts engineered under approval limits), or unit mixing. Not valid for constrained ranges (ages, percentages) or assigned numbers (ids, prices set from a list).

**33. Should you run a normality test (Shapiro–Wilk) before modeling? The junior did.** · `M`
**A:** Mostly no: at large n these tests reject trivially non-normal-but-fine data, and at small n they can't detect real problems — significance ≠ severity. Look at the QQ plot and ask *which* assumption matters for the method at hand (residual normality for small-sample inference; nothing about marginal X for tree models). Test-driven ritual is the junior tell.

**34. How do you compare a distribution across two groups, beyond comparing means?** · `M`
**A:** Overlaid ECDFs (differences at every quantile), quantile deltas (median, p90, p99 — tails often diverge when means don't), standardized mean difference for magnitude, KS statistic if a single number is needed. Report *where* the distributions differ; a mean comparison hides tail regressions.

---

## D — Outliers & Anomalies (35–44)

**35. Z-score vs IQR vs isolation-based outlier detection — assumptions of each?** · `M`
**A:** Z-score assumes roughly normal, symmetric data and is itself corrupted by the outliers it hunts (mean/σ not robust). IQR (Tukey fences) is robust and distribution-light but univariate and arbitrary at 1.5×. Isolation Forest / LOF handle multivariate and shape but need parameter care and produce scores, not verdicts. The senior point: detection flags candidates; *judgment* classifies them.

**36. Distinguish an outlier in X, an outlier in Y, and an influential point.** · `H`
**A:** Y-outlier: unusual response at a typical X — inflates error variance, mild effect on the fit. X-outlier (high leverage): extreme predictor value — *potentially* rotates the whole fit. Influential point: leverage × discrepancy actually moving coefficients (Cook's distance). Treatment differs: a leverage point that follows the trend is fine; a discrepant one owns your slope.

**37. Why does the 3σ rule fail exactly when you need it, and what's the robust alternative?** · `M`
**A:** With heavy tails or contamination, the mean and σ are dragged by the outliers themselves (masking), so 3σ flags too little; on skewed data it flags the wrong side. Use median ± k·MAD (robust to ~50% contamination) or quantile-based fences; for multimodal data, none of these — segment first.

**38. Give an example of a multivariate outlier invisible in every univariate view, and the tool for it.** · `H`
**A:** Height 150 cm (plausible) and weight 120 kg (plausible) — jointly improbable; or order_value ok and items_count ok but value-per-item impossible. Univariate fences pass it. Mahalanobis distance (with a robust covariance like MCD) or model-based scores (Isolation Forest) catch joint improbability; also engineered ratios turn joint checks into univariate ones.

**39. The junior asks: "should I remove outliers?" Give the decision framework.** · `H`
**A:** First classify: (1) data error (impossible values, unit mistakes) → correct or remove with a log; (2) genuine extreme from the population you must serve (whale customers) → keep; robustify the *model/loss*, not the data; (3) different process (fraud, test accounts) → route/model separately. Then: document every exclusion, re-run key results with and without (sensitivity), and never delete based on the target's residual alone — that's manufacturing fit.

**40. Winsorizing vs trimming vs robust losses — trade-offs?** · `M`
**A:** Trimming discards information and changes the estimand (mean of the middle 98%). Winsorizing keeps the row but lies about its value — gentler, still biased. Robust losses (Huber, quantile) keep all data and down-weight extremes inside the model — usually the cleanest for prediction. For *reporting*, prefer quantiles over doctored means.

**41. Outliers in the target vs outliers in features — does treatment differ?** · `M`
**A:** Yes. Feature outliers: verify validity, consider transforms/clipping with serving-time parity (same clip at inference). Target outliers: much more dangerous to remove — you may be deleting exactly the events that matter (huge orders); prefer loss functions/quantile models that accommodate them, and ask the business whether tails are the point.

**42. Two hundred rows are extreme, all sharing one timestamp/batch id. Read?** · `E`
**A:** Not a statistical outlier problem — an *incident*: a pipeline replay, a unit change in one batch, a stuck sensor, a promo you didn't know about. Cluster anomalies by metadata (time, source, loader version) before any per-row treatment; one root cause, one fix.

**43. Where does outlier handling sit relative to the split?** · `M`
**A:** Thresholds (fences, winsor limits, robust location/scale) are *fit on train only* and applied to test/serving — pooled thresholds leak test-set extremes into the rule. Same pipeline discipline as scaling/imputation. Cheap check: does the transform object get fit inside CV folds?

**44. You removed 1.3% of rows as outliers. What do you owe the reader of your analysis?** · `H`
**A:** A written rule (criteria, thresholds, counts by reason), the sensitivity delta (headline results with vs without), confirmation the rule is applied identically in production, and a stored artifact of excluded rows. Undocumented exclusions are how "reproducible" analyses stop reproducing — and how motivated cleaning becomes p-hacking.

---

## E — Relationships & Correlation (45–56)

**45. Pearson vs Spearman vs Kendall — when does each earn its place?** · `E`
**A:** Pearson: linear association on roughly continuous, outlier-tamed data. Spearman: monotonic association, robust to outliers and transforms — the safer default for EDA. Kendall: like Spearman but with cleaner interpretation (concordant-pair probability) and better small-sample behavior, at higher compute cost. Reporting Pearson on skewed, outlier-ridden data is the common junior error.

**46. Correlation ≈ 0. The junior writes "no relationship." What do you check?** · `E`
**A:** Plot it: U-shapes, thresholds, and interactions all yield r ≈ 0 with strong structure (Anscombe/datasaurus lesson). Check Spearman (monotone?), mutual information or a small tree's gain (any structure?), and correlation within segments (opposite-sign groups cancel). "r=0 ⇒ nothing" is the tell you're screening for.

**47. How do you measure association across mixed types (num–num, cat–cat, num–cat)?** · `M`
**A:** num–num: Pearson/Spearman. cat–cat: chi-square with Cramér's V for magnitude (bias-corrected for many levels). num–cat: correlation ratio (η) or ANOVA-flavored variance explained; binary–num: point-biserial. The point is one comparable "association strength" view across the whole table, not just the numeric corner.

**48. A stakeholder sees your correlation heatmap and starts drawing causal arrows. Your talk track?** · `M`
**A:** Correlation summarizes co-occurrence in *this* data under *this* selection: confounders, reverse causation, and selection effects (we only see approved orders) all produce arrows-free correlation. Offer what EDA *can* say (candidate hypotheses, effect sizes, segments) and what would license causal talk (experiment, quasi-experimental design, explicit assumptions).

**49. Features A and B correlate at 0.97. So what?** · `M`
**A:** For prediction: mostly harmless — models split the credit; drop one only for cost/simplicity. For *interpretation*: fatal — coefficients/importances become unstable and arbitrary between the twins (importance dilution in trees, sign flips in linear models). Also ask *why* 0.97: duplicated pipeline logic, one derived from the other, or a unit-converted copy.

**50. One point makes r jump from 0.1 to 0.6. What's the discipline?** · `M`
**A:** Always pair r with the scatterplot; compute a robust check (Spearman, or leave-one-out r); investigate that point's provenance. Report the honest pair: "r = 0.6 driven by a single extreme; 0.1 without it" — a correlation that one row owns is an anecdote wearing a statistic's clothes.

**51. Within every region the relationship is negative; pooled, it's positive. What now?** · `H`
**A:** Simpson's structure: region is associated with both variables. Report the stratified relationship as primary, identify the confounder explicitly, and warn every downstream consumer that the pooled number answers a different (usually wrong) question. In EDA writeups, any pooled correlation across heterogeneous groups should carry a stratified companion.

**52. Two KPIs correlate at 0.9 over three years of weekly data. Believe it?** · `M`
**A:** Not yet: both trend, and trending series correlate spuriously. Correlate the *changes* (first differences) or detrended residuals; check the relationship holds within shorter windows; if the differenced correlation collapses to ~0, the 0.9 was shared drift, not co-movement.

**53. The correlation matrix is 150×150. How do you actually read it?** · `M`
**A:** Don't eyeball a heatmap: cluster it (hierarchical clustering on 1−|r|) to reveal feature *blocks*; list top-k pairs by |r| with scatter thumbnails; extract a redundancy report (groups above 0.9) feeding feature selection; and correlate features with the target separately. Structure first, cells second.

**54. Your scatterplot of 5M points is a solid blob. Options?** · `E`
**A:** Alpha blending only goes so far: hexbin/2D histograms, density contours, or a stratified sample preserving tails; add a fitted smoother (LOESS) to expose the trend; facet by a key segment. The goal is density-aware plotting — a blob answers nothing.

**55. How do you *discover* an interaction during EDA, before any model?** · `H`
**A:** Condition and compare: the x–y relationship per level of z (faceted scatters/slopes; grouped means). Numeric z: bin it or plot slope-by-quantile. Confirm with a quick model comparison (with vs without the interaction term) or a shallow tree, which finds interactions natively. Example: discount lifts volume for small accounts, flat for large — invisible in any marginal view.

**56. Reading a cross-tab of two categoricals: what do juniors get wrong?** · `M`
**A:** Raw counts instead of the right percentages (row vs column % answer different questions — pick the one matching the conditional you care about); ignoring expected counts under independence (chi-square residuals show *which* cells deviate); and over-reading cells with tiny support. Add Cramér's V for overall strength.

---

## F — Categorical Data (57–66)

**57. Your cardinality audit: what are you looking for and why?** · `M`
**A:** Per-column unique counts and unique-ratio: near-1 ratios on strings = ids or free text (not features); high cardinality (city, SKU) = one-hot explosion risk and target-encoding leakage risk; unexpected low cardinality = a broken extract (all one value). Cardinality determines the entire encoding strategy, so it's a first-hour check.

**58. 400 categories, the bottom 300 cover 2% of rows. Group them?** · `M`
**A:** Usually yes — an "other" bucket stabilizes estimates and serving; but check first whether rare levels are *systematically different* (rare SKUs may be the high-margin ones) and whether rarity is temporal (new categories growing). Choose the threshold by coverage and support, keep the mapping versioned, and route genuinely-new levels to "other" at serving time.

**59. `df['city'].nunique()` returns 3× the real number of cities. Diagnose.** · `E`
**A:** Variant spellings: case ("Tabriz"/"TABRIZ"), whitespace, transliteration variants, encoding damage. Normalize (trim, casefold, unicode-normalize), then fuzzy-match survivors; keep the raw column and the mapping. The count-vs-expectation gap is itself the detector.

**60. A "category" column turns out to be free text. How do you tell, and what changes?** · `M`
**A:** Tells: unique-ratio near 1, long strings, punctuation, length variance. Changes everything: it's not one-hot-able; route to text processing (normalization → rules/embedding) or extract the structured part; check whether it hides PII. Treating free text as a factor silently creates thousands of one-row levels.

**61. New category levels appear every month. What's the EDA finding and the serving implication?** · `M`
**A:** Plot new-level rate over time: a living vocabulary (products, campaigns). Implication: any encoder must handle unseen levels at inference (`handle_unknown`, "other" routing, or hash/target encodings with fallbacks); training-time snapshots go stale; add unseen-level rate to monitoring.

**62. How do you decide whether a categorical is ordinal, and why does it matter?** · `E`
**A:** Semantics (S/M/L, satisfaction scales) plus a monotone relationship with the target when ordered. Encoding ordinal as one-hot discards the order (model must relearn it); encoding nominal as integers *invents* an order. Both errors cost accuracy and interpretability.

**63. One level holds 92% of rows. Consequences?** · `M`
**A:** The column carries little information for most rows (near-zero variance); rare levels have unstable statistics; stratification and CV folds may lack minority levels; interactions with it are data-starved. Consider binary-izing (dominant vs rest), verify the 92% isn't a default value masking missingness, and check whether the minority levels are where the signal (or the risk) concentrates.

**64. Train vs last-month category distributions — what do you compute and what threshold?** · `M`
**A:** PSI (or JS divergence) over levels including an "unseen" bucket: PSI > 0.25 = investigate. Also track top-level shares and the new-level rate. Categorical drift is often *business* drift (assortment, campaigns) and sometimes *pipeline* drift (a mapping table changed) — the fix differs, so diagnose the source.

**65. `customer_id` as a model feature improved validation AUC by 6 points. Interpret.** · `H`
**A:** The model memorizes per-customer outcomes — works only for seen customers, and if rows of one customer straddle the split it's leakage inflating validation. Grain question: if the task is per-customer prediction, use *aggregated history features*, not the raw id; if generalizing to new customers, group-split by id and watch the 6 points evaporate.

**66. What belongs in your standard categorical EDA checklist?** · `E`
**A:** Cardinality + unique ratio; level frequency table (top-k + coverage curve); missing/"Unknown" handling; variant normalization; temporal stability of levels (new/dying); association with target (with leakage caution); serving-time behavior for unseen levels. One reusable checklist beats re-improvising per project.

---

## G — Time-Aware EDA (67–76)

**67. First five checks on any time-series dataset?** · `M`
**A:** Is it sorted (never assume); duplicate timestamps per entity (grain!); gap structure (missing periods vs true zeros); timezone/DST handling; and volume-by-time plot (partitions missing, loads doubled). Only then look at the values themselves.

**68. The series is irregularly sampled. What are your options and their costs?** · `M`
**A:** Resample to a regular grid (aggregation choice changes meaning: sum vs mean vs last), model event streams directly (point-process view), or engineer time-since-last features. Regularizing hides burstiness; keeping it raw complicates most classical methods. The decision follows the question: totals per period vs dynamics between events.

**69. How do you *detect* seasonality rather than assume it?** · `M`
**A:** ACF (spikes at the seasonal lag and multiples), periodogram/FFT (dominant frequencies), and grouped means (by hour/day-of-week/month with CIs). Check multiple candidate periods — daily data often has weekly *and* yearly cycles — and check stability (seasonality can change regime).

**70. Rolling mean and rolling std as EDA — what are they for?** · `E`
**A:** Level shifts and trend show in the rolling mean; variance regimes show in the rolling std (constant mean can hide a volatility change); the window is a lens — sweep two or three widths. It's the fastest visual test for "is this series even stationary enough to summarize with one number."

**71. Multiple rows per (entity, timestamp). What are the possible truths?** · `M`
**A:** Wrong grain assumption (rows are order-lines, not orders), true duplicates (replayed loads), or multiple sources merged. Each has a different fix (aggregate to grain, dedupe, reconcile sources) — quantify each hypothesis before collapsing; blind `drop_duplicates` destroys line-level truth.

**72. What does joining a calendar table add to EDA?** · `E`
**A:** Explains "anomalies" (holidays, weekends, month-end), enables seasonality-by-calendar views, and surfaces regional effects (different holiday calendars — including lunar calendars whose dates drift yearly). Half of anomaly-hunting is calendar-joining; do it before flagging outliers in time.

**73. You suspect series A leads series B. How do you check honestly?** · `H`
**A:** Cross-correlation at lags — but on *detrended/differenced* series (shared trends fake lead-lag), with significance bounds, and validated out-of-sample (does lagged A actually improve B's forecast?). Also beware asymmetric smoothing/reporting delays creating artificial leads. CCF on raw trending series is the classic false discovery.

**74. During EDA you find `updated_at` columns and features like `days_to_resolution`. Alarm bells?** · `H`
**A:** Yes — leakage archaeology: any column written *after* the outcome (resolution fields, final statuses, updated timestamps) encodes the future relative to prediction time. EDA move: reconstruct feature availability timelines (what's known at prediction time t?) and quarantine post-outcome fields. This EDA step prevents the most expensive modeling failure there is.

**75. Retention "improves with account age" in a single snapshot. What's the EDA trap?** · `H`
**A:** Age–period–cohort confounding plus survivorship: old accounts are the survivors of their cohorts, and different cohorts joined under different conditions. Proper view: cohort curves (each signup cohort tracked over its own age), separating age effects from cohort composition. Snapshot cross-sections routinely reverse under cohort views.

**76. When should you re-index a dataset to event time (t=0 at signup) instead of calendar time?** · `M`
**A:** When the process is lifecycle-driven: onboarding, churn risk by tenure, post-purchase behavior. Event alignment makes cohorts comparable and lifecycle shape visible; calendar view mixes lifecycles with seasonality. Best EDA does both: lifecycle shape (event time) and seasonality/external shocks (calendar time).

---

## H — Data Quality & Integrity (77–88)

**77. "Remove duplicates" — why is that instruction underspecified?** · `M`
**A:** Duplicate *at what grain*? Exact-row dupes (loader replay), key dupes with differing values (which version wins — latest by `updated_at`?), and near-dupes (same customer, name variants) are different problems with different fixes. Quantify each class first; `drop_duplicates()` with no key argument is a decision made by accident.

**78. How do you validate a claimed primary key, and what if it fails?** · `E`
**A:** `df.groupby(key).size().max() == 1` (or `df[key].is_unique`); on failure inspect colliding groups: same values (dedupe) vs conflicting values (versioning problem — need a tiebreak rule) vs the key being genuinely composite (add the missing column). Never proceed to joins with an unvalidated key.

**79. Quantifying join health before trusting a merged dataset?** · `M`
**A:** Anti-joins both ways: % of left keys unmatched (coverage loss) and orphan right keys; row count before vs after (fan-out detector); `merge(..., validate=)` and `indicator=True` as guardrails. Report match rates by segment/time — join loss is rarely uniform, and non-uniform loss is a bias.

**80. Row count doubled after a merge. Walk through the diagnosis.** · `E`
**A:** The right table isn't unique on the join key: find duplicated keys, decide the true relationship (aggregate the many side first, or pick a version), re-merge with `validate='many_to_one'` so it can never silently recur. Then re-check any aggregates computed on the inflated frame.

**81. What "impossible value" checks belong in every EDA?** · `E`
**A:** Domain ranges (age 0–120, percentages 0–100, nonnegative quantities), date logic (end ≥ start, no future events in historical data), and unit sanity (order value vs items count). Encode them as assertions that run on every refresh — EDA findings should graduate into automated tests.

**82. Cross-field consistency: give examples and the payoff.** · `M`
**A:** Totals equal sum of parts (order total vs line items), geo consistency (postal code within stated city), status/date coherence ("delivered" rows have delivery dates). Violations localize the broken pipeline step and prevent the classic failure of modeling on internally contradictory rows.

**83. One numeric column looks bimodal with modes exactly 1000× apart. Hypothesis?** · `M`
**A:** Mixed units (grams vs kilograms, IRR vs toman, cents vs dollars) — typically split by source system or by an era (migration date). Confirm by conditioning on source/date; fix with a conversion rule, and add a magnitude-cluster check to ingestion. Modeling through mixed units produces confident nonsense.

**84. You see 'Ù…Ø­ØµÙˆÙ„' in a text column that should be Persian. What happened and what do you do?** · `M`
**A:** Mojibake: UTF-8 bytes decoded as Latin-1/Windows-1252 somewhere in the pipeline. Fix at the source read (correct encoding), or repair by re-encoding round-trip when possible; audit which loader/export stage corrupted it; add an encoding check (non-Latin script share) for multilingual data. Never "clean" it by dropping non-ASCII — that deletes the language.

**85. Column `revenue` exists throughout, but its *meaning* changed in March (net vs gross). How would EDA catch it, and what then?** · `H`
**A:** Level-shift + ratio checks against related fields (revenue/units jumps ~X%), and reconciliation to finance totals breaking after March. Then: treat as two eras — restate one era with the conversion if possible, else add an era flag and prevent naive year-over-year comparisons. Semantic drift is nastier than schema drift because nothing errors.

**86. The dataset merges three source systems. What must EDA do with `source`?** · `M`
**A:** Condition *everything* on it: distributions, missingness, category vocabularies, duplicate rates, target base rates. Source-level heterogeneity masquerades as signal (models learn "source" via proxies) and as drift when the mix shifts. If sources disagree on overlapping entities, that disagreement rate is a data-quality KPI.

**87. How do you assess freshness, and why does it belong in EDA?** · `E`
**A:** max(event_time) and max(load_time) vs now, per partition/source; lag distributions between event and availability. Because models trained on "yesterday" but served on data arriving with 3-day lag are silently forecasting with stale features — the feature you explored isn't the feature you'll have at prediction time.

**88. What is a tie-out, and when do you refuse to proceed without one?** · `M`
**A:** Reconciling your dataset's aggregates against an authoritative report (finance revenue, ops volumes) within tolerance. Refuse when the analysis will drive decisions denominated in those aggregates — if your total is 7% off finance's, every downstream conclusion inherits an unexplained 7%. Tie-out first; insight second.


---

## I — Target & Label EDA (89–98)

**89. Beyond the overall class ratio, what should a class-balance audit include?** · `M`
**A:** Balance *per segment and per time slice*: a 10% global positive rate can hide 0.5% in one region and 30% in another, which breaks a single global threshold and makes stratified splits mandatory. Also: absolute minority counts (2% of 1M is workable; 2% of 3k isn't), and whether the observed ratio matches deployment prevalence (eval sets are often artificially balanced).

**90. How do you audit label *quality* before trusting any model metric?** · `H`
**A:** Find near-identical feature rows with conflicting labels (upper-bounds achievable accuracy); measure annotator agreement (Cohen's/Fleiss' kappa) if human-labeled; inspect a random sample of positives and negatives yourself; check label source lineage (rule-generated labels inherit the rule's blind spots). Model ceilings are set by label noise — quantify it or chase ghosts later.

**91. What does target-leakage hunting look like *during EDA*, concretely?** · `H`
**A:** Rank features by univariate association with the target and treat "too good" as suspect (a single feature with AUC 0.95 is an indictment, not a gift); check feature availability timing (populated only after the outcome? correlated with `updated_at`?); train a shallow tree and read the top splits — leakage loves the root. Every leak found in EDA is a production incident prevented.

**92. Recent rows have systematically fewer positive labels. What's likely going on?** · `H`
**A:** Label maturity / right-censoring: outcomes take time to materialize (churn windows, fraud chargebacks land 60–90 days later), so the newest rows haven't finished becoming positive. Training or evaluating on immature rows deflates the positive rate and biases the model against recent patterns. Fix: enforce a maturity cutoff (only rows older than the label window) or model censoring explicitly.

**93. The positive rate differs 3× between data sources. What must you rule out before modeling?** · `M`
**A:** That the *labeling process* differs (one source auto-labels, another has human review; different definitions of "churn"), not just the population. If definitions differ, you have two tasks wearing one column — harmonize or separate. Conditioning the target on `source` is a two-minute check that regularly rewrites the project.

**94. The label is `clicked` but the business goal is revenue. What does EDA owe this gap?** · `M`
**A:** Quantify the proxy–target relationship: click→purchase conversion by segment, where they diverge (clickbait-adjacent items: high click, no purchase), and whether optimizing the proxy plausibly moves the goal. Report it as a modeling risk with numbers — proxy divergence discovered *after* deployment is the expensive version (Goodhart).

**95. How do you detect that the *definition* of the label changed over time?** · `H`
**A:** Plot the positive rate over time and look for step changes that don't track any business reality; diff the labeling code/rules history; compare feature–label relationships across eras (a stable feature whose association flips is a definition-change fingerprint). Treat eras as different label versions: train within era or map old to new.

**96. The target is a 1–5 satisfaction score and the junior fit MSE regression. Discussion points?** · `M`
**A:** The scale is ordinal: distances between levels aren't known to be equal (4→5 ≠ 1→2), MSE assumes they are; predictions come out as 3.7 (what does that mean operationally?). Alternatives: ordinal regression (cumulative logit), or classification with an ordinal-aware loss; also check heaping (people avoid extremes) and per-level base rates before choosing.

**97. The label column contains values like `"electronics, gifts"`. What did you just learn?** · `E`
**A:** The task is multi-*label*, not multi-class — categories co-occur. Consequences: one-hot per label with independent (or structured) classifiers, evaluation switches to per-label precision/recall and subset accuracy, and class-balance analysis becomes per-label. Splitting on commas isn't a parsing chore; it's a problem-definition discovery.

**98. When is collapsing rare target classes into "other" the right call, and what's the test?** · `M`
**A:** When rare classes lack support to learn or evaluate (single-digit counts), aren't operationally distinct (the action taken is the same), and hierarchy allows a parent class. The test: does the decision consumer treat them differently? If yes, collapsing optimizes a metric while destroying the product; consider a two-stage design (common classes + rare-class specialist/human routing) instead.

---

## J — Multivariate Structure (99–108)

**99. Running PCA as an EDA step — what's the correct procedure and what do you read from it?** · `M`
**A:** Standardize first (else PCA ranks features by variance/units), then read: the scree curve (how concentrated is the structure), loadings of the top components (which feature groups move together — name them), and score plots colored by candidate segments. PCA in EDA is a structure-summarization tool, not a preprocessing reflex.

**100. Your 2-D PCA plot explains 19% of variance. What can and can't you conclude from it?** · `M`
**A:** Can: inspect gross structure *in those two directions* (obvious clusters, outliers). Can't: conclude "no clusters" or judge separation — 81% of the variance is invisible, and structure may live there. Always annotate projection plots with explained variance; a beautiful 2-D story on 19% is mostly imagination.

**101. Using clustering as an EDA tool (not a product): what's the discipline?** · `M`
**A:** Treat clusters as *hypotheses*: profile them against features not used in clustering, check stability across seeds/resamples (unstable clusters are RNG, not structure), and try to name each cluster in business language — unnameable clusters are usually artifacts. The output of EDA clustering is questions and candidate segment variables, never shipped labels.

**102. What may you legitimately read from a t-SNE/UMAP plot, and what never?** · `M`
**A:** Legitimate: local neighborhoods (these points resemble each other), candidate groupings to verify elsewhere. Never: inter-cluster distances, cluster sizes/densities, or cluster *existence* by itself — both methods manufacture compact blobs from weak or no structure depending on perplexity/n_neighbors. Verification lives in the original space (metrics, stability, shuffled-data nulls).

**103. How do you produce a feature-redundancy map for 200 features?** · `M`
**A:** Hierarchical clustering on distance = 1−|corr| (Spearman for robustness), cut at a threshold (e.g., |r| > 0.9) to get redundancy groups; within each group keep the best-coverage/cheapest/most-interpretable representative. Deliverable: a named group list ("these 12 are all size proxies"), which is also the multicollinearity brief for any linear modeling later.

**104. Distances on mixed numeric + categorical features — what goes wrong and what's the fix?** · `M`
**A:** Euclidean on one-hots + raw numerics implicitly weights by scale and cardinality (a 100-level one-hot dominates or vanishes depending on encoding). Fixes: Gower distance (per-type contributions), deliberate standardization, or embed categories first. Deeper point: *scaling is the weighting* — every distance-based EDA (clustering, kNN, outliers) inherits whatever weights your preprocessing accidentally chose.

**105. How would you *demonstrate* the curse of dimensionality on your own dataset?** · `H`
**A:** Compute the ratio of each point's nearest- to farthest-neighbor distance as dimensions grow (sampled feature subsets): in high dimensions it approaches 1 — distances concentrate and "nearest" loses meaning. Practical readout: if the contrast ratio is near 1 on your full feature set, distance-based methods (kNN, LOF, k-means) are running on noise; reduce dimensions or select features first.

**106. What's in a sparsity/variance audit, and why bother?** · `E`
**A:** Near-zero-variance features (one value ≥ 99% of rows), density of sparse columns, and features that are constant within the training window but vary later. They waste computation, break some scalers, produce unstable CV importance, and — the useful part — often indicate a filter or bug upstream (a feature that "turned off"). Cheap to run, routinely finds pipeline problems.

**107. Training a quick gradient-boosted model just for EDA: what is it for, and what are the traps?** · `H`
**A:** For: a fast nonlinearity/interaction detector, a *leakage detector* (suspiciously dominant features), and a reality check on achievable signal. Traps: impurity importances are biased toward high-cardinality features; correlated features split credit arbitrarily (use permutation importance grouped by redundancy clusters); and the quick model's CV must already respect grouping/time or its "insights" are leakage artifacts themselves.

**108. What does it mean to anchor EDA to the decision, concretely?** · `H`
**A:** Start from the action the analysis feeds (who gets a discount? which SKUs get stocked?) and work backwards: which variables can the decision-maker actually change or condition on, at what grain, and with what lead time — then prioritize exploring *those* relationships and the data quality on *those* fields. It's the difference between an atlas of the dataset and a map to the destination; seniors produce maps.

---

## K — Text & Embeddings EDA (109–116)

**109. First checks on a new text corpus?** · `M`
**A:** Length distribution (chars *and* tokens — truncation exposure), language mix, exact and near-duplicate rate, boilerplate/template share, encoding health (mojibake scan), and empty/whitespace-only rate. Then read 30 random documents yourself — no profiling tool replaces contact with the actual text.

**110. Language-detection distribution shows 70/20/10 EN/TR/FA, but the product serves Iran. Discussion?** · `M`
**A:** The corpus doesn't match the deployment population — models will be strongest in the majority training language and weakest where most users live. Also audit detector reliability on short/mixed texts (code-switching is common), and report per-language *quality* expectations, not just counts. This single histogram sets the multilingual strategy.

**111. Why does near-duplicate detection matter in text EDA, and how do you do it at scale?** · `H`
**A:** Templates and boilerplate (auto-generated descriptions, email signatures) inflate corpus size, bias frequency statistics, leak across train/test splits, and make retrieval return ten copies of one document. Scale approach: shingling + MinHash/LSH for candidate pairs, then verify; report the deduplicated corpus size — often dramatically smaller, which changes every downstream plan.

**112. What does a vocabulary growth curve (Heaps' law) tell you?** · `M`
**A:** Plot unique tokens vs corpus size: the growth rate indicates how open the vocabulary is — steep unending growth means heavy OOV exposure for fixed-vocab methods and predicts how quickly embeddings/token statistics go stale as new data arrives. It also benchmarks how much more corpus would meaningfully expand coverage.

**113. Token-length distribution vs the model's max length — what are you quantifying?** · `M`
**A:** Truncation exposure: what share of documents exceeds the limit, *which classes/sources* they belong to (long documents are rarely a random sample — contracts, complaints), and what content lives in the truncated tails. Deliverable: truncation rate by segment plus a chunking-or-longer-context recommendation, before anyone trains on silently cut text.

**114. Using class-discriminative terms (log-odds) as EDA — what do you find besides "insights"?** · `H`
**A:** Leakage: template phrases that encode the label ("your refund has been approved" in the positive class), operator signatures correlated with outcomes, channel artifacts. Weighted log-odds with a prior (not raw counts) surfaces them; anything that's an *administrative* trace rather than customer language should be stripped before modeling, or the model learns the workflow, not the world.

**115. EDA on an embedding space you just computed — what's the checklist?** · `H`
**A:** Pairwise-similarity distribution (narrow high band = anisotropy/collapse); kNN sanity spot-checks on documents you know (are neighbors actually related?); isotropy/hubness check (do a few points appear in everyone's neighbor lists?); similarity separation between known-related and random pairs; and cluster-vs-metadata structure (grouping by language/source instead of meaning is a red flag for the intended use).

**116. Where does PII scanning fit in text EDA, and what's the minimum?** · `M`
**A:** Before the corpus is shared, embedded, or shipped anywhere — embeddings and prompts leak PII too. Minimum: pattern scans (emails, phones, national ID formats, card numbers) plus sampled manual review of free-text fields; quantify hit rates per field, and gate the pipeline on redaction. Finding PII late means re-processing everything; finding it in EDA costs an afternoon.

---

## L — Process, Pitfalls & Communication (117–124)

**117. How does EDA itself become p-hacking, and what's the guardrail?** · `H`
**A:** The garden of forking paths: dozens of implicit choices (which cuts, which outliers, which transforms) made *after seeing the data* mean any "finding" was selected from a multiverse of analyses — its nominal significance is fiction. Guardrails: split explore/confirm (hold out data the narrative must survive), log the paths tried, pre-register the confirmatory analysis, and label EDA outputs as hypotheses, not results.

**118. You eyeballed 200 features against the target and 9 "look related." Expected value of that finding?** · `M`
**A:** Roughly what chance produces: visual screening has its own false-positive rate, and 200 comparisons guarantee hits (look-elsewhere effect). Treat the 9 as candidates for out-of-sample confirmation with multiplicity control (FDR), and expect most to evaporate. The number to internalize: at α=0.05, 200 null features yield ~10 "discoveries."

**119. What should an EDA decision log contain, and why is it non-optional?** · `M`
**A:** Every consequential choice with its rationale: exclusions (rule, count, sensitivity), imputation and transform decisions, grain and key definitions, discovered quality issues and their fixes, assumptions taken on faith. Because six months later the model misbehaves and the question is always "what exactly did we do to the data" — the log is the difference between an audit and an archaeology dig.

**120. Which EDA activities are safe on the full dataset, and which must respect the split?** · `H`
**A:** Safe on everything: schema, grain, quality, missingness *patterns*, ranges, duplicates — structural facts. Split-respecting: anything that estimates parameters later reused (imputation values, scaling stats, encodings, feature selection, outlier fences) and any deep look at feature–*target* relationships that will guide modeling choices — that guidance is itself a leak channel into "held-out" performance. Rule of thumb: explore structure globally, estimate on train only.

**121. How do you know when to *stop* EDA?** · `M`
**A:** When the decision-driven checklist is exhausted, not when curiosity is: grain established, quality gates passed, target understood (balance, quality, leakage sweep), key relationships mapped at the depth the decision needs, and risks logged. Timebox it, publish findings, and let modeling iterations pull you back — EDA is a loop with modeling, not a phase you graduate from once.

**122. Presenting EDA to stakeholders: how do you carry uncertainty without losing the room?** · `M`
**A:** Lead with the finding and its decision implication, attach magnitude with an honest range, then caveat *selectively* — the two caveats that could reverse the conclusion, not all twenty. Separate "what the data shows" from "what we'd need to confirm it." Uncertainty as a footnote gets ignored; uncertainty as the headline gets you ignored; uncertainty priced into the recommendation gets used.

**123. Automated profiling reports (pandas-profiling-style): what do they do well and where are they blind?** · `E`
**A:** Well: mechanical univariate sweeps, missingness, cardinality, obvious correlations — a fine first pass and checklist executor. Blind: grain and key semantics, temporal leakage, label quality, cross-field consistency, source heterogeneity, and *meaning* generally — everything that requires knowing what the data is supposed to represent. Use them to free time for the judgment-heavy checks, not to replace them.

**124. Design the standard EDA checklist you'd institutionalize for any new dataset landing in the team.** · `H`
**A:** (1) Provenance & tie-out: source, freshness, reconciliation to an authority. (2) Grain & keys: one row =, key uniqueness, join health. (3) Quality gates: ranges, impossible values, duplicates, encoding, units. (4) Missingness: rates, patterns, mechanism hypotheses. (5) Distributions: per-column shapes, sentinels, heaping, tails. (6) Time: coverage, gaps, drift, era changes. (7) Target: balance, quality, maturity, leakage sweep. (8) Relationships: redundancy map, target associations with leakage caution. (9) Text/special types where present. (10) Decision log + graduated assertions into automated tests. The senior mark: findings become *tests*, so the same disease can't ship twice.

---

## How to score EDA answers

| Signal | Weak (1–2) | Strong senior (4–5) |
|---|---|---|
| **Process** | Tool-name recitation ("I'd use pandas-profiling") | A repeatable interrogation order with a hypothesis behind every check |
| **Mechanism link** | Describes the anomaly | Connects finding → modeling consequence → decision consequence |
| **Leakage instinct** | Never mentions timing | Asks "available at prediction time?" unprompted; treats too-good as guilty |
| **Quality instinct** | Trusts the frame as delivered | Reconciles, validates grain/keys, hunts pipeline causes behind statistical symptoms |
| **Split discipline** | Explores everything everywhere | Knows which EDA is safe globally vs train-only |
| **Communication** | Data dump | Decision-anchored findings with priced-in uncertainty and a written log |

**Usage.** Pick 5–7 questions spanning at least four sections for a 45-minute EDA round; start with A/H (mechanics) to calibrate, finish with one `H` from I or L (judgment). The single most predictive question in this bank for seniority is **120** — the split-discipline answer cleanly separates people who've been burned from people who haven't.
