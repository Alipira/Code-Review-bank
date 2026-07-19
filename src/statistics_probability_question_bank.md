# Senior Data Scientist — Statistics & Probability Question Bank (124 Questions)

> **What this is.** The stats/probability round as senior interviews actually run it: classic probability puzzles (with the worked numbers), estimation and testing questions where the *misinterpretations* are the real test, experimentation design, Bayesian reasoning, and the paradoxes that separate pattern-matchers from people who understand the machinery.
>
> **How to run one.** Ask for reasoning out loud, not the memorized answer. On computations, the setup is worth more than the arithmetic; on concepts, listen for what the candidate says a quantity is *not* (p-values, CIs) — that's where seniority shows.
>
> **Difficulty:** `E` screen · `M` core round · `H` senior signal.

---

## A — Probability Foundations & Classic Puzzles (1–12)

**1. A test for a disease with 1% prevalence has 95% sensitivity and 90% specificity. A random person tests positive — what's the probability they have the disease?** · `E`
**A:** Bayes: P(D|+) = (0.95·0.01) / (0.95·0.01 + 0.10·0.99) = 0.0095 / 0.1085 ≈ **8.8%**. The strong answer sets up base rates before arithmetic and names the fallacy being tested (confusing P(+|D) with P(D|+), i.e., base-rate neglect). Follow-up: what prevalence would make P(D|+) = 50%? (≈ 9.5%.)

**2. Explain the difference between independence and mutual exclusivity.** · `E`
**A:** Mutually exclusive events *can't* co-occur (P(A∩B)=0) — which makes them maximally **dependent**: knowing A happened tells you B didn't. Independence means P(A∩B)=P(A)P(B): knowing one changes nothing about the other. Two events with positive probability cannot be both.

**3. What is conditional independence, and where does an ML system assume it?** · `M`
**A:** A ⊥ B | C: given C, learning A tells you nothing more about B — even though A and B may be marginally dependent. Naive Bayes assumes features are conditionally independent given the class; symptom pairs correlate marginally but may be independent given the disease. The direction matters too: conditioning can also *create* dependence (colliders — see Berkson's, Q114... §J).

**4. A factory has three machines producing 50/30/20% of output with defect rates 1/2/4%. A random item is defective — which machine most likely made it?** · `E`
**A:** Total defect rate = 0.5·0.01 + 0.3·0.02 + 0.2·0.04 = 0.005+0.006+0.008 = 0.019. Posteriors: M1 = 5/19 ≈ 26%, M2 = 6/19 ≈ 32%, M3 = 8/19 ≈ **42%** — the 20%-share machine is the likeliest source. Law of total probability + Bayes; the lesson is that small-share/high-rate sources dominate posteriors.

**5. Probability of at least one six in four dice rolls?** · `E`
**A:** Complement: 1 − (5/6)⁴ = 1 − 625/1296 ≈ **0.518**. The tell is reaching for the complement instantly instead of summing cases; follow-up (de Méré): at least one double-six in 24 rolls of two dice = 1 − (35/36)²⁴ ≈ 0.491 — famously *less* than ½ despite the "same ratio."

**6. How many people in a room for >50% chance of a shared birthday, and why is the answer unintuitive?** · `E`
**A:** **23.** P(no match) = 365/365 · 364/365 · … · 343/365 ≈ 0.493. Unintuitive because intuition tracks "someone matches *me*" (~253 people needed) while the question counts *pairs* — 23 people have C(23,2)=253 pairs. Comparisons grow quadratically; this is also why hash collisions and duplicate ids appear "early."

**7. Monty Hall: you pick a door, the host (who knows) opens a goat door, offers a switch. Do you?** · `E`
**A:** Switch: win probability **2/3** vs 1/3 for staying. Cleanest argument: your initial pick is wrong 2/3 of the time, and in every such case the host's forced reveal leaves the car behind the other door. The senior follow-up: if the host opens a random door that merely *happens* to show a goat, switching is 1/2 — the host's knowledge (the selection mechanism) is what carries the information.

**8. "A family has two children; at least one is a boy. Probability both are boys?" And if "the *older* is a boy"?** · `M`
**A:** At least one boy: sample space {BB, BG, GB} → **1/3**. Older is a boy: {BB, BG} → **1/2**. The gap is the conditioning event's size — *how* you learned the information changes the posterior. Bonus depth: "at least one boy born on a Tuesday" shifts it to 13/27, showing that seemingly irrelevant detail changes the reference set.

**9. You have a coin with unknown bias p. Produce a fair 50/50 bit.** · `M`
**A:** Von Neumann: flip twice; HT → 0, TH → 1, HH/TT → discard and repeat. HT and TH both have probability p(1−p), so the output is exactly fair for any p (assuming independent flips). Expected pairs per bit = 1/(2p(1−p)). Reverse direction (biased from fair) — e.g., p = 1/3: generate bits until the binary expansion decides, or reject-sample over two fair flips.

**10. Expected number of rolls of a fair die until the first six? Until you've seen *all* faces?** · `E`
**A:** First six: geometric, E = 1/p = **6**. All faces: coupon collector, E = n·H_n = 6·(1+½+⅓+¼+⅕+⅙) = 6·2.45 = **14.7**. General n: ≈ n ln n + 0.577n (for n=50 coupons: ≈ 225). The structure — sum of geometric waits with shrinking success probability — is the transferable idea.

**11. n people throw their hats in a pile and each takes one at random. Expected number who get their own hat? And the variance's punchline?** · `M`
**A:** Indicators: P(person i gets own hat) = 1/n, so E = n·(1/n) = **1**, independent of n — linearity of expectation needs no independence. Punchline: Var is also 1 (for n ≥ 2), and the number of matches converges to Poisson(1) — P(no match) → 1/e ≈ 0.368.

**12. Two players alternate flipping a fair coin; first heads wins. What's the probability the first player wins?** · `M`
**A:** p = ½ + ¼·0 … cleanest: P(first wins) = ½ + (½)²·P(first wins) ⇒ P = **2/3**. The recursive/self-similar setup ("after two tails, the game restarts") is the skill being probed; it generalizes to any first-success race.

---

## B — Random Variables & Distributions (13–26)

**13. PMF vs PDF vs CDF — and why is P(X = x) = 0 for a continuous variable that clearly takes values?** · `E`
**A:** PMF assigns mass to points (discrete); PDF is a *density* — probability lives only in intervals, P(a<X<b) = ∫f; CDF F(x)=P(X≤x) unifies both. P(X=x)=0 because a point has zero width; density values can exceed 1 (they're not probabilities). Interviews use this to check the candidate isn't pattern-matching "f(x) = probability."

**14. What assumptions make a count Binomial(n, p), and give a realistic violation of each.** · `M`
**A:** Fixed n, independent trials, constant p. Violations: n random (visitors per day), dependence (word-of-mouth conversions cluster), varying p (mixed traffic sources → overdispersion). Overdispersed "binomial" data is the norm in the wild — beta-binomial or quasi-likelihoods; variance ≈ np(1−p) failing is the diagnostic.

**15. When is Poisson the right model for arrivals, and what breaks it?** · `M`
**A:** Events occurring independently at a constant rate, no simultaneous arrivals — the limit of many small-probability opportunities. Breaks under burstiness/self-excitation (marketing pushes, retries), diurnal rate variation (fix: inhomogeneous Poisson), and clustering (batch orders). Diagnostic: Poisson forces Var = mean; real counts usually have Var ≫ mean → negative binomial.

**16. Exponential memorylessness: state it, and assess "customer churn time is exponential."** · `M`
**A:** P(T > s+t | T > s) = P(T > t): elapsed time carries no information — a constant hazard. For churn it's usually wrong: hazards spike early (onboarding failures) and change at renewals — i.e., the hazard is time-varying. The senior move is to talk hazard shape (Weibull, piecewise) rather than accept convenience.

**17. Why does the normal distribution appear everywhere — and give three places it's the wrong default.** · `E`
**A:** CLT: sums/averages of many small independent effects → normal. Wrong defaults: multiplicative growth (incomes, latencies → lognormal), extremes/maxima (GEV territory), counts and heavy-tailed phenomena (file sizes, city sizes → power laws) where σ or even μ may be uninformative or nonexistent.

**18. Why do latencies and incomes come out lognormal?** · `M`
**A:** They're products of many small *multiplicative* factors (each stage scales the total); logs turn products into sums, CLT applies in log space. Consequences: report medians/geometric means, mean ≫ median, and exp(mean of logs) is the median not the mean — the retransformation trap.

**19. How do you check for a power-law/heavy tail, and what breaks if you ignore it?** · `M`
**A:** Log-log CCDF linearity (with proper fitting, not eyeballing a log-log histogram), tail index estimation; compare against lognormal (the usual rival). If the tail index ≤ 2 the variance is infinite — sample means converge painfully slowly, CLT-based CIs undercover, and "average user" is meaningless because sums are dominated by the largest observation.

**20. You have only Uniform(0,1) draws. Sample from an arbitrary distribution.** · `M`
**A:** Inverse-CDF: X = F⁻¹(U) has CDF F (for exponential: X = −ln(1−U)/λ). When F⁻¹ is unavailable: rejection sampling (accept U₂ ≤ f(x)/(M·g(x)) under proposal g) or transformation tricks (Box–Muller for normals). Knowing *why* inverse-CDF works — P(F⁻¹(U) ≤ x) = P(U ≤ F(x)) = F(x) — is the differentiator.

**21. Distribution of the sum of two fair dice — and E[max] and E[min] of two Uniform(0,1)s?** · `E`
**A:** Dice sum: triangular on 2–12, P(7) = 6/36 — convolution intuition. Uniforms: E[max] = 2/3, E[min] = 1/3 (general n: n/(n+1) and 1/(n+1)) via F_max(x) = xⁿ. Order statistics as "CDFs multiply for maxima" is the reusable tool.

**22. n independent Exp(λ) alarms: distribution of the first ring, and expected time until the last?** · `M`
**A:** Min of exponentials is Exp(nλ) — rates add (this powers reliability and queueing math). Last: E[max] = H_n/λ ≈ (ln n + 0.577)/λ, from summing the waits for each successive death (memorylessness restarts the clock with one fewer alarm). Same skeleton as coupon collector.

**23. What is the Beta distribution *for*, in one sentence — and why is it Binomial's best friend?** · `M`
**A:** A distribution over probabilities themselves (support [0,1], shaped by pseudo-counts α−1 successes, β−1 failures). Conjugacy: Beta prior + Binomial likelihood → Beta posterior with counts added — Bayesian updating becomes arithmetic (see §G). Uniform = Beta(1,1) makes "no prior knowledge" precise.

**24. Where does the Gamma distribution arise naturally?** · `E`
**A:** Waiting time for the k-th event of a Poisson process (sum of k Exp(λ)); also the conjugate prior for a Poisson rate and the variance-mixing distribution behind the negative binomial. If exponential is "time to first," gamma is "time to k-th."

**25. Your count data has variance 5× the mean. Name the phenomenon and the standard fix.** · `M`
**A:** Overdispersion — Poisson can't hold (it forces Var = mean). Standard fix: negative binomial (Poisson with Gamma-distributed rate; Var = μ + μ²/r), or quasi-Poisson for inference; if excess zeros specifically, zero-inflated/hurdle variants. Naming the *mechanism* (unmodeled rate heterogeneity) beats naming the distribution.

**26. Mixture of two normals, weights w and 1−w. Give E and Var, and the qualitative trap.** · `H`
**A:** E = wμ₁+(1−w)μ₂; Var = w(σ₁²+μ₁²)+(1−w)(σ₂²+μ₂²) − E² — equivalently law of total variance: mean of variances **plus variance of means**. Trap: the mixture's variance exceeds the average variance (between-component spread adds), and a mixture can be bimodal or deceptively unimodal; fitting a single Gaussian to a mixture gives a "center" where no data lives.

---

## C — Expectation, Variance & Moments (27–38)

**27. State linearity of expectation and why it's the most powerful cheap trick in interviews.** · `E`
**A:** E[ΣXᵢ] = ΣE[Xᵢ] with **no independence required**. Powerful because ugly dependent problems (hat matches, shared birthdays count, broken-stick pieces) decompose into indicator variables with trivial individual expectations. If a counting problem looks hard, the first question is "can I write it as a sum of indicators?"

**28. Var(aX + bY) = ? — and the portfolio insight hiding in it.** · `M`
**A:** a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y). Insight: negative covariance *reduces* combined variance below either component's — diversification; with ρ = −1, variance can hit zero. Also the reason error bars on differences aren't the sum of error bars unless independent.

**29. E[XY] vs E[X]E[Y] — what's the gap, and what does Cov = 0 not imply?** · `E`
**A:** The gap is Cov(X,Y) = E[XY] − E[X]E[Y]. Cov = 0 (uncorrelated) does **not** imply independence: Y = X² with symmetric X has zero covariance and total dependence — covariance sees only linear association. Independence ⇒ uncorrelated; never the converse (except special cases like joint normality).

**30. Expected number of pairs of people sharing a birthday among n = 30?** · `M`
**A:** Indicators over pairs: C(30,2)·(1/365) = 435/365 ≈ **1.19** expected shared-birthday pairs. Same linearity trick as hats; also explains why the birthday probability crosses ½ near 23 — that's where expected pairs ≈ 0.69 ≈ ln 2, matching the Poisson approximation P(≥1) ≈ 1 − e^{−λ}.

**31. Jensen's inequality: statement, direction, and a business place it bites.** · `M`
**A:** For convex f: E[f(X)] ≥ f(E[X]) (concave flips it). Bites: E[exp(ε)] > exp(E[ε]) — exponentiating log-space forecasts systematically underestimates means (retransformation bias); also "value of the average scenario" understates average value for convex payoffs, and log-utility explains risk aversion. Direction check via a two-point example is the reliable habit.

**32. Law of total expectation and law of total variance — write them and give the canonical use.** · `H`
**A:** E[Y] = E[E[Y|X]]; Var(Y) = E[Var(Y|X)] + Var(E[Y|X]) — *within* plus *between*. Uses: computing expectations by conditioning on the natural first stage (mixture problems, hierarchical models), and the variance decomposition that underlies ANOVA, mixed models, and "how much of the variance is explained by segment."

**33. St. Petersburg: a game pays 2^k for the first head on flip k. Expected payout is infinite — what would you actually pay, and why?** · `M`
**A:** E = Σ (1/2^k)·2^k = Σ1 = ∞, yet nobody pays much: resolutions are diminishing marginal utility (E[log payout] is finite — log utility values it around $2–4 plus bankroll effects) and the counterparty's finite bankroll capping the tail. The lesson for DS: expected value alone is not a decision rule when tails dominate.

**34. Roll a die; you may keep the value or re-roll once (must keep the second). Expected value of optimal play? And with two re-rolls?** · `M`
**A:** One re-roll: keep 4+ (anything above E=3.5): E = (4+5+6)/6·avg + (3/6)·3.5 → (1/2)(5) + (1/2)(3.5) = **4.25**. Two re-rolls: with continuation value 4.25, keep 5+ first: E = (2/6)·5.5 + (4/6)·4.25 = **4.67**. Backward induction on a toy — the pattern behind option pricing and sequential decisions.

**35. Why does the standard error of the mean shrink as 1/√n, and what's the practical bite?** · `E`
**A:** Var(X̄) = σ²/n (independent draws), so SD scales as σ/√n. Bite: halving uncertainty costs **4×** the data; the 10th-percentile improvement someone wants may need 100× — this square-root law is why "just collect more data" has a budget curve, and why tiny A/B effects need enormous samples.

**36. Standard deviation vs standard error — the confusion you're screening for.** · `E`
**A:** SD describes spread of the *data*; SE describes uncertainty of an *estimate* (SD of its sampling distribution). SD doesn't shrink with n; SE does. Error bars labeled ambiguously are how "significant-looking" plots lie — a senior asks which one is drawn before interpreting any figure.

**37. Chebyshev guarantees vs normal-based intervals — when do you need the weaker bound?** · `M`
**A:** Chebyshev: P(|X−μ| ≥ kσ) ≤ 1/k² for *any* distribution with finite variance (k=2 → ≤25%, vs normal's 4.6%). Needed when the distribution is unknown/heavy-tailed/small-sample — the price of assumption-freedom is looseness. Knowing both numbers for k=2 shows calibrated humility about normality.

**38. What is a moment generating function good for, practically — one honest use.** · `M`
**A:** M(t) = E[e^{tX}]: derivatives at 0 give moments, it determines the distribution when it exists, and — the workhorse — MGFs of independent sums *multiply*, which is how you prove sums of normals are normal, sums of Poissons are Poisson, and derive Chernoff/Hoeffding tail bounds. One honest sentence beats a memorized definition.

---

## D — Sampling, Estimators & the CLT (39–50)

**39. State the CLT precisely enough to use, and name two real situations where it fails you.** · `M`
**A:** For i.i.d. Xᵢ with finite variance, √n(X̄−μ)/σ → N(0,1): *averages* become normal regardless of the underlying shape, at a √n rate. Failures: heavy tails with infinite variance (tail index ≤ 2 — means wander, stable laws not normal apply) and dependence (time series, network effects — effective n is far smaller; the "n" in your SE is a lie). Also slow convergence for extreme skew: normal-ish only at painful n.

**40. LLN vs CLT — what does each actually promise?** · `E`
**A:** LLN: X̄ → μ (the estimate *converges* — consistency). CLT: how the *fluctuations* around μ behave on the way (shape and √n scale — enables CIs and tests). One says you'll arrive; the other describes the wobble. Conflating them produces "n is big so my mean is exactly right."

**41. Define bias and variance of an estimator and write the MSE decomposition.** · `E`
**A:** Bias = E[θ̂] − θ; Var = E[(θ̂ − E[θ̂])²]; **MSE = Bias² + Variance**. The point with teeth: unbiasedness is not sacred — a slightly biased, low-variance estimator (shrinkage, ridge) routinely beats the unbiased one on MSE. "Would you accept bias to reduce MSE?" separates textbook from practice.

**42. Why n−1 in the sample variance?** · `E`
**A:** Deviations are measured from X̄, which was itself fit to the data — the deviations satisfy one linear constraint (they sum to 0), leaving n−1 degrees of freedom; dividing by n systematically underestimates σ² (E[Σ(Xᵢ−X̄)²] = (n−1)σ²). Bonus nuance: for MSE of the variance estimate, dividing by n (or n+1) can actually be better — unbiasedness again isn't the only virtue.

**43. Derive the MLE for a Bernoulli p from k successes in n trials, and name two properties MLEs give you.** · `M`
**A:** L = p^k(1−p)^{n−k}; ∂/∂p log L = k/p − (n−k)/(1−p) = 0 ⇒ **p̂ = k/n**. Properties: consistency and asymptotic normality/efficiency (variance hits the Cramér–Rao bound asymptotically), plus invariance (MLE of g(θ) is g(θ̂)). Caveats worth volunteering: can be biased in finite samples (σ̂² divides by n) and degenerate at boundaries (k=0 → p̂=0).

**44. MLE vs MAP — the relationship in one line, and when they coincide.** · `M`
**A:** MAP maximizes likelihood × prior — MLE plus a log-prior penalty; with a flat prior they coincide, and as n grows the likelihood swamps any fixed prior so they converge. The bridge worth saying aloud: Gaussian prior = L2/ridge, Laplace prior = L1/lasso — regularization *is* MAP.

**45. Explain the bootstrap: what it estimates, why it works, and two places it fails.** · `M`
**A:** Resample n rows with replacement many times, recompute the statistic; the spread across resamples estimates the sampling distribution — works because the empirical distribution stands in for the true one (plug-in principle). Failures: dependent data (time series/clusters — need block or cluster bootstrap; resampling rows fakes independence) and extreme-value statistics (max, tail quantiles — the bootstrap is inconsistent there). Also small n: it can't invent information.

**46. Simple random vs stratified vs cluster sampling — when does stratification pay most?** · `M`
**A:** Stratified: sample within homogeneous strata → guarantees representation and cuts variance most when *between-stratum* differences are large relative to within (proportional or Neyman allocation). Cluster: sample whole groups for cost/logistics — *costs* precision (intra-cluster correlation shrinks effective n; design effect ≈ 1+(m−1)ρ). Choosing by variance structure and cost, not habit, is the senior tell.

**47. Give three selection biases that arise inside ordinary DS pipelines, and the question that catches each.** · `H`
**A:** (1) Survivorship: metrics computed on remaining users/products — "who left the denominator?" (2) Opt-in/consent bias: instrumented users differ from all users — "who is missing from logging?" (3) Approval/served-only data: you observe outcomes only for approved loans / shown items — "what happened to the rejected?" (reject inference, exploration). The unifying catch-question: *what process decided which rows I can see?*

**48. When does the finite population correction matter, and what does it do?** · `M`
**A:** When sampling a substantial fraction of a finite population (rule of thumb: n/N > 5–10%): Var(X̄) gets multiplied by (N−n)/(N−1), shrinking toward 0 as n→N — a census has no sampling error. Practical bite: surveying 400 of your 900 B2B customers, normal-theory CIs are meaningfully too wide without it.

**49. Your sample has known unequal inclusion probabilities. How do you estimate the population mean, and what's the price?** · `H`
**A:** Weight by inverse inclusion probability (Horvitz–Thompson / Hájek): Σwᵢyᵢ/Σwᵢ with wᵢ = 1/πᵢ. Price: variance inflates with weight variability (effective n ≈ n/(1+CV²(w))); a few huge weights can dominate — hence weight trimming as a bias-variance trade. Ignoring weights is estimating the *sample's* mean, not the population's.

**50. German tank problem: you observe serial numbers with maximum m from a k-sample. Estimate the total N.** · `H`
**A:** The classic (near-MVUE) estimator: **N̂ = m + m/k − 1** — the max plus the average gap. Intuition: the k observations split (0, N] into k+1 gaps of expected equal size; the unseen gap above m should be about m/k. Beloved in interviews because it tests estimator *reasoning* (sufficiency, bias correction) on one line of arithmetic; also genuinely used (WWII tank production, competitor volume from invoice numbers).

---

## E — Hypothesis Testing & Confidence Intervals (51–66)

**51. Define the p-value exactly — then list the three most common misreadings.** · `E`
**A:** P(data as or more extreme than observed | H₀ true, under the test's assumptions). It is **not**: (1) P(H₀ | data); (2) the probability the result is due to chance; (3) 1 − P(replication). And 1−p is not "probability the effect is real." A candidate who states the conditional direction cleanly and volunteers the misreadings passes; one who defines it *as* a misreading fails the round quietly.

**52. Type I, Type II, power, effect size — how do the four trade off?** · `E`
**A:** α = P(reject | H₀ true); β = P(fail to reject | H₁ true); power = 1−β. Power rises with effect size, n, and α, and falls with noise σ. The operational sentence: for fixed α and effect, the only honest lever is n (or variance reduction); tightening α at fixed n buys false-positive protection with false-negative currency.

**53. A 95% CI of [2.1, 4.7] — what may you say, and what's the forbidden sentence?** · `M`
**A:** Allowed: the *procedure* generating this interval covers the true value 95% of the time; values inside are compatible with the data (won't be rejected at 5%). Forbidden: "there's a 95% probability the parameter is in [2.1, 4.7]" — the parameter isn't random in this framework; that sentence belongs to Bayesian credible intervals (§G). Also worth volunteering: the CI is a better report than the p-value because it carries magnitude and precision.

**54. One-tailed vs two-tailed tests — when is one-tailed legitimate?** · `M`
**A:** Only when pre-specified and when effects in the untested direction would genuinely be treated as "no action" (not secretly interesting). Switching to one-tailed after seeing the direction halves the p-value fraudulently. Default two-tailed; document exceptions before data.

**55. Which t-test assumptions actually matter, and why is Welch the sane default?** · `M`
**A:** What matters: independence (fatal if violated — clustered/time data), and enough n for the CLT to handle non-normality (skew needs more; outliers dominate means regardless). Equal variances is the *dispensable* one: Welch's test drops it at essentially no cost, so defaulting to Student's equal-variance test is a habit with downside and no upside.

**56. Paired vs unpaired analysis — what does pairing buy and what's the classic mistake?** · `E`
**A:** Pairing differences out between-unit variance: analyze the per-pair deltas, often slashing the SE (power for free when correlation within pairs is high). Classic mistake: analyzing paired data (before/after, matched users) with an unpaired test — throwing away the design's power; the reverse error (pairing unrelated rows) manufactures dependence.

**57. What does the Mann–Whitney U test actually test? (Not "the medians.")** · `H`
**A:** Stochastic dominance: H₀ that P(X > Y) = ½ — whether a random draw from one group tends to exceed one from the other. It's a *median* test only under the added shift-model assumption (identical shapes). With unequal variances/shapes it can reject with identical medians. Precision about the tested hypothesis is exactly what this question screens.

**58. Chi-square test of independence: mechanics in one line, plus the two small-print rules.** · `E`
**A:** Compare observed vs expected counts under independence: Σ(O−E)²/E ~ χ² with (r−1)(c−1) df. Small print: expected counts ≥ ~5 per cell (else exact/Fisher or collapse categories), and it needs counts of *independent* units — feeding it events from repeated users violates it (same unit-of-analysis trap as everywhere).

**59. FWER vs FDR — define both, name the standard procedure for each, and say when each is right.** · `M`
**A:** FWER = P(≥1 false positive) — Bonferroni (α/m): right when any single false claim is costly (confirmatory, regulatory). FDR = expected fraction of discoveries that are false — Benjamini–Hochberg: right for screening many hypotheses (features, genes, experiment dashboards) where you'll validate hits downstream. The senior framing: choose by the *cost model of errors*, not by which is "stricter."

**60. p = 0.049 vs p = 0.051 — how do you handle the pair like an adult?** · `M`
**A:** They're the same evidence: p is continuous and noisy across replications; the 0.05 line is a convention, not a property of nature. Report effect sizes with CIs, treat both results as similar weak-to-moderate evidence, and decide with costs/priors — never let a dashboard binarize at the third decimal. ("The difference between significant and not significant is not itself significant.")

**61. n = 4M users, p < 10⁻¹², lift = 0.02%. Interpret.** · `M`
**A:** Statistically unambiguous, practically negligible: at huge n the test detects any nonzero effect — significance stops being informative and *magnitude with a CI* is the whole story. Decision framing: does 0.02% clear implementation and opportunity costs? Also check the tiny effect isn't a mix artifact or instrumentation quirk, which giant-n tests happily certify.

**62. Why is power analysis done before the experiment, and what's wrong with post-hoc power?** · `M`
**A:** Before: it sets the n needed to detect the minimum effect worth acting on — otherwise "no significance" is uninterpretable (see Q111-style traps). Post-hoc ("observed power" computed from the observed effect) is a deterministic function of the p-value — it adds zero information and is routinely used to launder null results. The honest post-experiment statement is the CI.

**63. How do you *demonstrate* two variants are equivalent, rather than fail to show a difference?** · `H`
**A:** Equivalence testing (TOST): pre-define the equivalence margin ±δ ("differences smaller than δ don't matter"), then run two one-sided tests that the effect is above −δ and below +δ — equivalently, show the CI lies entirely inside (−δ, δ). "p > 0.05 so they're the same" is the fallacy this question exists to catch; absence of evidence ≠ evidence of absence.

**64. Explain a permutation test and when you'd reach for it.** · `M`
**A:** Under H₀ of exchangeability, group labels are arbitrary: shuffle labels many times, recompute the statistic, and read the p-value as the rank of the observed value in that null distribution. Reach for it with weird statistics (ratio metrics, medians, custom KPIs), unknown distributions, or when you want assumptions you can state in one sentence. Caveat: exchangeability fails under clustering/time structure — permute at the right unit.

**65. Should you test residual normality before trusting a t-test/regression inference? The junior ran Shapiro–Wilk.** · `M`
**A:** Mostly a ritual: at large n the CLT protects the *mean's* inference against non-normality (and Shapiro rejects trivia); at small n the test lacks power to find real problems. What matters instead: independence, influential outliers, and heteroscedasticity — checked by plots and robust methods. Normality of *predictors* matters not at all; even residual normality matters mainly for small-n prediction intervals.

**66. Why 0.05? Defend or attack the convention in three sentences.** · `M`
**A:** It's Fisher's historical convenience, not an optimum: the right α balances the costs of false positives vs false negatives, which differ wildly between a marketing tweak (α=0.1 fine) and a medical claim (α=0.005 arguable). Fixed 0.05 plus publication filters produced a literature of inflated effects. A senior sets α per decision — and prefers estimation + costs over any bright line.

---

## F — A/B Testing & Causal Inference (67–80)

**67. What exactly does randomization buy you?** · `M`
**A:** Exchangeability: treatment assignment independent of *all* potential confounders, observed and unobserved — so the outcome difference identifies the causal effect without modeling the confounders. It also delivers a known assignment distribution, which is what licenses the inference (tests/CIs). Everything else in causal inference is an attempt to earn what randomization gives for free.

**68. Walk through the sample-size calculation for a two-proportion A/B (baseline 10%, want to detect +1pp, α=5%, power 80%).** · `M`
**A:** n/arm = 2(z_{α/2}+z_β)²·p̄(1−p̄)/δ² with (1.96+0.84)² ≈ 7.85 ⇒ ≈ 15.7·0.10·0.90/0.0001 ≈ **14,100 per arm**. The reusable rule: n/arm ≈ 16·p(1−p)/δ². Senior add-ons: this assumes independent users (not events), a fixed horizon (no peeking), and that δ = 1pp is the *minimum effect worth shipping* — the input people fake most often.

**69. Why does peeking at a running experiment inflate false positives, and what are the legitimate ways to peek?** · `M`
**A:** Under H₀ the p-value trajectory is a wandering process that will dip below 0.05 eventually if watched continuously — stopping at the first dip converts α=5% into 20–30%+. Legitimate peeking: group-sequential designs (O'Brien–Fleming/alpha-spending boundaries) or always-valid inference (mSPRT, confidence sequences) — methods that pay for the looks explicitly. Culturally: hide raw running p-values from dashboards.

**70. Explain CUPED in two sentences and say what determines how much variance it removes.** · `H`
**A:** Adjust each user's metric by their *pre-experiment* value: Y' = Y − θ(X_pre − X̄_pre), θ = Cov(Y,X)/Var(X); randomization guarantees the adjustment is unbiased. Variance shrinks by factor (1−ρ²) where ρ is the pre/post correlation — for sticky metrics (ρ≈0.7) that's ~2× effective sample size, i.e., the cheapest power in the building.

**71. Novelty and primacy effects — what are they and what's the design answer?** · `M`
**A:** Novelty: users engage with anything new, inflating early lift that decays; primacy: users resist change, deflating early lift that recovers. Design answers: run past the transient (decide on late-window data), analyze by exposure cohort / repeat visits, and check new users (who can't feel novelty) as a clean subgroup. Deciding at day 3 measures curiosity, not value.

**72. What is a sample-ratio mismatch, and why is it a stop-the-world alarm rather than a footnote?** · `M`
**A:** The realized assignment split deviates from design beyond binomial noise (chi-square test on counts) — e.g., 48.7% vs 50%. Stop-the-world because whatever mechanism ate users (bot filters, redirects, logging failure by arm) is almost never independent of the outcome, so *every* metric is biased in unknown direction. Rule: no metric readout until SRM passes.

**73. When must you randomize clusters (or time slices) instead of users?** · `H`
**A:** When units interfere: marketplaces (treated buyers deplete shared inventory → control looks worse), social features (spillovers along edges), pricing (arbitrage between arms), operational changes affecting shared queues. Cluster/geo randomization or switchbacks (alternate treatment over time blocks) contain interference at the cost of fewer effective units — power drops by the design effect, which is the price of validity (SUTVA).

**74. Your metric is a ratio (CTR = clicks/views) randomized by user. What's wrong with the naive t-test, and what's the fix?** · `H`
**A:** Naive per-event analysis treats correlated events from the same user as independent — SEs shrink falsely (heavy users dominate). Fixes: analyze per-user ratios or use the **delta method** for the ratio-of-sums (accounting for Var(clicks), Var(views), and their covariance), or cluster-robust/bootstrap-by-user. Validate the pipeline with A/A tests: false-positive rate should come out ≈ α.

**75. Someone finds a significant effect in the "Android users in Germany" subgroup. Protocol?** · `M`
**A:** Multiplicity discipline: pre-registered subgroups get tested with correction; discovered subgroups are hypotheses to *replicate*, not results (interaction test against the complement, not a subgroup-only p-value; expect shrinkage on replication — winner's curse operates hardest in subgroups). The formula: the more surprising and specific the subgroup, the more likely it's noise.

**76. What is an A/A test for, concretely?** · `M`
**A:** Runs the full experimentation machinery with no treatment difference: validates randomization (SRM), instrumentation, and that the analysis pipeline's false-positive rate ≈ α (catching unit-of-analysis bugs, dependence, broken variance estimates). Also yields empirical metric variances for power planning. Teams that never A/A discover their platform's bugs inside real launch decisions.

**77. Leadership asks for the *long-term* effect of a feature the A/B only measured for two weeks. Options?** · `H`
**A:** Long-term holdout: keep a small control group unexposed for months (costly but gold standard); measure learning/adaptation dynamics via exposure-time cohorts; use surrogate indices (validated short-term predictors of long-term outcomes) with explicit caveats. What not to do: extrapolate week-2 lift linearly — novelty decay and ecosystem adaptation (Goodhart on the metric) both bend it.

**78. Difference-in-differences: identifying assumption, and the standard falsification check.** · `M`
**A:** Parallel trends: absent treatment, treated and control would have moved in parallel — untestable in the post-period, but *pre*-period trends can falsify it (event-study plot with leads: any pre-trend divergence kills the design). Modern additions: synthetic control when no single control is parallel, and honest caveats that anticipation effects violate even good pre-trends.

**79. Confounder, mediator, collider — define via one DAG each, and say which adjusting for is a mistake.** · `H`
**A:** Confounder X←C→Y: adjust (else bias). Mediator X→M→Y: adjusting *blocks* the effect you're measuring (only adjust to isolate direct effects — deliberately). Collider X→K←Y: adjusting **creates** spurious association (Berkson) — never condition on it, including by sample selection (analyzing only "engaged users" conditions on a collider). "Should I control for everything?" — this question is the reason the answer is no.

**80. You can't randomize. Give the checklist that makes an observational causal claim survivable.** · `H`
**A:** (1) An explicit identification strategy: which design — regression with a defended backdoor set (DAG drawn), IV, DiD, RDD, matching — and why its assumptions hold here. (2) Falsification: placebo outcomes/periods, negative controls, pre-trends. (3) Sensitivity analysis: how strong an unobserved confounder would overturn it (E-values). (4) Triangulation across designs. Language honesty is part of the method: "associated with, consistent with a causal effect under assumptions A, B" — the checklist exists because "we controlled for stuff" is not identification.


---

## G — Bayesian Reasoning (81–92)

**81. What does "probability" mean to a Bayesian vs a frequentist, and where does the difference bite in practice?** · `M`
**A:** Frequentist: long-run frequency under repetition — parameters are fixed, data are random. Bayesian: degree of belief — parameters get distributions, updated by data. Bites: interpreting intervals (Q83), small-n problems where priors carry real weight, sequential decisions (bandits), and one-off events ("probability this launch succeeds") that have no frequentist repetition to appeal to.

**82. Prior Beta(1,1); you observe 7 conversions in 10 trials. Give the posterior, its mean, and contrast with the MLE.** · `E`
**A:** Conjugacy: posterior = **Beta(1+7, 1+3) = Beta(8, 4)**, mean 8/12 ≈ **0.667** vs MLE 0.7 — the uniform prior acts as one phantom success and one failure, shrinking toward ½. With Beta(20,20) prior (strong belief in ~0.5): posterior Beta(27,23), mean 0.54 — same data, different conclusion, prior made explicit. That explicitness is the sales pitch.

**83. Credible interval vs confidence interval — state each correctly, then say when they numerically agree.** · `M`
**A:** 95% credible: given the data and prior, the parameter lies in this interval with probability 0.95 — the sentence everyone *wants* to say. 95% confidence: the procedure covers the truth in 95% of repetitions. They coincide (asymptotically or exactly) with flat priors and nice likelihoods, which is why the misinterpretation of CIs usually goes unpunished — until priors matter (small n, boundaries).

**84. Why do conjugate priors matter, and name the three pairings every DS should know.** · `E`
**A:** Posterior stays in the prior's family — updating is count arithmetic instead of integration. Pairings: **Beta–Binomial** (rates), **Gamma–Poisson** (counts/rates, → negative binomial predictive), **Normal–Normal** (means, known variance; posterior mean = precision-weighted average of prior and data). Less critical now that MCMC exists, but they're the mental models behind smoothing and shrinkage everywhere.

**85. How do you choose a prior defensibly, and what's the mandatory robustness step?** · `H`
**A:** Sources: previous experiments/portfolio history (empirical priors), domain constraints (rates in [0,1], effects rarely >X%), weakly-informative defaults that regularize without dominating. Mandatory step: **prior sensitivity analysis** — rerun with 2–3 reasonable priors; if conclusions flip, the data are weak and the report must say so. A prior you can't justify to a skeptic in one paragraph shouldn't ship.

**86. What is the posterior predictive distribution, and what does it give you that a point estimate can't?** · `M`
**A:** P(new data | observed data) = ∫ P(new | θ)·P(θ | data)dθ — predictions that carry *both* parameter uncertainty and inherent noise. Uses: honest prediction intervals, model checking (does simulated data resemble real data?), and decision inputs (expected profit over the whole posterior, not at the point estimate). Point estimates silently pretend θ̂ is known — the predictive prices in that it isn't.

**87. Bayesian A/B: what do you report instead of a p-value, and what's the honest caveat about "stopping when sure"?** · `M`
**A:** Report P(B > A | data), the posterior distribution of the lift, and **expected loss** of shipping each variant (decision-theoretic stopping: ship when expected loss < threshold). Caveat: Bayesian inference is immune to peeking *for calibrated posteriors*, but if you stop on posterior thresholds your frequentist error rates still depend on the stopping rule — with honest priors expected-loss stopping controls what matters (cost of wrong decisions), which should be said out loud, not hidden.

**88. Twelve stores have 2 weeks of conversion data each; store #7's raw rate looks amazing. What's the empirical-Bayes move?** · `H`
**A:** Shrinkage: estimate the population distribution of store rates (e.g., fit a Beta across stores), then compute each store's posterior — small-sample stores get pulled hard toward the grand mean, high-volume stores barely move. Store #7's 2-week miracle mostly regresses. This is partial pooling (James–Stein flavor): it beats both "trust every raw rate" and "ignore store differences," and it's the correct default for any leaderboard of noisy small samples.

**89. Bayes factors vs p-values — one advantage, one practical weakness.** · `H`
**A:** Advantage: BF compares evidence *for* H₀ vs H₁ symmetrically — it can support the null (p-values can't) and composes with prior odds into posterior odds. Weakness: sensitive to the prior on the alternative's effect size (diffuse priors mechanically favor H₀ — Lindley's paradox), so a BF without its prior stated is uninterpretable. In practice: estimation with posteriors usually beats both testing camps.

**90. Explain Thompson sampling in three sentences, including why it balances explore/exploit.** · `M`
**A:** Keep a posterior per arm (Beta per variant for conversions); each round, *sample* one value from each posterior and play the argmax; update with the observed reward. Uncertain arms sometimes sample high (exploration), confident good arms usually win (exploitation) — randomization is exactly proportional to the probability each arm is best. It's Bayesian A/B that allocates traffic while it learns, with strong regret guarantees.

**91. Why does Laplace (add-one) smoothing work — what is it secretly?** · `M`
**A:** It's a Dirichlet/Beta prior in disguise: adding α pseudo-counts per category = posterior mean under Dirichlet(α) — MAP-flavored shrinkage that keeps unseen events from having probability zero (which would nuke products of probabilities in naive Bayes / language models). Choosing α = choosing prior strength; "add-one" is just the uniform unit prior.

**92. What is partial pooling in a hierarchical model, and when does it beat both extremes?** · `H`
**A:** Extremes: no pooling (each group fit alone — noisy for small groups) and complete pooling (one global fit — biased for distinct groups). Hierarchical models learn the between-group variance and shrink each group toward the population proportionally to its noise — beating both whenever groups are related but not identical (stores, regions, SKUs, languages). It's Q88 formalized, and the default answer to "we have 200 small segments."

---

## H — Regression & Correlation Statistics (93–102)

**93. Which OLS assumptions matter for *prediction* vs for *inference on coefficients*?** · `M`
**A:** Prediction: mostly that the functional form fits and train resembles serve — normality/homoscedasticity barely matter. Inference: independence and homoscedasticity drive SE validity (violations → wrong CIs even with perfect fit), no perfect collinearity for identifiability, and exogeneity (errors uncorrelated with X) for coefficients to mean anything causal. The senior habit: state *which use* before defending assumptions.

**94. Interpret β = 0.4 in log(y) ~ log(x) — and in y ~ log(x)?** · `E`
**A:** Log-log: elasticity — a 1% increase in x associates with ~0.4% increase in y. Level-log: a 1% increase in x associates with ~0.004 units of y (β/100 per percent). Getting transforms' interpretations right is where regression readouts most often silently lie in decks.

**95. Three ways R² misleads.** · `M`
**A:** (1) It never decreases when you add junk features (hence adjusted R² for comparison). (2) High R² ≠ correct model — confounded or leaky features score beautifully; low R² can be the honest ceiling of a noisy process. (3) Out-of-sample R² can be **negative** (worse than predicting the mean) — a sign convention juniors read as a bug. Bonus: R² compares against the mean baseline; for time series, beating the mean is not beating the naive lag.

**96. Walk the chain from a coefficient's standard error to its CI and t-statistic — and name the biggest threat to that SE.** · `M`
**A:** SE(β̂) from the residual variance and X's variation (inflated by collinearity via VIF); t = β̂/SE tests β=0; CI = β̂ ± t*·SE. Biggest practical threat: **dependence** — clustered or serially correlated errors make the nominal SE far too small (the n is a lie), before any normality concern matters.

**97. Residuals are heteroscedastic. What's actually broken, and what's the standard repair?** · `M`
**A:** β̂ remains unbiased/consistent — what breaks is the SEs (and thus tests/CIs), plus efficiency. Repairs: heteroscedasticity-robust (sandwich/HC) standard errors as the default, WLS if the variance structure is known, or model the variance (GLM). Saying "the coefficients are wrong" is the junior miss; it's the *inference* that's wrong.

**98. Same question for autocorrelated errors in a time-series regression.** · `M`
**A:** Positive autocorrelation makes OLS SEs dramatically overconfident — t-stats inflate, spurious significance abounds (compounding with spurious-trend regression). Repairs: Newey–West (HAC) SEs, explicit error models (AR terms/GLS), or differencing/detrending first. Diagnostic: residual ACF / Durbin–Watson — should be flat before any coefficient is quoted.

**99. Omitted-variable bias: give the direction rule and one worked example.** · `H`
**A:** Bias on β_x has the sign of (effect of omitted Z on y) × (correlation of Z with x). Example: regress salary on education omitting ability — ability raises salary (+) and correlates with education (+) → education's coefficient is biased **upward**. Reasoning the direction, not just naming OVB, is the senior signal; it's also how you argue whether an unmeasured confounder helps or hurts your story.

**100. Your predictor is measured with noise. What happens to its coefficient?** · `H`
**A:** Attenuation: classical measurement error biases β̂ toward zero by the reliability ratio σ²ₓ/(σ²ₓ+σ²ₑ) — noisier x, flatter estimated slope; in multivariate settings the bias can spill unpredictably onto other coefficients. Consequences: "no effect" findings on noisy features are suspect, and comparing coefficients across features with different measurement quality is unfair. Remedies: better measurement, instrumental variables, or errors-in-variables models.

**101. Why can't you read ridge/lasso coefficients (or their naive CIs) the way you read OLS output?** · `H`
**A:** Regularization deliberately biases coefficients toward zero (that's the variance trade) — magnitudes reflect penalty strength as much as effects, lasso's selection is unstable under correlated features, and naive post-selection CIs ignore that the *same data chose the model* (selective inference problem — intervals are far too narrow). Fixes exist (debiased lasso, sample-splitting, stability selection) but the default posture is: regularized fits are for prediction; inference needs a design.

**102. Interpret β = 0.69 on feature x in a logistic regression.** · `M`
**A:** Per unit of x, the **log-odds** rise by 0.69 — odds multiply by e^0.69 ≈ **2.0** (odds ratio, not "probability doubles"). Probability change depends on the baseline: from 1% → ~2%, but from 50% → ~67% — the nonlinearity is the trap. If stakeholders need probability language, report marginal effects at representative points.

---

## I — Stochastic Processes (103–110)

**103. A symmetric random walk after n steps: expectation, variance, and the recurrence punchline.** · `M`
**A:** E[Sₙ] = 0, Var(Sₙ) = n (SD grows as √n — why cumulative sums of noise *look* like trends). Punchline (Pólya): simple random walk returns to the origin with probability 1 in 1-D and 2-D, but not in 3-D — "a drunk man finds his way home; a drunk bird may not." Practical echo: cumulative metric plots of pure noise wander impressively; never read drift off a random walk by eye.

**104. Gambler's ruin: starting with i units, absorbing at 0 and N. P(reach N first) for a fair game — and the asymmetric punchline for an unfair one.** · `M`
**A:** Fair: **i/N** (elegant via the martingale/optional-stopping argument: expected fortune is constant). Unfair with per-step win probability p<½: P = (1−(q/p)^i)/(1−(q/p)^N) — ruin becomes near-certain exponentially fast even with a large bankroll; tiny house edges are decisive at scale. Same math prices "keep doubling until I win" strategies at their true worth.

**105. A two-state Markov chain flips 0→1 w.p. a and 1→0 w.p. b. Stationary distribution, and what "stationary" actually means.** · `M`
**A:** π = (b/(a+b), a/(a+b)) — balance the cross-flows (π₀a = π₁b). Stationary means: if the chain starts there it stays distributionally put, and (if ergodic) long-run time-averages converge to π regardless of start. DS uses: steady-state market share under churn/acquisition rates, session-state modeling, and the base of MCMC.

**106. Explain PageRank as a Markov chain in three sentences.** · `M`
**A:** A random surfer follows a uniformly random outlink each step; with probability 1−d teleports to a random page — the damping making the chain irreducible and aperiodic so a unique stationary distribution exists. PageRank *is* that stationary distribution: the long-run fraction of time spent on each page. The same construction ranks anything with a link/transition structure (products via co-purchase, entities in graphs).

**107. Poisson process facts every DS should be able to produce: interarrivals, superposition, thinning.** · `M`
**A:** Interarrival times are i.i.d. Exp(λ) (memoryless); merging independent Poisson streams gives Poisson with summed rates (superposition); independently keeping each event with probability p gives Poisson(pλ) (thinning) — and the thinned streams are independent. These three assemble most service/traffic models — and their failures (bursts, dependence) diagnose when Poisson is a fiction.

**108. Buses arrive every 10 minutes on average. Why is your expected wait *not* 5 minutes, and when is it exactly 10?** · `H`
**A:** Inspection paradox: arriving at a random time, you're more likely to land in a *long* gap — length-biased sampling gives E[wait] = E[X²]/(2E[X]) ≥ E[X]/2, with equality only for perfectly regular headways. For Poisson arrivals (exponential gaps), memorylessness makes the wait a full **10 minutes**. Same bias explains "average class size experienced by students > average class size," and why sampled-by-usage anything looks heavier than the catalog average.

**109. What is a martingale, and what does the optional stopping theorem have to do with A/B peeking?** · `H`
**A:** A process whose conditional expectation of the next value equals the current one — a fair game; optional stopping says clever stopping rules can't change the expected value *under conditions* (bounded time/values). The peeking connection: naive test statistics aren't the protected object — stopping at extremes biases them — while methods built on actual martingales (mSPRT, confidence sequences, e-values) inherit validity at *any* stopping time. That's the theory under "always-valid inference."

**110. Each user invites a random number of others with mean m. When does the viral process die out, and with what probability?** · `H`
**A:** Branching process: extinction is certain iff m ≤ 1 (critical case included); for m > 1, extinction probability is the smallest root of s = G(s) (the offspring PGF) — e.g., Poisson(1.5) offspring → extinction ≈ 0.42: even supercritical growth dies early with substantial probability. Same machinery as epidemic R₀ and cascade modeling; "m > 1 so it grows" ignores that near-critical processes usually fizzle.

---

## J — Paradoxes & Fallacies (111–118)

**111. Construct a Simpson's paradox with numbers, then state the resolution rule.** · `M`
**A:** Treatment A: small cases 81/87 (93%), large 192/263 (73%) — better in both; Treatment B: small 234/270 (87%), large 55/80 (69%). Pooled: A = 273/350 (78%) vs B = 289/350 (83%) — B "wins" because A got the hard cases. Resolution: identify the confounder (case severity → both treatment choice and outcome) and answer the *causal* question with stratified/adjusted comparisons; which aggregation is right depends on the question, not the arithmetic.

**112. Base-rate neglect: why do smart people fail Q1, and what's the one-line vaccine?** · `E`
**A:** The mind grabs P(evidence | hypothesis) — vivid, given — and swaps it for P(hypothesis | evidence), ignoring how rare the hypothesis is. Vaccine: force natural frequencies — "of 10,000 people: 100 sick, 95 caught; 9,900 healthy, 990 false alarms → 95/(95+990) ≈ 9%." Frequencies make the base rate impossible to drop.

**113. The prosecutor's fallacy — define it and give the courtroom-to-fraud-model translation.** · `H`
**A:** Presenting P(evidence | innocent) as if it were P(innocent | evidence): "the DNA match probability is 1 in a million, so he's guilty" ignores that in a city of 10M there are ~10 expected random matches — P(innocent | match) may be ~90%. Translation: a fraud alert with a 1-in-10,000 false-positive rate firing on 10M daily transactions produces 1,000 false alarms — the alert's *precision* depends on prevalence, not on the impressive-sounding conditional.

**114. Berkson's paradox: why do "talent" and "attractiveness" anti-correlate in the dating pool, and where does the same bug live in product analytics?** · `H`
**A:** Selection on a collider: if people date those who clear a *combined* bar (talented OR attractive enough), then within the selected pool the traits trade off — high on one permits low on the other — creating negative correlation absent in the population. Product version: analyzing only *engaged* users (engagement = collider caused by both price sensitivity and feature love) manufactures spurious negative relationships between its causes. The fix is recognizing the selection, not adding controls.

**115. Gambler's fallacy vs the hot-hand debate — and the subtle bias that reopened the latter.** · `H`
**A:** Gambler's fallacy: expecting independent trials to self-correct ("red is due"). Hot hand was long dismissed as the mirror illusion — but Miller–Sanjurjo showed the standard analysis (P(hit | previous hit) in finite sequences) is *biased downward*: in short sequences, conditioning on a hit having occurred makes the next-flip hit rate < p even for fair coins. Correcting the bias, real hot-hand effects reappeared. Meta-lesson: the *measurement* of streakiness had its own selection artifact — a beautiful trap for data scientists.

**116. The Will Rogers phenomenon: how can moving patients between groups improve *both* groups' averages?** · `M`
**A:** Move an element below group A's mean but above group B's mean from A to B: both means rise, nothing improved. Real version: better diagnostics reclassify borderline patients into "sick" (they're healthier than existing sick patients, raising that group's survival) and out of "healthy" (removing its sickest, raising that too) — stage migration fakes medical progress. DS version: re-segmentation "improving every segment's KPI" while the total is flat.

**117. Two analysts study whether a treatment helps, one using change scores (post−pre), one using ANCOVA (post adjusting for pre). They disagree. Who's wrong?** · `H`
**A:** Possibly neither — Lord's paradox: when groups differ at baseline, change-score and covariate-adjusted analyses answer *different questions* (unconditional change vs change conditional on baseline) and can conflict. In randomized data baselines balance and ANCOVA simply wins on power; in observational data the choice encodes a causal assumption about how baseline relates to assignment — the resolution is drawing the DAG, not picking the analysis with the preferred p-value.

**118. Ecological fallacy: give an example where group-level and individual-level correlations reverse, and the DS habit that prevents it.** · `M`
**A:** Classic: across regions, average income correlates positively with voting for party X, while within regions richer *individuals* vote for party Y — aggregation reverses the sign (a cousin of Simpson's). DS version: country-level "more app usage ↔ higher retention" while within any country heavy users churn more. Habit: match the inference level to the data level; never sell a group-level correlation as an individual-level mechanism, and disaggregate before believing any cross-unit scatter.

---

## K — Probability & Statistics for ML (119–124)

**119. Define calibration for a classifier, and explain why accuracy is not a "proper" score while log loss and Brier are.** · `H`
**A:** Calibrated: among cases scored 0.7, ~70% are positive — probabilities mean what they say (reliability diagrams/ECE measure it). Proper scoring rules are optimized in expectation by reporting the *true* probability: log loss and Brier qualify; accuracy (or anything thresholded) is improper — it rewards confident distortion and is blind between 0.51 and 0.99. Brier even decomposes into reliability − resolution + uncertainty, separating calibration from discrimination. If downstream decisions consume probabilities, train/select on proper scores.

**120. Show that minimizing MSE is Gaussian MLE and minimizing log loss is Bernoulli MLE — and why this framing pays rent.** · `M`
**A:** Gaussian likelihood: −log L ∝ Σ(y−ŷ)²/2σ² + const → MSE. Bernoulli: −log L = −Σ[y log p̂ + (1−y)log(1−p̂)] → cross-entropy. Every loss is a distributional assumption in disguise — so loss choice should follow the target's distribution: heavy tails → Laplace/Huber (MAE-flavored), counts → Poisson/negative-binomial deviance, zero-inflated demand → Tweedie/hurdle. When a model "mysteriously" mean-collapses or chases outliers, interrogate the implied likelihood first; it's the highest-leverage question in applied loss design.

**121. Entropy, cross-entropy, KL divergence: relate the three, and give the forward-vs-reverse KL intuition.** · `H`
**A:** H(p) = −Σp log p (irreducible surprise); CE(p,q) = H(p) + KL(p‖q) — so minimizing cross-entropy w.r.t. q minimizes KL to the truth; KL ≥ 0, asymmetric, not a metric. Direction intuition: forward KL(p‖q) punishes q for missing p's mass → *mass-covering* (broad, mode-averaging fits — MLE's direction); reverse KL(q‖p) punishes q for putting mass where p has none → *mode-seeking* (sharp, drops modes — variational inference's direction). The asymmetry explains real behavior differences between MLE-trained and variationally-trained models.

**122. Mutual information vs correlation for feature relevance — advantages, costs, and one caution.** · `M`
**A:** MI = KL between joint and product-of-marginals: captures *any* dependence (nonlinear, non-monotonic), zero iff independent — strictly more general than correlation. Costs: needs density/binning estimation (noisy in small samples, biased upward — high-cardinality features fake MI), no sign/direction, harder to compare across scales. Caution: like every univariate relevance screen, it misses features useful only in interaction and can crown leakage; treat as a screen, confirm in-model.

**123. Write the bias–variance decomposition and give the practical reading for model selection.** · `M`
**A:** Expected squared error at x = **Bias²(f̂) + Var(f̂) + σ²(irreducible)** — over resampled training sets. Reading: underfitting = bias-dominated (both train and val error high — more capacity/features, less regularization); overfitting = variance-dominated (train–val gap — regularize, simplify, more data); σ² is the floor no model beats, worth estimating (label-noise audits) so you stop chasing ghosts. Learning curves are this decomposition made visible.

**124. How large a test set to estimate accuracy within ±1% (95% confidence)? Derive it two ways.** · `H`
**A:** Hoeffding (distribution-free): n ≥ ln(2/δ)/(2ε²) = ln(40)/(2·0.0001) ≈ **18,400**. Normal approximation at worst case p = 0.5: n ≈ 1.96²·0.25/0.0001 ≈ **9,600** (smaller — it uses the variance rather than a worst-case bound). Practical extensions worth volunteering: per-segment guarantees multiply the requirement; *comparing* two models on the same set needs the paired analysis (McNemar/paired bootstrap), which can need far less than two independent estimates. This is the question behind "is our test set big enough" — most aren't, per segment.

---

## How to score stats & probability answers

| Signal | Weak (1–2) | Strong senior (4–5) |
|---|---|---|
| **Setup before arithmetic** | Jumps to formulas, hopes | Defines events/conditionals precisely; states assumptions before computing |
| **Definition hygiene** | Recites p-value/CI folklore | Says what quantities are *not*; catches the conditional's direction every time |
| **Mechanism** | Names the concept | Explains why it happens (selection, conditioning, variance structure) and predicts direction of bias |
| **Assumption audit** | "Assume normal/independent" reflexively | Names which assumption carries the result and what breaks it in production data |
| **Decision link** | Stops at the statistic | Converts to action: sample sizes, thresholds by cost, what to report to a stakeholder |

**Usage.** 6–8 questions per 45-minute round: one classic-puzzle warm-up from §A (setup quality), two from §E/§F (the misinterpretation minefield — most predictive section for real-world damage), one computation (§C/§D), one paradox (§J) for reasoning under surprise, one from §K to tie stats to the ML day job. Highest-signal single questions: **51** (p-value hygiene), **74** (unit-of-analysis), **79** (collider vs confounder), **120** (loss = likelihood — separates practitioners who understand their tools from those who inherit them).
