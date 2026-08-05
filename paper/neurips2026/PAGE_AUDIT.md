# Page-count audit (round-5)

**Status:** 2 pages over NeurIPS 9-page main-content limit.

## Layout

| Pages | Content |
|---|---|
| 1 | Abstract + §1 Introduction |
| 2-3 | §2 Related Work + Table 5 (positioning) |
| 4-7 | §3 Method (incl. Eq, Table 4 fidelity, Algorithm 1, Theoretical Properties) |
| 7-8 | §4 Experiments (incl. Table 4 head-to-head + Table 6 main results + Figs 2-3) |
| 9 | §5 Ablation + Table 5 importance + §6 Analysis |
| 10 | §6 Analysis cont. + §7 Limitations starts at line 85 |
| 11 | §7 cont. + §8 Conclusion + References starts at line 49 |
| 12 | References cont. |
| 13-19 | Appendix |

Main content (§1-§8) currently ends on page 11; references then start.
NeurIPS counts §1-§8 toward the 9-page limit.

## Trim plan (target: -2 pages)

| Action | Estimated savings | Difficulty |
|---|---|---|
| Move Table 5 (positioning Y/N) to appendix | -0.2 pg | low |
| Move Algorithm 1 box to appendix (keep prose summary) | -0.3 pg | low |
| Move Table 4 (per-model F(b) fidelity) to appendix | -0.3 pg | low |
| Compress §2 Related Work (very dense; one citation per method) | -0.4 pg | medium |
| Compress §7 Limitations from 6 numbered items to 4 (merge 5+6 with §3.2 finding) | -0.4 pg | medium |
| Move Figure 3 (Gini + importance curves) to appendix | -0.3 pg | low |
| Trim §3.5 Theoretical Properties paragraph to 3 sentences | -0.2 pg | medium |

Total potential trim: ~2.1 pages.

## Recommended trim order

1. **Move Algorithm 1 to appendix** — algorithm pseudocode is supplementary; the 1-paragraph prose description in §3.4 is enough.
2. **Move Table 5 (positioning) to appendix** — it's a feature comparison table, not a result.
3. **Move per-model F(b) Table 4 to appendix** — main paper just needs the calibrated values cited as a range.
4. **Trim §2 Related Work paragraph density** — replace dense citation lists with one canonical reference per method.
5. If still over: move Figure 3.

These are non-destructive moves (preserve content, change location). Recommend applying in order until fit is achieved.

## Note for /submit-prep

Run pdflatex after each trim to verify page count converges to 9 main + N refs/appendix.
