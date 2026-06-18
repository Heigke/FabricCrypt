# Phase 7 A/B/C/D — the killer ablation (30 seeds, ridge reservoir, real twins)

Two metrics × two eval hosts × four cells.

## C2 (self-anomaly autoencoder, AUROC, n=30 seeds per cell)

| Eval host  | Cell | Structure          | Data host | AUROC mean | AUROC std |
|------------|------|--------------------|-----------|------------|-----------|
| ikaros     | A    | ikaros-hash        | ikaros    | 0.8340     | 0.0254    |
| ikaros     | B    | random             | ikaros    | 0.8338     | 0.0246    |
| ikaros     | C    | ikaros-hash        | daedalus  | 0.4923     | 0.0414    |
| ikaros     | D    | random             | daedalus  | 0.4917     | 0.0411    |
| daedalus   | A    | daedalus-hash      | daedalus  | 0.8291     | 0.0456    |
| daedalus   | B    | random             | daedalus  | 0.8327     | 0.0471    |
| daedalus   | C    | daedalus-hash      | ikaros    | 0.4994     | 0.0462    |
| daedalus   | D    | random             | ikaros    | 0.4994     | 0.0462    |

Pre-reg gates:
- A − B ≥ 10%  →  FAIL (Δ = +0.0002 on ikaros, −0.0036 on daedalus)
- A − C large  →  PASS (Δ ≈ +0.34, but this just confirms data matters)
- (A − B) > (C − D)  →  FAIL (both ≈ 0)
- A > max(B,C,D) by ≥ 1σ  →  FAIL (B matches A within noise)

**Conclusion**: chassis-hash-derived structure adds **zero** capability over
arbitrary random structure given the same data. All signal is in the data
distribution, none in the chassis-bound init.

Same pattern observed on C1 (next-step prediction, NRMSE).

## Interpretation

The hash is being used as a *label* (a deterministic but informationally
inert mapping from chassis → 4-byte seed). The seed only chooses *which*
random reservoir we instantiate. Since random reservoirs of the same size
are statistically interchangeable (Yildiz et al. 2012; Lukoševičius 2012),
choosing one via SHA-256(chassi-id) vs. choosing one via `seed = 7` cannot
change the fitted ridge solution beyond seed noise.

The substrate never enters the forward pass. The model is *recognisable* by
the chassis (the data it was trained on is from this chassis) but it is
*not constitutive* (the chassis is not doing computation inside the model).
