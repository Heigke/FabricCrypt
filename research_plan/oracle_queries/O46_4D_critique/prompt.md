We are preparing a Mario brief v4.4 for NS-RAM. Today's findings:

(A) Application synthesis (oracle 3-way consensus, O44): NS-RAM is
positioned as IP-licensable spiking-neuron-core for MCU/SoC integration,
top apps = always-on KWS, edge co-processor, industrial anomaly det.

(B) v4.4 HEADLINE: HDC at N=1024 reaches 80.23% on UCI-HAR (n=20 seeds,
CI95 ±0.74pp) at 2.3 nJ/inference. Monotone log-N scaling 59→81% from
N=64→1024.

(C) Novel finding: NS-RAM body-state noise as Bayesian MCMC RNG passes
ESS-ratio gate (1.03× vs pseudo-RNG, n=10K MH steps).

(D) Negative results (honest):
  - Sebas IV transient replay: pyport over-predicts subthreshold I_d
    by ~1.67 dec systematically (z298b)
  - Original Mario+Sebas TCAD curves vs our pyport: 2-6 dec gap absolute
    current, 0.92 dec shape-only after offset correction (z299b)
  - Snapback shape gap: 4 candidate physics terms (Rs(V_d), self-heat,
    RaCBE, body 2nd-term) RULED OUT cleanly (z300)
  - KWS on Speech Commands: NS-RAM SNN at chance (8.3% for 12 classes)
  - NAB anomaly detection: NAB score ~17 (gate 30+) across 3 scoring
    variants

(E) Open: snapback gap appears to need heavier physics (avalanche
M(V_bc), velocity-sat feedback, hot-carrier into floating body).

QUESTIONS for each of you (please answer all):

Q1 (Falsification): What is the strongest scientific challenge you can
mount against the claim "NS-RAM is competitively viable as an IP-block
for standard-CMOS MCU integration in always-on sensing"? Cite the
findings above.

Q2 (Headline integrity): Is HDC 80.23% at N=1024 / 2.3 nJ a defensible
v4.4 headline given the negative results in (D)? What caveats MUST be
in the brief?

Q3 (Surprise): Is there a finding here that should be the v4.4 LEAD
(more interesting than HDC) we are under-valuing? (E.g., Bayesian RNG,
which is a paradigm-different claim about physical noise as a resource.)

Q4 (Cuts): Which of the negative results should we report explicitly
(integrity), and which are too minor to surface?

Q5 (Single-sentence verdict): Should we ship v4.4 to Mario now, or
gate it on closing one specific finding first? If gate, which one?
