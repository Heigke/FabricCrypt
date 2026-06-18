Look at these snapback I-V plots from NS-RAM 2T cell.

Context: dashed lines = measured (Sebas silicon), solid = our pyport model.
At V_G1=0.4, three V_G2 ∈ {0.0, 0.2, 0.4}. V_d swept 0-4V.

Files:
- z331_snapback_graph.png — 3-panel overview (forced-Vsint pyport vs silicon)
- z331_VG2=0.00.png, z331_VG2=0.20.png, z331_VG2=0.40.png — per-Vg2 zooms
- iv_vg1_0.2_V6.png, iv_vg1_0.4_V6.png, iv_vg1_0.6_V6.png — z328 solver V6 (free Vsint)

Expected behavior: I_d ramps with V_d, hits peak (knee), then "snapback"
where current drops sharply because body charge builds up and NPN fires.
User's complaint: "they go down and then some snap, they don't look good."

Q1: Per plot, does the SHAPE of model match silicon qualitatively?
Q2: Where does model fail visually (amplitude, peak position, slope)?
Q3: What's the most likely root-cause from visual inspection alone?
Q4: Compare z331 (forced-Vsint) plots to z328_V6 plots — which is closer to silicon?
Q5: What ONE plot anomaly should we prioritize fixing first?

≤500 words total.
