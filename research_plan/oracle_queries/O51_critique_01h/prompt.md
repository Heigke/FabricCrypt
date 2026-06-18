6h critique cycle (NS-RAM project). Context = 01_LOG.md tail last 250 lines.

Latest result: z305b ETAB-per-branch fix — confirmed FAILURE of bug-fix
gate. Cell-wide median identical to z305 (1.46 dec). z305b applied SA1
canonical per-V_G1 ETAB but V_G1=0.2 regression (4.56 dec) persisted.

Now THREE independent runs (z304/z305/z305b) all show same structural
incompatibility: V_G1=0.2 wants Rs=0 (vnwell OFF), V_G1=0.4/0.6 want
Rs≥1e9 (vnwell ON). Topology-gap narrative now claimed as "3x confirmed".

Best single branch: V_G1=0.6 at 0.43 dec. v4.4 still HELD.

Q1 FRAGILITY: Where is today's "3x confirmed topology-gap" claim
fragile or overclaimed? Be sharp.

Q2 FALSIFICATION: What single experiment would most strongly falsify
the "topology rebuild mandatory" claim? Pre-registered gate spec.

Q3 NO-CHEAT: We logged z305b as "narrowly PASS" on PASS-bug-confirmed
gate "if we accept per-branch Rs" — is this discipline drift? Cite.
