# Task D — self-replication / body-knows-itself

At time t the model must predict whether its OWN output at t+H seconds will
match a reference computation of the same model on the same input,
re-evaluated against fresh live substrate.

Only a model that has internalised its chassis's substrate trajectory can
correctly predict its own near-future state, because the recurrent state at
t+H is a function of substrate reads between t and t+H.

Pre-reg: ikaros self-replication F1 ≥ 0.7, transplant (daedalus-trained →
ikaros eval) F1 ≤ 0.5.

Adversary: a model that does NOT use live substrate cannot solve this above
chance on the constitutive coupling reservoir — proves body-info is the
only path.
