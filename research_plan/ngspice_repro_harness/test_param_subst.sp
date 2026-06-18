* Reproduction harness — ngspice silent .param fallback
* Demonstrates: bare-identifier .param refs inside .model body
* are NOT substituted; ngspice silently uses BSIM4 defaults.
*
* Run with:  ngspice -b test_param_subst.sp
* Expected OK:  toxe lands at 4e-9 (set), 5e-9 (braced), 7e-9 (braced)
* Expected BUG: toxe lands at 3e-9 (BSIM4 default) for bare-id case
* Run with debug-printf-instrumented build to confirm.

.title silent .param substitution test

.param toxn = 4e-9
.param toxp = 5e-9
.param toxe_explicit = 7e-9

* CASE 1 — explicit numeric literal (always works)
.model nmos_explicit nmos level=14 version=4.8.3
+ toxe = 4e-9    toxp = 4e-9    vth0 = 0.5

* CASE 2 — bare identifier (FAILS SILENTLY: toxe=3e-9 BSIM4 default)
.model nmos_bare nmos level=14 version=4.8.3
+ toxe = toxn    toxp = toxn    vth0 = 0.5

* CASE 3 — braced substitution (WORKS: toxe=4e-9 from .param)
.model nmos_braced nmos level=14 version=4.8.3
+ toxe = {toxn}    toxp = {toxn}    vth0 = 0.5

* CASE 4 — single-quote substitution (WORKS)
.model nmos_quoted nmos level=14 version=4.8.3
+ toxe = 'toxn'    toxp = 'toxn'    vth0 = 0.5

* CASE 5 — bare identifier on multi-assign line, mixed with literals
.model nmos_mixed nmos level=14 version=4.8.3
+ toxe = toxn    vth0 = 0.5    pvth0 = -1.45e-15
* Empirical: pvth0 also lands at 0 (likely line-desync side-effect
* of failed bare-id parse upstream).

* Trivial circuit just to exercise the parser
m1 d g 0 0 nmos_explicit  W=1u L=0.13u
m2 d g 0 0 nmos_bare      W=1u L=0.13u
m3 d g 0 0 nmos_braced    W=1u L=0.13u
m4 d g 0 0 nmos_quoted    W=1u L=0.13u
m5 d g 0 0 nmos_mixed     W=1u L=0.13u

vd  d  0  dc 1
vg  g  0  dc 0.5

.op
.print op v(d) v(g) i(vd)

.end
