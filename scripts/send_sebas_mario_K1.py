#!/usr/bin/env python3
"""Reply in the existing 'Zoom NSRAM' thread to Sebas Pazos, Mario Lanza, Robert Luciani.

Lead: the K1 finding (answers Sebas's 'anything that saves SPICE time' directly).
Also addresses Sebas's two open questions (fan-out architecture + testchip cell wish)
and re-pings Mario on proposal direction.

Threads via In-Reply-To + References headers so Gmail keeps it in the same conversation.

Run --dry-run to preview. No flag = live send.
Logs to results/mail_sebas_K1/mail_log.json.
"""
import argparse, json, smtplib, ssl, sys, time
from email.message import EmailMessage
from email.utils import make_msgid, formatdate
from pathlib import Path

PW = Path("/home/ikaros/Documents/claude_hive/hugue_tutorials_by_hugo/.secrets/gmail_app_pw").read_text().strip().replace(" ", "")
USER = "bergvall.eric@gmail.com"
TO_LIST = ["smpazos@ieee.org", "mlanza@nus.edu.sg", "robert@nervdynamics.io"]
SUBJECT = "Re: Zoom NSRAM"

# Threading — keep this email in the existing thread (Eric's May 11 bump)
IN_REPLY_TO = "<CAMCexNB3_c-qJ2GcN5AcbYO8Gdk0sp_gf2gY_NBDoRHLPFHUsg@mail.gmail.com>"
# Build References from the thread chain so Gmail collapses correctly
REFERENCES = (
    "<JH0PR06MB6341FB5B29723B4E0EEB3130E74CA@JH0PR06MB6341.apcprd06.prod.outlook.com> "
    "<CAMCexNBUncTsEbkEEhx02Ne7ppWkP6SWXrjQAodLpi8bZgY+RA@mail.gmail.com> "
    "<CAM9E1cowosxxFwKfioqqHaeSqXw8kq8XKGqCmh3Zd7G5rW2E0A@mail.gmail.com> "
    "<CAMCexNB3_c-qJ2GcN5AcbYO8Gdk0sp_gf2gY_NBDoRHLPFHUsg@mail.gmail.com>"
)

ATTACH_DIR = Path("/tmp/sebas_mail_attach")
ATTACHMENTS = [
    ATTACH_DIR / "verdict_K1_VG1_06.md",
    ATTACH_DIR / "scatter_fwd_bwd.png",
]

LOG_PATH = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/mail_sebas_K1/mail_log.json")

BODY = """Dear Sebas, Mario, Robert,

Quick introduction: this email is written by Eric's research agent — an
LLM that Eric uses to track and write up findings from the NS-RAM
modelling work. Eric has reviewed and OK'd everything below; replies
go to him directly (he's the human in the loop). I'm signing as the
agent so it's clear which messages are drafted this way vs typed by
Eric himself.

With that out of the way: bumping our thread with a concrete finding
from the fit work, and picking up the two openings Sebas left on the
30 April call.

The headline: one knob in the M1/M2 card may explain most of our DC fit
gap

We finished the per-bias decomposition of the pyport BSIM4 fit against
the 33-bias regression target. The remaining residual is dominated by
VG1 = 0.60 — top-5 outliers all sit there at ~1.99 dec each, with Imeas
saturated at 4.05e-5 A. The gap is symmetric fwd vs bwd (Spearman
rho = 0.95), so it's a static under-prediction of Vd-saturation, not a
hysteresis effect.

Tracing the parameters, it reduces to a single entry in your
2Tcell_BSIM_param_DC.csv:

   K1 = 0.41825   at VG1 = 0.60
   K1 = 0.53825   at VG1 = 0.20 and 0.40   (card value)

K1 at VG1=0.60 is 22 % below the card value you keep at the other
branches. Reverting only that entry to the card gives:

   K1 @ VG1=0.6 | all-bias median | VG1=0.6 triode RMSE
   -------------+-----------------+----------------------
   0.41825      | 1.163 dec       | 1.183 dec   (current)
   0.53825      | 0.883 dec       | 0.425 dec   (card revert)
   0.6459       | 0.528 dec       | 0.131 dec   (1.2x card)

VG1=0.4 and VG1=0.2 are unchanged (override only acts at VG1=0.6).

Sebas — was K1 = 0.41825 at VG1=0.60 in the CSV intentional (an
empirical body-bias correction you derived), or a transcription? If it
was a fit-time tweak and we can revert to the card value, ~24 % of our
residual closes today with no measurement.

What we already falsified internally (so you don't redo it)
  - NPN-OFF beats NPN-ON by 1.19 dec; T-coefficient runs the wrong sign.
    Parasitic NPN is not the missing parallel path.
  - BSIM4 §10.1 JTS-TAT at default: worsens joint fwd+bwd by 0.234 dec.
  - rbodymod=1 sweep up to R_body=1e6 ohm: regresses by +3.65 dec.
  - selfheatmod=1 sweep to Rth=1e8 K/W: Delta = -0.02 dec.
  - Hurkx-Gamma TAT: alpha=0 neutral; alpha>0 regresses.

Picking up the two openings you left on 30 April

(1) The fan-out / few-tens-of-neurons SPICE example you offered to run
    Taking you up on this. We have a 32-cell test topology ready —
    ER-style sparse (k=3) with one shared input neuron, ~96 cell
    instances counting fan-out, ngspice-feasible. If you tell us your
    runtime budget (rows/columns and seconds-per-bias), we'll package
    a netlist using your M1/M2 cards and the pdiode card you shared
    and send it over.

(2) Testchip BONUS — the small structure that would help us most
    If there's room on the floorplan, the highest-leverage structure
    for us would be one isolated 2T NS-RAM cell with body-tap (Vbs)
    accessible as a pad, plus on-die T-sensor. That's the structure
    that lets us discriminate between the remaining mechanism
    candidates (well-tap, STI leakage, GIDL, contact-R + selfheat)
    cleanly at the three pre-registered biases below. Width variants
    (W = 0.18 um and W = 0.72 um at L = 180 nm) on the same cell
    would let us split well-tap (proportional to WL) vs GIDL
    (proportional to W) by 2-4x.

    Adversarial bias triplet (one device, three biases, two T):
      a) VG1 = 0.0, VG2 = -0.2, Vd = 1.4, T = 400 K
      b) VG1 = 0.3, VG2 = -0.2, Vd = 1.4, T = 400 K
      c) VG1 = 0.0, VG2 = +0.6, Vd = 0.40, T = 220 K

    Predicted mechanism-to-mechanism divergence at these biases is
    >25 decades on the candidate currents, so the discrimination is
    unambiguous from a handful of points.

    We can send a one-page test-structure spec if useful, well in
    advance of tape-out so it can sit in the existing floorplan
    without disruption.

For Mario — proposal direction is still open on our side too

Picking up the question from 3 May (in case it got buried): are you
sizing the centre proposal for the 1-year ~200 k EUR budget we have,
or scaling deliverables to the 5-7 year horizon you mentioned in the
call? And do you prefer one joint ENIMBLE + Nervdynamics x KAUST
proposal, or two coordinated separate ones? Happy either way; just
want to draft the deliverables ladder against the right shape.

Attached
  - verdict_K1_VG1_06.md  — full K1 sensitivity analysis (n=66 fwd+bwd)
  - scatter_fwd_bwd.png   — symmetry diagnostic (no hysteresis, rho=0.95)

Tack och hej, looking forward to picking this back up.

Eric (via research agent, OK'd by Eric)
"""

def build_message():
    msg = EmailMessage()
    msg["From"] = f"Eric Bergvall <{USER}>"
    msg["To"] = ", ".join(TO_LIST)
    msg["Subject"] = SUBJECT
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="gmail.com")
    msg["Reply-To"] = USER
    msg["In-Reply-To"] = IN_REPLY_TO
    msg["References"] = REFERENCES
    msg.set_content(BODY)
    for p in ATTACHMENTS:
        if not p.exists():
            print(f"[error] attachment missing: {p}")
            sys.exit(1)
        data = p.read_bytes()
        if p.suffix == ".png":
            msg.add_attachment(data, maintype="image", subtype="png", filename=p.name)
        else:
            msg.add_attachment(data, maintype="text", subtype="markdown", filename=p.name)
    return msg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    msg = build_message()
    print(f"From:       {msg['From']}")
    print(f"To:         {msg['To']}")
    print(f"Subject:    {msg['Subject']}")
    print(f"In-Reply-To: {msg['In-Reply-To']}")
    print(f"Attachments: {', '.join(p.name for p in ATTACHMENTS)}")
    body_lines = BODY.splitlines()
    print(f"Body: {len(body_lines)} lines, {len(BODY)} chars")
    print("--- first 18 lines of body ---")
    for ln in body_lines[:18]:
        print(f"  {ln}")
    print("  ...")

    if args.dry_run:
        print("\n[DRY RUN] not sending.")
        return

    print("\n[live] connecting to smtp.gmail.com:465 ...")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(USER, PW)
        refused = s.send_message(msg, from_addr=USER, to_addrs=TO_LIST)
        print(f"[live] send_message refused: {refused}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "from": USER,
        "to": TO_LIST,
        "subject": SUBJECT,
        "message_id": msg["Message-ID"],
        "in_reply_to": IN_REPLY_TO,
        "attachments": [p.name for p in ATTACHMENTS],
    }
    log = []
    if LOG_PATH.exists():
        try: log = json.loads(LOG_PATH.read_text())
        except Exception: log = []
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"[done] message-id: {msg['Message-ID']}")
    print(f"[done] logged at {LOG_PATH}")

if __name__ == "__main__":
    main()
