"""NS-RAM Explainer Film — 6 scenes, manim CE 0.20.

Render: HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/manim -qh scripts/video/nsram_film.py Scene1Faucet Scene2Reservoir ...
"""
from manim import *
import numpy as np

# Palette
BG = "#0d1117"
C_CURRENT = "#58a6ff"   # blue
C_VOLT    = "#f0883e"   # orange
C_MEM     = "#3fb950"   # green
C_FIRE    = "#f85149"   # red
C_TEXT    = "#e6edf3"   # off-white
C_DIM     = "#8b949e"

config.background_color = BG
config.frame_rate = 30
config.pixel_width = 1280
config.pixel_height = 720

CAP_KW = dict(font="sans-serif", color=C_TEXT)


def make_title(s, size=36):
    t = Text(s, font_size=size, **CAP_KW)
    t.to_edge(UP, buff=0.4)
    return t


def make_caption(s, size=26):
    t = Text(s, font_size=size, **CAP_KW)
    t.to_edge(DOWN, buff=0.5)
    return t


def transistor_glyph(scale=1.0):
    """Simple FET symbol: gate (top), drain (right), source (left), body channel."""
    g = VGroup()
    # Substrate channel
    chan = Rectangle(width=3.0, height=0.4, color=C_DIM, fill_opacity=0.4)
    g.add(chan)
    # Source and drain pads
    src = Rectangle(width=0.6, height=0.6, color=C_CURRENT, fill_opacity=0.8).next_to(chan, LEFT, buff=0)
    drn = Rectangle(width=0.6, height=0.6, color=C_CURRENT, fill_opacity=0.8).next_to(chan, RIGHT, buff=0)
    # Gate
    gate = Rectangle(width=1.4, height=0.3, color=C_VOLT, fill_opacity=0.9).next_to(chan, UP, buff=0.15)
    # Gate wire
    gw = Line(gate.get_top(), gate.get_top() + UP*0.6, color=C_VOLT, stroke_width=4)
    # Labels
    sL = Text("S", font_size=22, color=C_TEXT).move_to(src.get_center())
    dL = Text("D", font_size=22, color=C_TEXT).move_to(drn.get_center())
    gL = Text("G", font_size=22, color=C_TEXT).next_to(gw, UP, buff=0.05)
    g.add(src, drn, gate, gw, sL, dL, gL)
    g.scale(scale)
    return g


# ─────────────────── Scene 1 ─────────────────── #
class Scene1Faucet(Scene):
    def construct(self):
        title = make_title("1. A regular transistor")
        self.play(Write(title))

        fet = transistor_glyph(scale=1.3).shift(DOWN*0.3)
        self.play(FadeIn(fet, shift=UP*0.3))

        # Voltage label going on gate
        vlabel = MathTex("V_G", color=C_VOLT, font_size=42).next_to(fet[4], UP, buff=1.0)
        self.play(Write(vlabel))

        # Pulse on gate -> arrow going down
        v_arrow = Arrow(vlabel.get_bottom(), fet[4].get_top()+UP*0.05, color=C_VOLT, buff=0.05, stroke_width=6)
        self.play(GrowArrow(v_arrow))

        # Gate "opens" — flash
        self.play(fet[2].animate.set_fill(C_VOLT, opacity=1.0), run_time=0.4)

        # Current flows S -> D
        current_dots = VGroup(*[
            Dot(radius=0.08, color=C_CURRENT).move_to(fet[0].get_center()+LEFT*1.0)
            for _ in range(5)
        ])
        for i, d in enumerate(current_dots):
            d.shift(LEFT*0.3*i)

        ilabel = MathTex("I_{DS}", color=C_CURRENT, font_size=36).next_to(fet, RIGHT, buff=1.0)
        self.play(FadeIn(current_dots), Write(ilabel))

        # Animate flow left -> right
        self.play(
            *[d.animate.shift(RIGHT*4.5) for d in current_dots],
            run_time=2.0, rate_func=linear
        )
        self.play(FadeOut(current_dots))

        # Turn gate off
        self.play(fet[2].animate.set_fill(C_VOLT, opacity=0.3))

        cap = make_caption("Voltage in. Current out. No memory.")
        self.play(Write(cap))
        self.wait(2.0)
        self.play(FadeOut(*self.mobjects))


# ─────────────────── Scene 2 ─────────────────── #
class Scene2Reservoir(Scene):
    def construct(self):
        title = make_title("2. What if it remembered?")
        self.play(Write(title))

        fet = transistor_glyph(scale=1.3).shift(DOWN*0.3)
        self.play(FadeIn(fet))

        # Add reservoir under channel: the body
        body = Rectangle(width=2.4, height=0.7, color=C_MEM, fill_opacity=0.15, stroke_width=3).next_to(fet[0], DOWN, buff=0.05).shift(RIGHT*0.7)
        body_label = Text("body", font_size=20, color=C_MEM).next_to(body, DOWN, buff=0.1)
        self.play(Create(body), Write(body_label))

        # Charge level overlay
        fill = Rectangle(width=2.4, height=0.05, color=C_MEM, fill_opacity=0.85, stroke_width=0).align_to(body, DOWN).shift(UP*0.025).align_to(body, LEFT)

        self.add(fill)

        # Pulse gate
        for cycle in range(2):
            self.play(fet[2].animate.set_fill(C_VOLT, opacity=1.0), run_time=0.3)
            new_h = 0.15 + cycle*0.25
            new_fill = Rectangle(width=2.4, height=new_h, color=C_MEM, fill_opacity=0.85, stroke_width=0)
            new_fill.align_to(body, DOWN).shift(UP*new_h/2).align_to(body, LEFT)
            self.play(Transform(fill, new_fill), run_time=0.8)
            self.play(fet[2].animate.set_fill(C_VOLT, opacity=0.3), run_time=0.3)

        # Avalanche: body fills, gate snaps open
        final_fill = Rectangle(width=2.4, height=0.65, color=C_FIRE, fill_opacity=0.9, stroke_width=0)
        final_fill.align_to(body, DOWN).shift(UP*0.325).align_to(body, LEFT)
        self.play(Transform(fill, final_fill), run_time=0.6)

        flash = Flash(body.get_center(), color=C_FIRE, line_length=0.6, num_lines=18, run_time=0.5)
        self.play(flash)

        # Snapback label
        snap = Text("SNAPBACK!", font_size=34, color=C_FIRE, weight=BOLD).next_to(body, RIGHT, buff=0.8)
        self.play(Write(snap))

        # Big current surge
        surge = VGroup(*[Dot(radius=0.1, color=C_FIRE).move_to(fet[0].get_center()+LEFT*0.3*i) for i in range(8)])
        self.play(FadeIn(surge), surge.animate.shift(RIGHT*5.0), run_time=1.5, rate_func=linear)
        self.play(FadeOut(surge))

        cap = make_caption("NS-RAM remembers how much current flowed in its body.")
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))


# ─────────────────── Scene 3 ─────────────────── #
class Scene3TwoTcell(Scene):
    def construct(self):
        title = make_title("3. The 2T cell")
        self.play(Write(title))

        # Two transistors stacked, sharing a floating body
        m1 = transistor_glyph(scale=0.9).shift(LEFT*3.0+DOWN*0.3)
        m2 = transistor_glyph(scale=0.9).shift(RIGHT*3.0+DOWN*0.3)
        # rename gate labels
        # m1 labels: index 6 = gL ("G")
        m1[6].become(Text("G1", font_size=22, color=C_TEXT).move_to(m1[6]))
        m2[6].become(Text("G2", font_size=22, color=C_TEXT).move_to(m2[6]))

        self.play(FadeIn(m1), FadeIn(m2))

        # Floating body — green island connecting them
        island = Ellipse(width=3.2, height=0.8, color=C_MEM, fill_opacity=0.25, stroke_width=3).move_to(DOWN*0.9)
        nwell = Ellipse(width=4.8, height=1.6, color=C_DIM, fill_opacity=0.08, stroke_width=2).move_to(DOWN*0.9)
        nwell_label = Text("deep N-well", font_size=18, color=C_DIM).next_to(nwell, DOWN, buff=0.05)
        body_label = Text("floating body", font_size=20, color=C_MEM).move_to(island.get_center())
        self.play(Create(nwell), Write(nwell_label))
        self.play(Create(island), Write(body_label))

        # Wires from M1 source/drain to island, M2 source/drain to island
        w1 = Line(m1[0].get_bottom(), island.get_left()+RIGHT*0.2, color=C_DIM, stroke_width=2)
        w2 = Line(m2[1].get_bottom(), island.get_right()+LEFT*0.2, color=C_DIM, stroke_width=2)
        self.play(Create(w1), Create(w2))

        # M1 fires a pulse -> body fills
        v1 = MathTex("V_{G1}\\;\\text{pulse}", color=C_VOLT, font_size=30).next_to(m1, UP, buff=0.8)
        self.play(Write(v1))
        self.play(m1[2].animate.set_fill(C_VOLT, opacity=1.0), run_time=0.3)
        # body charging
        charge_glow = island.copy().set_fill(C_MEM, opacity=0.7)
        self.play(Transform(island, charge_glow), run_time=1.0)
        self.play(m1[2].animate.set_fill(C_VOLT, opacity=0.3), run_time=0.3)

        # M2 reads -> fires spike
        v2 = MathTex("V_{G2}\\;\\text{read}", color=C_VOLT, font_size=30).next_to(m2, UP, buff=0.8)
        self.play(Write(v2))
        self.play(m2[2].animate.set_fill(C_VOLT, opacity=1.0), run_time=0.3)

        # Output spike
        spike_origin = m2[1].get_right()
        spike = Arrow(spike_origin, spike_origin+RIGHT*1.5, color=C_FIRE, stroke_width=8, buff=0)
        spike_label = Text("SPIKE!", font_size=28, color=C_FIRE, weight=BOLD).next_to(spike, RIGHT, buff=0.1)
        self.play(GrowArrow(spike), Write(spike_label))
        self.play(Flash(spike_origin, color=C_FIRE, run_time=0.4))

        cap = make_caption("M1 fills the memory. M2 fires when memory is full. A brain cell on silicon.")
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))


# ─────────────────── Scene 4 ─────────────────── #
class Scene4Modeling(Scene):
    def construct(self):
        title = make_title("4. Modeling silicon in software")
        self.play(Write(title))

        # Axes: I-V curve
        axes = Axes(
            x_range=[0, 5, 1], y_range=[-9, -3, 1],
            x_length=6, y_length=3.5,
            tips=False,
            axis_config={"color": C_DIM, "stroke_width": 2}
        ).shift(LEFT*2.5+DOWN*0.2)
        xlbl = Text("V_DS (V)", font_size=20, color=C_DIM).next_to(axes.x_axis, DOWN, buff=0.2)
        ylbl = Text("log I_D", font_size=20, color=C_DIM).next_to(axes.y_axis, LEFT, buff=0.2).rotate(PI/2)
        self.play(Create(axes), Write(xlbl), Write(ylbl))

        # Measured curve (with snapback fold)
        def measured(x):
            base = -8 + 1.4 * np.tanh((x-1.0)*1.2)
            # add snapback near x=3
            if x > 2.8:
                fold = -1.5*np.exp(-(x-3.2)**2/0.15)
                return base - fold + 0.7*(x>3.2)
            return base

        meas_pts = [axes.c2p(x, measured(x)) for x in np.linspace(0.05, 4.9, 80)]
        meas_curve = VMobject(color=C_CURRENT, stroke_width=4).set_points_smoothly(meas_pts)
        meas_label = Text("measured (Sebas chip)", font_size=18, color=C_CURRENT).move_to(axes.c2p(3.5, -4)).shift(UP*0.5)
        self.play(Create(meas_curve), Write(meas_label))

        # Model curve attempts
        # v1 — bad
        def model_v1(x): return -7 + 1.0*np.tanh((x-1.3)*0.9)
        m1_pts = [axes.c2p(x, model_v1(x)) for x in np.linspace(0.05, 4.9, 60)]
        m1_curve = VMobject(color=C_FIRE, stroke_width=3).set_points_smoothly(m1_pts)

        err_box = Rectangle(width=4.2, height=2.6, color=C_TEXT, stroke_width=2).shift(RIGHT*3.8+UP*0.3)
        err_title = Text("log-RMSE (dec)", font_size=22, color=C_TEXT).move_to(err_box.get_top()+DOWN*0.3)

        err_val = DecimalNumber(2.5, num_decimal_places=2, color=C_FIRE, font_size=72).move_to(err_box.get_center())
        err_status = Text("v1: way off", font_size=20, color=C_FIRE).next_to(err_val, DOWN, buff=0.2)

        self.play(Create(m1_curve), Create(err_box), Write(err_title), Write(err_val), Write(err_status))
        self.wait(1.2)

        # v2 — better
        def model_v2(x): return -8.2 + 1.3*np.tanh((x-1.1)*1.1)
        m2_pts = [axes.c2p(x, model_v2(x)) for x in np.linspace(0.05, 4.9, 60)]
        m2_curve = VMobject(color="#d29922", stroke_width=3).set_points_smoothly(m2_pts)
        self.play(Transform(m1_curve, m2_curve), err_val.animate.set_value(1.4).set_color("#d29922"),
                  Transform(err_status, Text("v2: closer", font_size=20, color="#d29922").next_to(err_val, DOWN, buff=0.2)))
        self.wait(1.0)

        # v3 — best DC
        def model_v3(x): return -8.0 + 1.38*np.tanh((x-1.0)*1.15)
        m3_pts = [axes.c2p(x, model_v3(x)) for x in np.linspace(0.05, 4.9, 60)]
        m3_curve = VMobject(color=C_MEM, stroke_width=3).set_points_smoothly(m3_pts)
        self.play(Transform(m1_curve, m3_curve), err_val.animate.set_value(0.99).set_color(C_MEM),
                  Transform(err_status, Text("v3: 0.99 dec DC fit", font_size=20, color=C_MEM).next_to(err_val, DOWN, buff=0.2)))
        self.wait(0.8)

        # Highlight the snapback gap
        gap_circle = Circle(radius=0.6, color=C_FIRE, stroke_width=4).move_to(axes.c2p(3.2, -5.5))
        gap_label = Text("snapback gap!", font_size=22, color=C_FIRE).next_to(gap_circle, DOWN, buff=0.05)
        self.play(Create(gap_circle), Write(gap_label))

        cap = make_caption("DC: 0.99 dec. Snapback shape: missing physics. Need new traps.")
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))


# ─────────────────── Scene 5 ─────────────────── #
class Scene5Networks(Scene):
    def construct(self):
        title = make_title("5. Networks of neurons")
        self.play(Write(title))

        # Grid of cells, zooming out
        N = 12
        cells = VGroup()
        for i in range(N):
            for j in range(N):
                d = Dot(radius=0.07, color=C_MEM).shift(LEFT*3.5+UP*1.0+RIGHT*0.4*j+DOWN*0.25*i)
                cells.add(d)
        self.play(LaggedStartMap(FadeIn, cells, lag_ratio=0.005, run_time=1.5))

        scale_text = Text("1024 → 4096 → 16,384 cells", font_size=24, color=C_TEXT).next_to(cells, DOWN, buff=0.3)
        self.play(Write(scale_text))

        # Activity icons (right)
        activities = ["walk", "sit", "stand", "lay", "up", "down"]
        icons = VGroup()
        for k, a in enumerate(activities):
            box = Rectangle(width=1.4, height=0.5, color=C_CURRENT, fill_opacity=0.15)
            t = Text(a, font_size=18, color=C_TEXT).move_to(box.get_center())
            grp = VGroup(box, t).shift(RIGHT*3.6+UP*1.4+DOWN*0.6*k)
            icons.add(grp)
        self.play(LaggedStartMap(FadeIn, icons, lag_ratio=0.1))

        # Accuracy meter
        acc_label = Text("Accuracy", font_size=22, color=C_TEXT).shift(LEFT*4.5+DOWN*1.8)
        acc_val = DecimalNumber(80.23, num_decimal_places=2, unit="\\%", color=C_CURRENT, font_size=44).next_to(acc_label, RIGHT, buff=0.3)

        # Energy
        eng_label = Text("Energy / inference", font_size=22, color=C_TEXT).shift(RIGHT*0.6+DOWN*1.8)
        eng_val = DecimalNumber(2.3, num_decimal_places=1, unit=" \\,nJ", color=C_VOLT, font_size=44).next_to(eng_label, RIGHT, buff=0.3)

        self.play(Write(acc_label), Write(acc_val), Write(eng_label), Write(eng_val))

        self.play(acc_val.animate.set_value(83.91), eng_val.animate.set_value(12.0), run_time=1.5)
        self.play(acc_val.animate.set_value(84.09), eng_val.animate.set_value(35.0), run_time=1.5)

        # Flash some cells to show activity
        sample = np.random.choice(len(cells), 30, replace=False)
        flashes = [cells[k].animate.set_color(C_FIRE).scale(1.6) for k in sample]
        self.play(*flashes, run_time=0.4)
        self.play(*[cells[k].animate.set_color(C_MEM).scale(1/1.6) for k in sample], run_time=0.4)

        cap = make_caption("16,384 cells. 84% accuracy. 35 nJ per guess. UCI-HAR.")
        self.play(Write(cap))
        self.wait(2.5)
        self.play(FadeOut(*self.mobjects))


# ─────────────────── Scene 6 ─────────────────── #
class Scene6Status(Scene):
    def construct(self):
        title = make_title("6. Where we are now")
        self.play(Write(title))

        # Three columns: WIN / HOLD / NEXT
        col_w = 3.7
        col_h = 4.2

        win_box = Rectangle(width=col_w, height=col_h, color=C_MEM, fill_opacity=0.08, stroke_width=3).shift(LEFT*4.3+DOWN*0.3)
        hold_box = Rectangle(width=col_w, height=col_h, color="#d29922", fill_opacity=0.08, stroke_width=3).shift(DOWN*0.3)
        next_box = Rectangle(width=col_w, height=col_h, color=C_CURRENT, fill_opacity=0.08, stroke_width=3).shift(RIGHT*4.3+DOWN*0.3)

        win_t = Text("WIN", font_size=30, color=C_MEM, weight=BOLD).move_to(win_box.get_top()+DOWN*0.3)
        hold_t = Text("HOLD", font_size=30, color="#d29922", weight=BOLD).move_to(hold_box.get_top()+DOWN*0.3)
        next_t = Text("NEXT", font_size=30, color=C_CURRENT, weight=BOLD).move_to(next_box.get_top()+DOWN*0.3)

        self.play(Create(win_box), Create(hold_box), Create(next_box),
                  Write(win_t), Write(hold_t), Write(next_t))

        win_items = [
            "HDC 84.09%",
            "Bayesian RNG",
            "NIST 5/5 PASS",
            "35 nJ/inf",
        ]
        hold_items = [
            "Snapback gap",
            "KWS at chance",
            "Model 10x off",
            "VG2 unclear",
        ]
        next_items = [
            "Add traps",
            "VNwell diode",
            "Talk to Sebas",
            "Rebuild topology",
        ]

        def fill_col(box, items, color):
            grp = VGroup()
            for i, it in enumerate(items):
                t = Text("- " + it, font_size=22, color=C_TEXT).move_to(box.get_top() + DOWN*(0.9 + 0.55*i)).align_to(box.get_left()+RIGHT*0.3, LEFT)
                grp.add(t)
            return grp

        wg = fill_col(win_box, win_items, C_MEM)
        hg = fill_col(hold_box, hold_items, "#d29922")
        ng = fill_col(next_box, next_items, C_CURRENT)

        self.play(LaggedStartMap(FadeIn, wg, lag_ratio=0.15), run_time=1.4)
        self.play(LaggedStartMap(FadeIn, hg, lag_ratio=0.15), run_time=1.4)
        self.play(LaggedStartMap(FadeIn, ng, lag_ratio=0.15), run_time=1.4)

        cap = make_caption("Real progress. Real gaps. We know what's missing.")
        self.play(Write(cap))
        self.wait(3.0)
        self.play(FadeOut(*self.mobjects))
