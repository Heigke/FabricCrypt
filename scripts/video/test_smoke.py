from manim import *

class Smoke(Scene):
    def construct(self):
        title = Text("NS-RAM").scale(2)
        self.play(Write(title), run_time=1)
        self.wait(0.5)
