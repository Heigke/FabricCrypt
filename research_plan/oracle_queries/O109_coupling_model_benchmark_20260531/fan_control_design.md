# Task C — closed-loop fan PWM control

Task: at each control step, output fan PWM duty (0..255). Reward:
`-(T - T_target)² - λ * (PWM)²`. Per-chassis thermal RC, ambient, paste
condition differ → ikaros-trained controller learns *ikaros's transfer
function*.

Pre-reg: ikaros-trained policy achieves RMS(T − T_target) at least 20%
lower than (a) constant-PWM baseline, (b) PID with default gains,
(c) daedalus-trained transplant.

Implementation:
- Try `/sys/class/hwmon/*/pwm1`. If non-writable, fall back to
  *simulated thermal RC* parameterized by recorded ikaros vs daedalus
  step-response (so the difference between chassis is preserved).
- Action latency in the loop (read T → act → next read 1 s later).
