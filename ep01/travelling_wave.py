"""Episode 1 «Що таке хвиля»: a travelling wave y = A·sin(kx − ωt) and v = λ·f numbers.

Run from the repo root: uv run ep01/travelling_wave.py
Drag the sliders under the plot to change A, λ and f; the defaults are set below.
"""

from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

A = 1.0  # amplitude
WAVELENGTH = 2.0  # λ, metres
FREQUENCY = 0.5  # f, hertz
RIBBON_X = 3.0  # the marked point that only bobs up and down
FPS = 30

STYLE = Path(__file__).resolve().parent.parent / "course.mplstyle"
RIBBON = "#ff7a59"  # the tracked point, as in the video
REST = "#3a4356"  # rest line and slider tracks
MEASURE = "#b9a8ff"  # λ and A markers

SOUND_SPEED = 343.0  # m/s, air at ~20 °C
LIGHT_SPEED = 2.998e8  # m/s


def print_examples():
    """The two worked examples from the episode: λ = v / f and f = v / λ."""
    note = 440.0
    print(f"Нота «ля» {note:.0f} Гц у повітрі: λ = v / f = {SOUND_SPEED / note:.2f} м")
    green = 550e-9
    print(f"Зелене світло {green * 1e9:.0f} нм: f = v / λ = {LIGHT_SPEED / green:.2e} Гц")


def wave(x, amplitude, wavelength, time_angle):
    """y = A·sin(kx − ωt), with ωt passed in as an accumulated angle."""
    k = 2 * np.pi / wavelength  # distance → angle
    return amplitude * np.sin(k * x - time_angle)


def animate():
    plt.style.use(STYLE)
    x = np.linspace(0, 10, 600)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.subplots_adjust(bottom=0.36)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y")
    ax.axhline(0, color=REST, lw=1)
    a_lines = [ax.axhline(s * A, color=MEASURE, lw=1, ls="--") for s in (1, -1)]
    (line,) = ax.plot(x, wave(x, A, WAVELENGTH, 0))
    (ribbon,) = ax.plot([RIBBON_X], [wave(RIBBON_X, A, WAVELENGTH, 0)], "o", color=RIBBON, ms=11)
    label = ax.set_title("")

    slider_style = {
        "color": "#5ad0ff",
        "track_color": REST,
        "handle_style": {"facecolor": "#e8ecf6"},
    }
    amp = Slider(fig.add_axes([0.15, 0.20, 0.62, 0.04]), "A", 0.1, 2.0, valinit=A, **slider_style)
    lam = Slider(
        fig.add_axes([0.15, 0.13, 0.62, 0.04]), "λ, м", 0.5, 5.0, valinit=WAVELENGTH, **slider_style
    )
    freq = Slider(
        fig.add_axes([0.15, 0.06, 0.62, 0.04]), "f, Гц", 0.1, 2.0, valinit=FREQUENCY, **slider_style
    )
    reset = Button(
        fig.add_axes([0.84, 0.06, 0.1, 0.06]), "Скинути", color="#222b42", hovercolor="#2a3552"
    )
    reset.on_clicked(lambda _: [s.reset() for s in (amp, lam, freq)])

    # ωt is accumulated frame by frame, so moving the f slider changes the speed
    # without making the wave jump.
    state = {"angle": 0.0, "t": 0.0}

    def update(_frame):
        state["t"] += 1 / FPS
        state["angle"] += 2 * np.pi * freq.val / FPS
        line.set_ydata(wave(x, amp.val, lam.val, state["angle"]))
        for sign, a_line in zip((1, -1), a_lines):
            a_line.set_ydata([sign * amp.val] * 2)
        ribbon.set_ydata([wave(RIBBON_X, amp.val, lam.val, state["angle"])])
        speed = lam.val * freq.val
        label.set_text(
            f"t = {state['t']:5.1f} с    λ = {lam.val:.2f} м,  f = {freq.val:.2f} Гц,  "
            f"v = λf = {speed:.2f} м/с"
        )
        return line, ribbon, label, *a_lines

    anim = FuncAnimation(fig, update, frames=count(), interval=1000 / FPS, cache_frame_data=False)
    plt.show()
    return anim


if __name__ == "__main__":
    print_examples()
    animate()
