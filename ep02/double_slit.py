"""Episode 2 «Інтерференція і дифракція»: brightness on a wall behind two slits.

Run from the repo root: uv run ep02/double_slit.py
Drag the sliders under the plot to change λ, d, a and L; the defaults are set below.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

WAVELENGTH = 650.0  # λ, nm (red)
SLIT_GAP = 0.25  # d, mm: distance between the slits
SLIT_WIDTH = 0.05  # a, mm: width of one slit
DISTANCE = 2.0  # L, m: slits → wall
Y_MAX = 75.0  # half-width of the wall shown, mm: fixed, so a farther screen visibly widens fringes
SAMPLES = 6000  # fine fringes (Δy ≈ 0.8 mm) must not alias

STYLE = Path(__file__).resolve().parent.parent / "course.mplstyle"
REST = "#3a4356"  # slider tracks
MEASURE = "#b9a8ff"  # envelope, as the measure colour in the videos


def wavelength_color(nm):
    """Rough visible-spectrum colour of a wavelength (Bruton's approximation), RGB in 0..1."""
    if nm < 440:
        rgb = ((440 - nm) / 60, 0.0, 1.0)
    elif nm < 490:
        rgb = (0.0, (nm - 440) / 50, 1.0)
    elif nm < 510:
        rgb = (0.0, 1.0, (510 - nm) / 20)
    elif nm < 580:
        rgb = ((nm - 510) / 70, 1.0, 0.0)
    elif nm < 645:
        rgb = (1.0, (645 - nm) / 65, 0.0)
    else:
        rgb = (1.0, 0.0, 0.0)
    if nm < 420:  # the eye fades only beyond the ends of 420–700 nm
        edge = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700:
        edge = 0.3 + 0.7 * (780 - nm) / 80
    else:
        edge = 1.0
    return np.array(rgb) * edge


def fringe_spacing_mm(wavelength_nm, gap_mm, distance_m):
    """Δy = λL/d between neighbouring bright fringes, in mm."""
    return wavelength_nm * 1e-9 * distance_m / (gap_mm * 1e-3) * 1e3


def spot_width_mm(wavelength_nm, width_mm, distance_m):
    """Width of the central bright spot, 2λL/a, in mm."""
    return 2 * wavelength_nm * 1e-9 * distance_m / (width_mm * 1e-3) * 1e3


def bright_fringes(gap_mm, width_mm):
    """Bright fringes inside the central spot: orders |m| < d/a (an order at m = d/a is missing)."""
    ratio = gap_mm / width_mm
    return 2 * (int(np.ceil(ratio - 1e-9)) - 1) + 1


def intensity(y_mm, wavelength_nm, gap_mm, width_mm, distance_m):
    """Return (two slits, envelope) brightness at height y on the wall, exact angle θ = atan(y/L).

    I = cos²(π d sinθ / λ) · sinc²(π a sinθ / λ), the central peak of two slits = 1.
    The envelope is the same sinc² factor; one slit alone gives only ¼ of the peak,
    so the envelope is the one-slit pattern times 4.
    """
    sin_theta = np.sin(np.arctan(y_mm * 1e-3 / distance_m))
    lam = wavelength_nm * 1e-9
    fringes = np.cos(np.pi * gap_mm * 1e-3 * sin_theta / lam) ** 2
    envelope = np.sinc(width_mm * 1e-3 * sin_theta / lam) ** 2  # np.sinc(u) = sin(πu)/(πu)
    return fringes * envelope, envelope


def print_examples():
    """The worked examples from the episode: Δy = λL/d, and λ from a measured Δy."""
    gap, width, dist = SLIT_GAP, SLIT_WIDTH, DISTANCE
    for name, nm in (("Червоне", 650.0), ("Зелене", 532.0)):
        dy = fringe_spacing_mm(nm, gap, dist)
        print(f"{name} світло {nm:.0f} нм: Δy = λL/d = {dy:.1f} мм".replace(".", ","))
    dy = fringe_spacing_mm(650.0, gap, dist)
    lam = dy * 1e-3 * gap * 1e-3 / dist * 1e9
    print(f"Навпаки: виміряно Δy = {dy:.1f} мм → λ = Δy·d/L = {lam:.0f} нм".replace(".", ","))
    spot = spot_width_mm(650.0, width, dist) / 10
    n = bright_fringes(gap, width)
    print(f"Центральна пляма 2λL/a ≈ {spot:.1f} см, світлих смуг у ній: {n}".replace(".", ","))


def draw_wall(wavelength_nm, brightness):
    """Wall picture: the colour of λ, brightness = I (one row, stretched by imshow)."""
    colour = wavelength_color(wavelength_nm)
    return (brightness[None, :, None] * colour[None, None, :]).clip(0, 1)


def show():
    plt.style.use(STYLE)
    y_max = Y_MAX
    y = np.linspace(-y_max, y_max, SAMPLES)
    fig = plt.figure(figsize=(9, 6.5))
    wall_ax = fig.add_axes([0.10, 0.74, 0.85, 0.12])
    ax = fig.add_axes([0.10, 0.36, 0.85, 0.32])
    wall_ax.set_xticks([])
    wall_ax.set_yticks([])
    wall_ax.set_title("Так виглядає екран")
    for side in wall_ax.spines.values():
        side.set_visible(False)

    wall = wall_ax.imshow(
        np.zeros((1, SAMPLES, 3)),
        extent=(-y_max, y_max, 0, 1),
        aspect="auto",
        interpolation="bilinear",
    )
    (curve,) = ax.plot(y, np.zeros_like(y))
    (envelope_line,) = ax.plot(
        y,
        np.zeros_like(y),
        color=MEASURE,
        lw=1.5,
        ls="--",
        label="одна щілина × 4 (одна дає ¼ піка)",
    )
    ax.set_xlim(-y_max, y_max)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("y, мм")
    ax.set_ylabel("яскравість")
    ax.legend(loc="upper right")
    label = fig.text(0.5, 0.94, "", ha="center", va="center")

    slider_style = {
        "color": "#5ad0ff",
        "track_color": REST,
        "handle_style": {"facecolor": "#e8ecf6"},
    }

    def make(row, name, lo, hi, init):
        return Slider(
            fig.add_axes([0.15, row, 0.62, 0.03]), name, lo, hi, valinit=init, **slider_style
        )

    lam = make(0.23, "λ, нм", 400, 700, WAVELENGTH)
    gap = make(0.17, "d, мм", 0.1, 0.5, SLIT_GAP)
    width = make(0.11, "a, мм", 0.03, 0.1, SLIT_WIDTH)
    dist = make(0.05, "L, м", 1.0, 3.0, DISTANCE)
    sliders = (lam, gap, width, dist)
    reset = Button(
        fig.add_axes([0.84, 0.05, 0.1, 0.06]), "Скинути", color="#222b42", hovercolor="#2a3552"
    )
    reset.on_clicked(lambda _: [s.reset() for s in sliders])

    def update(_value=None):
        y_max = Y_MAX
        y = np.linspace(-y_max, y_max, SAMPLES)
        colour = wavelength_color(lam.val)
        both, envelope = intensity(y, lam.val, gap.val, width.val, dist.val)
        wall.set_data(draw_wall(lam.val, both))
        wall.set_extent((-y_max, y_max, 0, 1))
        ax.set_xlim(-y_max, y_max)
        curve.set_data(y, both)
        curve.set_color(colour / colour.max())  # brightest channel = 1, hue kept
        envelope_line.set_data(y, envelope)
        dy = fringe_spacing_mm(lam.val, gap.val, dist.val)
        spot = spot_width_mm(lam.val, width.val, dist.val)
        label.set_text(
            f"Δy = λL/d = {dy:.1f} мм    пляма 2λL/a = {spot / 10:.1f} см    "
            f"світлих смуг у плямі: {bright_fringes(gap.val, width.val)}".replace(".", ",")
        )
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    update()
    plt.show()
    return sliders, reset


if __name__ == "__main__":
    print_examples()
    show()
