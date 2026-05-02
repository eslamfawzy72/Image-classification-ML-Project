"""Custom visual widgets built on top of customtkinter.

CustomTkinter doesn't natively render gradients, so we generate them with
Pillow and embed them as CTkImages. The components here re-render their
imagery on resize and on light/dark mode change.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import customtkinter as ctk
from PIL import Image, ImageDraw

from demo import theme as t


# ---------------------------------------------------------------------------
# Gradient banner
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _make_horizontal_gradient(width: int, height: int, stops: Sequence[str], radius: int = 0) -> Image.Image:
    width = max(width, 2)
    height = max(height, 2)
    img = Image.new("RGB", (width, height), stops[0])
    px = img.load()
    n = len(stops) - 1
    for x in range(width):
        # Map x -> position in stops list
        t_pos = (x / (width - 1)) * n
        i = min(int(t_pos), n - 1)
        f = t_pos - i
        r1, g1, b1 = _hex_to_rgb(stops[i])
        r2, g2, b2 = _hex_to_rgb(stops[i + 1])
        r = int(r1 + (r2 - r1) * f)
        g = int(g1 + (g2 - g1) * f)
        b = int(b1 + (b2 - b1) * f)
        for y in range(height):
            px[x, y] = (r, g, b)
    if radius > 0:
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
        out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        return out
    return img


class GradientFrame(ctk.CTkFrame):
    """A frame whose background is a multi-stop horizontal gradient."""

    def __init__(self, master, stops_light: Sequence[str], stops_dark: Sequence[str],
                 corner_radius: int = 0, **kwargs):
        super().__init__(master, corner_radius=0, fg_color="transparent", **kwargs)
        self._stops_light = stops_light
        self._stops_dark = stops_dark
        self._radius = corner_radius
        self._bg_label = ctk.CTkLabel(self, text="", anchor="center")
        self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self._bg_label.lower()
        self.bind("<Configure>", self._on_resize)
        self._last_size = (0, 0)

    def _on_resize(self, event=None) -> None:
        w = self.winfo_width()
        h = self.winfo_height()
        if (w, h) == self._last_size or w < 4 or h < 4:
            return
        self._last_size = (w, h)
        stops = self._stops_dark if t.is_dark() else self._stops_light
        img = _make_horizontal_gradient(w, h, stops, self._radius)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self._bg_label.configure(image=ctk_img)
        self._bg_label.image = ctk_img

    def refresh(self) -> None:
        self._last_size = (0, 0)
        self._on_resize()


# ---------------------------------------------------------------------------
# Pill / Chip
# ---------------------------------------------------------------------------

class Chip(ctk.CTkLabel):
    """A small rounded label used as a tag / pill."""

    def __init__(self, master, text: str, *, fg=t.PRIMARY_SOFT, text_color=t.PRIMARY,
                 padx_inner: int = 10, pady_inner: int = 4, **kwargs):
        super().__init__(
            master,
            text=text,
            font=t.caption_bold(),
            corner_radius=t.RADIUS_PILL,
            fg_color=fg,
            text_color=text_color,
            padx=padx_inner, pady=pady_inner,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Section / Step card
# ---------------------------------------------------------------------------

class StepCard(ctk.CTkFrame):
    """A bordered card with a numbered header (icon + step number + title)."""

    def __init__(self, master, *, step: str, icon: str, title: str, **kwargs):
        super().__init__(
            master,
            corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_1,
            border_width=1,
            border_color=t.BORDER_SOFT,
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=t.SPACE_L, pady=(t.SPACE_L, t.SPACE_S), sticky="ew")
        header.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            header, text=icon, font=t.font(20),
            text_color=t.PRIMARY,
        ).grid(row=0, column=0, padx=(0, t.SPACE_S))

        ctk.CTkLabel(
            header, text=step, font=t.caption_bold(),
            fg_color=t.PRIMARY_SOFT, text_color=t.PRIMARY,
            corner_radius=t.RADIUS_PILL, padx=8, pady=2,
        ).grid(row=0, column=1, padx=(0, t.SPACE_S))

        ctk.CTkLabel(
            header, text=title,
            font=t.heading(),
            text_color=t.TEXT_PRIMARY, anchor="w",
        ).grid(row=0, column=2, sticky="w")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, padx=t.SPACE_L, pady=(0, t.SPACE_L), sticky="nsew")
        self.body.grid_columnconfigure(0, weight=1)


# ---------------------------------------------------------------------------
# Model card (vertical list of clickable cards)
# ---------------------------------------------------------------------------

class ModelCard(ctk.CTkFrame):
    """A clickable card with model name + description. Supports a 'selected' state."""

    def __init__(self, master, *, title: str, description: str, icon: str,
                 on_select: Callable[[], None], **kwargs):
        super().__init__(
            master,
            corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_2,
            border_width=1,
            border_color=t.BORDER_SOFT,
            cursor="hand2",
            **kwargs,
        )
        self._title = title
        self._on_select = on_select
        self._selected = False

        self.grid_columnconfigure(1, weight=1)

        self.icon_lbl = ctk.CTkLabel(
            self, text=icon, font=t.font(18),
            text_color=t.TEXT_SECONDARY, width=32,
        )
        self.icon_lbl.grid(row=0, column=0, rowspan=2, padx=(t.SPACE_M, t.SPACE_S),
                           pady=t.SPACE_M, sticky="n")

        self.title_lbl = ctk.CTkLabel(
            self, text=title,
            font=t.body_bold(), text_color=t.TEXT_PRIMARY, anchor="w",
        )
        self.title_lbl.grid(row=0, column=1, padx=(0, t.SPACE_M),
                            pady=(t.SPACE_M, 0), sticky="ew")

        self.desc_lbl = ctk.CTkLabel(
            self, text=description, font=t.caption(),
            text_color=t.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=220,
        )
        self.desc_lbl.grid(row=1, column=1, padx=(0, t.SPACE_M),
                           pady=(2, t.SPACE_M), sticky="ew")

        for w in (self, self.icon_lbl, self.title_lbl, self.desc_lbl):
            w.bind("<Button-1>", lambda *_: self._on_select())
            w.bind("<Enter>", lambda *_: self._hover(True))
            w.bind("<Leave>", lambda *_: self._hover(False))

    def _hover(self, on: bool) -> None:
        if self._selected:
            return
        self.configure(fg_color=t.SURFACE_3 if on else t.SURFACE_2,
                       border_color=t.BORDER_STRONG if on else t.BORDER_SOFT)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.configure(fg_color=t.PRIMARY_SOFT, border_color=t.PRIMARY)
            self.icon_lbl.configure(text_color=t.PRIMARY)
            self.title_lbl.configure(text_color=t.PRIMARY)
        else:
            self.configure(fg_color=t.SURFACE_2, border_color=t.BORDER_SOFT)
            self.icon_lbl.configure(text_color=t.TEXT_SECONDARY)
            self.title_lbl.configure(text_color=t.TEXT_PRIMARY)


# ---------------------------------------------------------------------------
# Probability bars (canvas)
# ---------------------------------------------------------------------------

class ProbabilityBars(ctk.CTkFrame):
    """Horizontal bar chart showing class probabilities."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._rows: list[tuple[ctk.CTkLabel, ctk.CTkProgressBar, ctk.CTkLabel]] = []

    def show(self, labels: Iterable[str], scores: Iterable[float], highlight_idx: int | None) -> None:
        for lbl, bar, val in self._rows:
            lbl.destroy(); bar.destroy(); val.destroy()
        self._rows.clear()

        labels = list(labels)
        scores = list(scores)
        max_score = max(scores) if scores else 1.0
        for i, (label, s) in enumerate(zip(labels, scores)):
            color = t.PRIMARY if i == highlight_idx else t.BORDER_STRONG
            text_color = t.PRIMARY if i == highlight_idx else t.TEXT_SECONDARY

            lbl = ctk.CTkLabel(self, text=label, font=t.caption_bold(),
                               text_color=text_color, width=46, anchor="w")
            lbl.grid(row=i, column=0, padx=(0, t.SPACE_S), pady=2, sticky="w")

            bar = ctk.CTkProgressBar(self, height=10, corner_radius=t.RADIUS_PILL,
                                     progress_color=color, fg_color=t.SURFACE_3)
            bar.grid(row=i, column=1, sticky="ew", pady=2)
            bar.set(float(s) / (max_score if max_score > 0 else 1.0))

            val = ctk.CTkLabel(self, text=f"{s * 100:5.1f}%",
                               font=t.caption(), text_color=text_color, width=56, anchor="e")
            val.grid(row=i, column=2, padx=(t.SPACE_S, 0), pady=2, sticky="e")

            self._rows.append((lbl, bar, val))

        self.grid_columnconfigure(1, weight=1)

    def clear(self) -> None:
        for lbl, bar, val in self._rows:
            lbl.destroy(); bar.destroy(); val.destroy()
        self._rows.clear()


# ---------------------------------------------------------------------------
# Result tile (large prediction display with colored ring)
# ---------------------------------------------------------------------------

class ResultTile(ctk.CTkFrame):
    """Centered numeric display with a small caption above and bottom badge."""

    def __init__(self, master, *, caption: str, **kwargs):
        super().__init__(
            master,
            corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_2,
            border_width=1,
            border_color=t.BORDER_SOFT,
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)

        self.caption = ctk.CTkLabel(
            self, text=caption, font=t.caption_bold(),
            text_color=t.TEXT_TERTIARY,
        )
        self.caption.grid(row=0, column=0, pady=(t.SPACE_M, 0))

        self.value = ctk.CTkLabel(
            self, text="—", font=t.numeral(),
            text_color=t.TEXT_PRIMARY,
        )
        self.value.grid(row=1, column=0, padx=t.SPACE_L, pady=(0, t.SPACE_S))

        self.sub = ctk.CTkLabel(
            self, text="", font=t.caption(),
            text_color=t.TEXT_SECONDARY,
        )
        self.sub.grid(row=2, column=0, pady=(0, t.SPACE_M))

    def set_value(self, value: str, *, color=None, sub: str = "") -> None:
        self.value.configure(text=value, text_color=color or t.TEXT_PRIMARY)
        self.sub.configure(text=sub)

    def reset(self) -> None:
        self.value.configure(text="—", text_color=t.TEXT_PRIMARY)
        self.sub.configure(text="")


# ---------------------------------------------------------------------------
# Stat pill (used in hero)
# ---------------------------------------------------------------------------

class StatPill(ctk.CTkFrame):
    def __init__(self, master, *, icon: str, label: str, value: str, **kwargs):
        super().__init__(
            master,
            corner_radius=t.RADIUS_PILL,
            fg_color=("#5C57C7", "#3A356B"),
            **kwargs,
        )
        ctk.CTkLabel(self, text=icon, font=t.font(13),
                     text_color="#FFFFFF").grid(row=0, column=0, padx=(t.SPACE_M, 4), pady=6)
        ctk.CTkLabel(self, text=label + ":", font=t.caption(),
                     text_color="#E2E2F5").grid(row=0, column=1, padx=(0, 4), pady=6)
        ctk.CTkLabel(self, text=value, font=t.caption_bold(),
                     text_color="#FFFFFF").grid(row=0, column=2, padx=(0, t.SPACE_M), pady=6)
