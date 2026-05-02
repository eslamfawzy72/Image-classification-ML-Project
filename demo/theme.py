"""Design tokens for the demo UI.

Two palettes (dark / light), a typographic scale, and a spacing scale.
Each token is a (light, dark) tuple so CTk auto-switches with appearance.
"""

from __future__ import annotations

import customtkinter as ctk


# ---------------------------------------------------------------------------
# Color tokens — each value is (light_mode, dark_mode)
# ---------------------------------------------------------------------------

# Surface tiers (lowest -> highest elevation)
SURFACE_0 = ("#EEF1F6", "#0E1117")     # app background
SURFACE_1 = ("#FFFFFF", "#161A22")     # primary cards
SURFACE_2 = ("#F7F8FB", "#1D222C")     # nested cards / inputs
SURFACE_3 = ("#E8ECF3", "#262C38")     # hover / pressed

BORDER_SOFT = ("#E1E5EE", "#262C38")
BORDER_STRONG = ("#CDD3DF", "#323845")

# Text
TEXT_PRIMARY = ("#0F172A", "#E6E9EF")
TEXT_SECONDARY = ("#475569", "#A8B0BD")
TEXT_TERTIARY = ("#94A3B8", "#6B7280")
TEXT_INVERTED = ("#FFFFFF", "#0F172A")

# Brand / accent
PRIMARY = ("#5B5BF5", "#7C7CFF")             # indigo
PRIMARY_HOVER = ("#4747D8", "#6363EE")
PRIMARY_SOFT = ("#EEF0FF", "#21243A")
ACCENT_PINK = ("#EC4899", "#F472B6")
ACCENT_TEAL = ("#06B6D4", "#22D3EE")

# Semantic
SUCCESS = ("#10B981", "#34D399")
SUCCESS_SOFT = ("#DCFCE7", "#0F2A1F")
DANGER = ("#EF4444", "#F87171")
DANGER_SOFT = ("#FEE2E2", "#3A1414")
WARNING = ("#F59E0B", "#FBBF24")
INFO = ("#3B82F6", "#60A5FA")

# Gradient stops (light, dark) used by hero/CTA
GRADIENT_PRIMARY = (("#6366F1", "#8B5CF6", "#EC4899"), ("#7C7CFF", "#A78BFA", "#F472B6"))
GRADIENT_HERO = (("#312E81", "#4338CA", "#7C3AED"), ("#1E1B4B", "#312E81", "#5B21B6"))


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_MONO = "Consolas"


def font(size: int, weight: str = "normal", *, family: str | None = None) -> ctk.CTkFont:
    return ctk.CTkFont(family=family or FONT_FAMILY, size=size, weight=weight)


def display() -> ctk.CTkFont:    return font(28, "bold")
def title() -> ctk.CTkFont:      return font(20, "bold")
def heading() -> ctk.CTkFont:    return font(15, "bold")
def body() -> ctk.CTkFont:       return font(13)
def body_bold() -> ctk.CTkFont:  return font(13, "bold")
def caption() -> ctk.CTkFont:    return font(11)
def caption_bold() -> ctk.CTkFont: return font(11, "bold")
def mono() -> ctk.CTkFont:       return font(11, family=FONT_FAMILY_MONO)
def numeral() -> ctk.CTkFont:    return font(54, "bold")


# ---------------------------------------------------------------------------
# Spacing scale (used as padx/pady throughout)
# ---------------------------------------------------------------------------

SPACE_XS = 4
SPACE_S = 8
SPACE_M = 12
SPACE_L = 18
SPACE_XL = 24
SPACE_XXL = 32

RADIUS_S = 8
RADIUS_M = 14
RADIUS_L = 20
RADIUS_PILL = 999


def is_dark() -> bool:
    return ctk.get_appearance_mode().lower() == "dark"


def pick(token: tuple[str, str]) -> str:
    """Resolve a (light, dark) token to the active mode."""
    return token[1] if is_dark() else token[0]
