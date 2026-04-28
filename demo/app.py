"""CustomTkinter demo for the Image Classification ML Project.

Run from the project root:
    python -m demo.app
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time

import customtkinter as ctk
import numpy as np
from PIL import Image

# Make sure imports work whether launched as `python demo/app.py` or `-m demo.app`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demo import theme as t
from demo.components import (
    GradientFrame,
    Chip,
    StepCard,
    ModelCard,
    ProbabilityBars,
    ResultTile,
    StatPill,
)
from demo.data import get_test_images
from demo.registry import (
    PHASE_LABELS,
    FEATURE_LABELS,
    models_for,
)
from demo.trainer import train_or_load, predict_image


PHASE_INTERNAL = {v: k for k, v in PHASE_LABELS.items()}


class DemoApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("MNIST Classifier Showcase")
        self.geometry("1280x820")
        self.minsize(1100, 720)
        self.configure(fg_color=t.SURFACE_0)

        self._msg_q: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._test_images, self._test_labels = None, None
        self._current_index = 0
        self._predicting = False
        self._model_cards: dict[str, ModelCard] = {}
        self._selected_model_key: str | None = None
        self._feature_buttons: list[ctk.CTkRadioButton] = []
        self._gradient_frames: list[GradientFrame] = []

        self._build_layout()
        self._populate_models()
        self._load_initial_sample()
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_hero()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=t.SPACE_XL, pady=(0, t.SPACE_M))
        body.grid_columnconfigure(0, weight=0, minsize=320)
        body.grid_columnconfigure(1, weight=1, minsize=420)
        body.grid_columnconfigure(2, weight=0, minsize=360)
        body.grid_rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_center(body)
        self._build_right_panel(body)

        self._build_footer()

    # ------------------------------------------------------------------
    # Hero
    # ------------------------------------------------------------------

    def _build_hero(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="ew", padx=t.SPACE_XL, pady=(t.SPACE_XL, t.SPACE_L))
        outer.grid_columnconfigure(0, weight=1)

        hero = GradientFrame(
            outer,
            stops_light=t.GRADIENT_HERO[0],
            stops_dark=t.GRADIENT_HERO[1],
            corner_radius=t.RADIUS_L,
            height=180,
        )
        hero.grid(row=0, column=0, sticky="ew")
        hero.grid_propagate(False)
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)
        self._gradient_frames.append(hero)

        # Left: brand + title
        brand = ctk.CTkFrame(hero, fg_color="transparent")
        brand.grid(row=0, column=0, padx=t.SPACE_XXL, pady=(t.SPACE_XL, 0), sticky="nw")

        mark = ctk.CTkLabel(
            brand, text="◐",
            font=t.font(38, "bold"),
            text_color="#FFFFFF",
        )
        mark.grid(row=0, column=0, rowspan=2, padx=(0, t.SPACE_M))

        ctk.CTkLabel(
            brand, text="MNIST Classifier Showcase",
            font=t.display(),
            text_color="#FFFFFF",
        ).grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(
            brand,
            text="Compare every model from Phase 1 and Phase 2 on real MNIST samples.",
            font=t.body(),
            text_color="#E2E2F5",
        ).grid(row=1, column=1, sticky="w")

        pills = ctk.CTkFrame(hero, fg_color="transparent")
        pills.grid(row=1, column=0, padx=t.SPACE_XXL, pady=(t.SPACE_M, t.SPACE_XL), sticky="sw")
        StatPill(pills, icon="◆", label="Phase 1", value="5 models").grid(row=0, column=0, padx=(0, t.SPACE_S))
        StatPill(pills, icon="◇", label="Phase 2", value="4 models").grid(row=0, column=1, padx=(0, t.SPACE_S))
        StatPill(pills, icon="≣", label="Features", value="Flatten · PCA · HOG").grid(row=0, column=2)

        # Right: theme switch
        right = ctk.CTkFrame(hero, fg_color="transparent")
        right.grid(row=0, column=1, rowspan=2, padx=t.SPACE_XXL, pady=t.SPACE_XL, sticky="ne")

        self.theme_switch = ctk.CTkSwitch(
            right, text="Light mode",
            command=self._toggle_theme,
            text_color="#FFFFFF",
            progress_color="#FFFFFF",
            button_color="#FFFFFF",
            button_hover_color="#F1F5F9",
            fg_color="#473F8E",
        )
        self.theme_switch.grid(row=0, column=0, sticky="ne")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(self, parent) -> None:
        wrapper = ctk.CTkScrollableFrame(parent, fg_color="transparent", corner_radius=0)
        wrapper.grid(row=0, column=0, sticky="nsew", padx=(0, t.SPACE_M))
        wrapper.grid_columnconfigure(0, weight=1)

        # Step 1 — Phase
        phase_card = StepCard(wrapper, step="STEP 1", icon="◧", title="Choose phase")
        phase_card.grid(row=0, column=0, sticky="ew", pady=(0, t.SPACE_M))

        self.phase_var = ctk.StringVar(value=PHASE_LABELS["phase1"])
        self.phase_menu = ctk.CTkSegmentedButton(
            phase_card.body,
            values=[PHASE_LABELS["phase1"], PHASE_LABELS["phase2"]],
            variable=self.phase_var,
            command=lambda *_: self._on_phase_change(),
            selected_color=t.PRIMARY,
            selected_hover_color=t.PRIMARY_HOVER,
            unselected_color=t.SURFACE_2,
            unselected_hover_color=t.SURFACE_3,
            font=t.caption_bold(),
        )
        self.phase_menu.grid(row=0, column=0, sticky="ew")

        self.phase_hint = ctk.CTkLabel(
            phase_card.body,
            text="Binary task: predict whether the digit is a 0.",
            font=t.caption(),
            text_color=t.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=240,
        )
        self.phase_hint.grid(row=1, column=0, sticky="w", pady=(t.SPACE_S, 0))

        # Step 2 — Model
        model_card = StepCard(wrapper, step="STEP 2", icon="◈", title="Pick a model")
        model_card.grid(row=1, column=0, sticky="ew", pady=(0, t.SPACE_M))
        self.model_list = ctk.CTkFrame(model_card.body, fg_color="transparent")
        self.model_list.grid(row=0, column=0, sticky="ew")
        self.model_list.grid_columnconfigure(0, weight=1)

        # Step 3 — Feature
        feature_card = StepCard(wrapper, step="STEP 3", icon="≣", title="Feature extraction")
        feature_card.grid(row=2, column=0, sticky="ew", pady=(0, t.SPACE_M))
        self.feature_var = ctk.StringVar(value="pca")
        self.feature_frame = ctk.CTkFrame(feature_card.body, fg_color="transparent")
        self.feature_frame.grid(row=0, column=0, sticky="ew")
        self.feature_frame.grid_columnconfigure(0, weight=1)

        self.cache_chip = Chip(wrapper, text="Cache: empty",
                               fg=t.SURFACE_2, text_color=t.TEXT_SECONDARY)
        self.cache_chip.grid(row=3, column=0, sticky="w", pady=(0, t.SPACE_M))

    # ------------------------------------------------------------------
    # Center
    # ------------------------------------------------------------------

    def _build_center(self, parent) -> None:
        center = ctk.CTkFrame(parent, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=t.SPACE_M)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(0, weight=1)

        # Image stage
        stage = ctk.CTkFrame(
            center,
            corner_radius=t.RADIUS_L,
            fg_color=t.SURFACE_1,
            border_width=1,
            border_color=t.BORDER_SOFT,
        )
        stage.grid(row=0, column=0, sticky="nsew", pady=(0, t.SPACE_M))
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)

        stage_header = ctk.CTkFrame(stage, fg_color="transparent")
        stage_header.grid(row=0, column=0, sticky="ew", padx=t.SPACE_L, pady=(t.SPACE_L, 0))
        stage_header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            stage_header, text="◳", font=t.font(18),
            text_color=t.PRIMARY,
        ).grid(row=0, column=0, padx=(0, t.SPACE_S))

        ctk.CTkLabel(
            stage_header, text="Sample preview",
            font=t.heading(), text_color=t.TEXT_PRIMARY,
        ).grid(row=0, column=1, sticky="w")

        self.sample_chip = Chip(stage_header, text="#0 / 200",
                                fg=t.SURFACE_2, text_color=t.TEXT_SECONDARY)
        self.sample_chip.grid(row=0, column=2, sticky="e")

        # Image canvas card
        img_card = ctk.CTkFrame(
            stage,
            corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_2,
            border_width=1,
            border_color=t.BORDER_SOFT,
        )
        img_card.grid(row=1, column=0, sticky="nsew", padx=t.SPACE_L, pady=t.SPACE_M)
        img_card.grid_columnconfigure(0, weight=1)
        img_card.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(img_card, text="")
        self.image_label.grid(row=0, column=0, padx=t.SPACE_L, pady=t.SPACE_L, sticky="nsew")

        # Sample controls
        controls = ctk.CTkFrame(stage, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=t.SPACE_L, pady=(0, t.SPACE_L))
        controls.grid_columnconfigure(2, weight=1)

        self.prev_btn = ctk.CTkButton(
            controls, text="◀", width=44, height=36,
            corner_radius=t.RADIUS_M,
            font=t.body_bold(),
            fg_color=t.SURFACE_2, hover_color=t.SURFACE_3,
            text_color=t.TEXT_PRIMARY,
            command=lambda: self._step_index(-1),
        )
        self.prev_btn.grid(row=0, column=0, padx=(0, t.SPACE_XS))

        self.next_btn = ctk.CTkButton(
            controls, text="▶", width=44, height=36,
            corner_radius=t.RADIUS_M,
            font=t.body_bold(),
            fg_color=t.SURFACE_2, hover_color=t.SURFACE_3,
            text_color=t.TEXT_PRIMARY,
            command=lambda: self._step_index(1),
        )
        self.next_btn.grid(row=0, column=1, padx=(0, t.SPACE_M))

        self.index_entry = ctk.CTkEntry(
            controls, placeholder_text="Jump to index (0-199)",
            height=36, corner_radius=t.RADIUS_M,
            border_color=t.BORDER_SOFT, fg_color=t.SURFACE_2,
            font=t.body(),
        )
        self.index_entry.grid(row=0, column=2, sticky="ew")
        self.index_entry.bind("<Return>", lambda *_: self._goto_index())

        self.go_btn = ctk.CTkButton(
            controls, text="Go", width=70, height=36,
            corner_radius=t.RADIUS_M,
            font=t.body_bold(),
            fg_color=t.SURFACE_2, hover_color=t.SURFACE_3,
            text_color=t.TEXT_PRIMARY,
            command=self._goto_index,
        )
        self.go_btn.grid(row=0, column=3, padx=(t.SPACE_XS, t.SPACE_S))

        self.random_btn = ctk.CTkButton(
            controls, text="🎲  Surprise me", width=140, height=36,
            corner_radius=t.RADIUS_M,
            font=t.body_bold(),
            fg_color=t.PRIMARY, hover_color=t.PRIMARY_HOVER,
            text_color="#FFFFFF",
            command=self._pick_random,
        )
        self.random_btn.grid(row=0, column=4)

        # Result tiles row
        tiles_row = ctk.CTkFrame(center, fg_color="transparent")
        tiles_row.grid(row=1, column=0, sticky="ew")
        tiles_row.grid_columnconfigure(0, weight=1, uniform="tiles")
        tiles_row.grid_columnconfigure(1, weight=1, uniform="tiles")

        self.pred_tile = ResultTile(tiles_row, caption="PREDICTION")
        self.pred_tile.grid(row=0, column=0, sticky="nsew", padx=(0, t.SPACE_S))

        self.truth_tile = ResultTile(tiles_row, caption="GROUND TRUTH")
        self.truth_tile.grid(row=0, column=1, sticky="nsew", padx=(t.SPACE_S, 0))

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _build_right_panel(self, parent) -> None:
        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=2, sticky="nsew", padx=(t.SPACE_M, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        # Confidence card
        conf_card = ctk.CTkFrame(
            right, corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_1,
            border_width=1, border_color=t.BORDER_SOFT,
        )
        conf_card.grid(row=0, column=0, sticky="ew", pady=(0, t.SPACE_M))
        conf_card.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(conf_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=t.SPACE_L, pady=(t.SPACE_L, t.SPACE_S))
        head.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(head, text="◴", font=t.font(18),
                     text_color=t.PRIMARY).grid(row=0, column=0, padx=(0, t.SPACE_S))
        ctk.CTkLabel(head, text="Confidence", font=t.heading(),
                     text_color=t.TEXT_PRIMARY).grid(row=0, column=1, sticky="w")

        self.conf_status_chip = Chip(head, text="not run yet",
                                     fg=t.SURFACE_2, text_color=t.TEXT_TERTIARY)
        self.conf_status_chip.grid(row=0, column=2, sticky="e")

        self.prob_bars = ProbabilityBars(conf_card)
        self.prob_bars.grid(row=1, column=0, sticky="ew", padx=t.SPACE_L, pady=(0, t.SPACE_L))
        self.prob_placeholder = ctk.CTkLabel(
            conf_card,
            text="Run a prediction to see per-class scores.",
            font=t.caption(), text_color=t.TEXT_TERTIARY,
            anchor="w", justify="left", wraplength=300,
        )
        self.prob_placeholder.grid(row=2, column=0, sticky="w", padx=t.SPACE_L, pady=(0, t.SPACE_L))

        # Status card
        status_card = ctk.CTkFrame(
            right, corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_1,
            border_width=1, border_color=t.BORDER_SOFT,
        )
        status_card.grid(row=1, column=0, sticky="ew", pady=(0, t.SPACE_M))
        status_card.grid_columnconfigure(0, weight=1)

        head2 = ctk.CTkFrame(status_card, fg_color="transparent")
        head2.grid(row=0, column=0, sticky="ew", padx=t.SPACE_L, pady=(t.SPACE_L, t.SPACE_S))
        head2.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(head2, text="●", font=t.font(14),
                     text_color=t.SUCCESS).grid(row=0, column=0, padx=(0, t.SPACE_S))
        ctk.CTkLabel(head2, text="Status", font=t.heading(),
                     text_color=t.TEXT_PRIMARY).grid(row=0, column=1, sticky="w")

        self.status_chip = Chip(head2, text="Idle",
                                fg=t.SUCCESS_SOFT, text_color=t.SUCCESS)
        self.status_chip.grid(row=0, column=2, sticky="e")

        self.metric_label = ctk.CTkLabel(
            status_card, text="Pick a model and press Run prediction.",
            font=t.caption(), text_color=t.TEXT_SECONDARY,
            anchor="w", justify="left", wraplength=300,
        )
        self.metric_label.grid(row=1, column=0, sticky="w", padx=t.SPACE_L, pady=(0, t.SPACE_L))

        # Activity log card
        log_card = ctk.CTkFrame(
            right, corner_radius=t.RADIUS_M,
            fg_color=t.SURFACE_1,
            border_width=1, border_color=t.BORDER_SOFT,
        )
        log_card.grid(row=2, column=0, sticky="nsew")
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        log_head = ctk.CTkFrame(log_card, fg_color="transparent")
        log_head.grid(row=0, column=0, sticky="ew", padx=t.SPACE_L, pady=(t.SPACE_L, t.SPACE_S))
        ctk.CTkLabel(log_head, text="≡", font=t.font(18),
                     text_color=t.PRIMARY).grid(row=0, column=0, padx=(0, t.SPACE_S))
        ctk.CTkLabel(log_head, text="Activity log", font=t.heading(),
                     text_color=t.TEXT_PRIMARY).grid(row=0, column=1, sticky="w")

        self.log_box = ctk.CTkTextbox(
            log_card, wrap="word",
            font=t.mono(),
            fg_color=t.SURFACE_2,
            text_color=t.TEXT_SECONDARY,
            corner_radius=t.RADIUS_S,
            border_width=0,
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=t.SPACE_L, pady=(0, t.SPACE_L))
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Footer (CTA)
    # ------------------------------------------------------------------

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=t.SPACE_XL, pady=(0, t.SPACE_XL))
        footer.grid_columnconfigure(0, weight=1)

        cta_wrap = GradientFrame(
            footer,
            stops_light=t.GRADIENT_PRIMARY[0],
            stops_dark=t.GRADIENT_PRIMARY[1],
            corner_radius=t.RADIUS_L,
            height=72,
        )
        cta_wrap.grid(row=0, column=0, sticky="ew")
        cta_wrap.grid_propagate(False)
        cta_wrap.grid_columnconfigure(0, weight=1)
        cta_wrap.grid_columnconfigure(1, weight=0)
        self._gradient_frames.append(cta_wrap)

        self.cta_caption = ctk.CTkLabel(
            cta_wrap,
            text="Ready when you are — train and classify the selected sample.",
            font=t.body_bold(),
            text_color="#FFFFFF",
        )
        self.cta_caption.grid(row=0, column=0, sticky="w", padx=t.SPACE_XXL, pady=t.SPACE_L)

        self.predict_btn = ctk.CTkButton(
            cta_wrap, text="Run prediction  →",
            height=44, width=210,
            corner_radius=t.RADIUS_M,
            font=t.font(15, "bold"),
            fg_color="#FFFFFF",
            hover_color="#F1F5F9",
            text_color=t.PRIMARY,
            command=self._on_predict_click,
        )
        self.predict_btn.grid(row=0, column=1, sticky="e", padx=t.SPACE_XXL, pady=t.SPACE_M)

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _populate_models(self) -> None:
        for child in self.model_list.winfo_children():
            child.destroy()
        self._model_cards.clear()

        phase_key = PHASE_INTERNAL[self.phase_var.get()]
        specs = models_for(phase_key)
        for i, (key, spec) in enumerate(specs.items()):
            card = ModelCard(
                self.model_list,
                title=spec.label,
                description=spec.description,
                icon=spec.icon,
                on_select=lambda k=key: self._select_model(k),
            )
            card.grid(row=i, column=0, sticky="ew", pady=(0, t.SPACE_S))
            self._model_cards[key] = card

        first_key = next(iter(specs))
        self._select_model(first_key)

        if phase_key == "phase1":
            self.phase_hint.configure(text="Binary task: predict whether the digit is a 0.")
        else:
            self.phase_hint.configure(text="Multi-class task: predict which digit (0–9).")

    def _select_model(self, key: str) -> None:
        self._selected_model_key = key
        for k, card in self._model_cards.items():
            card.set_selected(k == key)
        phase_key = PHASE_INTERNAL[self.phase_var.get()]
        spec = models_for(phase_key)[key]
        self._render_features(spec.feature_methods, spec.default_feature)

    def _render_features(self, methods: list[str], default: str) -> None:
        for btn in self._feature_buttons:
            btn.destroy()
        self._feature_buttons.clear()
        self.feature_var.set(default)
        for i, m in enumerate(methods):
            btn = ctk.CTkRadioButton(
                self.feature_frame,
                text=FEATURE_LABELS.get(m, m),
                variable=self.feature_var,
                value=m,
                fg_color=t.PRIMARY,
                hover_color=t.PRIMARY_HOVER,
                border_color=t.BORDER_STRONG,
                font=t.body(),
                text_color=t.TEXT_PRIMARY,
            )
            btn.grid(row=i, column=0, padx=0, pady=(0 if i == 0 else 4, 0), sticky="w")
            self._feature_buttons.append(btn)

    # ------------------------------------------------------------------
    # Phase / sample browsing
    # ------------------------------------------------------------------

    def _on_phase_change(self) -> None:
        self._populate_models()
        self._update_truth_display()
        self.pred_tile.reset()
        self.prob_bars.clear()
        self.prob_placeholder.grid()
        self.conf_status_chip.configure(text="not run yet",
                                        fg_color=t.SURFACE_2,
                                        text_color=t.TEXT_TERTIARY)

    def _load_initial_sample(self) -> None:
        try:
            self._test_images, self._test_labels = get_test_images(n=200)
        except Exception as e:
            self._log(f"Failed to load MNIST: {e}")
            self._set_status("MNIST load error", danger=True)
            return
        self._show_index(0)

    def _show_index(self, idx: int) -> None:
        if self._test_images is None:
            return
        idx = max(0, min(idx, len(self._test_images) - 1))
        self._current_index = idx
        img = (self._test_images[idx] * 255).astype(np.uint8)
        pil = Image.fromarray(img, mode="L").resize((300, 300), Image.NEAREST)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(300, 300))
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img
        self.sample_chip.configure(text=f"Sample #{idx} / {len(self._test_images) - 1}")
        self._update_truth_display()
        self.pred_tile.reset()
        self.prob_bars.clear()
        self.prob_placeholder.grid()
        self.conf_status_chip.configure(text="not run yet",
                                        fg_color=t.SURFACE_2,
                                        text_color=t.TEXT_TERTIARY)

    def _update_truth_display(self) -> None:
        if self._test_labels is None:
            return
        true_digit = int(self._test_labels[self._current_index])
        phase_key = PHASE_INTERNAL[self.phase_var.get()]
        if phase_key == "phase1":
            text = "0" if true_digit == 0 else "Not 0"
            sub = f"actual digit: {true_digit}"
        else:
            text = str(true_digit)
            sub = f"digit class {true_digit}"
        self.truth_tile.set_value(text, sub=sub)

    def _step_index(self, delta: int) -> None:
        self._show_index(self._current_index + delta)

    def _goto_index(self) -> None:
        raw = self.index_entry.get().strip()
        if not raw:
            return
        try:
            self._show_index(int(raw))
        except ValueError:
            self._log(f"Invalid index: {raw!r}")

    def _pick_random(self) -> None:
        if self._test_images is None:
            return
        idx = int(np.random.randint(0, len(self._test_images)))
        self._show_index(idx)

    # ------------------------------------------------------------------
    # Prediction flow
    # ------------------------------------------------------------------

    def _on_predict_click(self) -> None:
        if self._predicting:
            return
        if self._test_images is None or self._selected_model_key is None:
            return

        phase_key = PHASE_INTERNAL[self.phase_var.get()]
        feature = self.feature_var.get()
        idx = self._current_index
        image = self._test_images[idx].copy()
        true_label = int(self._test_labels[idx])
        model_key = self._selected_model_key

        self._predicting = True
        self.predict_btn.configure(state="disabled", text="Working…")
        self._set_status("Working…", warn=True)
        self.pred_tile.set_value("…", color=t.TEXT_TERTIARY)

        thread = threading.Thread(
            target=self._predict_worker,
            args=(phase_key, model_key, feature, image, true_label),
            daemon=True,
        )
        thread.start()

    def _predict_worker(self, phase_key, model_key, feature, image, true_label) -> None:
        def log(msg):
            self._msg_q.put(("log", msg))

        try:
            t0 = time.time()
            model, transform, info = train_or_load(phase_key, model_key, feature, log)
            pred, score = predict_image(model, transform, image)
            elapsed = time.time() - t0
            self._msg_q.put((
                "result",
                {
                    "phase": phase_key,
                    "pred": pred,
                    "score": score,
                    "true": true_label,
                    "info": info,
                    "elapsed": elapsed,
                },
            ))
        except Exception as e:
            self._msg_q.put(("error", str(e)))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._msg_q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "result":
                    self._handle_result(payload)
                elif kind == "error":
                    self._handle_error(payload)
        except queue.Empty:
            pass
        finally:
            self.after(80, self._drain_queue)

    def _handle_result(self, payload: dict) -> None:
        self._predicting = False
        self.predict_btn.configure(state="normal", text="Run prediction  →")

        pred, true_label = payload["pred"], payload["true"]
        phase = payload["phase"]
        info = payload["info"]
        score = payload["score"]
        elapsed = payload["elapsed"]

        if phase == "phase1":
            pred_text = "0" if pred == 0 else "Not 0"
            true_text = "0" if true_label == 0 else "Not 0"
            correct = pred_text == true_text
            class_labels = ["0", "Not 0"]
            highlight_idx = int(pred)
        else:
            pred_text = str(pred)
            true_text = str(true_label)
            correct = pred == true_label
            class_labels = [str(i) for i in range(10)]
            highlight_idx = int(pred)

        color = t.SUCCESS if correct else t.DANGER
        self.pred_tile.set_value(
            pred_text,
            color=color,
            sub=("✓ matches truth" if correct else "✗ differs from truth"),
        )

        self._set_status("Correct" if correct else "Wrong",
                         danger=not correct, success=correct)

        if score is not None and len(score) == len(class_labels):
            self.prob_bars.show(class_labels, list(score), highlight_idx)
            self.prob_placeholder.grid_remove()
            top = float(np.max(score)) * 100
            self.conf_status_chip.configure(
                text=f"top {top:.1f}%",
                fg_color=t.SUCCESS_SOFT if correct else t.DANGER_SOFT,
                text_color=t.SUCCESS if correct else t.DANGER,
            )
        else:
            self.prob_bars.clear()
            self.prob_placeholder.grid()
            self.conf_status_chip.configure(
                text="no scores",
                fg_color=t.SURFACE_2,
                text_color=t.TEXT_TERTIARY,
            )

        self.cache_chip.configure(
            text=f"Cache: {info['model']} · {info['feature']}",
            fg_color=t.PRIMARY_SOFT, text_color=t.PRIMARY,
        )

        self.metric_label.configure(text=(
            f"Model: {info['model']}\n"
            f"Phase: {PHASE_LABELS[info['phase']]}\n"
            f"Features: {FEATURE_LABELS.get(info['feature'], info['feature'])} ({info['feature_dim']} dims)\n"
            f"Train samples: {info['train_samples']:,}\n"
            f"Train (one-off): {info['train_seconds']}s   ·   Total: {elapsed:.2f}s"
        ))

    def _handle_error(self, msg: str) -> None:
        self._predicting = False
        self.predict_btn.configure(state="normal", text="Run prediction  →")
        self._set_status("Error", danger=True)
        self._log(f"ERROR: {msg}")
        self.pred_tile.set_value("—", color=t.DANGER, sub="prediction failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str, *, success: bool = False, warn: bool = False, danger: bool = False) -> None:
        if danger:
            fg, fc = t.DANGER_SOFT, t.DANGER
        elif warn:
            fg, fc = ("#FEF3C7", "#3A2E14"), t.WARNING
        elif success:
            fg, fc = t.SUCCESS_SOFT, t.SUCCESS
        else:
            fg, fc = t.SURFACE_2, t.TEXT_SECONDARY
        self.status_chip.configure(text=text, fg_color=fg, text_color=fc)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}]  {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _toggle_theme(self) -> None:
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("light")
        else:
            ctk.set_appearance_mode("dark")
        for gf in self._gradient_frames:
            gf.refresh()


def main() -> None:
    app = DemoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
