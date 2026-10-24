import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
from pathlib import Path
import random

from core import (
    KERNELS, BAYER_MATRICES, PALETTES,
    diffuse_dither, ordered_dither,
)
from quantization import quantize_palette, quantize_gray
from glitches import (
    glitch_channel_shift, glitch_chromatic, glitch_row_displace,
    glitch_block_displace, glitch_scanlines, glitch_noise, glitch_mirror,
    glitch_vhs, glitch_pixel_sort, glitch_color_swap,
    glitch_pixel_stretch,
)
from effects import (
    bloom, epsilon_glow, halftone, jpeg_glitch,
    crt_warp, phosphor_glow, halation, vignette,
    datamosh, blur_sharpen,
)
from palette_io import load_palette_file, save_palette_file
from video_source import VideoSource, HAS_CV2

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tutorial import create_tutorial_widget, show_tutorial_window
    HAS_TUTORIAL = True
except ImportError:
    HAS_TUTORIAL = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

HINT = "#8a8a8a"
ENTRY_BG = "#343638"
ACCENT = "#1f538d"


def labeled_frame(parent, text):
    outer = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color="#3a3a3a")
    ctk.CTkLabel(
        outer, text=text, anchor="w",
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(fill="x", padx=12, pady=(8, 2))
    inner = ctk.CTkFrame(outer, fg_color="transparent")
    inner.pack(fill="x", padx=12, pady=(0, 10))
    return outer, inner


class DitherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PixelDither")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 720)

        self.source_image = None
        self.result_image = None
        self.preview_photo = None
        self.show_original = False
        self._job = None

        self.video_source = None
        self.video_path = None

        self.custom_palette = list(PALETTES["1-bit"])
        self.selected_pal_index = 0

        self.glitch_seed = 42

        self.mode_var = tk.StringVar(value="bw")
        self.algo_var = tk.StringVar(value="Floyd-Steinberg")
        self.pixel_var = tk.DoubleVar(value=160.0)
        self.levels_var = tk.IntVar(value=4)
        self.adaptive_colors_var = tk.IntVar(value=16)
        self.dither_strength_var = tk.DoubleVar(value=1.0)
        self.serpentine_var = tk.BooleanVar(value=True)
        self.palette_dither_var = tk.BooleanVar(value=True)

        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.gamma_var = tk.DoubleVar(value=1.0)

        self.output_scale_var = tk.IntVar(value=1)

        self.ch_r_var = tk.IntVar(value=0)
        self.ch_g_var = tk.IntVar(value=0)
        self.ch_b_var = tk.IntVar(value=0)
        self.chroma_var = tk.DoubleVar(value=0.0)
        self.row_shift_var = tk.IntVar(value=0)
        self.block_count_var = tk.IntVar(value=0)
        self.block_size_var = tk.IntVar(value=8)
        self.scanline_spacing_var = tk.IntVar(value=0)
        self.scanline_darkness_var = tk.DoubleVar(value=0.3)
        self.noise_var = tk.DoubleVar(value=0.0)
        self.mirror_var = tk.DoubleVar(value=0.0)
        self.vhs_var = tk.DoubleVar(value=0.0)
        self.sort_enabled_var = tk.BooleanVar(value=False)
        self.sort_threshold_var = tk.IntVar(value=128)
        self.sort_dir_var = tk.StringVar(value="horizontal")
        self.swap_rg_var = tk.BooleanVar(value=False)
        self.swap_rb_var = tk.BooleanVar(value=False)
        self.swap_gb_var = tk.BooleanVar(value=False)

        self.video_frame_var = tk.DoubleVar(value=0.0)
        self.per_frame_glitch_var = tk.BooleanVar(value=True)

        self.bloom_enabled = tk.BooleanVar(value=False)
        self.bloom_threshold = tk.DoubleVar(value=0.7)
        self.bloom_radius = tk.DoubleVar(value=8.0)
        self.bloom_intensity = tk.DoubleVar(value=0.5)
        self.bloom_tint = (255, 240, 200)

        self.epsilon_enabled = tk.BooleanVar(value=False)
        self.epsilon_radius = tk.DoubleVar(value=6.0)
        self.epsilon_intensity = tk.DoubleVar(value=0.7)

        self.halftone_enabled = tk.BooleanVar(value=False)
        self.halftone_size = tk.IntVar(value=4)
        self.halftone_angle = tk.IntVar(value=45)
        self.halftone_strength = tk.DoubleVar(value=0.8)

        self.jpeg_enabled = tk.BooleanVar(value=False)
        self.jpeg_quality = tk.IntVar(value=15)
        self.jpeg_strength = tk.DoubleVar(value=0.8)

        self.crt_enabled = tk.BooleanVar(value=False)
        self.crt_curvature = tk.DoubleVar(value=0.15)
        self.phosphor_enabled = tk.BooleanVar(value=False)
        self.phosphor_intensity = tk.DoubleVar(value=0.3)

        self.halation_enabled = tk.BooleanVar(value=False)
        self.halation_intensity = tk.DoubleVar(value=0.4)
        self.vignette_enabled = tk.BooleanVar(value=False)
        self.vignette_strength = tk.DoubleVar(value=0.4)

        self.datamosh_enabled = tk.BooleanVar(value=False)
        self.datamosh_intensity = tk.DoubleVar(value=0.3)
        self.blur_enabled = tk.BooleanVar(value=False)
        self.blur_radius = tk.DoubleVar(value=0.0)
        self.sharpen_amount = tk.DoubleVar(value=0.0)

        self.stretch_enabled = tk.BooleanVar(value=False)
        self.stretch_probability = tk.DoubleVar(value=0.1)
        self.stretch_max = tk.IntVar(value=20)

        self._build_menu()
        self._build_ui()
        self._on_mode_change()

    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Открыть…", accelerator="Ctrl+O", command=self.open_image)
        f.add_command(label="Сохранить…", accelerator="Ctrl+S", command=self.save_image)
        f.add_separator()
        f.add_command(label="Выход", accelerator="Ctrl+Q", command=self.root.quit)
        m.add_cascade(label="Файл", menu=f)

        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Туториал", command=self._show_tutorial)
        m.add_cascade(label="Справка", menu=h)

        self.root.config(menu=m)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    def _build_ui(self):
        top = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        top.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkButton(top, text="Открыть", width=100, command=self.open_image).pack(side="left", padx=2)
        ctk.CTkButton(top, text="Сохранить", width=100, command=self.save_image).pack(side="left", padx=2)

        ctk.CTkLabel(top, text="Масштаб ×").pack(side="left", padx=(12, 4))
        self._make_spinbox(top, self.output_scale_var, 1, 8, width=50).pack(side="left")

        self.lbl_file = ctk.CTkLabel(top, text="Нет изображения", text_color=HINT)
        self.lbl_file.pack(side="left", padx=12)

        btn_cmp = ctk.CTkButton(top, text="Сравнить (удерж.)", width=140)
        btn_cmp.pack(side="right")
        btn_cmp.bind("<ButtonPress-1>", lambda e: self._set_compare(True))
        btn_cmp.bind("<ButtonRelease-1>", lambda e: self._set_compare(False))

        split = tk.PanedWindow(self.root, orient="horizontal", bg="#1a1a1a",
                               sashwidth=6, bd=0, sashrelief="flat")
        split.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        left = ctk.CTkFrame(split, corner_radius=0)
        split.add(left, minsize=300, width=440)

        self.tabview = ctk.CTkTabview(left)
        self.tabview.pack(fill="both", expand=True, padx=2, pady=2)

        tab_mode = self.tabview.add("Режим")
        tab_pal  = self.tabview.add("Палитра")
        tab_tune = self.tabview.add("Настройка")
        tab_glitch = self.tabview.add("Глитч")
        tab_post = self.tabview.add("Post FX")
        tab_video = self.tabview.add("Видео")
        tab_tutor = self.tabview.add("Туториал")

        self._build_mode_tab(tab_mode)
        self._build_palette_tab(tab_pal)
        self._build_tune_tab(tab_tune)
        self._build_glitch_tab(tab_glitch)
        self._build_post_tab(tab_post)
        self._build_video_tab(tab_video)
        self._build_tutorial_tab(tab_tutor)

        right = ctk.CTkFrame(split, corner_radius=0, fg_color="#1a1a1a")
        split.add(right, minsize=400)

        self.preview_canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", lambda e: self._redraw_preview())
        self.preview_canvas.bind("<Button-1>", lambda e: self._set_compare(True))
        self.preview_canvas.bind("<ButtonRelease-1>", lambda e: self._set_compare(False))

        self.status = ctk.CTkLabel(self.root, text="Готово", text_color=HINT, anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 6))

    def _build_mode_tab(self, parent):
        inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True)

        mf, mf_in = labeled_frame(inner, "Режим")
        mf.pack(fill="x", padx=6, pady=6)
        for text, val in [
            ("Ч/Б", "bw"),
            ("Оттенки серого (N уровней)", "gray"),
            ("Палитра (своя)", "palette"),
            ("Адаптивный (N цветов)", "adaptive"),
        ]:
            ctk.CTkRadioButton(mf_in, text=text, variable=self.mode_var,
                               value=val, command=self._on_mode_change).pack(
                                   anchor="w", pady=2)

        af, af_in = labeled_frame(inner, "Алгоритм дизеринга (Ч/Б)")
        af.pack(fill="x", padx=6, pady=6)
        algo_names = list(KERNELS.keys()) + list(BAYER_MATRICES.keys())
        self.algo_combo = ctk.CTkComboBox(
            af_in, variable=self.algo_var, values=algo_names,
            state="readonly", command=lambda _v: self.schedule_update(),
        )
        self.algo_combo.pack(fill="x")

        self.strength_scale = self._slider(af_in, "Сила", self.dither_strength_var, 0.0, 2.0)

        self.serp_check = ctk.CTkCheckBox(
            af_in, text="Змеевидный проход",
            variable=self.serpentine_var, command=self.schedule_update,
        )
        self.serp_check.pack(anchor="w", pady=(6, 0))

        gf, gf_in = labeled_frame(inner, "Уровни серого")
        gf.pack(fill="x", padx=6, pady=6)
        self.levels_spin = self._make_spinbox(
            gf_in, self.levels_var, 2, 64, command=self.schedule_update)
        self.levels_spin.pack(fill="x")

        adf, adf_in = labeled_frame(inner, "Адаптивные цвета")
        adf.pack(fill="x", padx=6, pady=6)
        self.adaptive_spin = self._make_spinbox(
            adf_in, self.adaptive_colors_var, 2, 256, command=self.schedule_update)
        self.adaptive_spin.pack(fill="x")

        pf, pf_in = labeled_frame(inner, "Параметры палитры")
        pf.pack(fill="x", padx=6, pady=6)
        self.pal_dither_check = ctk.CTkCheckBox(
            pf_in, text="Дизеринг в палитру",
            variable=self.palette_dither_var, command=self.schedule_update,
        )
        self.pal_dither_check.pack(anchor="w")

    def _on_mode_change(self):
        mode = self.mode_var.get()
        gr = "normal" if mode == "gray" else "disabled"
        ad = "normal" if mode == "adaptive" else "disabled"

        self.algo_combo.configure(state="readonly" if mode == "bw" else "disabled")
        self.strength_scale.configure(state="normal" if mode == "bw" else "disabled")
        self.serp_check.configure(state="normal" if mode == "bw" else "disabled")
        self.levels_spin.config(state=gr)
        self.adaptive_spin.config(state=ad)
        self.pal_dither_check.configure(state="normal" if mode == "palette" else "disabled")
        self.schedule_update()

    def _build_palette_tab(self, parent):
        pf, pf_in = labeled_frame(parent, "Пресеты")
        pf.pack(fill="x", padx=6, pady=6)
        self.preset_var = tk.StringVar(value="1-bit")
        row = ctk.CTkFrame(pf_in, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkComboBox(row, variable=self.preset_var,
                        values=list(PALETTES.keys()), state="readonly",
                        width=200).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(row, text="Загрузить", width=100,
                      command=self._load_preset).pack(side="right")

        sf, sf_in = labeled_frame(parent, "Цвета")
        sf.pack(fill="both", expand=False, padx=6, pady=6)
        self.pal_canvas = tk.Canvas(sf_in, height=180, bg="#222", highlightthickness=0)
        sb = ctk.CTkScrollbar(sf_in, orientation="vertical",
                              command=self.pal_canvas.yview)
        self.pal_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.pal_canvas.pack(side="left", fill="both", expand=True)
        self.pal_canvas.bind("<Button-1>", self._on_palette_click)
        self.pal_canvas.bind("<Configure>", lambda e: self._draw_palette())

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", padx=6, pady=4)
        for text, cmd in [
            ("+ Добавить", self._pal_add),
            ("Изменить", self._pal_edit),
            ("− Удалить", self._pal_remove),
            ("Очистить", self._pal_clear),
        ]:
            ctk.CTkButton(bf, text=text, command=cmd).pack(
                side="left", expand=True, fill="x", padx=2)

        ff = ctk.CTkFrame(parent, fg_color="transparent")
        ff.pack(fill="x", padx=6, pady=4)
        ctk.CTkButton(ff, text="Загрузить файл…",
                      command=self._pal_load_file).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(ff, text="Сохранить файл…",
                      command=self._pal_save_file).pack(side="left", expand=True, fill="x", padx=2)

        ctk.CTkLabel(parent, text="Форматы: .gpl · .pal · .hex · .txt",
                     text_color=HINT).pack(fill="x", padx=10, pady=(0, 6))

        self._draw_palette()

    def _load_preset(self, *_):
        name = self.preset_var.get()
        if name in PALETTES:
            self.custom_palette = list(PALETTES[name])
            self.selected_pal_index = 0
            self._draw_palette()
            self.schedule_update()

    def _draw_palette(self):
        c = self.pal_canvas
        c.delete("all")
        colors = self.custom_palette
        sw, gap = 24, 4
        if not colors:
            c.config(scrollregion=(0, 0, 300, 40))
            c.create_text(150, 20, text="Нет цветов — добавьте!", fill="#888")
            return
        w = max(c.winfo_width(), 260)
        cols = max(1, (w - gap) // (sw + gap))
        for i, (r, g, b) in enumerate(colors):
            row, col = divmod(i, cols)
            x = gap + col * (sw + gap)
            y = gap + row * (sw + gap)
            hexc = f"#{r:02x}{g:02x}{b:02x}"
            outline = "#ff3b30" if i == self.selected_pal_index else "#111"
            width = 2 if i == self.selected_pal_index else 1
            c.create_rectangle(x, y, x + sw, y + sw, fill=hexc,
                               outline=outline, width=width, tags=("sw", str(i)))
        rows = (len(colors) + cols - 1) // cols
        c.config(scrollregion=(0, 0, w, gap + rows * (sw + gap)))

    def _on_palette_click(self, e):
        x = self.pal_canvas.canvasx(e.x)
        y = self.pal_canvas.canvasy(e.y)
        for item in self.pal_canvas.find_overlapping(x, y, x, y):
            for tag in self.pal_canvas.gettags(item):
                if tag.isdigit():
                    self.selected_pal_index = int(tag)
                    self._draw_palette()
                    return

    def _pal_add(self):
        res = colorchooser.askcolor(title="Добавить цвет")
        if res and res[0]:
            self.custom_palette.append(tuple(int(c) for c in res[0]))
            self.selected_pal_index = len(self.custom_palette) - 1
            self._draw_palette()
            self.schedule_update()

    def _pal_edit(self):
        if not (0 <= self.selected_pal_index < len(self.custom_palette)):
            return
        cur = self.custom_palette[self.selected_pal_index]
        res = colorchooser.askcolor(color=cur, title="Изменить цвет")
        if res and res[0]:
            self.custom_palette[self.selected_pal_index] = tuple(int(c) for c in res[0])
            self._draw_palette()
            self.schedule_update()

    def _pal_remove(self):
        if not (0 <= self.selected_pal_index < len(self.custom_palette)):
            return
        self.custom_palette.pop(self.selected_pal_index)
        self.selected_pal_index = max(0, self.selected_pal_index - 1)
        self._draw_palette()
        self.schedule_update()

    def _pal_clear(self):
        self.custom_palette = []
        self.selected_pal_index = -1
        self._draw_palette()
        self.schedule_update()

    def _pal_load_file(self):
        path = filedialog.askopenfilename(
            title="Загрузить палитру",
            filetypes=[("Палитра", "*.gpl *.pal *.hex *.txt"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            colors = load_palette_file(path)
            if not colors:
                messagebox.showwarning("Пусто", "В файле не найдено цветов.")
                return
            self.custom_palette = colors
            self.selected_pal_index = 0
            self._draw_palette()
            self.status.configure(text=f"Палитра загружена: {len(colors)} цветов")
            self.schedule_update()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить палитру:\n{e}")

    def _pal_save_file(self):
        if not self.custom_palette:
            messagebox.showinfo("Пусто", "Палитра пуста.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить палитру",
            defaultextension=".gpl",
            filetypes=[("GIMP палитра", "*.gpl"), ("JASC палитра", "*.pal"),
                       ("Список hex", "*.hex")],
        )
        if not path:
            return
        try:
            save_palette_file(path, self.custom_palette)
            self.status.configure(text=f"Палитра сохранена: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def _build_tune_tab(self, parent):
        inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True)

        pf, pf_in = labeled_frame(inner, "Пикселизация")
        pf.pack(fill="x", padx=6, pady=6)
        self.lbl_pixel = ctk.CTkLabel(pf_in, text="160", text_color=HINT, anchor="e")
        self.lbl_pixel.pack(fill="x")
        sc = ctk.CTkSlider(pf_in, from_=24, to=1024, variable=self.pixel_var,
                           command=self._on_pixel_change)
        sc.pack(fill="x")

        af, af_in = labeled_frame(inner, "Коррекция")
        af.pack(fill="x", padx=6, pady=6)
        self._slider(af_in, "Яркость", self.brightness_var, 0.2, 2.0)
        self._slider(af_in, "Контраст", self.contrast_var, 0.2, 3.0)
        self._slider(af_in, "Насыщенность", self.saturation_var, 0.0, 2.0)
        self._slider(af_in, "Гамма", self.gamma_var, 0.3, 3.0)

        ctk.CTkButton(inner, text="Сбросить коррекцию",
                      command=self._reset_adjust).pack(fill="x", padx=6, pady=6)

    def _reset_adjust(self):
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.gamma_var.set(1.0)
        self.schedule_update()

    def _on_pixel_change(self, val):
        try:
            self.lbl_pixel.configure(text=str(int(round(float(val)))))
        except (TypeError, ValueError):
            pass
        self.schedule_update()

    def _build_glitch_tab(self, parent):
        inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True)

        sf, sf_in = labeled_frame(inner, "Сид")
        sf.pack(fill="x", padx=6, pady=6)
        row = ctk.CTkFrame(sf_in, fg_color="transparent")
        row.pack(fill="x")
        self.lbl_seed = ctk.CTkLabel(row, text=str(self.glitch_seed), anchor="w")
        self.lbl_seed.pack(side="left")
        ctk.CTkButton(row, text="Перегенерировать", width=140,
                      command=self._reroll_seed).pack(side="right")
        ctk.CTkButton(row, text="Сбросить всё", width=120,
                      command=self._reset_glitch).pack(side="right", padx=4)

        cf, cf_in = labeled_frame(inner, "Каналы / Цвет")
        cf.pack(fill="x", padx=6, pady=6)
        self._int_slider(cf_in, "R сдвиг", self.ch_r_var, -10, 10)
        self._int_slider(cf_in, "G сдвиг", self.ch_g_var, -10, 10)
        self._int_slider(cf_in, "B сдвиг", self.ch_b_var, -10, 10)
        self._slider(cf_in, "Хроматика", self.chroma_var, 0.0, 1.0)
        ctk.CTkLabel(cf_in, text="Обмен каналов:", text_color=HINT, anchor="w").pack(
            anchor="w", pady=(6, 0))
        row = ctk.CTkFrame(cf_in, fg_color="transparent")
        row.pack(fill="x")
        for label, var in [("R↔G", self.swap_rg_var),
                           ("R↔B", self.swap_rb_var),
                           ("G↔B", self.swap_gb_var)]:
            ctk.CTkCheckBox(row, text=label, variable=var,
                            command=self.schedule_update).pack(side="left", padx=4)

        stf, stf_in = labeled_frame(inner, "Структура")
        stf.pack(fill="x", padx=6, pady=6)
        self._int_slider(stf_in, "Сдвиг строк ±", self.row_shift_var, 0, 20)
        self._int_slider(stf_in, "Блоков", self.block_count_var, 0, 20)
        self._int_slider(stf_in, "Размер блока", self.block_size_var, 2, 40)
        self._slider(stf_in, "Зеркало %", self.mirror_var, 0.0, 1.0)
        self._slider(stf_in, "VHS", self.vhs_var, 0.0, 1.0)

        scf, scf_in = labeled_frame(inner, "Строки развёртки")
        scf.pack(fill="x", padx=6, pady=6)
        self._int_slider(scf_in, "Промежуток", self.scanline_spacing_var, 0, 10)
        self._slider(scf_in, "Затемнение", self.scanline_darkness_var, 0.0, 0.9)

        nf, nf_in = labeled_frame(inner, "Шум")
        nf.pack(fill="x", padx=6, pady=6)
        self._slider(nf_in, "Сила", self.noise_var, 0.0, 0.5)

        psf, psf_in = labeled_frame(inner, "Сортировка пикселей")
        psf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(psf_in, text="Включить", variable=self.sort_enabled_var,
                        command=self.schedule_update).pack(anchor="w")
        self._int_slider(psf_in, "Порог", self.sort_threshold_var, 0, 255)
        row = ctk.CTkFrame(psf_in, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkRadioButton(row, text="Гориз.", variable=self.sort_dir_var,
                           value="horizontal",
                           command=self.schedule_update).pack(side="left", padx=4)
        ctk.CTkRadioButton(row, text="Верт.", variable=self.sort_dir_var,
                           value="vertical",
                           command=self.schedule_update).pack(side="left", padx=4)

    def _reroll_seed(self):
        self.glitch_seed = random.randint(0, 10_000_000)
        self.lbl_seed.configure(text=str(self.glitch_seed))
        self.schedule_update()

    def _reset_glitch(self):
        for v in (self.ch_r_var, self.ch_g_var, self.ch_b_var, self.row_shift_var,
                  self.block_count_var, self.scanline_spacing_var):
            v.set(0)
        for v in (self.chroma_var, self.noise_var, self.mirror_var, self.vhs_var):
            v.set(0.0)
        self.swap_rg_var.set(False)
        self.swap_rb_var.set(False)
        self.swap_gb_var.set(False)
        self.sort_enabled_var.set(False)
        self.schedule_update()

    def _build_post_tab(self, parent):
        inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True)

        bf, bf_in = labeled_frame(inner, "Bloom / Glow")
        bf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(bf_in, text="Включить Bloom", variable=self.bloom_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(bf_in, "Порог", self.bloom_threshold, 0.0, 1.0)
        self._slider(bf_in, "Радиус", self.bloom_radius, 1.0, 30.0)
        self._slider(bf_in, "Сила", self.bloom_intensity, 0.0, 2.0)
        ctk.CTkButton(bf_in, text="Цвет свечения…",
                      command=self._pick_bloom_tint).pack(fill="x", pady=4)

        ef, ef_in = labeled_frame(inner, "Epsilon Glow")
        ef.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(ef_in, text="Включить", variable=self.epsilon_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(ef_in, "Радиус", self.epsilon_radius, 1.0, 20.0)
        self._slider(ef_in, "Сила", self.epsilon_intensity, 0.0, 2.0)

        hf, hf_in = labeled_frame(inner, "Halftone")
        hf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(hf_in, text="Включить", variable=self.halftone_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._int_slider(hf_in, "Размер точки", self.halftone_size, 2, 12)
        self._int_slider(hf_in, "Угол°", self.halftone_angle, 0, 90)
        self._slider(hf_in, "Сила", self.halftone_strength, 0.0, 1.0)

        jf, jf_in = labeled_frame(inner, "JPEG Glitch")
        jf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(jf_in, text="Включить", variable=self.jpeg_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._int_slider(jf_in, "Качество", self.jpeg_quality, 1, 50)
        self._slider(jf_in, "Сила", self.jpeg_strength, 0.0, 1.0)

        cf, cf_in = labeled_frame(inner, "CRT")
        cf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(cf_in, text="Искривление экрана", variable=self.crt_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(cf_in, "Кривизна", self.crt_curvature, 0.0, 0.5)
        ctk.CTkCheckBox(cf_in, text="Фосфорное свечение",
                        variable=self.phosphor_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(cf_in, "Сила свечения", self.phosphor_intensity, 0.0, 1.0)

        af, af_in = labeled_frame(inner, "Аналоговые эффекты")
        af.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(af_in, text="Halation", variable=self.halation_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(af_in, "Сила halation", self.halation_intensity, 0.0, 1.0)
        ctk.CTkCheckBox(af_in, text="Vignette", variable=self.vignette_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(af_in, "Сила виньетки", self.vignette_strength, 0.0, 1.0)

        mf, mf_in = labeled_frame(inner, "Прочее")
        mf.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(mf_in, text="Datamosh", variable=self.datamosh_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(mf_in, "Сила datamosh", self.datamosh_intensity, 0.0, 1.0)
        ctk.CTkCheckBox(mf_in, text="Pixel Stretch", variable=self.stretch_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(mf_in, "Вероятность", self.stretch_probability, 0.0, 0.5)
        self._int_slider(mf_in, "Макс. длина", self.stretch_max, 2, 60)
        ctk.CTkCheckBox(mf_in, text="Blur / Sharpen", variable=self.blur_enabled,
                        command=self.schedule_update).pack(anchor="w")
        self._slider(mf_in, "Размытие", self.blur_radius, 0.0, 5.0)
        self._slider(mf_in, "Резкость", self.sharpen_amount, 0.0, 1.0)

    def _pick_bloom_tint(self):
        res = colorchooser.askcolor(color=self.bloom_tint, title="Цвет bloom")
        if res and res[0]:
            self.bloom_tint = tuple(int(c) for c in res[0])
            self.schedule_update()

    def _build_video_tab(self, parent):
        inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        inner.pack(fill="both", expand=True)

        of, of_in = labeled_frame(inner, "Источник")
        of.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(of_in, text="Открыть видео…",
                      command=self.open_video).pack(fill="x")

        self.lbl_video_info = ctk.CTkLabel(
            of_in, text="Видео не загружено", text_color=HINT,
            wraplength=340, justify="left", anchor="w")
        self.lbl_video_info.pack(fill="x", pady=(6, 0))

        if not HAS_CV2:
            ctk.CTkLabel(
                of_in,
                text="OpenCV не установлен.\nДля работы с видео установите:\n"
                     "    pip install opencv-python",
                text_color="#d33", wraplength=340, justify="left", anchor="w",
            ).pack(fill="x", pady=(6, 0))

        ff, ff_in = labeled_frame(inner, "Просмотр кадра")
        ff.pack(fill="x", padx=6, pady=6)
        self.lbl_video_frame = ctk.CTkLabel(
            ff_in, text="Кадр — / —", text_color=HINT, anchor="w")
        self.lbl_video_frame.pack(fill="x")
        self.video_slider = ctk.CTkSlider(
            ff_in, from_=0, to=1, variable=self.video_frame_var,
            command=self._on_video_frame_change, state="disabled")
        self.video_slider.pack(fill="x", pady=(4, 0))

        ef, ef_in = labeled_frame(inner, "Экспорт")
        ef.pack(fill="x", padx=6, pady=6)
        ctk.CTkCheckBox(
            ef_in, text="Разный глитч на кадрах",
            variable=self.per_frame_glitch_var,
            command=self.schedule_update,
        ).pack(anchor="w")
        self.btn_export_video = ctk.CTkButton(
            ef_in, text="Экспортировать видео…",
            command=self.export_video, state="disabled")
        self.btn_export_video.pack(fill="x", pady=(6, 0))

        ctk.CTkLabel(
            inner,
            text=("Для видео лучше использовать упорядоченные алгоритмы "
                  "(Bayer) или режимы без тяжёлого дизеринга. "
                  "Floyd-Steinberg и подобные работают медленно."),
            text_color=HINT, wraplength=340, justify="left", anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 6))

    def _on_video_frame_change(self, val):
        if self.video_source is None:
            return
        try:
            idx = int(float(val))
        except (TypeError, ValueError):
            return
        frame = self.video_source.get_frame(idx)
        if frame is not None:
            self.source_image = frame
            self.lbl_video_frame.configure(
                text=f"Кадр {idx + 1} / {self.video_source.frame_count}")
            self.schedule_update()

    def open_video(self):
        if not HAS_CV2:
            messagebox.showerror(
                "OpenCV не установлен",
                "Для работы с видео установите OpenCV:\n\n"
                "    pip install opencv-python\n\n"
                "и перезапустите программу.")
            return
        path = filedialog.askopenfilename(
            title="Открыть видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
                       ("Все файлы", "*.*")],
        )
        if not path:
            return
        try:
            if self.video_source is not None:
                self.video_source.close()
            self.video_source = VideoSource(path)
            self.video_path = path
            self.video_frame_var.set(0)
            first = self.video_source.get_frame(0)
            if first is None:
                raise RuntimeError("Не удалось прочитать первый кадр.")
            self.source_image = first
            self.video_slider.configure(
                from_=0, to=max(0, self.video_source.frame_count - 1),
                state="normal")
            self.lbl_video_frame.configure(
                text=f"Кадр 1 / {self.video_source.frame_count}")
            self.lbl_video_info.configure(
                text=(f"Файл: {Path(path).name}\n"
                      f"Кадров: {self.video_source.frame_count} · "
                      f"FPS: {self.video_source.fps:.2f}\n"
                      f"{self.video_source.width}×{self.video_source.height}"))
            self.btn_export_video.configure(state="normal")
            self.lbl_file.configure(text=Path(path).name)
            self.status.configure(
                text=f"Видео загружено: {self.video_source.frame_count} кадров")
            self.schedule_update()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть видео:\n{e}")

    def export_video(self):
        if self.video_source is None:
            messagebox.showinfo("Нет видео", "Сначала откройте видео.")
            return
        if not HAS_CV2:
            messagebox.showerror("OpenCV не установлен",
                                 "Установите: pip install opencv-python")
            return

        path = filedialog.asksaveasfilename(
            title="Экспорт видео", defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")])
        if not path:
            return

        progress_win = ctk.CTkToplevel(self.root)
        progress_win.title("Экспорт видео")
        progress_win.geometry("420x180")
        progress_win.transient(self.root)
        progress_win.grab_set()

        lbl = ctk.CTkLabel(progress_win, text="Подготовка…")
        lbl.pack(pady=(20, 8))
        pb = ctk.CTkProgressBar(progress_win, width=380)
        pb.set(0)
        pb.pack(pady=4, padx=20)

        cancel_flag = [False]

        def cancel():
            cancel_flag[0] = True
            lbl.configure(text="Отмена…")

        ctk.CTkButton(progress_win, text="Отмена", width=100,
                      command=cancel).pack(pady=(8, 16))

        def progress(i, total, msg=""):
            pb.set(i / max(1, total))
            lbl.configure(text=msg or f"Кадр {i} из {total}")
            self.root.update_idletasks()

        error = None
        try:
            self._do_export_video(path, progress, cancel_flag)
        except Exception as e:
            error = e
        finally:
            progress_win.destroy()

        if error is not None:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать видео:\n{error}")
        elif cancel_flag[0]:
            self.status.configure(text="Экспорт отменён")
        else:
            messagebox.showinfo("Готово", f"Видео сохранено:\n{path}")
            self.status.configure(text=f"Видео сохранено: {path}")

    def _do_export_video(self, path, progress, cancel_flag):
        scale = max(1, int(self.output_scale_var.get()))
        total = self.video_source.frame_count

        progress(0, total, "Подготовка…")
        first = self.video_source.get_frame(0)
        if first is None:
            raise RuntimeError("Не удалось прочитать первый кадр.")
        first_out = self._render(first, seed=self.glitch_seed)
        out_w = first_out.width * scale
        out_h = first_out.height * scale
        out_w += out_w % 2
        out_h += out_h % 2

        ext = Path(path).suffix.lower()
        fourcc_str = "MJPG" if ext == ".avi" else "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, self.video_source.fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(
                "Не удалось открыть.\nПопробуйте другой кодек или формат (.avi).")

        try:
            for i in range(total):
                if cancel_flag[0]:
                    break
                frame = self.video_source.get_frame(i)
                if frame is None:
                    continue
                seed = (self.glitch_seed + i) if self.per_frame_glitch_var.get() \
                    else self.glitch_seed
                out = self._render(frame, seed=seed)
                if scale > 1 or out.size != (out_w, out_h):
                    out = out.resize((out_w, out_h), Image.NEAREST)
                bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if i % 3 == 0 or i == total - 1:
                    progress(i + 1, total)
        finally:
            writer.release()

    def _build_tutorial_tab(self, parent):
        if HAS_TUTORIAL:
            try:
                create_tutorial_widget(parent)
            except Exception as e:
                ctk.CTkLabel(parent, text=f"Не удалось загрузить туториал:\n{e}",
                             text_color="#d33", wraplength=340).pack(padx=10, pady=20)
        else:
            ctk.CTkLabel(
                parent,
                text=("Не найден модуль tutorial.py.\n\n"
                      "Положите файл tutorial.py рядом с основным скриптом, "
                      "чтобы включить встроенный туториал."),
                text_color=HINT, wraplength=340, justify="left",
            ).pack(padx=10, pady=20)

    def _show_tutorial(self):
        if HAS_TUTORIAL:
            show_tutorial_window(self.root)
        else:
            messagebox.showinfo(
                "Туториал",
                "Файл tutorial.py не найден.\nПоложите его рядом с основным скриптом.")

    def _make_spinbox(self, parent, var, from_, to, command=None, width=60):
        spin = tk.Spinbox(
            parent, from_=from_, to=to, textvariable=var,
            bg=ENTRY_BG, fg="#dce4ee", insertbackground="#dce4ee",
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor=ACCENT,
            font=("Segoe UI", 11), width=width,
        )
        if command:
            spin.configure(command=command)
            spin.bind("<KeyRelease>", lambda e: command())
        return spin

    def _slider(self, parent, label, var, lo, hi):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(frame, text=label, anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(frame, text="", anchor="e", width=60, text_color=HINT)
        val_lbl.pack(side="right")

        def upd(*_):
            try:
                val_lbl.configure(text=f"{var.get():.2f}")
            except Exception:
                pass

        def on_change(_v=None):
            upd()
            self.schedule_update()

        sc = ctk.CTkSlider(parent, from_=lo, to=hi, variable=var, command=on_change)
        sc.pack(fill="x")
        upd()
        return sc

    def _int_slider(self, parent, label, var, lo, hi):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(frame, text=label, anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(frame, text="", anchor="e", width=60, text_color=HINT)
        val_lbl.pack(side="right")

        def upd(*_):
            try:
                val_lbl.configure(text=str(int(var.get())))
            except Exception:
                pass

        def on_change(_v=None):
            upd()
            self.schedule_update()

        sc = ctk.CTkSlider(parent, from_=lo, to=hi, variable=var, command=on_change)
        sc.pack(fill="x")
        upd()
        return sc

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Открыть изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            self.source_image = img
            self.lbl_file.configure(text=Path(path).name)
            self.status.configure(text=f"Загружено: {img.width}×{img.height}")
            self.schedule_update()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{e}")

    def save_image(self):
        if self.result_image is None:
            messagebox.showinfo("Нечего сохранять", "Сначала откройте изображение.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            img = self.result_image
            s = max(1, int(self.output_scale_var.get()))
            if s > 1:
                img = img.resize((img.width * s, img.height * s), Image.NEAREST)
            img.save(path)
            self.status.configure(text=f"Сохранено: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def schedule_update(self, *_):
        if self._job:
            self.root.after_cancel(self._job)
        self._job = self.root.after(80, self._process)

    def _process(self):
        self._job = None
        if self.source_image is None:
            return
        try:
            if self.video_source is not None and self.per_frame_glitch_var.get():
                seed = self.glitch_seed + int(self.video_frame_var.get())
            else:
                seed = self.glitch_seed
            self.result_image = self._render(self.source_image, seed=seed)
            self._redraw_preview()
            self.status.configure(
                text=f"Обработано · {self.result_image.width}×{self.result_image.height}")
        except Exception as e:
            self.status.configure(text=f"Ошибка: {e}")

    def _render(self, pil_img, seed=None):
        target_w = max(8, int(round(self.pixel_var.get())))
        aspect = pil_img.height / pil_img.width
        target_h = max(1, int(round(target_w * aspect)))

        img = pil_img.resize((target_w, target_h), Image.LANCZOS)
        img = self._adjust(img)

        if seed is None:
            seed = self.glitch_seed
        np.random.seed(seed)
        img = self._glitch(img)

        img = self._apply_mode(img)
        img = self._post_process(img)
        return img

    def _adjust(self, img):
        b = self.brightness_var.get()
        c = self.contrast_var.get()
        s = self.saturation_var.get()
        g = self.gamma_var.get()

        if abs(b - 1.0) > 0.01:
            img = ImageEnhance.Brightness(img).enhance(b)
        if abs(c - 1.0) > 0.01:
            img = ImageEnhance.Contrast(img).enhance(c)
        if abs(s - 1.0) > 0.01:
            img = ImageEnhance.Color(img).enhance(s)
        if abs(g - 1.0) > 0.01:
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.power(arr, 1.0 / g)
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        return img

    def _glitch(self, img):
        rs, gs, bs = self.ch_r_var.get(), self.ch_g_var.get(), self.ch_b_var.get()
        if rs or gs or bs:
            img = glitch_channel_shift(img, rs, gs, bs)

        img = glitch_chromatic(img, self.chroma_var.get())
        img = glitch_row_displace(img, self.row_shift_var.get())
        img = glitch_block_displace(img, self.block_count_var.get(),
                                    self.block_size_var.get())
        img = glitch_mirror(img, self.mirror_var.get())

        if self.sort_enabled_var.get():
            img = glitch_pixel_sort(img, self.sort_threshold_var.get(),
                                    self.sort_dir_var.get())

        img = glitch_color_swap(img, self.swap_rg_var.get(),
                                self.swap_rb_var.get(), self.swap_gb_var.get())
        img = glitch_vhs(img, self.vhs_var.get())
        img = glitch_scanlines(img, self.scanline_spacing_var.get(),
                               self.scanline_darkness_var.get())
        img = glitch_noise(img, self.noise_var.get())
        return img

    def _post_process(self, img):
        if self.halftone_enabled.get():
            img = halftone(
                img, dot_size=max(2, int(self.halftone_size.get())),
                angle=int(self.halftone_angle.get()),
                strength=float(self.halftone_strength.get()))

        if self.jpeg_enabled.get():
            img = jpeg_glitch(
                img, quality=max(1, int(self.jpeg_quality.get())),
                strength=float(self.jpeg_strength.get()))

        if self.bloom_enabled.get():
            img = bloom(
                img, threshold=float(self.bloom_threshold.get()),
                radius=float(self.bloom_radius.get()),
                intensity=float(self.bloom_intensity.get()),
                tint=self.bloom_tint)
        if self.epsilon_enabled.get():
            img = epsilon_glow(
                img, radius=float(self.epsilon_radius.get()),
                intensity=float(self.epsilon_intensity.get()))

        if self.crt_enabled.get():
            img = crt_warp(img, curvature=float(self.crt_curvature.get()))
        if self.phosphor_enabled.get():
            img = phosphor_glow(img, intensity=float(self.phosphor_intensity.get()))

        if self.halation_enabled.get():
            img = halation(img, intensity=float(self.halation_intensity.get()))
        if self.vignette_enabled.get():
            img = vignette(img, strength=float(self.vignette_strength.get()))

        if self.datamosh_enabled.get():
            img = datamosh(img, intensity=float(self.datamosh_intensity.get()))
        if self.stretch_enabled.get():
            img = glitch_pixel_stretch(
                img,
                probability=float(self.stretch_probability.get()),
                max_stretch=max(2, int(self.stretch_max.get())))
        if self.blur_enabled.get():
            img = blur_sharpen(
                img, blur_radius=float(self.blur_radius.get()),
                sharpen_amount=float(self.sharpen_amount.get()))

        return img

    def _apply_mode(self, img):
        mode = self.mode_var.get()

        if mode == "bw":
            gray = np.array(img.convert("L"))
            algo = self.algo_var.get()
            strength = float(self.dither_strength_var.get())
            if algo in BAYER_MATRICES:
                out = ordered_dither(gray, BAYER_MATRICES[algo], strength)
            else:
                out = diffuse_dither(gray, KERNELS[algo], strength,
                                     bool(self.serpentine_var.get()))
            return Image.fromarray(out, "L").convert("RGB")

        if mode == "gray":
            return quantize_gray(img, max(2, int(self.levels_var.get())))

        if mode == "palette":
            pal = self.custom_palette if self.custom_palette else PALETTES["1-bit"]
            return quantize_palette(img, pal, bool(self.palette_dither_var.get()))

        if mode == "adaptive":
            n = max(2, min(256, int(self.adaptive_colors_var.get())))
            return img.quantize(colors=n, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")

        return img

    def _set_compare(self, on):
        self.show_original = on
        self._redraw_preview()

    def _redraw_preview(self):
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        img = self.source_image if self.show_original else self.result_image
        if img is None:
            img = self.source_image
        if img is None:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                cw // 2, ch // 2,
                text="Откройте изображение для начала",
                fill="#888", font=("Segoe UI", 14))
            return

        scale = min(cw / img.width, ch / img.height)
        nw = max(1, int(img.width * scale))
        nh = max(1, int(img.height * scale))
        resample = Image.NEAREST if scale >= 1.0 else Image.LANCZOS
        scaled = img.resize((nw, nh), resample)
        self.preview_photo = ImageTk.PhotoImage(scaled)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                                         anchor="nw", image=self.preview_photo)