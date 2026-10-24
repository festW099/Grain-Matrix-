import customtkinter as ctk

TUTORIAL_SECTIONS = [
    {
        "title": "В разработке",
        "body": "В разработке",
    },
    {
        "title": "Некоторые функции могут не работать",
        "body": "они экспериментальные",
    },
]


def create_tutorial_widget(parent):
    """Создаёт скроллируемый виджет с секциями туториала внутри parent."""
    inner = ctk.CTkScrollableFrame(parent, fg_color="transparent")
    inner.pack(fill="both", expand=True)

    for section in TUTORIAL_SECTIONS:
        frame = ctk.CTkFrame(inner, corner_radius=8, border_width=1,
                             border_color="#3a3a3a")
        frame.pack(fill="x", padx=8, pady=6)

        ctk.CTkLabel(
            frame, text=section["title"], anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(fill="x", padx=12, pady=(8, 2))

        lbl = ctk.CTkLabel(
            frame, text=section["body"], justify="left", anchor="w",
            wraplength=320, font=ctk.CTkFont(size=11),
        )
        lbl.pack(fill="x", padx=12, pady=(0, 10))

        def _on_conf(e, l=lbl):
            try:
                l.configure(wraplength=max(120, e.width - 30))
            except Exception:
                pass

        frame.bind("<Configure>", _on_conf)

    return inner


def show_tutorial_window(parent, title="Туториал PixelDither"):
    win = ctk.CTkToplevel(parent)
    win.title(title)
    win.geometry("640x720")
    win.minsize(420, 400)
    win.transient(parent)
    create_tutorial_widget(win)
    return win