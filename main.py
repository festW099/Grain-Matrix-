import customtkinter as ctk
from app import DitherApp


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    DitherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()