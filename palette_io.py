from pathlib import Path


def load_palette_file(path):
    p = Path(path)
    text = p.read_text(errors="ignore")
    colors = []
    suf = p.suffix.lower()

    if suf == ".gpl" or "GIMP Palette" in text[:100]:
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("GIMP") \
               or s.startswith("Name:") or s.startswith("Columns:"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    colors.append(tuple(int(x) for x in parts[:3]))
                except ValueError:
                    pass
    elif suf == ".pal" or text.lstrip().startswith("JASC-PAL"):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        idx = 3 if lines[0] == "JASC-PAL" else 0
        for line in lines[idx:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    colors.append(tuple(int(x) for x in parts[:3]))
                except ValueError:
                    pass
    elif suf == ".hex" or all(len(l.strip().lstrip("#")) == 6 for l in text.splitlines() if l.strip()):
        for line in text.splitlines():
            s = line.strip().lstrip("#")
            if len(s) == 6:
                try:
                    colors.append((int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)))
                except ValueError:
                    pass
    else:
        for line in text.splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    colors.append(tuple(int(x) for x in parts[:3]))
                except ValueError:
                    pass
    return colors


def save_palette_file(path, colors, name="Custom"):
    p = Path(path)
    if p.suffix.lower() == ".pal":
        lines = ["JASC-PAL", "0100", str(len(colors))]
        lines += [f"{r} {g} {b}" for r, g, b in colors]
        p.write_text("\n".join(lines))
    elif p.suffix.lower() == ".hex":
        p.write_text("\n".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors))
    else:
        lines = ["GIMP Palette", f"Name: {name}", "Columns: 8", "#"]
        lines += [f"{r:3d} {g:3d} {b:3d}" for r, g, b in colors]
        p.write_text("\n".join(lines))