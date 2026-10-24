from PIL import Image


def quantize_palette(img_rgb, palette, dither=True):
    pal_img = Image.new("P", (1, 1))
    flat = []
    for c in palette:
        flat.extend(c[:3])
    flat += [0] * (768 - len(flat))
    pal_img.putpalette(flat)
    d = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return img_rgb.quantize(palette=pal_img, dither=d).convert("RGB")


def quantize_gray(img_rgb, levels, dither=True):
    d = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return img_rgb.convert("L").quantize(colors=levels, dither=d).convert("RGB")