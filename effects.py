import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _to_float(img):
    return np.array(img).astype(np.float32) / 255.0


def _to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def bloom(img, threshold=0.7, radius=8, intensity=0.6, tint=(255, 255, 255)):
    if intensity <= 0.005:
        return img
    arr = _to_float(img)
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.float32)
    mask = np.array(Image.fromarray((mask * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(radius * 0.5))) / 255.0
    bright = arr * mask[:, :, None]
    blurred = np.array(Image.fromarray((bright * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))) / 255.0
    tint_arr = np.array(tint, dtype=np.float32) / 255.0
    glow = blurred * tint_arr[None, None, :] * intensity
    out = np.clip(arr + glow, 0, 1)
    return _to_img(out)


def epsilon_glow(img, radius=6, intensity=0.8, edge_boost=2.0):
    if intensity <= 0.005:
        return img
    arr = _to_float(img)
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0
    edges = np.array(Image.fromarray((edges * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(radius * 0.3))) / 255.0
    edges = np.clip(edges * edge_boost, 0, 1)
    blurred = np.array(Image.fromarray((edges * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))) / 255.0
    glow = blurred[:, :, None] * np.array([1.0, 0.9, 0.7])[None, None, :] * intensity
    out = np.clip(arr + glow, 0, 1)
    return _to_img(out)

def halftone(img, dot_size=4, angle=45, strength=0.8, invert=False):
    if dot_size <= 1 or strength <= 0.005:
        return img
    arr = _to_float(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.deg2rad(angle)
    u = xx * np.cos(rad) + yy * np.sin(rad)
    v = -xx * np.sin(rad) + yy * np.cos(rad)
    cell = dot_size
    fu = (u % cell) / cell - 0.5
    fv = (v % cell) / cell - 0.5
    dist = np.sqrt(fu ** 2 + fv ** 2)
    lum = arr.mean(axis=2)
    radius = np.sqrt(lum) * 0.5
    mask = (dist < radius).astype(np.float32)
    if invert:
        mask = 1 - mask
    out = arr * (1 - strength) + (arr * mask[:, :, None]) * strength
    return _to_img(out)

def jpeg_glitch(img, quality=15, block_size=8, strength=1.0):
    if strength <= 0.005:
        return img
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    crushed = Image.open(buf).convert("RGB")
    arr = np.array(crushed).astype(np.float32)
    h, w = arr.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            if block.size:
                block += np.random.normal(0, 10 * strength, block.shape)
    crushed = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return Image.blend(img, crushed, strength)

def crt_warp(img, curvature=0.15):
    if curvature <= 0.005:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx ** 2 + ny ** 2
    factor = 1 + curvature * r2
    sx = np.clip(nx * factor * cx + cx, 0, w - 1).astype(np.int32)
    sy = np.clip(ny * factor * cy + cy, 0, h - 1).astype(np.int32)
    return Image.fromarray(arr[sy, sx])


def phosphor_glow(img, intensity=0.4, tint=(0, 255, 180)):
    if intensity <= 0.005:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(6))
    tint_arr = np.array(tint, dtype=np.float32) / 255.0
    arr = _to_float(img) + _to_float(blurred) * tint_arr[None, None, :] * intensity
    return _to_img(np.clip(arr, 0, 1))

def halation(img, threshold=0.75, radius=10, intensity=0.5):
    if intensity <= 0.005:
        return img
    arr = _to_float(img)
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.float32)
    bright = arr * mask[:, :, None]
    blurred = np.array(Image.fromarray((bright * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))) / 255.0
    glow = blurred * np.array([1.0, 0.4, 0.3])[None, None, :] * intensity
    return _to_img(np.clip(arr + glow, 0, 1))


def vignette(img, strength=0.5, radius=0.8):
    if strength <= 0.005:
        return img
    arr = _to_float(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    mask = np.clip(1 - strength * (dist / radius) ** 2, 0, 1)
    return _to_img(arr * mask[:, :, None])

def datamosh(img, block_size=16, intensity=0.4):
    if intensity <= 0.005:
        return img
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    bs = block_size
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            if np.random.rand() < intensity:
                sy = np.clip(y + np.random.randint(-bs, bs), 0, h - bs)
                sx = np.clip(x + np.random.randint(-bs, bs), 0, w - bs)
                arr[y:y+bs, x:x+bs] = arr[sy:sy+bs, sx:sx+bs]
    return Image.fromarray(arr)

def blur_sharpen(img, blur_radius=0.0, sharpen_amount=0.0):
    out = img
    if blur_radius > 0.1:
        out = out.filter(ImageFilter.GaussianBlur(blur_radius))
    if sharpen_amount > 0.05:
        out = out.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(sharpen_amount * 200), threshold=3))
    return out