import numpy as np
from PIL import Image


def glitch_channel_shift(img, rs, gs, bs):
    if rs == 0 and gs == 0 and bs == 0:
        return img
    a = np.array(img)
    r = np.roll(a[:, :, 0], rs, axis=1)
    g = np.roll(a[:, :, 1], gs, axis=1)
    b = np.roll(a[:, :, 2], bs, axis=1)
    return Image.fromarray(np.stack([r, g, b], -1))


def glitch_chromatic(img, amount):
    if amount <= 0.005:
        return img
    a = np.array(img)
    h, w = a.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def sample(ch, scale):
        sy = np.clip(((yy - cy) / scale + cy).astype(np.int32), 0, h - 1)
        sx = np.clip(((xx - cx) / scale + cx).astype(np.int32), 0, w - 1)
        return ch[sy, sx]

    r = sample(a[:, :, 0], 1 + amount * 0.15)
    b = sample(a[:, :, 2], 1 - amount * 0.15)
    return Image.fromarray(np.stack([r, a[:, :, 1], b], -1))


def glitch_row_displace(img, max_shift):
    if max_shift <= 0:
        return img
    a = np.array(img)
    h = a.shape[0]
    shifts = np.random.randint(-max_shift, max_shift + 1, size=h)
    for y in range(h):
        a[y] = np.roll(a[y], shifts[y], axis=0)
    return Image.fromarray(a)


def glitch_block_displace(img, count, size):
    if count <= 0:
        return img
    a = np.array(img).copy()
    h, w = a.shape[:2]
    size = min(size, h, w)
    for _ in range(count):
        by = np.random.randint(0, max(1, h - size + 1))
        bx = np.random.randint(0, max(1, w - size + 1))
        dy = np.random.randint(-size, size + 1)
        dx = np.random.randint(-size, size + 1)
        block = a[by:by + size, bx:bx + size].copy()
        ty = np.clip(by + dy, 0, h - size)
        tx = np.clip(bx + dx, 0, w - size)
        a[ty:ty + size, tx:tx + size] = block
    return Image.fromarray(a)


def glitch_scanlines(img, spacing, darkness):
    if spacing <= 1 or darkness <= 0.005:
        return img
    a = np.array(img).astype(np.float32)
    a[::spacing] *= (1 - darkness)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def glitch_noise(img, amount):
    if amount <= 0.005:
        return img
    a = np.array(img).astype(np.float32)
    n = np.random.normal(0, amount * 255, a.shape)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def glitch_mirror(img, probability):
    if probability <= 0.005:
        return img
    a = np.array(img).copy()
    h = a.shape[0]
    for y in range(h):
        if np.random.rand() < probability:
            a[y] = a[y, ::-1]
    return Image.fromarray(a)


def glitch_vhs(img, strength):
    if strength <= 0.005:
        return img
    a = np.array(img).astype(np.float32)
    h = a.shape[0]
    for y in range(h):
        offset = int(strength * 3 * np.sin(y * 0.2))
        if offset:
            a[y] = np.roll(a[y], offset, axis=0)
    shift = int(strength * 2)
    a[:, :, 0] = np.roll(a[:, :, 0], shift, axis=1)
    a[:, :, 2] = np.roll(a[:, :, 2], -shift, axis=1)
    n = np.random.normal(0, strength * 25, a.shape[:2])[:, :, None]
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


def glitch_pixel_sort(img, threshold, direction="horizontal"):
    a = np.array(img).copy()
    h, w = a.shape[:2]
    gray = np.array(img.convert("L"))

    if direction == "horizontal":
        for y in range(h):
            row = a[y]
            above = gray[y] > threshold
            x = 0
            while x < w:
                if above[x]:
                    start = x
                    while x < w and above[x]:
                        x += 1
                    if x - start > 1:
                        order = np.argsort(gray[y, start:x])
                        row[start:x] = row[start:x][order]
                else:
                    x += 1
    else:
        for x in range(w):
            col = a[:, x]
            above = gray[:, x] > threshold
            y = 0
            while y < h:
                if above[y]:
                    start = y
                    while y < h and above[y]:
                        y += 1
                    if y - start > 1:
                        order = np.argsort(gray[start:y, x])
                        col[start:y] = col[start:y][order]
                else:
                    y += 1
    return Image.fromarray(a)


def glitch_color_swap(img, rg, rb, gb):
    if not (rg or rb or gb):
        return img
    a = np.array(img).copy()
    r, g, b = a[:, :, 0].copy(), a[:, :, 1].copy(), a[:, :, 2].copy()
    if rg: r, g = g, r
    if rb: r, b = b, r
    if gb: g, b = b, g
    a[:, :, 0] = r
    a[:, :, 1] = g
    a[:, :, 2] = b
    return Image.fromarray(a)

def glitch_pixel_stretch(img, probability=0.1, max_stretch=20):
    if probability <= 0.005:
        return img
    a = np.array(img).copy()
    h, w = a.shape[:2]
    for y in range(h):
        x = 0
        while x < w:
            if np.random.rand() < probability:
                length = np.random.randint(1, max_stretch)
                x2 = min(w, x + length)
                a[y, x:x2] = a[y, x]
                x = x2
            else:
                x += 1
    return Image.fromarray(a)