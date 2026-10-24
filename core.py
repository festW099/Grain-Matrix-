import numpy as np


def generate_bayer(n):
    if n == 1:
        return np.array([[0]], dtype=np.int32)
    prev = generate_bayer(n // 2)
    s = prev.shape[0]
    out = np.zeros((2 * s, 2 * s), dtype=np.int32)
    out[:s, :s] = 4 * prev
    out[:s, s:] = 4 * prev + 2
    out[s:, :s] = 4 * prev + 3
    out[s:, s:] = 4 * prev + 1
    return out


def make_halftone_dot(size=8):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    dx = np.abs(xx - c); dx = np.minimum(dx, size - dx)
    dy = np.abs(yy - c); dy = np.minimum(dy, size - dy)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return (1.0 - dist / dist.max()).astype(np.float32)


def make_halftone_line(size=8):
    yy, _ = np.mgrid[0:size, 0:size].astype(np.float32)
    return (0.5 + 0.5 * np.cos(yy / size * 2 * np.pi)).astype(np.float32)


def make_white_noise(size=16, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.permutation(size * size).reshape(size, size) / (size * size)).astype(np.float32)

def make_blue_noise(size=64, seed=123):
    rng = np.random.default_rng(seed)
    points = rng.random((size, size)) < 0.3
    try:
        from scipy.ndimage import gaussian_filter
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
    if HAS_SCIPY:
        for _ in range(30):
            density = gaussian_filter(points.astype(np.float32), sigma=1.5)
            void = density.argmin()
            cluster = density.argmax()
            vy, vx = np.unravel_index(void, points.shape)
            cy, cx = np.unravel_index(cluster, points.shape)
            points[vy, vx] = True
            points[cy, cx] = False
    flat = points.flatten().astype(np.float32)
    order = np.argsort(flat)
    matrix = np.zeros(size * size, dtype=np.float32)
    matrix[order] = np.linspace(0, 1, size * size)
    return matrix.reshape(size, size)


def make_pattern_matrix(kind="waves", size=8):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == "waves":
        return (0.5 + 0.5 * np.sin(xx * 1.5 + yy * 0.8)).astype(np.float32)
    if kind == "honeycomb":
        u = (xx % 3) / 3
        v = (yy % 3) / 3
        return np.abs(u - v).astype(np.float32)
    if kind == "stars":
        cx, cy = size / 2, size / 2
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        return (0.5 + 0.5 * np.cos(d * 1.2)).astype(np.float32)
    if kind == "circuit":
        return ((xx // 2 + yy // 2) % 2).astype(np.float32)
    return np.random.rand(size, size).astype(np.float32)


def diffuse_dither(gray, kernel, strength=1.0, serpentine=True):
    img = gray.astype(np.float32).copy()
    h, w = img.shape
    for y in range(h):
        if serpentine and y % 2 == 1:
            xs = range(w - 1, -1, -1); sgn = -1
        else:
            xs = range(w); sgn = 1
        row = img[y]
        for x in xs:
            old = row[x]
            new = 255.0 if old >= 128.0 else 0.0
            row[x] = new
            err = (old - new) * strength
            if err == 0.0:
                continue
            for dy, dx, f in kernel:
                ny, nx = y + dy, x + dx * sgn
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += err * f
    return np.clip(img, 0, 255).astype(np.uint8)


def ordered_dither(gray, matrix, strength=1.0):
    h, w = gray.shape
    th, tw = matrix.shape
    tile = np.tile(matrix, (h // th + 1, w // tw + 1))[:h, :w]
    effective = 0.5 + (tile - 0.5) * strength
    return np.where(gray.astype(np.float32) / 255.0 > effective, 255, 0).astype(np.uint8)


KERNELS = {
    "Floyd-Steinberg": [(0, 1, 7/16), (1, -1, 3/16), (1, 0, 5/16), (1, 1, 1/16)],
    "False Floyd": [(0, 1, 3/8), (1, 0, 3/8), (1, 1, 2/8)],
    "Atkinson": [(0, 1, 1/8), (0, 2, 1/8), (1, -1, 1/8), (1, 0, 1/8), (1, 1, 1/8), (2, 0, 1/8)],
    "Jarvis": [
        (0, 1, 7/48), (0, 2, 5/48),
        (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48),
        (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48),
    ],
    "Stucki": [
        (0, 1, 8/42), (0, 2, 4/42),
        (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42),
        (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42),
    ],
    "Burkes": [(0, 1, 8/32), (0, 2, 4/32), (1, -2, 2/32), (1, -1, 4/32),
               (1, 0, 8/32), (1, 1, 4/32), (1, 2, 2/32)],
    "Sierra": [
        (0, 1, 5/32), (0, 2, 3/32),
        (1, -2, 2/32), (1, -1, 4/32), (1, 0, 5/32), (1, 1, 4/32), (1, 2, 2/32),
        (2, -1, 2/32), (2, 0, 3/32), (2, 1, 2/32),
    ],
    "Two-Row Sierra": [(0, 1, 4/16), (0, 2, 3/16), (1, -2, 1/16), (1, -1, 2/16),
                       (1, 0, 3/16), (1, 1, 2/16), (1, 2, 1/16)],
    "Sierra Lite": [(0, 1, 2/4), (1, -1, 1/4), (1, 0, 1/4)],
}

BAYER_MATRICES = {
    "Bayer 2x2": generate_bayer(2).astype(np.float32) / 4.0,
    "Bayer 4x4": generate_bayer(4).astype(np.float32) / 16.0,
    "Bayer 8x8": generate_bayer(8).astype(np.float32) / 64.0,
    "Bayer 16x16": generate_bayer(16).astype(np.float32) / 256.0,
    "Halftone Dot": make_halftone_dot(8),
    "Halftone Line": make_halftone_line(8),
    "White Noise": make_white_noise(16),
    "Blue Noise 64": make_blue_noise(64),
    "Pattern Waves": make_pattern_matrix("waves", 8),
    "Pattern Honeycomb": make_pattern_matrix("honeycomb", 8),
    "Pattern Stars": make_pattern_matrix("stars", 8),
    "Pattern Circuit": make_pattern_matrix("circuit", 8),
}

PALETTES = {
    "1-bit": [(0, 0, 0), (255, 255, 255)],
    "Game Boy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
    "GB Pocket": [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)],
    "CGA": [
        (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
        (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
        (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
        (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
    ],
    "C64": [
        (0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238),
        (204, 68, 204), (0, 204, 85), (0, 0, 170), (238, 238, 119),
        (221, 136, 85), (102, 68, 0), (255, 119, 119), (51, 51, 51),
        (119, 119, 119), (170, 255, 102), (0, 136, 255), (187, 187, 187),
    ],
    "PICO-8": [
        (0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
        (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
        (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
        (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170),
    ],
    "Sweetie 16": [
        (26, 28, 44), (93, 39, 93), (177, 62, 83), (239, 125, 87),
        (255, 205, 117), (167, 240, 112), (56, 183, 100), (37, 113, 121),
        (41, 54, 111), (59, 93, 201), (65, 166, 246), (115, 239, 247),
        (244, 244, 244), (148, 176, 194), (86, 108, 134), (51, 60, 87),
    ],
    "Sepia": [(50, 30, 20), (90, 60, 35), (135, 100, 65), (180, 140, 100), (220, 195, 160)],
    "NES": [
        (124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188),
        (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0),
        (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0),
        (0, 64, 88), (0, 0, 0), (188, 188, 188), (0, 120, 248),
    ],
}