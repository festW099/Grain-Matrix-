from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoSource:
    def __init__(self, path):
        if not HAS_CV2:
            raise RuntimeError(
                "OpenCV не установлен.\nУстановите командой:\n"
                "    pip install opencv-python"
            )
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть видеофайл.")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, idx):
        idx = max(0, min(self.frame_count - 1, int(idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None