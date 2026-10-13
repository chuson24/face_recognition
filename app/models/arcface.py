import cv2
import numpy as np

from .common import create_session


class ArcFaceRecognizer:
    """ArcFace ResNet50 - crop mat 112x112 -> embedding 512 chieu (da L2-normalize)."""

    INPUT_SIZE = 112

    def __init__(self, model_path):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, face_bgr):
        img = cv2.resize(face_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        img = (img - 127.5) / 127.5
        chw = img.transpose(2, 0, 1)[None].astype(np.float32)

        out = self.session.run([self.output_name], {self.input_name: chw})[0][0]
        norm = np.linalg.norm(out) + 1e-9
        return (out / norm).astype(np.float32)
