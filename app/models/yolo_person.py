import numpy as np

from .common import create_session, letterbox


class YoloPersonDetector:
    """YOLO26s (end2end, NMS da duoc nhung san trong model) - chi lay class 'person'."""

    INPUT_SIZE = 640
    PERSON_CLASS_ID = 0

    def __init__(self, model_path):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame_bgr, conf_threshold=0.45):
        img, r, (left, top) = letterbox(frame_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        chw = rgb.transpose(2, 0, 1)[None].astype(np.float32)

        out = self.session.run([self.output_name], {self.input_name: chw})[0]
        rows = out[0]  # [300, 6] -> x1,y1,x2,y2,conf,cls (toa do trong khong gian 640x640)

        h, w = frame_bgr.shape[:2]
        results = []
        for x1, y1, x2, y2, conf, cls in rows:
            if conf < conf_threshold or int(round(float(cls))) != self.PERSON_CLASS_ID:
                continue
            x1o = max(0.0, min(w - 1, (x1 - left) / r))
            y1o = max(0.0, min(h - 1, (y1 - top) / r))
            x2o = max(0.0, min(w - 1, (x2 - left) / r))
            y2o = max(0.0, min(h - 1, (y2 - top) / r))
            if x2o <= x1o or y2o <= y1o:
                continue
            results.append((float(x1o), float(y1o), float(x2o), float(y2o), float(conf)))
        return results
