import numpy as np

from .common import create_session, letterbox


class ScrfdFaceDetector:
    """SCRFD - phat hien khuon mat. Model nay khong co output landmark (5 diem)."""

    INPUT_SIZE = 640
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, model_path):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._anchor_cache = {}

    def _anchors_for_stride(self, stride):
        if stride in self._anchor_cache:
            return self._anchor_cache[stride]
        fm = self.INPUT_SIZE // stride
        ys, xs = np.mgrid[:fm, :fm]
        centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centers = centers.reshape(-1, 2)
        centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)
        self._anchor_cache[stride] = centers
        return centers

    def detect(self, frame_bgr, conf_threshold=0.5, nms_threshold=0.4):
        if frame_bgr.size == 0:
            return []

        img, r, (left, top) = letterbox(frame_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        blob = (img[:, :, ::-1].astype(np.float32) - 127.5) / 128.0  # BGR->RGB + normalize
        chw = blob.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = self.session.run(self.output_names, {self.input_name: chw})
        out_map = dict(zip(self.output_names, outputs))

        all_boxes, all_scores = [], []
        for stride in self.STRIDES:
            scores = out_map[f"score_{stride}"].reshape(-1)
            bbox_preds = out_map[f"bbox_{stride}"].reshape(-1, 4) * stride
            centers = self._anchors_for_stride(stride)

            keep = scores >= conf_threshold
            if not np.any(keep):
                continue

            c, b, s = centers[keep], bbox_preds[keep], scores[keep]
            x1 = c[:, 0] - b[:, 0]
            y1 = c[:, 1] - b[:, 1]
            x2 = c[:, 0] + b[:, 2]
            y2 = c[:, 1] + b[:, 3]
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            all_scores.append(s)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep_idx = self._nms(boxes, scores, nms_threshold)

        h, w = frame_bgr.shape[:2]
        results = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes[i]
            x1o = max(0.0, min(w - 1, (x1 - left) / r))
            y1o = max(0.0, min(h - 1, (y1 - top) / r))
            x2o = max(0.0, min(w - 1, (x2 - left) / r))
            y2o = max(0.0, min(h - 1, (y2 - top) / r))
            if x2o <= x1o or y2o <= y1o:
                continue
            results.append((float(x1o), float(y1o), float(x2o), float(y2o), float(scores[i])))
        return results

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep
