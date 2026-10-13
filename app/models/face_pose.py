import cv2
import numpy as np

from .common import create_session


class FacePoseEstimator:
    """MTCNN ONet (xuat sang ONNX tu facenet-pytorch) - 5 diem landmark tren 1 box mat.

    Dung de uoc luong goc quay ngang (yaw) cua mat khi dang ky, vi model SCRFD
    dang dung khong co output landmark.
    """

    INPUT_SIZE = 48

    def __init__(self, model_path):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def landmarks(self, frame_bgr, box):
        """Tra ve (points[5,2] toa do trong frame, prob) hoac (None, 0.0).

        Thu tu diem: mat trai anh, mat phai anh, mui, mieng trai, mieng phai.
        """
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        side = max(w, h)  # ONet duoc train voi box vuong (rerec trong MTCNN)
        sx1 = x1 + w / 2 - side / 2
        sy1 = y1 + h / 2 - side / 2
        ix1, iy1 = int(round(sx1)), int(round(sy1))
        ix2, iy2 = ix1 + int(round(side)), iy1 + int(round(side))

        fh, fw = frame_bgr.shape[:2]
        # Pad khi box vuong vuot ra ngoai frame, giu dung ty le.
        pad = [max(0, -iy1), max(0, iy2 - fh), max(0, -ix1), max(0, ix2 - fw)]
        crop = frame_bgr[max(0, iy1):min(fh, iy2), max(0, ix1):min(fw, ix2)]
        if crop.size == 0:
            return None, 0.0
        if any(pad):
            crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_CONSTANT, value=0)

        img = cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        img = (img[:, :, ::-1].astype(np.float32) - 127.5) * 0.0078125  # BGR->RGB
        chw = img.transpose(2, 0, 1)[None]

        _reg, pts, prob = self.session.run(None, {self.input_name: chw})
        pts, prob = pts[0], float(prob[0][1])
        size = ix2 - ix1
        xs = pts[:5] * size + ix1
        ys = pts[5:] * size + iy1
        return np.stack([xs, ys], axis=1), prob

    @staticmethod
    def yaw_score(points):
        """Uoc luong goc quay ngang tu vi tri mui so voi 2 mat.

        Tra ve so trong khoang ~[-1, 1]: 0 = chinh dien. Gia tri DUONG = mui lech
        ve phia ben PHAI cua anh = nguoi dang quay mat sang TRAI cua ho
        (camera khong lat guong).
        """
        le, re, nose = points[0], points[1], points[2]
        eye_dx = re[0] - le[0]
        if eye_dx <= 1e-3:
            # Mat nghieng qua nhieu (2 mat chong len nhau) -> ra bien.
            return 1.0 if nose[0] > (le[0] + re[0]) / 2 else -1.0
        ratio = (nose[0] - le[0]) / eye_dx  # 0.5 = mui nam giua 2 mat
        return float(np.clip((ratio - 0.5) * 2.0, -1.5, 1.5))

    @staticmethod
    def nose_vertical_ratio(points):
        """Vi tri doc cua mui giua duong mat (0) va duong mieng (1).

        Ngang mat len -> dau mui nho ra phia truoc nen bi day len gan mat ->
        ty le GIAM; cui xuong -> ty le TANG. Gia tri chinh dien khac nhau theo
        tung nguoi (~0.5-0.65) nen can so voi moc rieng cua nguoi do.
        """
        eye_y = (points[0][1] + points[1][1]) / 2
        mouth_y = (points[3][1] + points[4][1]) / 2
        span = mouth_y - eye_y
        if span <= 1e-3:
            return None
        return float((points[2][1] - eye_y) / span)
