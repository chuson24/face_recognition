import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "deployed"
DATA_DIR = BASE_DIR / "data"
ENROLL_DIR = DATA_DIR / "enroll_captures"
FACE_DB_PATH = DATA_DIR / "face_db.json"
# Bo embedding export tu he thong khac (chi doc), nap them vao luc khoi dong.
IMPORTED_EMBEDDINGS_DIR = DATA_DIR / "imported_embeddings"
RTSP_FILE = BASE_DIR / "rtsp.txt"

YOLO_MODEL_PATH = MODELS_DIR / "yolo_detector.onnx"
SCRFD_MODEL_PATH = MODELS_DIR / "scrfd_detector.onnx"
ARCFACE_MODEL_PATH = MODELS_DIR / "arcface_r50.onnx"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ENROLL_DIR.mkdir(parents=True, exist_ok=True)

RTSP_HEADER = (
    "# Nhap duong link RTSP camera CCTV vao dong KHONG bat dau bang #\n"
    "# Vi du: rtsp://admin:matkhau@192.168.1.64:554/Streaming/Channels/101\n"
    "# Dung so 0 de test nhanh bang webcam laptop (khi chua co link RTSP that)\n"
)

# Gioi han tai nhan dien de luong video khong bi keo cham (CPU chay SCRFD +
# ArcFace mat ~350 ms cho moi nguoi).
MAX_FACE_CHECKS_PER_CYCLE = 2  # so track toi da chay SCRFD+ArcFace moi lan infer
UNKNOWN_RETRY_FRAMES = 6  # track "Unknown"/chua thay mat: thu lai sau N lan infer

ONET_MODEL_PATH = MODELS_DIR / "onet_landmarks.onnx"

# Dang ky nhieu goc mat (giong Face ID).
#   yaw   ~ [-1, 1]: 0 = chinh dien, duong = quay sang trai cua nguoi do.
#   pitch ~ tan(goc ngang/cui): 0 = chinh dien, duong = ngang len.
ENROLL_SIDE_RATIO = 0.2  # ty le anh moi ben trai/phai
ENROLL_VERT_RATIO = 0.15  # ty le anh moi huong len/xuong; con lai la chinh dien
ENROLL_FRONT_MAX_YAW = 0.15  # |yaw| <= nguong nay -> chinh dien
ENROLL_SIDE_MIN_YAW = 0.25  # |yaw| trong [min, max] -> nghieng trai/phai
ENROLL_SIDE_MAX_YAW = 1.0  # qua nguong nay -> quay qua nhieu (gan nhu nhin ngang)
ENROLL_FRONT_MAX_PITCH = 0.2  # |pitch| <= nguong nay -> chinh dien (~11 do)
ENROLL_VERT_MIN_PITCH = 0.3  # |pitch| trong [min, max] -> ngang len / cui xuong
ENROLL_VERT_MAX_PITCH = 1.0  # (~17-45 do)
ENROLL_VERT_MAX_YAW = 0.25  # khi chup len/xuong khong duoc quay ngang qua muc nay
ENROLL_SIDE_MAX_PITCH = 0.3  # khi chup trai/phai khong duoc ngang/cui qua muc nay
# pitch = (moc - ty_le_doc_mui) / ENROLL_PITCH_SCALE. Moc ban dau la gia tri
# trung binh; sau khi co du anh chinh dien thi dung trung vi cua chinh nguoi do.
ENROLL_PITCH_PRIOR = 0.58
ENROLL_PITCH_SCALE = 0.6
ENROLL_MIN_FRONT_FOR_BASE = 3  # so anh chinh dien can co truoc khi chup len/xuong
ENROLL_MAX_ROLL_DEG = 25  # dau nghieng sang vai qua nhieu -> bo
ENROLL_MIN_LANDMARK_PROB = 0.6  # ONet xac nhan day dung la mat (loc nhan nham)

# Anh MJPEG gui len trinh duyet: thu nho de encode JPEG nhanh hon.
STREAM_MAX_WIDTH = 1280
STREAM_JPEG_QUALITY = 80

DEFAULT_SETTINGS = {
    "person_conf_threshold": 0.45,
    "face_conf_threshold": 0.5,
    "min_face_size": 60,
    "similarity_threshold": 0.40,
    "process_every_n_frames": 2,
    "recheck_interval_frames": 60,
    "enroll_min_face_size": 75,
    "enroll_min_sharpness": 60.0,
    "gallery_mode": "all",  # all | local | imported
}

SETTINGS_CHOICES = {"gallery_mode": ("all", "local", "imported")}

SETTINGS_RANGES = {
    "person_conf_threshold": (0.1, 0.95),
    "face_conf_threshold": (0.1, 0.95),
    "min_face_size": (20, 300),
    "similarity_threshold": (0.1, 0.9),
    "process_every_n_frames": (1, 10),
    "recheck_interval_frames": (5, 600),
    "enroll_min_face_size": (40, 300),
    "enroll_min_sharpness": (0.0, 500.0),
}


def read_rtsp_source():
    if not RTSP_FILE.exists():
        RTSP_FILE.write_text(RTSP_HEADER + "0\n", encoding="utf-8")
    for line in RTSP_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return "0"


def write_rtsp_source(value):
    value = value.strip()
    RTSP_FILE.write_text(RTSP_HEADER + value + "\n", encoding="utf-8")


class Settings:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = dict(DEFAULT_SETTINGS)

    def get_all(self):
        with self._lock:
            return dict(self._values)

    def get(self, key):
        with self._lock:
            return self._values[key]

    def update(self, partial):
        with self._lock:
            for key, value in partial.items():
                if key not in self._values:
                    continue
                if key in SETTINGS_CHOICES:
                    if value in SETTINGS_CHOICES[key]:
                        self._values[key] = value
                    continue
                caster = type(self._values[key])
                value = caster(value)
                lo, hi = SETTINGS_RANGES.get(key, (None, None))
                if lo is not None:
                    value = max(lo, min(hi, value))
                self._values[key] = value
            return dict(self._values)


settings = Settings()
