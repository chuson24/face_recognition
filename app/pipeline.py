import threading
import time
import unicodedata
from collections import deque

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from . import config
from .camera import camera_stream
from .face_db import face_db
from .models.arcface import ArcFaceRecognizer
from .models.face_pose import FacePoseEstimator
from .models.scrfd_face import ScrfdFaceDetector
from .models.yolo_person import YoloPersonDetector

yolo = YoloPersonDetector(config.YOLO_MODEL_PATH)
scrfd = ScrfdFaceDetector(config.SCRFD_MODEL_PATH)
arcface = ArcFaceRecognizer(config.ARCFACE_MODEL_PATH)
face_pose = FacePoseEstimator(config.ONET_MODEL_PATH)
tracker = ByteTrackTracker()

# _track_state chi duoc doc/ghi trong luong infer. Luong hien thi chi doc ban
# snapshot _overlay (detections + ban sao track_state) de ve len frame.
_track_state = {}  # track_id -> {label, similarity, last_check, face_box}

_overlay_lock = threading.Lock()
_overlay = (sv.Detections.empty(), {})

# JPEG da encode san cho MJPEG stream: encode 1 lan / frame, dung chung cho
# moi trinh duyet dang xem.
_jpeg_cond = threading.Condition()
_jpeg_bytes = None
_jpeg_seq = 0

_enroll_lock = threading.Lock()
_enroll_state = {
    "active": False,
    "auto": True,
    "name": None,
    "target": 0,
    "captured": 0,
    "last_capture_frame": -999,
    "thumb": None,
    "embeddings": [],
    "poses": {},  # goc -> so anh da chup
    "quotas": {},  # goc -> so anh can chup
    "front_ratios": [],  # ty le doc cua mui o cac anh chinh dien (moc pitch)
    "last_check": None,  # ket qua kiem tra gan nhat (hien thi tren web)
    "reject_counts": {},  # ly do -> so lan bi loai
    "last_reason": {},  # loai ly do -> thoi diem ghi log gan nhat
    "log": deque(maxlen=80),
}
_ENROLL_GAP_FRAMES = 5  # tu dong chup: cach nhau toi thieu N frame camera
_MANUAL_MIN_FACE = 40  # chup thu cong: chi tu choi mat qua nho (px)
_LOG_REPEAT_SEC = 3.0  # cung 1 loai ly do bi loai: ghi log toi da 1 lan / N giay

# Yeu cau chup thu cong tu API -> luong infer xu ly o frame ke tiep.
_manual_lock = threading.Lock()
_manual_request = None  # {"event": Event, "result": dict}

# Khung mat dang xet khi dang ky, ve len video (do = bi loai, xanh = dat).
_enroll_overlay = None  # {"box", "ok", "text", "t"}

_started = False
_start_lock = threading.Lock()


def _laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _imwrite_unicode(path, image):
    # cv2.imwrite loi am tham (tra ve False) voi duong dan co ky tu Unicode
    # tren Windows (vd: thu muc du an co dau tieng Viet) -> dung imencode + ghi
    # file bang Python de an toan.
    ok, buf = cv2.imencode(".jpg", image)
    if ok:
        buf.tofile(str(path))
    return ok


def _ascii(text):
    # cv2.putText chi ve duoc ASCII: bo dau tieng Viet de ten khong thanh "???".
    text = text.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _make_placeholder(text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


# ---------------------------------------------------------------- enrollment
#
# Dang ky nhieu goc giong Face ID: chia so anh thanh 5 nhom chinh dien / quay
# trai / quay phai / ngang len / cui xuong (trai-phai tinh theo NGUOI dung truoc
# camera). Goc mat uoc luong tu 5 diem landmark cua ONet:
#   - yaw (ngang): vi tri mui so voi 2 mat.
#   - pitch (doc): vi tri doc cua mui giua mat va mieng, so voi MOC rieng cua
#     nguoi do (trung vi cac anh chinh dien da chup) -> phai chup chinh dien
#     truoc (>= ENROLL_MIN_FRONT_FOR_BASE anh) moi chup duoc len/xuong.
# Huong dan nguoi dung lan luot theo thu tu POSES, nhung goc nao con thieu ma
# xuat hien truoc thi van chup.

POSES = [
    ("front", "chinh dien", "Nhin THANG vao camera"),
    ("left", "quay trai", "Tu tu quay mat sang TRAI (cua ban) khoang 30-45 do"),
    ("right", "quay phai", "Tu tu quay mat sang PHAI (cua ban) khoang 30-45 do"),
    ("up", "ngang len", "Tu tu NGANG mat LEN khoang 20-30 do (mat van huong ve camera)"),
    ("down", "cui xuong", "Tu tu CUI mat XUONG khoang 20-30 do"),
]
_POSE_LABEL = {k: label for k, label, _ in POSES}
_POSE_HINT = {k: hint for k, _, hint in POSES}


def _pose_quotas(target):
    side = max(1, round(target * config.ENROLL_SIDE_RATIO))
    vert = max(1, round(target * config.ENROLL_VERT_RATIO))
    return {"front": target - 2 * side - 2 * vert, "left": side, "right": side,
            "up": vert, "down": vert}


def _pitch_base():
    """Moc ty le doc cua mui khi nhin thang. Phai goi khi DANG giu _enroll_lock.

    Tra ve (moc, da_hieu_chinh): dung trung vi anh chinh dien cua chinh nguoi
    dang ky khi da du anh, neu chua thi dung gia tri trung binh.
    """
    ratios = _enroll_state["front_ratios"]
    if len(ratios) >= config.ENROLL_MIN_FRONT_FOR_BASE:
        return float(np.median(ratios)), True
    return config.ENROLL_PITCH_PRIOR, False


def _next_pose(counts, quotas):
    """Goc tiep theo can chup theo thu tu huong dan, None neu da du."""
    for key, _label, _hint in POSES:
        if counts.get(key, 0) < quotas.get(key, 0):
            return key
    return None


def _log_enroll(msg, level="info"):
    """Ghi 1 dong log dang ky. Phai goi khi DANG giu _enroll_lock."""
    _enroll_state["log"].append({"t": time.strftime("%H:%M:%S"), "level": level, "msg": msg})


def start_enrollment(name, target, auto=True):
    name = name.strip()
    if not name:
        raise ValueError("Ten khong duoc de trong")
    target = max(10, min(50, int(target)))
    quotas = _pose_quotas(target)
    with _enroll_lock:
        _enroll_state.update(
            {
                "active": True,
                "auto": bool(auto),
                "name": name,
                "target": target,
                "captured": 0,
                "last_capture_frame": -999,
                "thumb": None,
                "embeddings": [],
                "poses": {k: 0 for k in quotas},
                "quotas": quotas,
                "front_ratios": [],
                "last_check": None,
                "reject_counts": {},
                "last_reason": {},
            }
        )
        _enroll_state["log"].clear()
        mode = "tu dong + thu cong" if auto else "chi chup thu cong"
        plan = ", ".join(f"{_POSE_LABEL[k]} {q}" for k, q in quotas.items())
        _log_enroll(f"Bat dau dang ky '{name}': {target} anh ({plan}) - {mode}")
        _log_enroll(f"Buoc 1: {_POSE_HINT['front']}")


def cancel_enrollment():
    with _enroll_lock:
        if _enroll_state["active"]:
            _log_enroll("Da huy dang ky", "warn")
        _enroll_state["active"] = False


def set_enroll_auto(auto):
    with _enroll_lock:
        _enroll_state["auto"] = bool(auto)
        if _enroll_state["active"]:
            _log_enroll("Bat tu dong chup" if auto else "Tat tu dong chup (chi chup thu cong)")


def enrollment_status():
    with _enroll_lock:
        last = _enroll_state["last_check"]
        if last is not None:
            last = dict(last, age=round(time.time() - last["t"], 1))
        counts = _enroll_state.get("poses", {})
        quotas = _enroll_state.get("quotas", {})
        nxt = _next_pose(counts, quotas) if quotas else None
        if not _enroll_state["active"]:
            hint = None
        elif nxt is None:
            hint = "Da du anh"
        else:
            hint = _POSE_HINT[nxt]
        return {
            "active": _enroll_state["active"],
            "auto": _enroll_state["auto"],
            "name": _enroll_state["name"],
            "target": _enroll_state["target"],
            "captured": _enroll_state["captured"],
            "thumb": _enroll_state["thumb"],
            "poses": [
                {"key": k, "label": label, "captured": counts.get(k, 0), "quota": quotas.get(k, 0)}
                for k, label, _ in POSES
            ],
            "next_pose": nxt,
            "hint": hint,
            "last_check": last,
            "reject_counts": dict(_enroll_state["reject_counts"]),
            "log": list(_enroll_state["log"])[::-1],  # moi nhat len dau
        }


def request_manual_capture(timeout=3.0):
    """Goi tu API: yeu cau luong infer chup ngay o frame ke tiep, cho ket qua."""
    global _manual_request
    with _enroll_lock:
        if not _enroll_state["active"]:
            return {"ok": False, "msg": "Chua bat dau dang ky - nhap ten va bam Bat dau truoc"}
    req = {"event": threading.Event(), "result": None}
    with _manual_lock:
        _manual_request = req
    if not req["event"].wait(timeout):
        with _manual_lock:
            if _manual_request is req:
                _manual_request = None
        return {"ok": False, "msg": "Het thoi gian cho - camera/AI khong phan hoi"}
    return req["result"]


def _classify_pose(yaw, pitch, base_ready, manual):
    """Xep goc mat vao 1 nhom trong POSES, None neu o vung trung gian."""
    ay, ap = abs(yaw), abs(pitch)
    side = "left" if yaw > 0 else "right"  # yaw > 0: nguoi quay sang trai cua ho
    vert = "up" if pitch > 0 else "down"
    if manual:
        # Thu cong: xep vao nhom gan nhat, khong co vung "trung gian".
        if (ay <= (config.ENROLL_FRONT_MAX_YAW + config.ENROLL_SIDE_MIN_YAW) / 2
                and ap <= (config.ENROLL_FRONT_MAX_PITCH + config.ENROLL_VERT_MIN_PITCH) / 2):
            return "front"
        if ay / config.ENROLL_SIDE_MIN_YAW >= ap / config.ENROLL_VERT_MIN_PITCH or not base_ready:
            return side
        return vert
    if ay <= config.ENROLL_FRONT_MAX_YAW and ap <= config.ENROLL_FRONT_MAX_PITCH:
        return "front"
    if (config.ENROLL_SIDE_MIN_YAW <= ay <= config.ENROLL_SIDE_MAX_YAW
            and ap <= config.ENROLL_SIDE_MAX_PITCH):
        return side
    if (base_ready and ay <= config.ENROLL_VERT_MAX_YAW
            and config.ENROLL_VERT_MIN_PITCH <= ap <= config.ENROLL_VERT_MAX_PITCH):
        return vert
    return None


def _pose_reason(yaw, pitch, pose, base_ready, counts, quotas):
    """Ly do bi loai lien quan den goc mat (auto), None neu goc nay can chup."""
    nxt = _next_pose(counts, quotas)
    angles = f"yaw {yaw:+.2f}, pitch {pitch:+.2f}"
    if pose is None:
        if abs(yaw) > config.ENROLL_SIDE_MAX_YAW:
            return f"Quay ngang qua nhieu ({angles}) - quay lai ve phia camera mot chut"
        if abs(pitch) > config.ENROLL_VERT_MAX_PITCH:
            return f"Ngang/cui qua nhieu ({angles}) - dua mat ve gan chinh dien mot chut"
        if not base_ready and abs(pitch) >= config.ENROLL_VERT_MIN_PITCH:
            return f"Can chup chinh dien truoc ({angles}) - {_POSE_HINT['front']}"
        # Vung trung gian giua cac goc.
        if nxt == "front":
            return f"Mat chua thang ({angles}) - nhin thang vao camera"
        if nxt is not None:
            return f"Chua dung goc ({angles}) - {_POSE_HINT[nxt]}"
        return f"Goc trung gian ({angles})"
    if counts.get(pose, 0) >= quotas.get(pose, 0):
        hint = _POSE_HINT[nxt] if nxt else "chup xong"
        return (f"Da du anh {_POSE_LABEL[pose]} ({counts.get(pose, 0)}/{quotas.get(pose, 0)})"
                f" - {hint}")
    return None


def _evaluate_enroll_face(frame, s, manual):
    """Tim mat lon nhat trong frame va kiem tra dieu kien chup.

    Tra ve dict: ok, reason (None neu dat), box, size, score, sharpness, yaw,
    roll, pose, faces, crop. Che do thu cong chi yeu cau co mat >= _MANUAL_MIN_FACE
    px va goc mat con thieu anh; cac nguong khac chi dua ra canh bao.
    """
    info = {"ok": False, "reason": None, "warnings": [], "box": None, "size": None,
            "score": None, "sharpness": None, "yaw": None, "pitch": None, "roll": None,
            "nose_ratio": None, "pose": None, "faces": 0, "crop": None}

    # Detect voi nguong thap de con bao duoc "do tin cay thap" thay vi "khong thay mat".
    faces = scrfd.detect(frame, conf_threshold=min(0.3, s["face_conf_threshold"]))
    info["faces"] = len(faces)
    if not faces:
        info["reason"] = "Khong thay khuon mat nao trong khung hinh"
        return info

    faces.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    x1, y1, x2, y2, score = faces[0]
    fw, fh = x2 - x1, y2 - y1
    size = int(min(fw, fh))
    info.update(box=(x1, y1, x2, y2), size=size, score=round(float(score), 2))

    mx, my = fw * 0.2, fh * 0.2
    xi1, yi1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    xi2, yi2 = min(frame.shape[1], int(x2 + mx)), min(frame.shape[0], int(y2 + my))
    crop = frame[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        info["reason"] = "Vung mat nam ngoai khung hinh"
        return info
    sharp = float(_laplacian_var(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    info.update(sharpness=round(sharp, 1), crop=crop)

    with _enroll_lock:
        counts = dict(_enroll_state["poses"])
        quotas = dict(_enroll_state["quotas"])
        base, base_ready = _pitch_base()

    pts, lm_prob = face_pose.landmarks(frame, (x1, y1, x2, y2))
    nose_ratio = face_pose.nose_vertical_ratio(pts) if pts is not None else None
    if nose_ratio is not None:
        yaw = face_pose.yaw_score(pts)
        pitch = (base - nose_ratio) / config.ENROLL_PITCH_SCALE
        dx, dy = pts[1] - pts[0]
        roll = float(np.degrees(np.arctan2(dy, dx)))
        info.update(yaw=round(yaw, 2), pitch=round(pitch, 2), roll=round(roll, 1),
                    nose_ratio=nose_ratio,
                    pose=_classify_pose(yaw, pitch, base_ready, manual))
    else:
        pts = None

    checks = [
        (score >= s["face_conf_threshold"],
         f"Do tin cay mat thap ({score:.2f} < {s['face_conf_threshold']:.2f}) - nhin ve phia camera"),
        (size >= s["enroll_min_face_size"],
         f"Mat qua nho ({size}px < {int(s['enroll_min_face_size'])}px) - lai gan camera hon"),
        (sharp >= s["enroll_min_sharpness"],
         f"Anh bi mo (do net {sharp:.0f} < {s['enroll_min_sharpness']:.0f}) - dung yen"),
        (pts is not None and lm_prob >= config.ENROLL_MIN_LANDMARK_PROB,
         f"Khong xac dinh duoc mat/mui (p {lm_prob:.2f}) - mat bi che hoac nhan nham"),
        (info["roll"] is None or abs(info["roll"]) <= config.ENROLL_MAX_ROLL_DEG,
         f"Dau nghieng sang vai ({info['roll'] or 0:.0f} do) - giu dau thang"),
    ]
    failed = [msg for passed, msg in checks if not passed]

    if manual:
        if size < _MANUAL_MIN_FACE:
            info["reason"] = f"Mat qua nho de chup ({size}px < {_MANUAL_MIN_FACE}px)"
            return info
        if info["pose"] is None:
            info["reason"] = "Khong uoc luong duoc goc mat"
            return info
        if counts[info["pose"]] >= quotas[info["pose"]]:
            nxt = _next_pose(counts, quotas)
            info["reason"] = (f"Da du anh {_POSE_LABEL[info['pose']]} - "
                              f"{_POSE_HINT[nxt] if nxt else 'chup xong'}")
            return info
        info["warnings"] = failed
    else:
        if failed:
            info["reason"] = failed[0]
            return info
        reason = _pose_reason(info["yaw"], info["pitch"], info["pose"], base_ready,
                              counts, quotas)
        if reason:
            info["reason"] = reason
            return info

    if len(faces) > 1:
        info["warnings"].append(f"Co {len(faces)} khuon mat - dang lay mat LON NHAT")
    info["ok"] = True
    return info


def _save_enroll_capture(frame_idx, info, manual):
    """Luu anh + embedding. Tra ve (ok, message)."""
    pose = info["pose"]
    with _enroll_lock:
        name = _enroll_state["name"]
        idx = _enroll_state["captured"] + 1

    embedding = arcface.get_embedding(info["crop"])
    person_dir = config.ENROLL_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{idx:02d}_{pose}.jpg"
    if not _imwrite_unicode(person_dir / fname, info["crop"]):
        return False, f"Loi ghi file anh {fname}"

    with _enroll_lock:
        if not _enroll_state["active"] or _enroll_state["name"] != name:
            return False, "Dang ky da bi huy"
        counts, quotas = _enroll_state["poses"], _enroll_state["quotas"]
        _enroll_state["embeddings"].append(embedding)
        _enroll_state["captured"] = idx
        _enroll_state["last_capture_frame"] = frame_idx
        _enroll_state["thumb"] = f"/captures/{name}/{fname}"
        counts[pose] += 1
        if pose == "front":
            _was_ready = _pitch_base()[1]
            _enroll_state["front_ratios"].append(info["nose_ratio"])
            base, ready = _pitch_base()
            if ready and not _was_ready:
                _log_enroll(f"Da hieu chinh moc ngang/cui cho '{name}' (ty le mui {base:.2f})")

        kind = "thu cong" if manual else "tu dong"
        msg = (f"Chup anh {idx}/{_enroll_state['target']} - {_POSE_LABEL[pose]} "
               f"{counts[pose]}/{quotas[pose]} ({kind}) - mat {info['size']}px, "
               f"yaw {info['yaw']:+.2f}, pitch {info['pitch']:+.2f}, "
               f"do net {info['sharpness']:.0f}")
        if info["warnings"]:
            msg += " | Luu y: " + "; ".join(info["warnings"])
        _log_enroll(msg, "warn" if info["warnings"] else "ok")

        nxt = _next_pose(counts, quotas)
        if nxt is None:
            face_db.add_person(name, _enroll_state["embeddings"])
            _enroll_state["active"] = False
            _log_enroll(f"Hoan thanh! Da luu '{name}' voi {idx} anh", "ok")
        elif counts[pose] == quotas[pose]:
            _log_enroll(f"Xong goc {_POSE_LABEL[pose]}. Tiep theo: {_POSE_HINT[nxt]}", "ok")
    return True, msg


def _record_check(info, frame_idx):
    """Cap nhat trang thai kiem tra gan nhat, bo dem ly do, overlay va log."""
    global _enroll_overlay
    with _enroll_lock:
        _enroll_state["last_check"] = {
            "t": time.time(),
            "ok": info["ok"],
            "reason": info["reason"],
            "size": info["size"],
            "score": info["score"],
            "sharpness": info["sharpness"],
            "yaw": info["yaw"],
            "pitch": info["pitch"],
            "roll": info["roll"],
            "pose": info["pose"],
            "faces": info["faces"],
        }
        reason = info["reason"]
        if reason is not None:
            key = reason.split(" (")[0].split(" - ")[0]  # gom nhom theo loai ly do
            counts = _enroll_state["reject_counts"]
            counts[key] = counts.get(key, 0) + 1
            # Moi loai ly do ghi log toi da 1 lan / _LOG_REPEAT_SEC, tranh spam
            # khi ly do nhay qua lai giua cac frame.
            now = time.time()
            if now - _enroll_state["last_reason"].get(key, 0) >= _LOG_REPEAT_SEC:
                _log_enroll(f"Chua chup: {reason}", "warn")
                _enroll_state["last_reason"][key] = now

    if info["box"] is not None:
        yaw = (f" yaw {info['yaw']:+.2f} pitch {info['pitch']:+.2f}"
               if info["yaw"] is not None else "")
        pose = f" {info['pose'].upper()}" if info["pose"] else ""
        if info["ok"]:
            text = f"OK{pose} {info['size']}px{yaw}"
        else:
            text = f"{info['size']}px conf {info['score']:.2f}{yaw}"
        _enroll_overlay = {"box": info["box"], "ok": info["ok"], "text": text, "t": time.time()}


def _handle_enrollment(frame, frame_idx):
    global _manual_request
    with _manual_lock:
        manual_req, _manual_request = _manual_request, None

    with _enroll_lock:
        active = _enroll_state["active"]
        auto = _enroll_state["auto"]
        last_capture_frame = _enroll_state["last_capture_frame"]

    if not active:
        if manual_req is not None:
            manual_req["result"] = {"ok": False, "msg": "Chua bat dau dang ky"}
            manual_req["event"].set()
        return

    s = config.settings.get_all()

    if manual_req is not None:
        info = _evaluate_enroll_face(frame, s, manual=True)
        _record_check(info, frame_idx)
        if info["ok"]:
            ok, msg = _save_enroll_capture(frame_idx, info, manual=True)
        else:
            ok, msg = False, f"Khong chup duoc: {info['reason']}"
            with _enroll_lock:
                _log_enroll(f"Chup thu cong that bai: {info['reason']}", "error")
        manual_req["result"] = {"ok": ok, "msg": msg}
        manual_req["event"].set()
        return

    if not auto or frame_idx - last_capture_frame < _ENROLL_GAP_FRAMES:
        return

    info = _evaluate_enroll_face(frame, s, manual=False)
    _record_check(info, frame_idx)
    if info["ok"]:
        _save_enroll_capture(frame_idx, info, manual=False)


def _draw_enroll_overlay(frame):
    with _enroll_lock:
        active = _enroll_state["active"]
    ov = _enroll_overlay
    if not active or ov is None or time.time() - ov["t"] > 1.0:
        return
    x1, y1, x2, y2 = (int(v) for v in ov["box"])
    color = (0, 220, 0) if ov["ok"] else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, "DANG KY: " + ov["text"], (x1, y2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ------------------------------------------------------------- recognition


def _recognize_track(frame, frame_idx, box, state, s):
    x1, y1, x2, y2 = box
    state["last_check"] = frame_idx

    mx1, my1 = max(0, int(x1)), max(0, int(y1))
    mx2, my2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
    if mx2 <= mx1 or my2 <= my1:
        return
    person_crop = frame[my1:my2, mx1:mx2]
    if person_crop.size == 0:
        return

    faces = scrfd.detect(person_crop, conf_threshold=s["face_conf_threshold"])
    if not faces:
        return
    faces.sort(key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    fx1, fy1, fx2, fy2, _fscore = faces[0]
    fw, fh = fx2 - fx1, fy2 - fy1
    if min(fw, fh) < s["min_face_size"]:
        return

    state["face_box"] = (mx1 + fx1, my1 + fy1, mx1 + fx2, my1 + fy2)

    if not face_db.has_people(s["gallery_mode"]):
        # Chua dang ky ai -> khong can chay ArcFace (~180 ms tren CPU).
        state["label"] = "Unknown"
        state["similarity"] = 0.0
        return

    mx, my = fw * 0.2, fh * 0.2
    cx1, cy1 = max(0, int(fx1 - mx)), max(0, int(fy1 - my))
    cx2, cy2 = min(person_crop.shape[1], int(fx2 + mx)), min(
        person_crop.shape[0], int(fy2 + my)
    )
    face_crop = person_crop[cy1:cy2, cx1:cx2]
    if face_crop.size == 0:
        return

    embedding = arcface.get_embedding(face_crop)
    name, sim = face_db.match(embedding, s["gallery_mode"])
    if name is not None and sim >= s["similarity_threshold"]:
        state["label"] = name
        state["similarity"] = sim
    else:
        state["label"] = "Unknown"
        state["similarity"] = max(sim, 0.0)


def _process_tracks(frame, frame_idx, detections):
    s = config.settings.get_all()
    active_ids = set()
    due = []  # (last_check, index, track_id)

    for i in range(len(detections)):
        if detections.tracker_id is None:
            break
        track_id = int(detections.tracker_id[i])
        if track_id < 0:
            continue  # track chua duoc xac nhan (chua du so frame lien tiep)
        active_ids.add(track_id)

        state = _track_state.setdefault(
            track_id, {"label": None, "similarity": 0.0, "last_check": -999, "face_box": None}
        )
        known = state["label"] not in (None, "Unknown")
        interval = s["recheck_interval_frames"] if known else config.UNKNOWN_RETRY_FRAMES
        if frame_idx - state["last_check"] >= interval:
            due.append((state["last_check"], i, track_id))

    # Moi lan infer chi nhan dien toi da vai track, uu tien track lau chua
    # kiem tra nhat -> thoi gian moi vong infer khong tang theo so nguoi.
    due.sort()
    for _last, i, track_id in due[: config.MAX_FACE_CHECKS_PER_CYCLE]:
        _recognize_track(frame, frame_idx, detections.xyxy[i], _track_state[track_id], s)

    for tid in list(_track_state.keys()):
        if tid not in active_ids:
            del _track_state[tid]


def _draw_overlay(frame, detections, track_state, connected):
    for i in range(len(detections)):
        track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        if track_id < 0:
            continue  # track chua duoc xac nhan, chua ve
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        state = track_state.get(track_id, {})
        label = state.get("label")
        sim = state.get("similarity", 0.0)

        if label and label not in (None, "Unknown"):
            color, text = (0, 200, 0), f"ID{track_id} {_ascii(label)} {sim * 100:.0f}%"
        elif label == "Unknown":
            color, text = (0, 0, 220), f"ID{track_id} Unknown ({sim * 100:.0f}%)"
        else:
            color, text = (0, 200, 200), f"ID{track_id} ..."

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, text, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        face_box = state.get("face_box")
        if face_box is not None:
            fx1, fy1, fx2, fy2 = (int(v) for v in face_box)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 1)

    if not connected:
        cv2.putText(
            frame,
            "Mat ket noi camera - dang thu lai...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
    return frame


# ----------------------------------------------------------------- main loops
#
# Hai luong tach biet:
#   - _infer_loop: lay frame MOI NHAT tu camera, chay YOLO/SCRFD/ArcFace (cham,
#     ~0.1-0.5 s/vong tren CPU), cap nhat _overlay. Frame den trong luc dang
#     infer se bi bo qua -> khong bao gio bi don ung do tre.
#   - _display_loop: moi frame camera moi -> ve _overlay moi nhat len -> encode
#     JPEG 1 lan. Video muot theo FPS camera, khong phu thuoc toc do AI.


def _infer_loop():
    global _overlay

    last_id = None
    last_processed_id = -999
    while True:
        frame, frame_id = camera_stream.get_frame_with_id(last_id)
        if frame is None:
            time.sleep(0.01)
            continue
        s = config.settings.get_all()
        if frame_id - last_processed_id < max(1, s["process_every_n_frames"]):
            last_id = frame_id
            time.sleep(0.005)
            continue
        last_id = last_processed_id = frame_id

        try:
            _handle_enrollment(frame, frame_id)

            boxes = yolo.detect(frame, conf_threshold=s["person_conf_threshold"])
            if boxes:
                xyxy = np.array([b[:4] for b in boxes], dtype=np.float32)
                confidence = np.array([b[4] for b in boxes], dtype=np.float32)
                class_id = np.zeros(len(boxes), dtype=int)
                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            else:
                detections = sv.Detections.empty()

            detections = tracker.update(detections)
            _process_tracks(frame, frame_id, detections)
            snapshot = {tid: dict(st) for tid, st in _track_state.items()}
            with _overlay_lock:
                _overlay = (detections, snapshot)
        except Exception as exc:  # pragma: no cover - defensive, keep loop alive
            print(f"[pipeline] Loi xu ly frame: {exc}")
            time.sleep(0.1)


def _encode_jpeg(frame):
    h, w = frame.shape[:2]
    if w > config.STREAM_MAX_WIDTH:
        scale = config.STREAM_MAX_WIDTH / w
        frame = cv2.resize(
            frame, (config.STREAM_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _publish_jpeg(jpg):
    global _jpeg_bytes, _jpeg_seq
    if jpg is None:
        return
    with _jpeg_cond:
        _jpeg_bytes = jpg
        _jpeg_seq += 1
        _jpeg_cond.notify_all()


def _display_loop():
    placeholder = _encode_jpeg(_make_placeholder("Dang ket noi camera..."))
    last_id = None
    last_frame = None
    last_render = 0.0

    while True:
        frame, frame_id = camera_stream.get_frame_with_id(last_id)
        connected = camera_stream.is_connected()

        if frame is not None:
            last_id = frame_id
            last_frame = frame
        elif last_frame is None:
            if time.monotonic() - last_render > 0.5:
                _publish_jpeg(placeholder)
                last_render = time.monotonic()
            time.sleep(0.02)
            continue
        elif time.monotonic() - last_render < 0.5:
            # Chua co frame moi: cho; van ve lai moi 0.5 s de hien canh bao mat ket noi.
            time.sleep(0.005)
            continue

        with _overlay_lock:
            detections, track_state = _overlay
        try:
            annotated = _draw_overlay(last_frame.copy(), detections, track_state, connected)
            _draw_enroll_overlay(annotated)
        except Exception as exc:  # pragma: no cover
            annotated = last_frame.copy()
            print(f"[pipeline] Loi ve overlay: {exc}")
        _publish_jpeg(_encode_jpeg(annotated))
        last_render = time.monotonic()


def wait_for_jpeg(last_seq, timeout=1.0):
    """Cho den khi co JPEG moi hon last_seq. Tra ve (jpeg_bytes, seq)."""
    with _jpeg_cond:
        _jpeg_cond.wait_for(lambda: _jpeg_seq != last_seq, timeout=timeout)
        return _jpeg_bytes, _jpeg_seq


def get_annotated_jpeg():
    with _jpeg_cond:
        return _jpeg_bytes


def reload_camera(source):
    camera_stream.start(source)


def camera_connected():
    return camera_stream.is_connected()


def start():
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
        camera_stream.start(config.read_rtsp_source())
        threading.Thread(target=_infer_loop, daemon=True).start()
        threading.Thread(target=_display_loop, daemon=True).start()
