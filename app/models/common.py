import os

import cv2
import numpy as np
import onnxruntime as ort


def _cuda_available():
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    # onnxruntime-gpu can cac DLL CUDA/cuDNN; nap tu cac goi pip nvidia-*
    # (cai bang: pip install "onnxruntime-gpu[cuda,cudnn]").
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            print(f"[onnx] Khong nap duoc DLL CUDA: {exc}")
    return True


_USE_CUDA = _cuda_available()


def create_session(model_path):
    # Gioi han so thread/session: de mac dinh (dung toan bo core cho moi lan
    # infer) gay tranh chap CPU va lam do tre tang dan khi chay lien tuc (CCTV
    # phai infer nhieu frame/giay) do CPU bi throttle nhiet.
    so = ort.SessionOptions()
    so.intra_op_num_threads = min(4, os.cpu_count() or 4)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if _USE_CUDA:
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    print(f"[onnx] {model_path.name}: {session.get_providers()[0]}")
    return session


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (left, top)
