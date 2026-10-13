import threading
import time

import cv2


class CameraStream:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0  # tang moi khi doc duoc frame moi
        self._connected = False
        self._source = None
        self._cap = None
        self._stop_flag = False
        self._thread = None
        self._generation = 0

    def start(self, source):
        self.stop()
        self._stop_flag = False
        self._source = source
        self._generation += 1
        self._thread = threading.Thread(target=self._run, args=(self._generation,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open(self, source):
        try:
            source_cast = int(source)
        except (TypeError, ValueError):
            source_cast = source
        return cv2.VideoCapture(source_cast)

    def _run(self, generation):
        while not self._stop_flag and generation == self._generation:
            cap = self._open(self._source)
            self._cap = cap
            if not cap.isOpened():
                with self._lock:
                    self._connected = False
                time.sleep(2)
                continue

            with self._lock:
                self._connected = True

            while not self._stop_flag and generation == self._generation:
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1

            cap.release()
            with self._lock:
                self._connected = False
            if not self._stop_flag and generation == self._generation:
                time.sleep(2)

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_frame_with_id(self, last_id=None):
        """Tra ve (frame, frame_id). Neu frame_id == last_id (chua co frame moi)
        thi tra ve (None, last_id) de tranh copy/xu ly lai cung mot frame."""
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def is_connected(self):
        with self._lock:
            return self._connected


camera_stream = CameraStream()
