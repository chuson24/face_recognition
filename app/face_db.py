import json
import threading
import time

import numpy as np

from . import config


def _load_imported_gallery(path):
    """Doc 1 file export embedding (dinh dang CS500: {"embeddings": [{personId,
    fullName, vector, modelVersion, ...}]}). Tra ve {ten_hien_thi: info}.

    Ten hien thi duoc gan hau to "[TAG]" (TAG lay tu tien to ten file, vd
    "cs500-..." -> "[CS500]") de biet ket qua nhan dien den tu bo du lieu nao va
    khong trung voi nguoi dang ky tai cho.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    tag = path.stem.split("-")[0].upper()
    dim = int(raw.get("dim", 512))

    by_person = {}
    for e in raw.get("embeddings", []):
        vec = e.get("vector")
        if not vec or len(vec) != dim:
            continue
        pid = e.get("personId") or e.get("fullName")
        person = by_person.setdefault(
            pid,
            {"full_name": (e.get("fullName") or str(pid)).strip(), "vectors": [],
             "models": set(), "created_at": e.get("createdAt")},
        )
        person["vectors"].append(vec)
        person["models"].add(e.get("modelVersion"))

    # Hai personId khac nhau trung ten -> them doan dau personId de phan biet.
    name_counts = {}
    for p in by_person.values():
        name_counts[p["full_name"]] = name_counts.get(p["full_name"], 0) + 1

    people = {}
    for pid, p in by_person.items():
        name = p["full_name"]
        if name_counts[name] > 1:
            name = f"{name} #{str(pid)[-4:]}"
        vecs = np.asarray(p["vectors"], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        people[f"{name} [{tag}]"] = {
            "embeddings": vecs,
            "created_at": (p["created_at"] or "")[:19].replace("T", " "),
            "source": path.name,
            "models": sorted(m for m in p["models"] if m),
        }
    return people


class FaceDB:
    """Bo khuon mat de nhan dien = nguoi dang ky tai cho (face_db.json, sua/xoa
    duoc) + cac bo embedding import trong data/imported_embeddings/*.json (chi doc).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._people = {}
        self._imported = {}
        self._matrix = np.zeros((0, 512), dtype=np.float32)
        self._labels = []
        self._load()

    def _load(self):
        if config.FACE_DB_PATH.exists():
            try:
                raw = json.loads(config.FACE_DB_PATH.read_text(encoding="utf-8"))
                self._people = raw.get("people", {})
            except Exception:
                self._people = {}

        config.IMPORTED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(config.IMPORTED_EMBEDDINGS_DIR.glob("*.json")):
            try:
                people = _load_imported_gallery(path)
            except Exception as exc:
                print(f"[face_db] Bo qua {path.name}: {exc}")
                continue
            self._imported.update(people)
            total = sum(len(p["embeddings"]) for p in people.values())
            print(f"[face_db] Import {path.name}: {len(people)} nguoi, {total} embedding")
        self._rebuild_index()

    def _rebuild_index(self):
        """Gop moi embedding thanh 1 ma tran de match bang 1 phep nhan. Goi khi
        dang giu _lock (hoac trong __init__)."""
        mats, labels = [], []
        for name, info in self._people.items():
            embs = info.get("embeddings", [])
            if embs:
                mats.append(np.asarray(embs, dtype=np.float32))
                labels += [name] * len(embs)
        for name, info in self._imported.items():
            mats.append(info["embeddings"])
            labels += [name] * len(info["embeddings"])
        self._matrix = np.concatenate(mats) if mats else np.zeros((0, 512), np.float32)
        self._labels = labels
        self._is_imported = np.array([n in self._imported for n in labels], dtype=bool)

    def _save(self):
        payload = {"people": self._people}
        config.FACE_DB_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_people(self):
        with self._lock:
            local = [
                {
                    "name": name,
                    "num_images": len(info.get("embeddings", [])),
                    "created_at": info.get("created_at"),
                    "source": "local",
                }
                for name, info in self._people.items()
            ]
            imported = [
                {
                    "name": name,
                    "num_images": len(info["embeddings"]),
                    "created_at": info.get("created_at"),
                    "source": info["source"],
                }
                for name, info in self._imported.items()
            ]
            return local + imported

    def _mask(self, mode):
        if mode == "imported":
            return self._is_imported
        if mode == "local":
            return ~self._is_imported
        return np.ones(len(self._labels), dtype=bool)

    def has_people(self, mode="all"):
        with self._lock:
            return bool(self._mask(mode).any())

    def add_person(self, name, embeddings):
        with self._lock:
            self._people[name] = {
                "embeddings": [
                    e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings
                ],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save()
            self._rebuild_index()

    def delete_person(self, name):
        # Chi xoa duoc nguoi dang ky tai cho; bo import la chi doc.
        with self._lock:
            if name in self._people:
                del self._people[name]
                self._save()
                self._rebuild_index()
                return True
            return False

    def match(self, embedding, mode="all"):
        """mode: "all" | "local" (chi nguoi dang ky tai cho) | "imported" (chi bo import)."""
        with self._lock:
            mask = self._mask(mode)
            if not mask.any():
                return None, -1.0
            sims = self._matrix @ np.asarray(embedding, dtype=np.float32)
            sims = np.where(mask, sims, -np.inf)
            i = int(np.argmax(sims))
            return self._labels[i], float(sims[i])


face_db = FaceDB()
