import os
import cv2
import time
import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Load .env so USE_GCS_CROPS and GCS_... flags can be set in project .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# Normalize GOOGLE_APPLICATION_CREDENTIALS to absolute path if provided relative in .env
try:
    import os as _os
    cred = _os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if cred and not _os.path.isabs(cred):
        repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
        abs_cred = _os.path.abspath(_os.path.join(repo_root, cred))
        if _os.path.exists(abs_cred):
            _os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_cred
        else:
            pass
except Exception:
    pass



class CropWriter:
    """
    Asynchronous writer that saves cropped ROI images.
    - Local filesystem mode (default): images/YYYY-MM-DD/<camera_id>/<persistent_id>/
    - GCS mode (optional): upload JPEG bytes to a GCS bucket under the same path prefix
    - Bounded queue with drop-oldest policy to avoid blocking producers
    - One writer thread per camera recommended
    """

    def __init__(
        self,
        camera_id: int,
        base_path: str = 'images',
        queue_size: int = 20,
        jpg_quality: int = 85,
        use_gcs: bool | None = None,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = 'images',
    ):
        self.camera_id = camera_id
        self.base_path = base_path  # relative path per user request
        self.jpg_quality = int(jpg_quality)
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._running = False
        self._dirs_created = set()

        # GCS settings
        if use_gcs is None:
            use_gcs = os.getenv('USE_GCS_CROPS', 'false').lower() == 'true'
        self.use_gcs = bool(use_gcs)
        self.gcs_bucket_name = gcs_bucket or os.getenv('GCS_BUCKET')
        self.gcs_prefix = gcs_prefix or os.getenv('GCS_PREFIX', 'images')
        self._gcs_client = None
        self._gcs_bucket = None

        # Metrics (best-effort, non-atomic)
        self.enqueued = 0
        self.saved = 0
        self.dropped_full = 0
        self.errors = 0

    def start(self):
        if not self._running:
            self._running = True
            self._thread.start()

    def stop(self, wait: bool = True):
        self._running = False
        try:
            # Nudge the thread out of get(timeout)
            self.queue.put_nowait({"_poison": True})
        except queue.Full:
            # Drop one and insert poison
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait({"_poison": True})
            except queue.Full:
                pass
        if wait:
            self._thread.join(timeout=2.0)

    def enqueue(self, job: Dict[str, Any]) -> bool:
        """
        Enqueue a crop job.
        Job schema:
          {
            'roi': np.ndarray (H x W x 3, BGR),
            'persistent_id': int,
            'camera_id': int,
            'ts_ms': int,
          }
        """
        try:
            self.queue.put_nowait(job)
            self.enqueued += 1
            return True
        except queue.Full:
            # Drop oldest, then add
            try:
                _ = self.queue.get_nowait()
                self.dropped_full += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(job)
                self.enqueued += 1
                return True
            except queue.Full:
                self.dropped_full += 1
                return False

    def _ensure_dir(self, path: str):
        if path in self._dirs_created:
            return
        os.makedirs(path, exist_ok=True)
        self._dirs_created.add(path)

    def _init_gcs(self):
        if not self.use_gcs or self._gcs_client is not None:
            return
        from google.cloud import storage  # lazy import
        self._gcs_client = storage.Client()
        if not self.gcs_bucket_name:
            raise RuntimeError("GCS bucket name is not set")
        self._gcs_bucket = self._gcs_client.bucket(self.gcs_bucket_name)

    def _upload_to_gcs(self, date_str: str, cam: int, pid: int, filename: str, jpg_bytes: bytes):
        if not self.use_gcs:
            return False
        if self._gcs_client is None:
            self._init_gcs()
        key = f"{self.gcs_prefix}/{date_str}/{cam}/{pid}/{filename}"
        blob = self._gcs_bucket.blob(key)
        blob.upload_from_string(jpg_bytes, content_type='image/jpeg')
        return True

    def _run(self):
        while self._running:
            try:
                job = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if job is None or job.get("_poison"):
                break

            try:
                roi = job.get('roi')
                pid = job.get('persistent_id')
                cam = job.get('camera_id', self.camera_id)
                ts_ms = int(job.get('ts_ms', int(time.time() * 1000)))

                if roi is None or roi.size == 0 or pid is None:
                    continue

                # Build path parts
                dt = datetime.fromtimestamp(ts_ms / 1000.0)
                date_str = dt.strftime('%Y-%m-%d')
                fname_ts = dt.strftime('%Y%m%d_%H%M%S_') + f"{ts_ms % 1000:03d}"
                filename = f"{pid}_{fname_ts}.jpg"

                # Replacement policy (local mode only)
                replace_latest = bool(job.get('replace_latest'))

                # Encode JPEG once
                ok, enc = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality])
                if not ok:
                    continue
                jpg_bytes = enc.tobytes()

                if self.use_gcs:
                    # Upload to GCS only (GCP-first mode)
                    try:
                        self._upload_to_gcs(date_str, cam, pid, filename, jpg_bytes)
                        self.saved += 1
                    except Exception:
                        self.errors += 1
                        continue
                else:
                    # Local filesystem mode
                    subdir = os.path.join(self.base_path, date_str, str(cam), str(pid))
                    self._ensure_dir(subdir)
                    fpath = os.path.join(subdir, filename)

                    if replace_latest:
                        try:
                            files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.startswith(f"{pid}_") and f.endswith('.jpg')]
                            if files:
                                newest = max(files, key=lambda p: os.path.getmtime(p))
                                fpath = newest
                        except Exception:
                            pass

                    try:
                        with open(fpath, 'wb') as f:
                            f.write(jpg_bytes)
                        self.saved += 1
                    except Exception:
                        self.errors += 1
                        continue
            except Exception:
                self.errors += 1
                continue

        # Drain any remaining poison items
        while True:
            try:
                job = self.queue.get_nowait()
                if job is None or job.get("_poison"):
                    break
            except queue.Empty:
                break

