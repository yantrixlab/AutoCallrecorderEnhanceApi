import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import job_store, processing
from app.auth import require_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhance_api")

app = FastAPI(title="Auto Call Recorder Plus - Enhance API")

SUPPORTED_EXTENSIONS = {"m4a", "mp3", "wav"}
CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60  # sweep every 6h
JOB_TTL_SECONDS = 48 * 60 * 60  # delete anything older than 48h regardless of status


@app.on_event("startup")
def on_startup() -> None:
    job_store.init_db()
    processing.start_worker()
    threading.Thread(target=_cleanup_loop, name="enhance-cleanup", daemon=True).start()


def _cleanup_loop() -> None:
    while True:
        try:
            for job_id in job_store.get_jobs_older_than(JOB_TTL_SECONDS):
                shutil.rmtree(processing.job_dir(job_id), ignore_errors=True)
                job_store.delete_job(job_id)
        except Exception:
            logger.exception("Cleanup sweep failed")
        time.sleep(CLEANUP_INTERVAL_SECONDS)


def _extension_of(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()


@app.post("/v1/enhance", status_code=202, dependencies=[Depends(require_api_key)])
async def start_enhance(file: UploadFile):
    extension = _extension_of(file.filename or "")
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{extension}")

    job_id = job_store.create_job(extension, file.filename or f"enhanced.{extension}")
    directory = processing.job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)

    dest = processing.input_path(job_id, extension)
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    if dest.stat().st_size == 0:
        job_store.delete_job(job_id)
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    processing.enqueue(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/enhance/{job_id}", dependencies=[Depends(require_api_key)])
async def get_status(job_id: str):
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    response = {"job_id": job_id, "status": row["status"], "error": row["error"]}
    if row["status"] == "done":
        response["download_url"] = f"/v1/enhance/{job_id}/download"
    return response


@app.get("/v1/enhance/{job_id}/download", dependencies=[Depends(require_api_key)])
async def download(job_id: str):
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if row["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done yet (status: {row['status']})")

    extension = row["extension"]
    file_path = processing.output_path(job_id, extension)
    if not file_path.exists():
        raise HTTPException(status_code=410, detail="Result file no longer available")

    media_types = {"m4a": "audio/mp4", "mp3": "audio/mpeg", "wav": "audio/wav"}
    download_name = row["original_filename"] or f"enhanced.{extension}"
    return FileResponse(
        path=file_path,
        media_type=media_types.get(extension, "application/octet-stream"),
        filename=download_name,
    )


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _job_disk_usage(job_id: str) -> int:
    directory = processing.job_dir(job_id)
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


@app.get("/v1/admin/jobs", dependencies=[Depends(require_api_key)])
async def admin_list_jobs():
    """Every job the server knows about, newest first, with per-job disk usage.
    leftover_intermediates flags jobs whose chunk/denoised working files never
    got cleaned up - normally impossible (they're removed in a `finally` block),
    but a SIGKILLed worker process (e.g. the OOM case this API has hit before)
    skips `finally` entirely, so this is a real signal an admin should be able
    to see rather than just accumulating silently."""
    now = time.time()
    jobs = []
    for row in job_store.get_all_jobs():
        job_id = row["job_id"]
        directory = processing.job_dir(job_id)
        leftover_intermediates = directory.exists() and any(
            p.name in ("chunks", "denoised", "concat_list.txt", "denoised_full.wav")
            for p in directory.iterdir()
        )
        disk_usage = _job_disk_usage(job_id)
        jobs.append({
            "job_id": job_id,
            "status": row["status"],
            "extension": row["extension"],
            "original_filename": row["original_filename"],
            "error": row["error"],
            "created_at": datetime.fromtimestamp(row["created_at"], tz=timezone.utc).isoformat(),
            "age_hours": round((now - row["created_at"]) / 3600, 2),
            "disk_usage_bytes": disk_usage,
            "disk_usage_human": _human_size(disk_usage),
            "leftover_intermediates": leftover_intermediates,
        })
    return {"count": len(jobs), "jobs": jobs}


@app.get("/v1/admin/storage", dependencies=[Depends(require_api_key)])
async def admin_storage_summary():
    """High-level dashboard numbers: how many jobs exist (by status), how much
    disk this API's own job data is using, and the actual VPS disk's
    total/used/free - the last of those matters because job data isn't the
    only thing that can fill a disk, and this API has no visibility into
    anything else running on the same box."""
    rows = job_store.get_all_jobs()
    by_status: dict = {}
    total_bytes = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        total_bytes += _job_disk_usage(row["job_id"])

    disk = shutil.disk_usage(processing.DATA_DIR)
    return {
        "job_count": len(rows),
        "jobs_by_status": by_status,
        "enhance_api_storage_bytes": total_bytes,
        "enhance_api_storage_human": _human_size(total_bytes),
        "job_ttl_hours": JOB_TTL_SECONDS / 3600,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "disk_total_human": _human_size(disk.total),
        "disk_used_human": _human_size(disk.used),
        "disk_free_human": _human_size(disk.free),
    }


@app.delete("/v1/admin/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def admin_delete_job(job_id: str):
    """Manual purge, for freeing space without waiting out the 48h sweep -
    e.g. once the app has already confirmed downloading a result."""
    row = job_store.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    freed_bytes = _job_disk_usage(job_id)
    shutil.rmtree(processing.job_dir(job_id), ignore_errors=True)
    job_store.delete_job(job_id)
    return {"deleted": job_id, "freed_bytes": freed_bytes, "freed_human": _human_size(freed_bytes)}


@app.get("/health")
async def health():
    return {"status": "ok"}
