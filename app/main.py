import logging
import shutil
import threading
import time
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

    job_id = job_store.create_job(extension)
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
    return FileResponse(
        path=file_path,
        media_type=media_types.get(extension, "application/octet-stream"),
        filename=f"enhanced.{extension}",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
