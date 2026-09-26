# Auto Call Recorder Plus - Enhance API

Self-hosted audio enhancement service for the Auto Call Recorder Plus app's
"Enhance Audio (HD)" feature. Runs entirely on your own server - no
third-party account, no per-use cost.

Pipeline per uploaded recording: **ffmpeg loudness normalization** (EBU R128)
**-> DeepFilterNet noise suppression** (open-source, CPU-only, no GPU
required) **-> ffmpeg re-encode** back to the original format.

## API

All endpoints require `Authorization: Bearer <API_SECRET>`.

- `POST /v1/enhance` - multipart upload, field name `file`. Returns
  `{ "job_id": "...", "status": "queued" }` (202).
- `GET /v1/enhance/{job_id}` - returns
  `{ "job_id", "status": "queued"|"processing"|"done"|"failed", "error", "download_url"? }`.
  `download_url` is present only once `status` is `"done"`.
- `GET /v1/enhance/{job_id}/download` - the processed audio file.
- `GET /health` - no auth required, for uptime checks.

Jobs are processed one at a time (a single background worker thread) to
avoid CPU/memory contention on a modest VPS. Job files are deleted
automatically 48 hours after creation regardless of status, so disk usage
stays bounded even if the app never confirms a download.

## Deploying with Coolify

Same flow as the `AutoCallRecorderWebsite` project - a Dockerfile-based app
pointed at this repo.

1. Create a new application in Coolify, source = this Git repo, build pack =
   Dockerfile.
2. Set the environment variable `API_SECRET` to a long random value (e.g.
   `openssl rand -hex 32`) - this is the key the Android app's Settings
   screen needs.
3. Mount a persistent volume at `/data` (job files + the SQLite job-tracking
   DB live there) so they survive redeploys.
4. Expose port `8000`, point your domain/subdomain at it.
5. Deploy. Verify with `curl https://your-domain/health` -> `{"status":"ok"}`.

## Local testing without Docker

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt deepfilternet
# ffmpeg must also be installed and on PATH locally.
API_SECRET=test-secret DATA_DIR=./data/jobs DB_PATH=./data/jobs.db \
  uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/v1/enhance \
  -H "Authorization: Bearer test-secret" \
  -F "file=@sample.m4a"

curl http://localhost:8000/v1/enhance/<job_id> \
  -H "Authorization: Bearer test-secret"

curl http://localhost:8000/v1/enhance/<job_id>/download \
  -H "Authorization: Bearer test-secret" -o enhanced.m4a
```
