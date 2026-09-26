# Auto Call Recorder Plus - Enhance API

Self-hosted audio enhancement service for the Auto Call Recorder Plus app's
"Enhance Audio (HD)" feature. Runs entirely on your own server - no
third-party account, no per-use cost.

## Pipeline

Per uploaded recording, in order:

1. **ffmpeg static cleanup** - high-pass filter (removes sub-80Hz rumble) +
   a narrow notch filter (removes a fixed-frequency electrical/hardware
   whine, if present; a no-op otherwise). No loudness change at this stage.
2. **Split into 30-second chunks.** DeepFilterNet loads and processes an
   entire file in memory in one pass with no streaming, so a long real call
   recording can exceed available memory. Chunking keeps memory bounded
   regardless of recording length.
3. **DeepFilterNet3** (open-source, CPU-only, no GPU required) - the main
   noise-removal stage, run once over the whole batch of chunks
   (`--noisy-dir`). `--atten-lim 20` caps how aggressively it can attenuate
   any single frame, so it can't fully erase quiet speech by mistaking it
   for noise. `--pf` (post-filter) pushes suppression harder on the
   noisiest sections, safe to enable specifically because the attenuation
   cap already puts a floor under it.
4. **Stitch the denoised chunks back together** (ffmpeg concat).
5. **ffmpeg mop-up + loudness + re-encode**:
   - `afftdn` - a second, classical spectral denoiser cleaning up residual
     hiss the neural model left behind.
   - `acompressor` - mild compression so quiet passages are more
     consistently audible.
   - `loudnorm`, two-pass/measured ("linear" mode) targeting -14 LUFS -
     measured first, then applied with the measured values. Single-pass
     "dynamic" mode is only a heuristic and was observed silently
     undershooting badly on unusually quiet source material.
   - `alimiter` - brick-wall safety ceiling against any transient near full
     scale.
   - Re-encoded at a fixed 48kHz / 128kbps for lossy formats (ffmpeg's
     default bitrate for mono AAC/MP3 is far too low and its own
     quantization noise becomes audible once normalized louder).
6. **Safety check**: if the final output's mean volume comes back below
   -50dB (real speech never does), the job is failed rather than marked
   done - a destroyed/silent result is a worse outcome than an honest
   failure.

There is deliberately **no noise gate** in this pipeline. One was tried and
removed after it was confirmed (via direct before/after measurement on a
real call) to be muting genuine quiet speech, not just noise - a single
fixed threshold can't safely tell "background noise" apart from "this
caller's quieter delivery" across every voice, and for a call-recording app
losing real speech is a far worse failure than a bit of residual hiss
between words.

## API

All endpoints below except `/health` require `Authorization: Bearer
<API_SECRET>`.

### `POST /v1/enhance`
Start a new enhancement job.

- Body: multipart form-data, file field named `file`. Supported extensions:
  `m4a`, `mp3`, `wav`.
- Response `202`: `{ "job_id": "...", "status": "queued" }`
- Errors: `400` unsupported file type or empty upload, `401`/`403` bad API
  key.

### `GET /v1/enhance/{job_id}`
Poll job status.

- Response `200`:
  ```json
  {
    "job_id": "...",
    "status": "queued" | "processing" | "done" | "failed",
    "error": null,
    "download_url": "/v1/enhance/{job_id}/download"
  }
  ```
  `download_url` is present only once `status` is `"done"`. `error` is
  populated only when `status` is `"failed"`. `download_url` is a relative
  path - prepend your server's base URL before requesting it.
- Errors: `404` unknown job_id (already swept, or never existed).

### `GET /v1/enhance/{job_id}/download`
Download the enhanced audio. Returns the raw audio bytes with the correct
`Content-Type` and a `Content-Disposition` filename matching whatever you
originally uploaded.

- Errors: `404` unknown job_id, `409` job isn't done yet, `410` the result
  file is no longer on disk (e.g. swept by the 48h cleanup before you
  downloaded it).

### `GET /v1/admin/jobs`
Every job the server knows about, newest first.

- Response `200`:
  ```json
  {
    "count": 2,
    "jobs": [
      {
        "job_id": "...",
        "status": "done",
        "extension": "m4a",
        "original_filename": "OUTGOING_198_....m4a",
        "error": null,
        "created_at": "2026-09-26T13:51:14.197182+00:00",
        "age_hours": 1.2,
        "disk_usage_bytes": 8000,
        "disk_usage_human": "7.8 KB",
        "leftover_intermediates": false
      }
    ]
  }
  ```
  `leftover_intermediates: true` flags a job whose chunk/denoised working
  files never got cleaned up - normally impossible (they're removed in a
  `finally` block), but a worker process killed outright (e.g. an
  out-of-memory kill) skips that cleanup, so this is a real signal
  something went wrong at the OS level for that job.

### `GET /v1/admin/storage`
Dashboard summary.

- Response `200`:
  ```json
  {
    "job_count": 5,
    "jobs_by_status": {"done": 3, "failed": 1, "processing": 1},
    "enhance_api_storage_bytes": 10485760,
    "enhance_api_storage_human": "10.0 MB",
    "job_ttl_hours": 48.0,
    "disk_total_bytes": 509722226688,
    "disk_used_bytes": 425222021120,
    "disk_free_bytes": 84500205568,
    "disk_total_human": "474.7 GB",
    "disk_used_human": "396.0 GB",
    "disk_free_human": "78.7 GB"
  }
  ```
  The `disk_*` fields report the actual VPS disk, not just this API's own
  usage - useful since job data isn't the only thing that can fill it up.

### `DELETE /v1/admin/jobs/{job_id}`
Manually purge a job's files and database row immediately, instead of
waiting out the 48-hour automatic sweep.

- Response `200`: `{ "deleted": "...", "freed_bytes": 8000, "freed_human": "7.8 KB" }`
- Errors: `404` unknown job_id.

### `GET /health`
No auth required. Returns `{"status": "ok"}`. Use this for uptime checks.

## Data retention

- Uploaded originals and enhanced results are **not** deleted immediately
  after download - both persist on disk.
- A background sweep runs every 6 hours and deletes any job (files +
  database row) older than **48 hours**, regardless of status or whether it
  was ever downloaded. In practice a file can live anywhere from just under
  48 hours to up to ~54 hours depending on sweep timing.
- Use `DELETE /v1/admin/jobs/{job_id}` to purge a specific job sooner (e.g.
  right after your client confirms a successful download).
- Jobs are processed **one at a time** on a single background worker
  thread, to avoid CPU/memory contention on a modest VPS.

## Security notes

- `API_SECRET` should be a long random value (`openssl rand -hex 32`), not
  something guessable - it's the only thing standing between the public
  internet and both your call recordings and the admin endpoints (which
  expose file listings and sizes).
- Serve this over **HTTPS**, not plain HTTP. Android blocks unencrypted
  HTTP traffic from apps by default (`CLEARTEXT communication ... not
  permitted`), and more importantly your recordings are sensitive personal
  data that shouldn't travel in plaintext. If you're on a `sslip.io`-style
  address via Coolify, prefix the domain with `https://` in Coolify's
  Domains field and redeploy - it's a real, publicly resolvable domain, so
  Let's Encrypt can issue a certificate for it automatically.

## Deploying with Coolify

1. Create a new application in Coolify, source = this Git repo, build pack =
   Dockerfile.
2. Set the environment variable `API_SECRET` to a long random value (see
   Security notes above) - this is the key the Android app's Settings
   screen needs.
3. Mount a persistent volume at `/data` (job files + the SQLite job-tracking
   DB live there) so they survive redeploys.
4. Expose port `8000`, point your domain/subdomain at it with `https://`.
5. Deploy. Verify with `curl https://your-domain/health` -> `{"status":"ok"}`.

## Local testing without Docker

```bash
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
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

curl http://localhost:8000/v1/admin/jobs \
  -H "Authorization: Bearer test-secret"

curl http://localhost:8000/v1/admin/storage \
  -H "Authorization: Bearer test-secret"

curl -X DELETE http://localhost:8000/v1/admin/jobs/<job_id> \
  -H "Authorization: Bearer test-secret"
```
