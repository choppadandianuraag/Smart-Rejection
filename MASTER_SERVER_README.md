# Master Webhook Server - Deployment Guide

## Overview

The master webhook server combines all workflow endpoints into a single FastAPI server for easy deployment.

**Previously:** 3 separate servers (ports 8000, 8001, 8002)
**Now:** 1 unified server (port 8000)

## Quick Start

### 1. Start the Master Server

```bash
# From project root
python master_webhook_server.py
```

Or with uvicorn:
```bash
uvicorn master_webhook_server:app --host 0.0.0.0 --port 8000
```

### 2. Verify All Services

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "workflow1_ready": true,
  "workflow3_ready": true,
  "services": {
    "resume_processor": true,
    "supabase": true,
    "llm": true
  }
}
```

## API Endpoints

### Global Endpoints

- **GET /** - API documentation and endpoint list
- **GET /health** - Global health check for all workflows
- **GET /docs** - Interactive Swagger UI documentation

### Workflow 1: Resume Preprocessing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/workflow1/process` | POST | Process resume from URL or base64 |
| `/workflow1/upload` | POST | Process resume from file upload |
| `/workflow1/health` | GET | Workflow 1 health check |

**Example - Process Resume:**
```bash
curl -X POST http://localhost:8000/workflow1/process \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "resume_url": "https://example.com/resume.pdf",
    "filename": "john_resume.pdf"
  }'
```

**Example - Upload Resume:**
```bash
curl -X POST http://localhost:8000/workflow1/upload \
  -F "file=@/path/to/resume.pdf" \
  -F "name=John Doe" \
  -F "email=john@example.com" \
  -F "phone=+1234567890"
```

### Workflow 3: Feedback Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/workflow3/feedback` | POST | Generate single feedback email |
| `/workflow3/process-rejections` | POST | Batch process all rejections |
| `/workflow3/health` | GET | Workflow 3 health check |

**Example - Generate Feedback:**
```bash
curl -X POST http://localhost:8000/workflow3/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "123e4567-e89b-12d3-a456-426614174000",
    "job_id": "987fcdeb-51a2-43d1-b789-123456789abc"
  }'
```

**Example - Batch Process Rejections:**
```bash
curl -X POST http://localhost:8000/workflow3/process-rejections \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "987fcdeb-51a2-43d1-b789-123456789abc",
    "threshold": 0.7
  }'
```

## Environment Variables

Make sure your `.env` file contains:

```env
# Supabase (required for both workflows)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# HuggingFace (required for workflow 3)
HF_TOKEN=your_huggingface_token

# Optional
PORT=8000
```

## Deployment Options

### Option 1: Local Development
```bash
python master_webhook_server.py
```

### Option 2: Production with Gunicorn
```bash
gunicorn master_webhook_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Option 3: Docker (create Dockerfile)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "master_webhook_server.py"]
```

Build and run:
```bash
docker build -t smart-rejection-master .
docker run -p 8000:8000 --env-file .env smart-rejection-master
```

### Option 4: Cloud Deployment (Railway, Render, Fly.io)

1. Push code to GitHub
2. Connect repository to platform
3. Set environment variables in platform dashboard
4. Deploy command: `python master_webhook_server.py`

## Integration with Make.com

Update your Make.com webhooks to use the new endpoints:

**Old URLs:**
- `http://your-domain:8000/webhook/process` → **New:** `http://your-domain:8000/workflow1/process`
- `http://your-domain:8001/webhook/feedback` → **New:** `http://your-domain:8000/workflow3/feedback`

## Logs

Logs are saved to:
- Console: Real-time colored output
- File: `logs/master_server_YYYY-MM-DD.log`
- Retention: 7 days

Log prefixes:
- `[WF1]` - Workflow 1 (Preprocessing)
- `[WF3]` - Workflow 3 (Feedback)

## Monitoring

### Health Checks

**Global Health:**
```bash
curl http://localhost:8000/health
```

**Workflow 1 Only:**
```bash
curl http://localhost:8000/workflow1/health
```

**Workflow 3 Only:**
```bash
curl http://localhost:8000/workflow3/health
```

### Status Codes

- `200` - Success
- `400` - Bad Request (missing parameters, invalid data)
- `404` - Resource not found (applicant, job)
- `500` - Server error (service not initialized)
- `503` - Service unavailable (processor/client not ready)

## Troubleshooting

### Issue: "Resume processor not initialized"
**Solution:** Check if workflow 1 dependencies are installed and OCR tools are available.

### Issue: "Supabase client not initialized"
**Solution:** Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env` file.

### Issue: "LLM client not initialized"
**Solution:** Verify `HF_TOKEN` in `.env` file and check HuggingFace credits.

### Issue: Import errors
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

## Benefits of Master Server

✅ **Single deployment** - One server instead of three
✅ **Simplified orchestration** - No need to manage multiple ports
✅ **Unified logging** - All logs in one place with workflow prefixes
✅ **Easier monitoring** - Single health check endpoint
✅ **Resource efficient** - Shared memory and initialization
✅ **Simplified CI/CD** - One deployment process

## Migration from Old Servers

If you were running individual servers:

1. **Stop old servers:**
   ```bash
   # Kill processes on ports 8000, 8001
   lsof -ti:8000 | xargs kill -9
   lsof -ti:8001 | xargs kill -9
   ```

2. **Start master server:**
   ```bash
   python master_webhook_server.py
   ```

3. **Update webhook URLs** in Make.com or other integrations

4. **Old servers remain available** for backward compatibility if needed

## Next Steps

- Visit http://localhost:8000/docs for interactive API documentation
- Test all endpoints with the Swagger UI
- Update your Make.com workflows with new endpoint URLs
- Deploy to your preferred cloud platform
