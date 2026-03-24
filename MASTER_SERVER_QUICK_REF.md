# Master Server - Quick Reference

## 🚀 Start the Server

```bash
# Option 1: Using start script
./start_master_server.sh

# Option 2: Direct command
source venv/bin/activate
python master_webhook_server.py

# Option 3: With custom port
PORT=8080 python master_webhook_server.py
```

## 📋 Endpoint Mapping

### Old → New URLs

| Old Endpoint | New Endpoint | Workflow |
|--------------|--------------|----------|
| `http://localhost:8000/webhook/process` | `http://localhost:8000/workflow1/process` | W1 - Resume preprocessing |
| `http://localhost:8000/webhook/upload` | `http://localhost:8000/workflow1/upload` | W1 - Resume upload |
| `http://localhost:8000/health` | `http://localhost:8000/workflow1/health` | W1 - Health check |
| `http://localhost:8001/webhook/feedback` | `http://localhost:8000/workflow3/feedback` | W3 - Single feedback |
| `http://localhost:8001/webhook/process-rejections` | `http://localhost:8000/workflow3/process-rejections` | W3 - Batch feedback |
| `http://localhost:8001/health` | `http://localhost:8000/workflow3/health` | W3 - Health check |

### New Global Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API overview and endpoint list |
| `/health` | GET | Global health check (all workflows) |
| `/docs` | GET | Interactive Swagger UI documentation |

## 🧪 Test the Server

```bash
# Test all endpoints
python test_master_server.py

# Or manually test health
curl http://localhost:8000/health

# View interactive docs
open http://localhost:8000/docs
```

## 📊 Log Prefixes

When viewing logs, look for these prefixes:

- `[WF1]` - Workflow 1 (Resume Preprocessing)
- `[WF3]` - Workflow 3 (Feedback Generation)

Example:
```
16:32:15 | INFO     | [WF1] Processing: /tmp/resume_abc123.pdf
16:32:18 | SUCCESS  | [WF1] Processed successfully: 123e4567-e89b-12d3-a456-426614174000
16:32:20 | INFO     | [WF3] Generating feedback for John Doe
16:32:25 | SUCCESS  | [WF3] Feedback generated for John Doe (DB ID: 42)
```

## 🔧 Troubleshooting

### Check all services are ready:
```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected output:
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

### Individual workflow checks:
```bash
curl http://localhost:8000/workflow1/health
curl http://localhost:8000/workflow3/health
```

## 🌐 Update Make.com Webhooks

1. Open your Make.com scenario
2. Find HTTP/webhook modules
3. Update URLs:
   - Old: `http://your-domain:8000/webhook/process`
   - New: `http://your-domain:8000/workflow1/process`
   - Old: `http://your-domain:8001/webhook/feedback`
   - New: `http://your-domain:8000/workflow3/feedback`
4. Test & activate

## 📦 Deployment Checklist

- [ ] Environment variables set in `.env`:
  - [ ] `SUPABASE_URL`
  - [ ] `SUPABASE_KEY`
  - [ ] `HF_TOKEN`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Server starts without errors: `python master_webhook_server.py`
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Make.com webhooks updated
- [ ] Test resume upload
- [ ] Test feedback generation

## 🎯 Benefits Summary

✅ **Single deployment** - 3 servers → 1 server
✅ **One port** - Easy firewall config
✅ **Unified logs** - All workflows in one place
✅ **Simpler monitoring** - Single health endpoint
✅ **Faster startup** - Shared initialization
✅ **Easy scaling** - One process to manage

## 📞 Support

- Check logs in: `logs/master_server_YYYY-MM-DD.log`
- API documentation: http://localhost:8000/docs
- Test script: `python test_master_server.py`
