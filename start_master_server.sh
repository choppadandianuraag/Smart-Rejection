#!/bin/bash
# Start the Master Webhook Server

echo "=============================================="
echo "Starting Master Webhook Server"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create .env with:"
    echo "  - SUPABASE_URL"
    echo "  - SUPABASE_KEY"
    echo "  - HF_TOKEN"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the server
echo ""
echo "🚀 Starting server on http://localhost:8000"
echo "📚 API docs: http://localhost:8000/docs"
echo "❤️  Health check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python master_webhook_server.py
