#!/usr/bin/env python3
"""
Quick test script for Master Webhook Server
Tests all endpoints to verify they're accessible
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, expected_status=None):
    """Test a single endpoint."""
    url = f"{BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data, timeout=5)

        status = "✓" if (expected_status is None or response.status_code == expected_status) else "✗"
        print(f"{status} {method} {endpoint} - Status: {response.status_code}")

        return response.status_code
    except requests.exceptions.ConnectionError:
        print(f"✗ {method} {endpoint} - Connection failed (is server running?)")
        return None
    except Exception as e:
        print(f"✗ {method} {endpoint} - Error: {e}")
        return None

def main():
    print("=" * 60)
    print("Master Webhook Server - Endpoint Test")
    print("=" * 60)
    print()

    # Test global endpoints
    print("Global Endpoints:")
    test_endpoint("GET", "/")
    test_endpoint("GET", "/health")
    print()

    # Test workflow 1
    print("Workflow 1 (Preprocessing):")
    test_endpoint("GET", "/workflow1/health")
    # Don't test POST endpoints without data
    print("  - POST /workflow1/process (requires data - skipped)")
    print("  - POST /workflow1/upload (requires file - skipped)")
    print()

    # Test workflow 3
    print("Workflow 3 (Feedback):")
    test_endpoint("GET", "/workflow3/health")
    # Don't test POST endpoints without data
    print("  - POST /workflow3/feedback (requires data - skipped)")
    print("  - POST /workflow3/process-rejections (requires data - skipped)")
    print()

    print("=" * 60)
    print("Note: POST endpoints require valid data and are not tested here.")
    print("Use the interactive docs at http://localhost:8000/docs to test them.")
    print("=" * 60)

if __name__ == "__main__":
    print()
    print("Make sure the master server is running!")
    print("Start it with: python master_webhook_server.py")
    print()
    input("Press Enter to start tests...")
    print()

    main()
