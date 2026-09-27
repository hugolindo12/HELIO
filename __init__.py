"""
HEILO package initialization
"""
import sys

# Ensure UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure") and sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure") and sys.stderr:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
