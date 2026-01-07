# AgenWatch/utils/safe_traceback.py
import traceback

def safe_print_exc():
    try:
        traceback.print_exc()
    except Exception:
        pass

__INTERNAL__ = True
