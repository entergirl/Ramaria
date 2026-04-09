import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
print(f"sys.path: {sys.path[:3]}")

# Try importing
try:
    from ramaria.storage.database import new_session, get_setting
    print("SUCCESS: database module loaded")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
