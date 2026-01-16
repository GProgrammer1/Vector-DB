import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"Path: {sys.path}")

try:
    import fastapi
    print("fastapi: found")
except ImportError as e:
    print(f"fastapi: {e}")

try:
    import vector_db
    print("vector_db: found")
except ImportError as e:
    print(f"vector_db: {e}")
