
import sys
from pathlib import Path

# Add src directory to Python path so tests can import vector_db
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

