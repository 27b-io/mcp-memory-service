import os
import sys

# Force CPU-only mode for tests (avoids CUDA compatibility issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force integration tests to use in-memory Qdrant to avoid exclusive lock contention
# This prevents the infinite loop issue discovered in mm-nt7lt where multiple pytest
# processes blocked on embedded Qdrant's exclusive file lock
if "QDRANT_URL" not in os.environ and "QDRANT_STORAGE_PATH" not in os.environ:
    os.environ["QDRANT_URL"] = ":memory:"

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

