import os
import sys

# Force CPU-only mode for tests (avoids CUDA compatibility issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
