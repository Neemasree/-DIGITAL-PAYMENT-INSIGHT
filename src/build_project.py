"""
Digital Payment Insight — synthetic educational pipeline.
Run from repository root: python src/build_project.py
All labels/data are simulated and must not be interpreted as real fraud outcomes.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "src" / "pipeline_impl.py"), run_name="__main__")
