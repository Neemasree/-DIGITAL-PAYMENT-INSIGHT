from pathlib import Path
import runpy
ROOT = Path(__file__).resolve().parents[1]
# Original generator currently expects /mnt/data/digital-payment-insight.
# Copy the project folder there or adjust ROOT in the source for your environment.
runpy.run_path(str(ROOT / "src" / "pipeline_impl.py"), run_name="__main__")
