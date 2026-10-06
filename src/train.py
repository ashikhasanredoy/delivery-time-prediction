"""
train.py
--------
Training runner that delegates to final_model.run_final_pipeline().
"""

from .final_model import run_final_pipeline

if __name__ == "__main__":
    run_final_pipeline()
