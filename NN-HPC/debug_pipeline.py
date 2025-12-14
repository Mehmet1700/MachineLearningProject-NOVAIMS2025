
import sys
import os
sys.path.append(os.getcwd())

from utils.imputation import get_imputation_pipeline
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline

def check_pipeline():
    print("Checking imputation pipeline...")
    pipe = get_imputation_pipeline()
    print(f"Type of pipe: {type(pipe)}")
    
    for name, step in pipe.steps:
        print(f"Step '{name}': {type(step)}")
        if isinstance(step, SklearnPipeline) or isinstance(step, ImblearnPipeline):
            print(f"  -> WARNING: Step '{name}' is a Pipeline!")

if __name__ == "__main__":
    check_pipeline()
