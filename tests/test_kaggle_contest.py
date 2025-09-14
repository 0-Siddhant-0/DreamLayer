#!/usr/bin/env python3
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kaggle_contest import score

def test_scoring():
    submission_zip = "/Users/abhijeetbajaj/Coding/ Side_Projects/DreamLayerProd/DreamLayer/tests/submissions.zip"
    
    # Create mock solution DataFrame
    solution_df = pd.DataFrame({'id': [1]})
    
    print("Testing full scoring pipeline...")
    try:
        final_score = score(
            solution=solution_df,
            submission_zip_path=submission_zip,
            row_id_column_name='id'
        )
        print(f"✅ Final Score: {final_score}")
        print(f"Score breakdown: 0.5 × CLIPScore - 0.5 × FID_norm = {final_score}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scoring()
