import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.predict_video import predict_video

def run_browser_prediction_sanity_check():
    print("=" * 80)
    print("PHASE 12 BROWSER & MODEL PREDICTION SANITY CHECK EVALUATION")
    print("=" * 80)

    dataset_dir = PROJECT_ROOT / "Dataset" / "Video"
    output_csv = PROJECT_ROOT / "data" / "videos" / "results" / "browser_prediction_sanity_check.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Define category targets and sample videos
    categories = [
        ("original", "REAL", 10),
        ("Deepfakes", "FAKE", 10),
        ("Face2Face", "FAKE", 10),
        ("FaceShifter", "FAKE", 10),
        ("FaceSwap", "FAKE", 10),
        ("NeuralTextures", "FAKE", 10),
        ("DeepFakeDetection", "FAKE", 10),
    ]

    records = []

    for cat_folder, ground_truth, target_count in categories:
        cat_path = dataset_dir / cat_folder
        if not cat_path.exists():
            print(f"Warning: Category folder {cat_folder} not found at {cat_path}")
            continue

        # Get first N mp4 files deterministically
        mp4_files = sorted(list(cat_path.glob("*.mp4")))[:target_count]
        print(f"\nProcessing Category [{cat_folder}] ({len(mp4_files)} videos, Ground Truth: {ground_truth})...")

        for mp4_f in mp4_files:
            rel_path = str(mp4_f.relative_to(PROJECT_ROOT).as_posix())
            res = predict_video(str(mp4_f))

            if res.get("status") == "success":
                pred_label = res["prediction"]["label"]
                fake_p = res["prediction"]["fake_probability"]
                real_p = res["prediction"]["real_probability"]
                frames_n = res["video"]["frames_analyzed"]
                feats_n = res["video"]["features_extracted"]
                is_correct = (pred_label == ground_truth)

                records.append({
                    "video_path": rel_path,
                    "category": cat_folder,
                    "ground_truth": ground_truth,
                    "prediction": pred_label,
                    "real_probability": real_p,
                    "fake_probability": fake_p,
                    "frames_analyzed": frames_n,
                    "features_extracted": feats_n,
                    "correct": is_correct
                })
                print(f"  {mp4_f.name:<35} GT: {ground_truth:<4} | Pred: {pred_label:<4} (FakeProb: {fake_p:.4f}) | Correct: {is_correct}")
            else:
                print(f"  {mp4_f.name:<35} ERROR: {res.get('error_message')}")

    # 2. Export CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 80)
    print(f"Saved sanity check results to: {output_csv}")
    print("=" * 80)

    # 3. Calculate metrics
    real_df = df[df["ground_truth"] == "REAL"]
    fake_df = df[df["ground_truth"] == "FAKE"]

    real_correct = real_df["correct"].sum()
    real_total = len(real_df)
    real_acc = (real_correct / real_total) if real_total > 0 else 0.0
    real_fpr = 1.0 - real_acc

    fake_correct = fake_df["correct"].sum()
    fake_total = len(fake_df)
    fake_acc = (fake_correct / fake_total) if fake_total > 0 else 0.0

    print("\n--- SANITY CHECK SUMMARY METRICS ---")
    print(f"Known-REAL Videos Analyzed:      {real_total}")
    print(f"Known-REAL Correctly Classified: {real_correct} / {real_total} (Accuracy: {real_acc * 100:.2f}%)")
    print(f"REAL False Positive Rate (FPR):  {real_fpr * 100:.2f}%")
    print(f"Known-FAKE Videos Analyzed:      {fake_total}")
    print(f"Known-FAKE Correctly Classified: {fake_correct} / {fake_total} (Accuracy: {fake_acc * 100:.2f}%)")

    print("\n--- CATEGORY-LEVEL ACCURACIES ---")
    for cat in df["category"].unique():
        sub_df = df[df["category"] == cat]
        acc_c = sub_df["correct"].mean()
        print(f"  Category [{cat:<18}]: {sub_df['correct'].sum()}/{len(sub_df)} ({acc_c * 100:.2f}% Accuracy)")

    print("=" * 80)

if __name__ == "__main__":
    run_browser_prediction_sanity_check()
