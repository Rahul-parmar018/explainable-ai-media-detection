import os
import sys
import csv
import math
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

METADATA_COLUMNS = [
    "video_path",
    "category",
    "label",
    "source_id",
    "frame_count",
    "fps",
    "width",
    "height",
]

def extract_actors_from_path(video_path: str, category: str) -> List[str]:
    """
    Extracts all subject/actor IDs involved in a video file.
    For original videos (e.g. 006.mp4) -> ['006']
    For FF++ manipulated videos (e.g. 006_002.mp4) -> ['006', '002']
    For DFD videos (e.g. 01_02__...) -> ['01', '02']
    """
    stem = Path(video_path).stem
    if category == "original":
        return [stem]
    elif category in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
        parts = stem.split("_")
        if len(parts) >= 2:
            return [parts[0], parts[1]]
        return [parts[0]]
    elif category == "DeepFakeDetection":
        parts = stem.split("_")
        if len(parts) >= 2:
            return [parts[0], parts[1]]
        return [parts[0]]
    return [stem]


def build_connected_component_groups(df: pd.DataFrame) -> Dict[str, str]:
    """
    Builds an undirected actor graph across all video records and extracts
    connected components to assign a leak-free connected_component_id to every video path.

    Returns:
        Dict[video_path, connected_component_id]
    """
    actor_graph: Dict[str, Set[str]] = {}

    def add_edge(u: str, v: str):
        if u not in actor_graph:
            actor_graph[u] = set()
        if v not in actor_graph:
            actor_graph[v] = set()
        actor_graph[u].add(v)
        actor_graph[v].add(u)

    # 1. Populate graph edges
    video_actors_map = {}
    for idx, row in df.iterrows():
        vpath = str(row["video_path"])
        cat = str(row["category"])
        actors = extract_actors_from_path(vpath, cat)
        video_actors_map[vpath] = actors

        if len(actors) == 1:
            if actors[0] not in actor_graph:
                actor_graph[actors[0]] = set()
        else:
            for a1 in actors:
                for a2 in actors:
                    if a1 != a2:
                        add_edge(a1, a2)

    # 2. Extract connected components via BFS
    visited_nodes: Set[str] = set()
    component_map: Dict[str, str] = {}
    comp_counter = 0

    for node in sorted(actor_graph.keys()):
        if node not in visited_nodes:
            comp_counter += 1
            comp_id = f"COMP_{comp_counter:03d}"
            queue = [node]
            visited_nodes.add(node)

            while queue:
                curr = queue.pop(0)
                component_map[curr] = comp_id
                for neighbor in actor_graph[curr]:
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append(neighbor)

    # 3. Map each video_path to its connected component ID
    video_group_map = {}
    for vpath, actors in video_actors_map.items():
        comp_ids = set(component_map[a] for a in actors if a in component_map)
        if comp_ids:
            video_group_map[vpath] = min(comp_ids)
        else:
            video_group_map[vpath] = "COMP_UNCATEGORIZED"

    return video_group_map


def audit_dataset_integrity(
    csv_path: str = "data/videos/features/video_features_stage_b.csv",
    output_dir: str = "data/videos/results"
) -> Dict[str, Any]:
    """
    Performs a comprehensive integrity and data leakage audit on the video feature CSV.
    Generates data/videos/results/source_group_summary.csv and audit dictionary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Target dataset CSV {csv_path} does not exist.")

    df = pd.read_csv(csv_path)
    total_rows = len(df)

    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]

    # Integrity Checks
    duplicate_paths = df["video_path"].duplicated().sum()
    duplicate_rows = df.duplicated(subset=feature_cols).sum()

    nan_count = df[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df[feature_cols].values).sum()

    cat_counts = df["category"].value_counts().to_dict()
    label_counts = df["label"].value_counts().to_dict()

    # Source ID Analysis
    unique_sources = df["source_id"].nunique()

    # Build connected components for leak-free grouping
    video_comp_map = build_connected_component_groups(df)
    df["connected_component_id"] = df["video_path"].map(video_comp_map)
    unique_components = df["connected_component_id"].nunique()

    # Generate Source Group Summary DataFrame
    group_summary_rows = []
    for sid, group_df in df.groupby("source_id"):
        cats = sorted(list(group_df["category"].unique()))
        labels = sorted(list(group_df["label"].unique()))
        comps = sorted(list(group_df["connected_component_id"].unique()))
        sample_vids = list(group_df["video_path"].head(3))

        group_summary_rows.append({
            "source_id": sid,
            "video_count": len(group_df),
            "categories_count": len(cats),
            "categories_present": "|".join(cats),
            "labels_present": "|".join(labels),
            "connected_components": "|".join(comps),
            "sample_videos": "|".join(sample_vids),
        })

    summary_df = pd.DataFrame(group_summary_rows).sort_values(by="video_count", ascending=False).reset_index(drop=True)
    summary_path = out_path / "source_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    audit_passed = (
        duplicate_paths == 0 and
        duplicate_rows == 0 and
        len(feature_cols) == 216 and
        nan_count == 0 and
        inf_count == 0 and
        "REAL" in label_counts and
        "FAKE" in label_counts
    )

    print("=" * 80)
    print("PHASE 10B: DATASET INTEGRITY & DATA LEAKAGE AUDIT")
    print("=" * 80)
    print(f"Target CSV File:                 {csv_path}")
    print(f"Total Rows Verified:             {total_rows}")
    print(f"Feature Columns Verified:        {len(feature_cols)} (Expected: 216)")
    print(f"Unique Primary Source IDs:      {unique_sources}")
    print(f"Unique Connected Components:     {unique_components}")
    print(f"Duplicate Video Paths:           {duplicate_paths}")
    print(f"Duplicate Feature Vectors:       {duplicate_rows}")
    print(f"NaN Values Count:                {nan_count}")
    print(f"Inf Values Count:                {inf_count}")
    print(f"Source Group Summary CSV:        {summary_path}")
    print("=" * 80)
    print(f"AUDIT VERDICT:                  {'PASS' if audit_passed else 'FAIL'}")
    print("=" * 80)

    return {
        "csv_path": csv_path,
        "total_rows": total_rows,
        "feature_count": len(feature_cols),
        "unique_sources": unique_sources,
        "unique_components": unique_components,
        "duplicate_paths": int(duplicate_paths),
        "duplicate_rows": int(duplicate_rows),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "cat_counts": cat_counts,
        "label_counts": label_counts,
        "summary_csv": str(summary_path),
        "audit_passed": audit_passed,
    }

if __name__ == "__main__":
    audit_dataset_integrity()
