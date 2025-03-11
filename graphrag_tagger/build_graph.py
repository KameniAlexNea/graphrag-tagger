import argparse
import json
import os

from .graph.graph_manager import GraphManager


def process_graph(
    input_folder: str,
    output_folder: str,
    threshold_percentile: float = 97.5,
    content_type_filter="",
):
    print("Processing graph...")
    pattern = "chunk_*.json"
    if threshold_percentile < 1:
        threshold_percentile *= 100
    os.makedirs(output_folder, exist_ok=True)

    graph_manager = GraphManager()
    raws = graph_manager.load_raw_files(input_folder, pattern, content_type_filter)
    scores_map = graph_manager.compute_scores(raws)
    G = graph_manager.build_graph(raws, scores_map)
    G_pruned = graph_manager.prune_graph(G, threshold_percentile)
    component_map = graph_manager.update_graph_components(G_pruned)

    output_file = os.path.join(output_folder, "connected_components.json")
    with open(output_file, "w") as f:
        json.dump(component_map, f)
    print(f"Connected components map saved to {output_file}")
    print("Graph processing complete.")
    return G_pruned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process graph from JSON files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/test/results",
        help="Folder with input JSON files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/test",
        help="Folder to save output JSON.",
    )
    parser.add_argument(
        "--threshold_percentile",
        type=float,
        default=97.5,
        help="Percentile threshold for pruning edges.",
    )
    parser.add_argument(
        "--content_type_filter",
        type=str,
        default="",
        help="Percentile threshold for pruning edges.",
    )
    args = parser.parse_args()

    process_graph(
        args.input_folder,
        args.output_folder,
        threshold_percentile=args.threshold_percentile,
        content_type_filter=args.content_type_filter,
    )
