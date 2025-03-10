import json
import os
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


class GraphManager:
    def __init__(self):
        pass

    def load_raw_files(
        self, input_folder: str, pattern: str = "*.json", content_type_filter=""
    ):
        files = sorted(glob(os.path.join(input_folder, pattern)))
        print(f"Found {len(files)} files in {input_folder}.")
        raws = []
        if content_type_filter:
            print(f"Filtering by content type: {content_type_filter}")
        for f in tqdm(files, desc="Loading raw files"):
            raw = json.load(open(f))
            raw["all_raw"] = raw["classification"]
            if content_type_filter:
                content_type = raw["classification"].get("content_type", None)
                if content_type not in content_type_filter:
                    continue
            raw["classification"] = raw["classification"].get("topics", [])
            raws.append(raw)
        print(f"Loaded {len(raws)} raw documents.")
        return raws

    def compute_scores(self, raws: list[dict]) -> dict:
        print("Computing scores...")
        # Count topic rankings
        counter: dict[str, dict] = {}
        for raw in raws:
            for i, topic in enumerate(raw["classification"], start=1):
                if topic not in counter:
                    counter[topic] = {}
                counter[topic][i] = counter[topic].get(i, 0) + 1
        # Compute scores
        df = pd.DataFrame(counter).T
        df.fillna(0, inplace=True)
        scores = df.apply(lambda x: sum(i * j for i, j in x.items()), axis=1)
        totals = scores.sum()
        scores = scores.apply(lambda x: np.log(totals / x))
        scores_map = scores.to_dict()
        print("Scores computed.")
        return scores_map

    def build_graph(self, raws: list[dict], scores_map: dict[str, float]):
        print("Building graph...")
        chunks = raws
        G = nx.Graph()
        for idx, chunk in enumerate(chunks):
            G.add_node(
                idx,
                chunk_text=chunk["chunk"],
                source=chunk["source_file"],
                classifications=chunk["classification"],
            )

        def compute_contribution(rank1: int, rank2: int, topic: str) -> float:
            return 1.0 / (rank1 + 1) + 1.0 / (rank2 + 1) + scores_map[topic]

        for i in tqdm(range(len(chunks)), desc="Building nodes & edges"):
            for j in range(i + 1, len(chunks)):
                common_details = []
                total_weight = 0.0
                classifications_i: list = list(chunks[i]["classification"])
                classifications_j: list = list(chunks[j]["classification"])
                topics = set(classifications_i).intersection(classifications_j)
                for topic in topics:
                    rank_i = classifications_i.index(topic)
                    rank_j = classifications_j.index(topic)
                    contribution = compute_contribution(rank_i, rank_j, topic)
                    common_details.append(
                        {
                            "topic": topic,
                            "rank_i": rank_i,
                            "rank_j": rank_j,
                            "contribution": contribution,
                        }
                    )
                    total_weight += contribution
                if common_details:
                    G.add_edge(i, j, weight=total_weight, common=common_details)
        print("Graph built. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
        return G

    def prune_graph(self, G: nx.Graph, threshold_percentile: float):
        print("Starting graph pruning...")
        edge_weights = [data.get("weight", 0) for _, _, data in G.edges(data=True)]
        print("Min weight:", min(edge_weights))
        print("Max weight:", max(edge_weights))
        print("Mean weight:", np.mean(edge_weights))
        print("Median weight:", np.median(edge_weights))
        threshold = np.percentile(edge_weights, threshold_percentile)
        print(f"Pruning threshold ({threshold_percentile}th percentile):", threshold)
        G_pruned = G.copy()
        edges_to_remove = [
            (u, v)
            for u, v, data in G_pruned.edges(data=True)
            if data.get("weight", 0) < threshold
        ]
        print(
            f"Removing {len(edges_to_remove)} edges out of {G_pruned.number_of_edges()}..."
        )
        G_pruned.remove_edges_from(edges_to_remove)
        print(
            "Graph pruned. Nodes:",
            G_pruned.number_of_nodes(),
            "Edges:",
            G_pruned.number_of_edges(),
        )
        return G_pruned

    def update_graph_components(self, G: nx.Graph):
        print("Computing connected components...")
        components = list(nx.connected_components(G))
        print("Number of connected components:", len(components))
        component_map = {}
        for comp_id, comp_nodes in enumerate(components):
            for node in comp_nodes:
                component_map[node] = comp_id
        nx.set_node_attributes(G, component_map, "component_id")
        component_sizes = [len(comp) for comp in components]
        print(
            "Component sizes (min, max, mean):",
            np.min(component_sizes),
            np.max(component_sizes),
            np.mean(component_sizes),
        )
        return component_map
