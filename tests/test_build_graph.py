import json
import os

import networkx as nx
import pytest

from graphrag_tagger.build_graph import process_graph
from graphrag_tagger.graph.graph_manager import GraphManager


@pytest.fixture()
def data1():
    return {
        "chunk": "doc1",
        "source_file": "f1",
        "classification": {"topics": ["a", "b"]},
    }


@pytest.fixture()
def data2():
    return {
        "chunk": "doc2",
        "source_file": "f2",
        "classification": {"topics": ["b", "c"]},
    }


@pytest.fixture()
def data3():
    return {
        "chunk": "doc3",
        "source_file": "f3",
        "classification": {"topics": ["a", "c"]},
    }


# Test load_raw_files using tmp_path fixture
def test_load_raw_files(tmp_path, data1, data2):
    # Create dummy JSON files
    file1 = tmp_path / "chunk_1.json"
    file2 = tmp_path / "chunk_2.json"
    file1.write_text(json.dumps(data1))
    file2.write_text(json.dumps(data2))

    graph_manager = GraphManager()
    raws = graph_manager.load_raw_files(str(tmp_path))
    assert len(raws) == 2
    # Basic checks on loaded data
    assert raws[0]["chunk"] == "doc1"
    assert raws[1]["classification"] == ["b", "c"]


def test_compute_scores(data1, data2):
    raws = [
        {"chunk": "doc1", "source_file": "f1", "classification": ["a", "b"]},
        {"chunk": "doc2", "source_file": "f2", "classification": ["a", "c"]},
    ]
    graph_manager = GraphManager()
    scores_map = graph_manager.compute_scores(raws)
    # Check that required keys exist
    for topic in ["a", "b", "c"]:
        assert topic in scores_map
        assert isinstance(scores_map[topic], float)


def test_build_graph():
    # Two documents with one common classification "a"
    raws = [
        {"chunk": "doc1", "source_file": "f1", "classification": ["a", "b"]},
        {"chunk": "doc2", "source_file": "f2", "classification": ["a", "c"]},
    ]
    scores_map = {"a": 1.0, "b": 1.0, "c": 1.0}
    graph_manager = GraphManager()
    G = graph_manager.build_graph(raws, scores_map)
    # Expect 2 nodes and an edge because of common "a"
    assert G.number_of_nodes() == 2
    assert G.has_edge(0, 1)
    # Calculate expected weight: 1/(0+1) + 1/(0+1) + scores_map["a"] = 1 + 1 + 1 = 3
    edge_data = G.get_edge_data(0, 1)
    assert abs(edge_data["weight"] - 3) < 1e-6


def test_prune_and_update_components():
    # Create a simple graph with three nodes and two edges.
    G = nx.Graph()
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 2, weight=1)  # lower weight edge to be pruned
    # Prune using a threshold that only retains weight >= 2
    graph_manager = GraphManager()
    G_pruned = graph_manager.prune_graph(
        G, 50
    )  # 50th percentile, edge weight 3 remains
    # After pruning, only edge (0,1) should remain.
    assert G_pruned.number_of_edges() == 1
    component_map = graph_manager.update_graph_components(G_pruned)
    # Check that nodes in the main connected component have proper component IDs.
    assert 0 in component_map and 1 in component_map
    # Node 2 is isolated: update_graph_components does not remove nodes so it should be in its own comp
    assert 2 in component_map
    # Components should be different for isolated node.
    assert component_map[2] != component_map[0]


def test_process_graph(tmp_path, data1, data2, data3):
    # Create temporary directories for input and output
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "chunk_1.json").write_text(json.dumps(data1))
    (input_dir / "chunk_2.json").write_text(json.dumps(data2))
    (input_dir / "chunk_3.json").write_text(json.dumps(data3))

    # Run process_graph
    G_pruned = process_graph(str(input_dir), str(output_dir), threshold_percentile=97.5)
    # Check that output file exists
    output_file = os.path.join(str(output_dir), "connected_components.json")
    assert os.path.exists(output_file)
    with open(output_file, "r") as f:
        comp_map = json.load(f)
    # Basic check: component map should have keys for each node
    assert len(comp_map) == G_pruned.number_of_nodes()
