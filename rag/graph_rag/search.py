import networkx as nx
from thefuzz import process, fuzz

def search_graph(
    question: str,
    graph: nx.Graph,
    node_to_texts: dict,
    hop: int = 1,
    max_context: int = 20,
    threshold: int = 60  # Ngưỡng độ tương đồng (0-100)
) -> str:
    # 1. Tách từ khóa
    keywords = [k.lower() for k in question.split() if len(k) > 2]
    if not keywords:
        return ""

    all_nodes = list(graph.nodes())
    seed_nodes = set()

    # 2. Tìm các "Seed Nodes" sử dụng Fuzzy Matching
    for k in keywords:
        # extractBests trả về danh sách (tên_node, score)
        matches = process.extractBests(k, all_nodes, scorer=fuzz.partial_ratio, score_cutoff=threshold)
        for match_node, score in matches:
            seed_nodes.add(match_node)

    # Nếu vẫn không tìm thấy, lấy fallback
    if not seed_nodes:
        seed_nodes = all_nodes[:3]

    visited_nodes = set()
    visited_edges = set()
    used_texts = set()
    context_parts = []

    # 3. Duyệt qua từng seed node và mở rộng (Hop)
    for seed in seed_nodes:
        sub_nodes = nx.single_source_shortest_path_length(graph, seed, cutoff=hop)

        for node in sub_nodes:
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            # Lấy thông tin [TEXT]
            texts = node_to_texts.get(node, [])
            for text in texts:
                short_text = text[:300].replace("\n", " ")
                if short_text not in used_texts:
                    used_texts.add(short_text)
                    context_parts.append(f"[TEXT] {node}: {short_text}...")
                    break 

            # Lấy thông tin [GRAPH]
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted([str(node), str(neighbor)]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    rel = graph.get_edge_data(node, neighbor).get("relation", "RELATED_TO")
                    context_parts.append(f"[GRAPH] {node} --({rel})--> {neighbor}")

    # 4. Ranking (Sử dụng fuzz.token_set_ratio để xếp hạng context hiệu quả hơn)
    def calculate_score(item):
        return fuzz.token_set_ratio(question, item)

    context_parts = sorted(context_parts, key=calculate_score, reverse=True)

    return "\n".join(context_parts[:max_context])