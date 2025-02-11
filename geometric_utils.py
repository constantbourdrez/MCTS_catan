import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def hexagon_coordinates(center, size=1):
    """Generates the coordinates of a hexagon based on its center."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return [(center[0] + size * np.cos(angle), center[1] + size * np.sin(angle)) for angle in angles]

def find_existing_node(position, position_to_id, tol=1e-6):
    """Finds an existing node close to a given position within a tolerance."""
    for existing_pos, node_id in position_to_id.items():
        if np.allclose(existing_pos, position, atol=tol):
            return node_id
    return None

def create_hexagonal_graph(n):
    """Creates a hexagonal tiling of `n` hexagons forming a Catan-like board."""
    G = nx.Graph()
    hex_size = 1
    centers = []

    # Generate hexagon centers in a hexagonal grid pattern
    for q in range(-n, n + 1):
        for r in range(-n, n + 1):
            if abs(q + r) > n:
                continue
            x = hex_size * (3 / 2) * q
            y = hex_size * (np.sqrt(3) / 2) * (2 * r + q)
            centers.append((x, y))

    # Dictionary to store nodes based on their positions (avoids duplicates)
    position_to_id = {}
    existing_edges = set()

    # Add nodes and edges based on hexagon corners
    current_id = 0
    for center in centers:
        corners = hexagon_coordinates(center, hex_size)

        corner_ids = []
        for corner in corners:
            node_id = find_existing_node(corner, position_to_id)
            if node_id is None:
                position_to_id[corner] = current_id
                G.add_node(current_id, pos=corner)
                corner_ids.append(current_id)
                current_id += 1
            else:
                corner_ids.append(node_id)

        # **Fix: Ensure each hexagon is properly formed by connecting its corners in sequence**
        for i in range(6):
            edge = (corner_ids[i], corner_ids[(i + 1) % 6])  # Connect consecutive corners
            if edge not in existing_edges:
                G.add_edge(*edge)
                existing_edges.add(edge)

    # Get the node positions for plotting
    positions = {node: G.nodes[node]['pos'] for node in G.nodes}

    print(f"Created {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G, positions

def rotate_positions(positions, angle):
    """Rotate all positions by the given angle in radians."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return {node: (x * cos_a - y * sin_a, x * sin_a + y * cos_a) for node, (x, y) in positions.items()}

if __name__ == "__main__":
    hex_board, positions = create_hexagonal_graph(2)  # Use 3 instead of 6 for better visualization
    positions = rotate_positions(positions, np.pi / 2)

    plt.figure(figsize=(8, 8))
    nx.draw(hex_board, positions, with_labels=False, node_color='lightblue', edge_color='black', node_size=50)
    plt.show()
