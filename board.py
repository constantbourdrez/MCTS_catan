import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import queue

from player import Player, Structure
from geometric_utils import hexagon_coordinates, create_hexagonal_graph, rotate_positions



class Tile:
    def __init__(self, nodes, edges, center, number, resource, coordinates, tile_id):
        self.nodes = nodes            # List of Node objects that form the corners
        self.edges = edges            # List of Edge objects along tile borders
        self.center = center          # Center coordinates of the tile
        self.number = number          # Dice number associated with the tile (0 for desert)
        self.resource = resource      # Resource type (or "desert")
        self.coordinates = coordinates  # Coordinates of the hexagon’s corners
        self.has_thief = False
        self.tile_id = tile_id

class Node:
    def __init__(self, coord, node_id):
        self.position = coord
        self.node_id = node_id         # Unique node identifier
        self.structure = Structure.none
        self.player = None             # Which player built here (if any)
        self.tiles = []                # Tiles adjacent to this node
        self.color = None              # For display purposes

    def add_tile(self, tile):
        """Associate the node with a tile."""
        self.tiles.append(tile)

class Edge:
    def __init__(self, node1, node2):
        # Store references to the two Node objects (order does not matter)
        self.nodes = (node1, node2)
        self.structure = Structure.none
        self.player = None
        self.adjacent_tiles = []  # Tiles sharing this edge
        
    def is_buildable(self, player):
        # Edge is buildable if it is unoccupied and touches at least one node with the player's settlement or city.
        if self.structure != Structure.none:
            return False
        for node in self.nodes:
            if node.player == player and node.structure in [Structure.house, Structure.city]:
                return True
        return False

class Board:
    def __init__(self, names, ai_or_not, verbose = True):
        self.num_players = len(names)
        self.gameOver = False
        self.maxPoints = 13
        self.turn = 1
        self.playerQueue = queue.Queue(self.num_players)
        self.gameSetup = True
        self.verbose = verbose

        # Bank of cards (resource counts, development cards, etc.)
        self.bank = {
            "brick": 19, "buildRoad": 2, "knight": 13, "monopoly": 2,
            "plenty": 1, "sheep": 19, "stone": 19, "victoryPoint": 5,
            "wheat": 19, "wood": 19
        }

        # Create the hexagonal graph (the board outline)
        self.G, self.node_positions, self.centers = create_hexagonal_graph(2)
        # Create Node objects for each node in the graph
        self.nodes = {}
        for node_id, pos in self.node_positions.items():
            self.nodes[node_id] = Node(coord=pos, node_id=node_id)
        # Build a mapping from position to node for fast lookup
        self.positions_to_node = {pos: self.nodes[node_id] for node_id, pos in self.node_positions.items()}

        # Build board tiles (and later update edges) 
        self.initialize_board()
        # Create all board edges from the graph
        self.create_edges()
        # Now that edges exist, update each tile’s edge list to refer to Edge objects.
        self.update_tile_edges()

        
        self.build_initial_settlements(names, ai_or_not)
        self.plot_board()
        
    def reset(self):
        """
        Reset the board and all player state for a new game,
        preserving the player objects and their identities (e.g. colors, AI flags).
        """
        # Reset game state
        self.gameOver = False
        self.turn = 1
        self.gameSetup = True

        # Recreate graph structure and tiles
        self.G, self.node_positions, self.centers = create_hexagonal_graph(2)
        
        # Recreate Nodes
        self.nodes = {}
        for node_id, pos in self.node_positions.items():
            self.nodes[node_id] = Node(coord=pos, node_id=node_id)
        self.positions_to_node = {pos: self.nodes[node_id] for node_id, pos in self.node_positions.items()}

        # Reset tiles, edges
        self.initialize_board()
        self.create_edges()
        self.update_tile_edges()

        # Reset the bank
        self.bank = {
            "brick": 19, "buildRoad": 2, "knight": 13, "monopoly": 2,
            "plenty": 1, "sheep": 19, "stone": 19, "victoryPoint": 5,
            "wheat": 19, "wood": 19
        }

        # Reset player states
        for player in self.players:
            player.reset()

        # Re-run setup phase
        self.build_initial_settlements([p.name for p in self.players], [p.isAI for p in self.players], reset=True)
        

  # No need to recreate players

        # Optional: plot the board
        # self.plot_board(title="Reset Board")



    def initialize_board(self):
        """ Initialize game board tiles with resources and numbers """
        # Define resource distribution (non-desert)
        resources = {"brick": 3, "sheep": 4, "stone": 3, "wheat": 4, "wood": 4}
        # Pre-defined dice numbers (for non-desert tiles)
        numbers = [10, 2, 9, 10, 8, 5, 11, 6, 5, 8, 9, 12, 6, 4, 3, 4, 3, 11]
        self.tiles = {}
        thief_placed = False

        for i, center in enumerate(self.centers):
            if center == (0.0, 0.0):
                # Use desert tile – no number and with the thief initially
                resource = "desert"
                number = 0
                thief_placed = True
            else:
                resource = np.random.choice(list(resources.keys()))
                resources[resource] -= 1
                if resources[resource] == 0:
                    del resources[resource]
                number = np.random.choice(numbers)
                numbers.remove(number)

            # Determine the hexagon corner coordinates for the tile
            corners = hexagon_coordinates(center, 1)
            tile_nodes = []
            for corner in corners:
                node = self.find_closest_node(corner)
                tile_nodes.append(node)
                # Also add this tile to the node’s adjacent tiles list
                # (We add the tile later after creation)
            # Temporarily, store edge tuples (will be updated later)
            tile_edge_ids = []
            for j in range(6):
                # Use the Node objects directly
                tile_edge_ids.append((tile_nodes[j], tile_nodes[(j + 1) % 6]))
            tile = Tile(nodes=tile_nodes, edges=tile_edge_ids, center=center, number=number, resource=resource,
                        coordinates=corners, tile_id=i)
            tile.has_thief = thief_placed
            thief_placed = False
            self.tiles[i] = tile

            # Associate this tile with its nodes
            for node in tile_nodes:
                node.add_tile(tile)

    def find_closest_node(self, corner):
        """Find the closest existing node to the given corner coordinates."""
        for pos, node in self.positions_to_node.items():
            if np.isclose(pos[0], corner[0]) and np.isclose(pos[1], corner[1]):
                return node
        raise KeyError(f"No matching node found for corner {corner}")

    def create_edges(self):
        """ Create Edge objects for each edge in the board graph """
        self.edges = {}
        # Use the networkx graph’s edges; each edge is between two node IDs
        for n1, n2 in self.G.edges():
            node1 = self.nodes[n1]
            node2 = self.nodes[n2]
            key = tuple(sorted((node1.node_id, node2.node_id)))
            e = Edge(node1, node2)
            self.edges[key] = e

    def update_tile_edges(self):
        """ Update each tile’s edges to be the corresponding Edge objects """
        for tile in self.tiles.values():
            edge_objs = []
            for n1, n2 in tile.edges:
                key = tuple(sorted((n1.node_id, n2.node_id)))
                if key in self.edges:
                    edge_obj = self.edges[key]
                    edge_objs.append(edge_obj)
                    # Also add the tile to the edge’s adjacent tiles
                    edge_obj.adjacent_tiles.append(tile)
                else:
                    raise KeyError(f"Edge {key} not found in board edges.")
            tile.edges = edge_objs

    def get_valid_nodes(self, player=None):
        """
        Determine valid nodes for building a settlement.
        A valid node must:
        - Be unoccupied
        - Be at least 2 nodes away from any other settlement/city
        - Be adjacent to one of the player's roads (except during initial setup)
        """
        valid_nodes = []
        for node in self.nodes.values():
            if node.structure != Structure.none:
                continue

            # 1. Distance rule: no adjacent node is occupied
            conflict = False
            for neighbor in self.get_neighbor_nodes(node):
                if neighbor.structure != Structure.none:
                    conflict = True
                    break
            if conflict:
                continue

            # 2. Road connection requirement (skip if game setup)
            if not self.gameSetup:
                connected = False
                for edge in self.get_edges_of_node(node):
                    if edge.structure == Structure.road and edge.player == player:
                        connected = True
                        break
                if not connected:
                    continue

            valid_nodes.append(node)
        return valid_nodes


    def get_valid_edges(self, player):
        """
        Determine valid edges for building a road.
        Valid if edge is not occupied and touches either a settlement/city or an existing road of the player.
        """
        valid_edges = []
        # Find all edges adjacent to player's settlements or roads
        candidate_edges = set()
        for node in self.nodes.values():
            if node.player == player:
                for edge in self.get_edges_of_node(node):
                    candidate_edges.add(edge)
        for edge in candidate_edges:
            if edge.structure == Structure.none:
                valid_edges.append(edge)
        return valid_edges

    def get_neighbor_nodes(self, node):
        """Return neighbor nodes (adjacent in the graph) for a given node."""
        neighbors = []
        for neighbor_id in self.G.neighbors(node.node_id):
            neighbors.append(self.nodes[neighbor_id])
        return neighbors

    def get_edges_of_node(self, node):
        """Return all edges incident to a given node."""
        incident_edges = []
        for edge in self.edges.values():
            if node in edge.nodes:
                incident_edges.append(edge)
        return incident_edges

    def build_initial_settlements(self, names, ai_or_not, reset = False):
        """
        Build the initial settlements and roads.
        For human players the board will call player.choose_node and choose_edge.
        For AI players, their initial_setup method is used.
        """
        playerColors = ['black', 'darkblue', 'purple', 'orange']
        # Create players and add them to the queue
        if reset:
            pass
        else:
            for i in range(self.num_players):
                newPlayer = Player(names[i], i, ai_or_not[i])
                newPlayer.color = playerColors[i % len(playerColors)]
                self.playerQueue.put(newPlayer)

            self.players = list(self.playerQueue.queue)

        # Decide starting order using a dice roll
        max_dice = -1
        starting_player = None
        for player in self.players:
            player.roll_dice()  # Assume player has a roll_dice method that sets player.dice
            if player.dice > max_dice:
                max_dice = player.dice
                starting_player = player

        # Rotate players so that the highest roller is first
        while self.players[0] != starting_player:
            self.players.append(self.players.pop(0))

        # Refill the queue with the new order
        self.playerQueue = queue.Queue()
        for player in self.players:
            self.playerQueue.put(player)

        # First round: each player builds a settlement and a road in turn order.
        for player in self.players:
            if player.isAI:
                player.initial_setup(self)  # AI takes its own decisions
            else:
                self.build(player, action='SETTLE')
                self.build(player, action='ROAD')

        # Second round: reverse order for the second settlement/road.
        for player in reversed(self.players):
            if player.isAI:
                player.initial_setup(self)
            else:
                self.build(player, action='SETTLE')
                self.build(player, action='ROAD')
                # For the second settlement, grant resources from adjacent tiles
                self.grant_initial_resources(player)

        self.gameSetup = False

    def build(self, player, action):
        """Handle building of settlements (houses) and roads."""
        if action == 'SETTLE':
            valid_nodes = self.get_valid_nodes(player)
            # Ask the player to choose a node (this method must be defined in Player)
            chosen_node = player.choose_node(self, valid_nodes)
            if chosen_node is None:
                raise ValueError("No valid settlement placement chosen.")
            # Place a settlement (house) on the chosen node
            chosen_node.structure = Structure.house
            chosen_node.player = player
            player.settlements.append(chosen_node)
            if self.verbose:
                print(f"{player.name} built a settlement at node {chosen_node.node_id}")
        elif action == 'ROAD':
            valid_edges = self.get_valid_edges(player)
            chosen_edge = player.choose_edge(self, valid_edges)
            if chosen_edge is None:
                raise ValueError("No valid road placement chosen.")
            chosen_edge.structure = Structure.road
            chosen_edge.player = player
            player.roads.append(chosen_edge)
            if self.verbose:
                print(f"{player.name} built a road between nodes {chosen_edge.nodes[0].node_id} and {chosen_edge.nodes[1].node_id}")
        elif action == 'CITY':
            # Upgrade a settlement (house) to a city if the player owns it.
            valid_settlements = [node for node in player.settlements if node.structure == Structure.house]
            chosen_node = player.choose_settlement_to_upgrade(self, valid_settlements)
            if chosen_node is None:
                raise ValueError("No valid settlement chosen for upgrade.")
            chosen_node.structure = Structure.city
            player.cities.append(chosen_node)
            player.settlements.remove(chosen_node)
            if self.verbose:    
                print(f"{player.name} upgraded settlement at node {chosen_node.node_id} to a city")
        else:
            raise ValueError("Unknown build action")

    def grant_initial_resources(self, player):
        """For the second settlement placement, grant one resource for each adjacent tile (if not desert)."""
        # Assume the last settlement placed is the second one.
        settlement = player.settlements[-1]
        for tile in settlement.tiles:
            if tile.resource != "desert":
                player.add_resource(tile.resource, 1)
                if self.verbose:
                    print(f"{player.name} received 1 {tile.resource} from tile {tile.tile_id}")

    def roll_dice(self):
        """Simulate rolling two dice and process resource distribution."""
        dice = np.random.randint(1, 7) + np.random.randint(1, 7)
        if self.verbose:
            print(f"Dice rolled: {dice}")
        if dice == 7:
            self.handle_robber()
        else:
            self.distribute_resources(dice)
        return dice

    def distribute_resources(self, dice):
        """Distribute resources to players based on dice roll."""
        for tile in self.tiles.values():
            if tile.number == dice and not tile.has_thief:
                # For every node around the tile, give resources if a settlement or city is present.
                for node in tile.nodes:
                    if node.structure == Structure.house:
                        node.player.add_resource(tile.resource, 1)
                        if self.verbose:  
                            print(f"{node.player.name} receives 1 {tile.resource} from tile {tile.tile_id}")
                    elif node.structure == Structure.city:
                        node.player.add_resource(tile.resource, 2)
                        if self.verbose:
                            print(f"{node.player.name} receives 2 {tile.resource} from tile {tile.tile_id}")

    def handle_robber(self):
        """Handle the robber when a 7 is rolled."""
        # Remove thief from its current tile
        for tile in self.tiles.values():
            if tile.has_thief:
                tile.has_thief = False
        # For simplicity, choose a random non-desert tile to move the thief to.
        non_desert_tiles = [tile for tile in self.tiles.values() if tile.resource != "desert"]
        chosen_tile = np.random.choice(non_desert_tiles)
        chosen_tile.has_thief = True
        if self.verbose:
            print(f"Robber moved to tile {chosen_tile.tile_id}")

    def next_turn(self):
        """Advance to the next turn."""
        self.turn += 1
        if self.verbose:
            print(f"Turn {self.turn}")
        # Rotate the player queue
        self.players.append(self.players.pop(0))
        self.playerQueue = queue.Queue()
        for player in self.players:
            self.playerQueue.put(player)
        # Each turn the current player rolls the dice and may build
        current_player = self.players[0]
        dice = self.roll_dice()
        # (Additional phases like trading and building may be implemented here.)
        return current_player
    
    def available_road_placements(self, player):
        valid_edges = []
        for edge in self.edges.values():
            # Assume edge has an attribute like is_buildable(player)
            if edge.is_buildable(player):
                valid_edges.append(edge)
        return valid_edges

    
    def trade_with_bank(self, player, offer_resource, request_resource, ratio=4):
        """
        Trade with the bank.
        Player must offer 'ratio' units of one resource in exchange for 1 unit of another resource.
        Parameters:
          - player: The Player object initiating the trade.
          - offer_resource: Resource the player is giving.
          - request_resource: Resource the player wants.
          - ratio: The trade ratio (default is 4:1).
        Returns True if the trade was successful, otherwise False.
        """
        # Check if player has enough of the offered resource.
        if player.resources.get(offer_resource, 0) < ratio:
            print(f"{player.name} does not have enough {offer_resource} to trade (needs {ratio}).")
            return False

        # Check if bank has at least 1 unit of the requested resource.
        if self.bank.get(request_resource, 0) < 1:
            if self.verbose:
                print(f"The bank does not have any {request_resource} available for trade.")
            return False

        # Execute the trade: subtract resources from player and add the offered resources to the bank.
        player.resources[offer_resource] -= ratio
        self.bank[offer_resource] += ratio

        # Give one unit of the requested resource to the player.
        player.resources[request_resource] = player.resources.get(request_resource, 0) + 1
        self.bank[request_resource] -= 1
        if self.verbose:
            print(f"{player.name} traded {ratio} {offer_resource} for 1 {request_resource} with the bank.")
        return True
    
    def dfs_longest(self, player, current_node, coming_from_edge, visited_edges):
        """Recursively traverse player's road network starting from current_node.
        Avoid reusing an edge (using visited_edges) to compute the length of the chain."""
        max_length = 0
        for edge in self.get_edges_of_node(current_node):
            # Consider only edges that belong to the player and that haven't been used in this chain.
            if edge.player == player and edge != coming_from_edge and edge not in visited_edges:
                new_visited = visited_edges.union({edge})
                # Determine the other end of the edge.
                next_node = edge.nodes[0] if edge.nodes[1] == current_node else edge.nodes[1]
                length = 1 + self.dfs_longest(player, next_node, edge, new_visited)
                if length > max_length:
                    max_length = length
        return max_length

    def compute_longest_road(self, player):
        """Compute the longest continuous road length for the given player."""
        best = 0
        # Consider each road's endpoints as a starting point.
        for edge in player.roads:
            for node in edge.nodes:
                current_length = self.dfs_longest(player, node, None, set())
                if current_length > best:
                    best = current_length
        return best

    def update_longest_road(self):
        """Update players’ longest road flags.
        A player must have a continuous road length of at least 5 to claim the bonus."""
        best_length = 0
        best_player = None
        for player in self.players:
            lr = self.compute_longest_road(player)
            # Only consider roads of length >= 5 as valid for the bonus.
            if lr > best_length and lr >= 5:
                best_length = lr
                best_player = player
        for player in self.players:
            player.longest_road = (player == best_player)
            # Optionally, you can also log the longest road length for debugging:
            # print(f"{player.name} longest road: {self.compute_longest_road(player)}")
    
    def update_largest_army(self):
        """Update players’ largest army flags.
        A player must have at least 3 played knight cards to claim the bonus."""
        best = 0
        best_player = None
        for player in self.players:
            # Assume player.knight_count holds the number of played knight cards.
            count = getattr(player, "knight_count", 0)
            if count >= 3 and count > best:
                best = count
                best_player = player
        for player in self.players:
            player.largest_army = (player == best_player)
    
    def check_winner(self):
        """Check if any player has reached or exceeded the victory point threshold.
        This method now updates longest road and largest army status before comparing victory points.
        It assumes that player.victory_points() includes bonus points for longest road and largest army."""
        self.update_longest_road()
        self.update_largest_army()
        for player in self.players:
            if player.victory_points() >= self.maxPoints:
                self.gameOver = True
                print(f"Player {player.name} wins with {player.victory_points()} points!")
                return player
        return None


    def plot_board(self, show=True, title="Catan Board"):
        """Plot the game board, including tiles, roads, settlements, and cities."""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_title(title)

        # Base: draw the graph layout
        nx.draw(self.G, self.node_positions, with_labels=False, node_color='lightblue', edge_color='gray', node_size=100)

        # 1. Draw tiles with color fill
        colors = {"brick": "red", "sheep": "green", "stone": "gray", "wheat": "khaki", "wood": "saddlebrown", "desert": "orange"}
        for tile in self.tiles.values():
            x = [corner[0] for corner in tile.coordinates]
            y = [corner[1] for corner in tile.coordinates]
            plt.fill(x, y, colors.get(tile.resource, "white"), edgecolor='black', linewidth=2)
            if tile.number != 0:
                plt.text(tile.center[1], tile.center[0], f"{tile.number}", fontsize=12, ha='center', va='center', color='black', weight='bold')
            if tile.has_thief:
                plt.text(tile.center[1], tile.center[0] - 0.3, "👺", fontsize=16, ha='center')

        # 2. Draw roads
        for edge in self.edges.values():
            if edge.structure == Structure.road and edge.player is not None:
                x1, y1 = edge.nodes[0].position
                x2, y2 = edge.nodes[1].position
                plt.plot([x1, x2], [y1, y2], color=edge.player.color, linewidth=4, alpha=0.8)

        # 3. Draw settlements and cities
        for node in self.nodes.values():
            x, y = node.position
            if node.structure == Structure.house and node.player is not None:
                plt.scatter(x, y, c=node.player.color, s=150, edgecolors='black', zorder=3, label="Settlement")
            elif node.structure == Structure.city and node.player is not None:
                plt.scatter(x, y, c=node.player.color, s=200, edgecolors='black', marker='s', zorder=3, label="City")

        # 4. Final cleanup
        ax.set_aspect('equal')
        ax.axis('off')

        # Legend (optional: only once per player)
        legend_elements = []
        seen_colors = set()
        for player in self.players:
            if player.color not in seen_colors:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=player.name,
                                                markerfacecolor=player.color, markersize=10))
                seen_colors.add(player.color)
        plt.legend(handles=legend_elements, loc='upper right')

        if show:
            plt.show()
            
        

# If running as a script, you might create a board instance and simulate a few turns:
if __name__ == "__main__":
    # Example players: names and a flag indicating if they are controlled by an AI.
    names = ["Alice", "Bob", "Carol", "Dave"]
    ai_or_not = [False, False, True, True]
    board = Board(names, ai_or_not)

    # Example game loop (without trading logic)
    while not board.gameOver and board.turn < 50:
        current_player = board.next_turn()
        # Here you could let the current player decide on building/upgrading
        # For instance, if the player has sufficient resources, they might build or upgrade.
        # board.build(current_player, 'SETTLE') or board.build(current_player, 'ROAD')
        # After actions, check for winner.
        winner = board.check_winner()
        if winner:
            break
