import random
import numpy as np
 

class Structure:
    none = "none"
    road = "road"
    house = "house"
    city = "city"

class Player:
    def __init__(self, name, number, isAI=False):
        self.name = name
        self.number = number
        self.isAI = isAI
        self.resources = {"brick": 0, "sheep": 0, "stone": 0, "wheat": 0, "wood": 0}
        self.cards = {"buildRoad": 0, "knight": 0, "monopoly": 0, "plenty": 0, "victoryPoint": 0}
        self.dice = 0
        self.settlements = []  # List of Node objects where settlements (houses) are built
        self.roads = []        # List of Edge objects for roads
        self.cities = []       # List of Node objects upgraded to cities

    def roll_dice(self):
        """Roll two six-sided dice and store the result."""
        self.dice = random.randint(1, 6) + random.randint(1, 6)
        return self.dice

    def victory_points(self):
        """
        Compute victory points.
        Each settlement is worth 1 point, each city 2 points, plus any victory point cards.
        """
        vp = len(self.settlements) + 2 * len(self.cities) + self.cards.get("victoryPoint", 0)
        return vp

    def add_resource(self, resource, quantity):
        """Add resource cards to the player."""
        if resource in self.resources:
            self.resources[resource] += quantity
        else:
            self.resources[resource] = quantity

    def choose_node(self, board, valid_nodes):
        """
        Choose a node from valid_nodes.
        For now, simply choose randomly.
        """
        if not valid_nodes:
            return None
        return np.random.choice(valid_nodes)

    def choose_edge(self, board, valid_edges):
        """
        Choose an edge from valid_edges.
        For now, simply choose randomly.
        """
        if not valid_edges:
            return None
        return np.random.choice(valid_edges)

    def choose_settlement_to_upgrade(self, board, valid_settlements):
        """
        Choose one of the player's settlements to upgrade to a city.
        For now, simply choose randomly.
        """
        if not valid_settlements:
            return None
        return np.random.choice(valid_settlements)

    def initial_setup(self, board):
        """
        AI initial setup:
        Build one settlement and one road randomly from available valid choices.
        """
        # Settlement placement
        valid_nodes = board.get_valid_nodes(self)
        chosen_node = self.choose_node(board, valid_nodes)
        if chosen_node is None:
            raise ValueError("No valid node available for settlement placement.")
        chosen_node.structure = "house"  # Using string directly from Structure
        chosen_node.player = self
        self.settlements.append(chosen_node)
        print(f"{self.name} (AI) built an initial settlement at node {chosen_node.node_id}")

        # Road placement
        valid_edges = board.get_valid_edges(self)
        chosen_edge = self.choose_edge(board, valid_edges)
        if chosen_edge is None:
            raise ValueError("No valid edge available for road placement.")
        chosen_edge.structure = "road"
        chosen_edge.player = self
        self.roads.append(chosen_edge)
        print(f"{self.name} (AI) built an initial road between nodes {chosen_edge.nodes[0].node_id} and {chosen_edge.nodes[1].node_id}")
        
    def reset(self):
        # Reset all resource counts to 0
        self.resources = {
            "brick": 0,
            "wood": 0,
            "wheat": 0,
            "sheep": 0,
            "stone": 0
        }

        # Reset structures
        self.settlements = []
        self.cities = []
        self.roads = []

        # Reset dev cards & special points
        self.cards = {"buildRoad": 0, "knight": 0, "monopoly": 0, "plenty": 0, "victoryPoint": 0}
        self.dev_cards = 0
        self.knight_count = 0
        self.development_points = 0

        # Reset bonus flags
        self.longest_road = False
        self.largest_army = False

