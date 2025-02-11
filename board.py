import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import queue

from player import Player
from geometric_utils import hexagon_coordinates, create_hexagonal_graph, rotate_positions

class Structure:
    none = "none"
    road = "road"
    house = "house"
    city = "city"

class Tile:
    def __init__(self, nodes, edges, center, number, resource, coordinates, tile_id):
        self.nodes = nodes  # List of Node IDs
        self.edges = edges  # List of Edge objects with node IDs
        self.center = center
        self.number = number
        self.resource = resource
        self.coordinates = coordinates
        self.has_thief = False
        self.tile_id = tile_id
class Node:
    def __init__(self, coord, node_id):
        self.position = coord
        self.node_id = node_id  # Unique node identifier
        self.structure = Structure.none
        self.player = None
        self.tiles = []  # List of tiles that share this node
        self.color = None

    def add_tile(self, tile):
        """Associate the node with a tile."""
        self.tiles.append(tile)

class Edge:
    def __init__(self, node1_id, node2_id):
        self.position = (node1_id, node2_id)  # Store node IDs
        self.structure = Structure.none
        self.player = None
        self.tile = []


class Board:
    def __init__(self, names, ai_or_not):
        self.num_players = len(names)
        self.gameOver = False
        self.maxPoints = 10
        self.turn = 1
        self.playerQueue = queue.Queue(self.num_players)
        self.gameSetup = True

        self.bank = {
            "brick": 19, "buildRoad": 2, "knight": 13, "monopoly": 2,
            "plenty": 1, "sheep": 19, "stone": 19, "victoryPoint": 5,
            "wheat": 19, "wood": 19
        }
        self.G, self.node_positions, self.centers = create_hexagonal_graph(2)
        self.node_positions = rotate_positions(self.node_positions, np.pi / 2)
        self.centers = [(-y, x) for x, y in self.centers]
        self.initialize_board()
        self.plot_board()

    def initialize_board(self, *args):
        """ Initialize game board """
        resources = {"brick": 3, "sheep": 4, "stone": 3, "wheat": 4, "wood": 4}
        numbers = [10, 2, 9, 10, 8, 5, 11, 6, 5, 8, 9, 12, 6, 4, 3, 4, 3, 11]
        self.tiles = {}
        has_thief = False
        for i, center in enumerate(self.centers):
            if center == (0.0, 0.0):
                resource = "desert"
                has_thief = True

            else:
                resource = np.random.choice(list(resources.keys()))
                resources[resource] -= 1
                if resources[resource] == 0:
                    del resources[resource]
            # nodes = [Node((i, j)) for j in range(6)]
            # edges = [Edge((i, j), (i, (j + 1) % 6)) for j in range(6)]

            # tile = Tile(nodes, edges, center, numbers[i], resource, hexagon_coordinates(center, 1), i)
            # self.tiles[i] = tile




    def build_initial_settlements(self, *args):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        self.playerQueue.put(self.players[self.current_player])
        self.players[self.current_player].color = playerColors[self.current_player-1]
        order_players = [ self.current_player + i  if self.current_player + i - self.num_players < 0 else self.current_player + i - self.num_players for i in range(1, self.num_players)]
        for i in order_players:
            self.playerQueue.put(self.players[i])
            self.players[i].color = playerColors[i-1]
        playerList = list(self.playerQueue.queue)
        #Build Settlements and roads of each player forwards
        for player_i in playerList:
            if(player_i.isAI):
                player_i.initial_setup(self.board)
            else:
                self.build(player_i, 'SETTLE')
                self.build(player_i, 'ROAD')

        #Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList:
            if(player_i.isAI):
                player_i.initial_setup(self.board)
            else:
                self.build(player_i, 'SETTLE')
                self.build(player_i, 'ROAD')

            #Initial resource generation


        self.gameSetup = False

        return

    def build(self, player, action, *args):
        """ Determine valid initial house and road placements """
        actions = []
        log = []
        if action == 'SETTLE':
            valid_nodes = self.get_valid_nodes()
            if self.gameSetup:
                chosen_node = player.choose_node(self)

    def get_valid_nodes(self, *args):
        """ Determine valid nodes for initial house placement """
        valid_nodes = []
        for tile in self.tiles:
            for node in tile.nodes:
                if node.structure == Structure.none:
                    # find all adjacent nodes to the current node
                    adjacent_edges = [node.edges[i][1] for i in range(6)]

                    if node not in valid_nodes:
                        valid_nodes.append(node)
        return valid_nodes

    def get_valid_edges(self, *args):
        pass


    def plot_board(self, *args):
            """ Plot the game board """
            plt.figure(figsize=(8, 8))
            nx.draw(self.G, self.node_positions, with_labels=False, node_color='lightblue', edge_color='gray')

            # Display tile IDs at their center positions
            for tile in self.tiles.values():
                if tile.number != 0:
                    plt.text(tile.center[0], tile.center[1], str(tile.number)+", "+tile.resource, fontsize=12, ha='center', va='center', color='black', weight='bold')
            # Color tiles according to their resource type
            # colors = {"brick": "red", "sheep": "green", "stone": "gray", "wheat": "yellow", "wood": "brown", "desert": "orange"}
            # for tile in self.tiles.values():
            #     #fill tile with the desired color
            #     plt.fill(*zip(*tile.coordinates.values()), colors[tile.resource], edgecolor='black', linewidth=2)
            plt.show()
