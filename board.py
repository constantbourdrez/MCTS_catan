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
        self.positions_to_id = {pos: node for node, pos in self.node_positions.items()}
        self.initialize_board()

        self.plot_board()
        self.build_initial_settlements()


    def initialize_board(self, *args):
        """ Initialize game board """
        resources = {"brick": 3, "sheep": 4, "stone": 3, "wheat": 4, "wood": 4}
        numbers = [10, 2, 9, 10, 8, 5, 11, 6, 5, 8, 9, 12, 6, 4, 3, 4, 3, 11]
        self.tiles = {i: None for i in range(len(self.centers))}
        has_thief = False

        for i, center in enumerate(self.centers):

            if center == (0.0, 0.0):
                resource = "desert"
                has_thief = True
                number = 0


            else:
                resource = np.random.choice(list(resources.keys()))
                resources[resource] -= 1
                if resources[resource] == 0:
                    del resources[resource]
                number = np.random.choice(numbers)
                numbers.remove(number)

            # Find all the nodes that belong to the tile
            corners = hexagon_coordinates(center, 1)
            nodes = []
            for corner in corners:
                node_id = self.find_closest_node(corner)
                nodes.append(node_id)
            #Find all the edges that belong to the tile
            edges = []
            for j in range(6):
                edge = (nodes[j], nodes[(j + 1) % 6])
                edges.append(edge)



            tile = Tile(nodes= nodes, edges= edges, center=center, number=number, resource=resource,
                        coordinates= corners, tile_id = i)
            tile.has_thief = has_thief
            has_thief = False
            self.tiles[i] = tile



    def find_closest_node(self, corner):
        """Find the closest existing node to the given corner coordinates."""
        for existing_corner in self.positions_to_id:
            if np.isclose(existing_corner[0], corner[0]) and np.isclose(existing_corner[1], corner[1]):
                return self.positions_to_id[existing_corner]
        raise KeyError(f"No matching node found for corner {corner}")


    def build_initial_settlements(self, *args):
        #Initialize new players with names and colors
        playerColors = ['black', 'darkslateblue', 'magenta4', 'orange1']
        for i in range(self.num_players -1):
            newPlayer = Player(names[i], i, ai_or_not[i])
            self.playerQueue.put(newPlayer)

        self.current_player = self.num_players - 1
        self.players = list(self.playerQueue.queue)  # Get players from queue
        max_dice_player = {0: None}

        # Determine the player with the highest dice roll
        for player in self.players:
            player.color = playerColors[player.number]
            player.roll_dice()  # Roll dice for each player
            if player.dice > max_dice_player[0]:
                max_dice_player = {player.dice: player}

        # Set the current player
        self.current_player = max_dice_player[0].number

        # Rotate the queue so that the highest dice roller is first
        while self.players[0].number != self.current_player:
            self.players.append(self.players.pop(0))  # Rotate left

        # Clear and refill the queue with the new order
        self.playerQueue = queue.Queue()
        for player in self.players:
            self.playerQueue.put(player)

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
                    #rotation by oriented degree pi/2 because this is how de did the hexagon coordinates
                    plt.text(-tile.center[1], tile.center[0], str(tile.number)+", "+tile.resource, fontsize=12, ha='center', va='center', color='black', weight='bold')
            #Color tiles according to their resource type
            colors = {"brick": "red", "sheep": "green", "stone": "gray", "wheat": "yellow", "wood": "brown", "desert": "orange"}
            for tile in self.tiles.values():
                list_x = np.array([element[0] for element in tile.coordinates])
                list_y = np.array([element[1] for element in tile.coordinates])
                plt.fill(list_x, list_y, colors[tile.resource], edgecolor='black', linewidth=2)
            plt.show()
