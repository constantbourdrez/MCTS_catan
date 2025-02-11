class Player():
    def __init__(self, name, number, is_ai=False):
        self.name = name
        self.number = number
        self.is_ai = is_ai
        self.ressources = {"brick": 0, "sheep": 0, "stone": 0, "wheat": 0, "wood": 0}
        self.victory_points = 0
        self.cards = {"buildRoad": 0, "knight": 0, "monopoly": 0, "plenty": 0, "victoryPoint": 0}

    def chose_node(self, node):
        pass
