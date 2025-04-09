import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from board import *
from player import *
from geometric_utils import *
from copy import deepcopy
import queue
import os
import time

# ===========================================================
# 1. Global Deep Network Architecture (DeepCatanNet)
# ===========================================================

class ResidualBlock(nn.Module):
    """
    Residual block using a 5x3 kernel as described in Deep Catan.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1))
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out)

class DeepCatanNet(nn.Module):
    """
    Global network for Catan that takes two inputs:
      - A board representation as a 2D tensor with 29 channels (23x13 image)
      - A game_info vector with additional game state features.
    Outputs a policy (log-probabilities over actions) and a scalar value prediction.
    """
    def __init__(self, board_channels=29, board_height=23, board_width=13, game_info_dim=200, action_size=6, num_res_blocks=5):
        super(DeepCatanNet, self).__init__()
        self.board_conv = nn.Conv2d(board_channels, 128, kernel_size=(5, 3), padding=(2, 1))
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_res_blocks)])
        self.board_fc = nn.Linear(128 * board_height * board_width, 256)
        self.game_info_fc = nn.Linear(game_info_dim, 256)
        self.combined_fc = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.policy_head = nn.Linear(256, action_size)
        # Changed value head to output a single scalar.
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, board, game_info):
        x = F.relu(self.board_conv(board))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        board_repr = F.relu(self.board_fc(x))
        game_info_repr = F.relu(self.game_info_fc(game_info))
        combined = torch.cat([board_repr, game_info_repr], dim=1)
        combined = F.relu(self.combined_fc(combined))
        combined = self.dropout(combined)
        # Policy branch: log probabilities with log_softmax.
        policy_logits = self.policy_head(combined)
        policy = F.log_softmax(policy_logits, dim=1)
        # Value branch: outputs a single scalar per sample.
        value = self.value_head(combined)
        return policy, value

# ===========================================================
# 2. Local Value Network
# ===========================================================

class LocalValueNet(nn.Module):
    """
    A fully connected network that evaluates local move features.
    Input dimension: game_info_dim + action_size.
    Outputs a scalar prediction of the move's AMAF score.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(LocalValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def extract_move_features(state, action, action_size):
    """
    Extract local move features from the state and chosen action.
    For simplicity, we use the game_info vector concatenated with a one-hot encoding of the action.
    """
    board, game_info = state
    game_info_tensor = torch.FloatTensor(game_info)  # shape (game_info_dim,)
    one_hot = torch.zeros(action_size)
    one_hot[action] = 1.0
    features = torch.cat([game_info_tensor, one_hot])  # Dimension = game_info_dim + action_size
    return features

# ===========================================================
# 3. Catan Environment Wrapper (Simplified)
# ===========================================================
class CatanGame:
    def __init__(self, names, ai_or_not, verbose=False):
        self.board = Board(names, ai_or_not, verbose=verbose)
        self.num_players = self.board.num_players
        self.current_player = self.board.players[0]
        self.board_channels = 29
        self.board_height = 23
        self.board_width = 13
        self.game_info_dim = 200
        self.action_size = 6

    def get_initial_state(self):
        board = np.zeros((self.board_channels, self.board_height, self.board_width), dtype=np.float32)
        game_info = np.zeros((self.game_info_dim,), dtype=np.float32)
        return (board, game_info)

    def encode_state(self):
        board = np.zeros((self.board_channels, self.board_height, self.board_width), dtype=np.float32)
        game_info = np.zeros((self.game_info_dim,), dtype=np.float32)
        # --- Encoding board features (tiles, roads, settlements, etc.) ---
        # ... Your detailed implementation goes here ...
        # **Encode Player Information**
        player = self.current_player
        game_info[0:self.num_players] = [p.victory_points() for p in self.board.players]
        resource_types = ["brick", "wood", "wheat", "sheep", "stone"]
        for i, r in enumerate(resource_types):
            game_info[10 + i] = player.resources.get(r, 0)
        game_info[20] = len(player.settlements)
        game_info[21] = len(player.cities)
        game_info[22] = len(player.roads)
        dev_cards = ["knight", "victoryPoint", "roadBuilding", "yearOfPlenty", "monopoly"]
        for i, card in enumerate(dev_cards):
            game_info[30 + i] = player.cards.get(card, 0)
        game_info[40:40 + self.num_players] = 0
        game_info[40 + self.board.players.index(player)] = 1
        return board, game_info

    def legal_actions(self):
        legal = []
        player = self.current_player
        # 0 - Build Settlement
        cost_settlement = {"brick": 1, "wood": 1, "wheat": 1, "sheep": 1}
        if (all(player.resources.get(r, 0) >= cost_settlement[r] for r in cost_settlement) and
            len(self.board.get_valid_nodes(player)) > 0):
            legal.append(0)

        # 1 - Build Road
        cost_road = {"brick": 1, "wood": 1}
        if (all(player.resources.get(r, 0) >= cost_road[r] for r in cost_road) and 
            len(self.board.get_valid_edges(player)) > 0):
            legal.append(1)

        # 2 - Upgrade to City
        cost_city = {"stone": 3, "wheat": 2}
        has_valid_settlement = any(node.structure == Structure.house for node in player.settlements)
        if (all(player.resources.get(r, 0) >= cost_city[r] for r in cost_city) and 
            has_valid_settlement):
            legal.append(2)

        # 3 - Trade with Bank
        if player.resources.get("brick", 0) >= 4:
            legal.append(3)

        # 4 - Buy Development Card
        cost_dev = {"stone": 1, "wheat": 1, "sheep": 1}
        if all(player.resources.get(r, 0) >= cost_dev[r] for r in cost_dev):
            legal.append(4)

        # 5 - End Turn (always allowed)
        legal.append(5)

        return legal

    def step(self, action):
        player = self.current_player
        done = False
        reward = 0.0
        info = {}
        if action == 0:
            cost = {"brick": 1, "wood": 1, "wheat": 1, "sheep": 1}
            if all(player.resources[r] >= cost[r] for r in cost):
                for r in cost:
                    player.resources[r] -= cost[r]
                self.board.build(player, 'SETTLE')
                reward = 10.0
            else:
                reward = -0.2
        elif action == 1:
            cost = {"brick": 1, "wood": 1}
            if all(player.resources[r] >= cost[r] for r in cost):
                for r in cost:
                    player.resources[r] -= cost[r]
                self.board.build(player, 'ROAD')
                reward = 9.5
            else:
                reward = -0.2
        elif action == 2:
            cost = {"stone": 3, "wheat": 2}
            if all(player.resources[r] >= cost[r] for r in cost):
                for r in cost:
                    player.resources[r] -= cost[r]
                self.board.build(player, 'CITY')
                reward = 10.0
            else:
                reward = -0.3
        elif action == 3:
            success = self.board.trade_with_bank(player, "brick", "wood", ratio=4)
            reward = 2.0 if success else -0.2
        elif action == 4:
            cost = {"stone": 1, "wheat": 1, "sheep": 1}
            if all(player.resources[r] >= cost[r] for r in cost):
                for r in cost:
                    player.resources[r] -= cost[r]
                player.cards["victoryPoint"] += 1
                reward = 3.0
            else:
                reward = -0.2
        elif action == 5:
            reward = -10
        next_state = self.encode_state()
        if self.board.check_winner():
            done = True
            reward += 500.0
        self.current_player = self.board.next_turn()
        return next_state, reward, done, info

    def is_terminal(self):
        return self.board.check_winner() is not None

    def current_player_index(self):
        return self.current_player.number

# ===========================================================
# 4. Monte Carlo Tree Search (MCTS) with Local Value Integration
# ===========================================================

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # (board, game_info)
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0

    def q_value(self):
        return 0 if self.visit_count == 0 else self.total_value / self.visit_count

class MCTS:
    def __init__(self, game, net, local_value_net, simulations=50, c_puct=1.0, c2=1.0):
        self.game = game
        self.net = net
        self.local_value_net = local_value_net
        self.simulations = simulations
        self.c_puct = c_puct
        self.c2 = c2
        self.local_value_data = []  # To collect (features, target) for local network training

    def search(self, state):
        root = MCTSNode(state)
        board, game_info = state
        state_tensor = torch.FloatTensor(board).unsqueeze(0)
        game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0)
        with torch.no_grad():
            log_policy, value = self.net(state_tensor, game_info_tensor)
        policy = torch.exp(log_policy).squeeze(0).numpy()
        legal = self.game.legal_actions()
        for action in legal:
            child_state = deepcopy(state)
            child_node = MCTSNode(child_state, parent=root)
            child_node.prior = policy[action]
            root.children[action] = child_node

        for _ in range(self.simulations):
            self._simulate(root)
        return root

    def _simulate(self, node):
        current = node
        search_path = [current]
        while current.children:
            best_score = -float('inf')
            best_action, best_child = None, None
            for action, child in current.children.items():
                # Standard UCB term
                ucb = child.q_value() + self.c_puct * child.prior * np.sqrt(current.visit_count + 1) / (1 + child.visit_count)
                # Add local value evaluation term:
                features = extract_move_features(current.state, action, self.game.action_size)
                local_eval = self.local_value_net(features.unsqueeze(0))
                ucb += self.c2 * local_eval.item()
                if ucb > best_score:
                    best_score = ucb
                    best_action = action
                    best_child = child
            current = best_child
            search_path.append(current)

        board, game_info = current.state
        state_tensor = torch.FloatTensor(board).unsqueeze(0)
        game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0)
        with torch.no_grad():
            log_policy, value = self.net(state_tensor, game_info_tensor)
        # Since the network now returns a single scalar per state, use value.item()
        leaf_value = value.item()
        if not self.game.is_terminal():
            policy = torch.exp(log_policy).squeeze(0).numpy()
            legal = self.game.legal_actions()
            for action in legal:
                if action not in current.children:
                    child_state = deepcopy(current.state)
                    child_node = MCTSNode(child_state, parent=current)
                    child_node.prior = policy[action]
                    current.children[action] = child_node
                    # Record local features training sample: (features, leaf_value)
                    feat = extract_move_features(current.state, action, self.game.action_size)
                    self.local_value_data.append((feat, leaf_value))
        for node in search_path:
            node.visit_count += 1
            node.total_value += leaf_value

    def get_action_probabilities(self, root, temperature=1.0):
        counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(self.game.action_size)])
        if temperature == 0:
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
            return probs
        counts = counts ** (1.0 / temperature)
        return counts / np.sum(counts)

# ===========================================================
# 5. Training Loop and Self-Play (with Local Value Training)
# ===========================================================

def self_play_episode(game, net, local_value_net, mcts_simulations=50, temperature=4.0, max_moves=400, plotting=False):
    examples = []
    state = game.get_initial_state()
    done = False
    episode_length = 0
    mcts = MCTS(game, net, local_value_net, simulations=mcts_simulations, c_puct=1.5, c2=0.5)
    while not done and episode_length < max_moves:
        root = mcts.search(state)
        action_probs = mcts.get_action_probabilities(root, temperature)
        examples.append((state, action_probs, None))  # Reward will be assigned later
        action = np.random.choice(range(game.action_size), p=action_probs)
        state, reward, done, _ = game.step(action)
        episode_length += 1
    if not done:
        reward = -0.5
    if plotting:
        print("Episode finished. Final state:")
        game.board.plot_board()
        print(f"Episode length: {episode_length}, Final reward: {reward}")
    
    # Assign the final outcome as the reward for all moves of this episode.
    episode_data = [(s, p, reward) for s, p, _ in examples]
    return episode_data, reward, episode_length, mcts.local_value_data

def train_deepnet(net, optimizer, training_data, batch_size=32, epochs=5):
    net.train()
    epoch_losses = []
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            boards = torch.FloatTensor([x[0][0] for x in batch])
            game_infos = torch.FloatTensor([x[0][1] for x in batch])
            target_policies = torch.FloatTensor([x[1] for x in batch])
            target_values = torch.FloatTensor([[x[2]] for x in batch])
            optimizer.zero_grad()
            out_policies, out_values = net(boards, game_infos)
            # Policy loss: negative log likelihood
            loss_policy = -torch.mean(torch.sum(target_policies * out_policies, dim=1))
            # Value loss: use MSE directly between scalar outputs and targets.
            loss_value = F.mse_loss(out_values, target_values)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        print(f"Global Net Epoch {epoch+1}: avg loss {avg_loss:.4f}")
    return np.mean(epoch_losses)

def train_local_value_net(local_net, optimizer_local, training_data, batch_size=32, epochs=5):
    local_net.train()
    epoch_losses = []
    if not training_data:
        return 0.0
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            features = torch.stack([x[0] for x in batch])
            targets = torch.FloatTensor([[x[1]] for x in batch])
            optimizer_local.zero_grad()
            outputs = local_net(features)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer_local.step()
            losses.append(loss.item())
        avg_local_loss = np.mean(losses)
        epoch_losses.append(avg_local_loss)
        print(f"Local Value Net Epoch {epoch+1}: avg loss {avg_local_loss:.4f}")
    return np.mean(epoch_losses)

def main_training_loop():
    log_dir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    names = ["Alice", "Bob", "Carol", "Dave"]
    ai_or_not = [True, True, True, True]
    game = CatanGame(names, ai_or_not, verbose =  False)
    net = DeepCatanNet(board_channels=game.board_channels,
                       board_height=game.board_height,
                       board_width=game.board_width,
                       game_info_dim=game.game_info_dim,
                       action_size=game.action_size)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    input_dim_local = game.game_info_dim + game.action_size
    local_value_net = LocalValueNet(input_dim=input_dim_local, hidden_dim=128)
    optimizer_local = optim.Adam(local_value_net.parameters(), lr=0.001)
    
    training_data = []
    local_training_data = []
    num_iterations = 1000
    for iteration in range(num_iterations):
        print(f"Self-play iteration {iteration+1}")
        if iteration > 0:
            game.board.reset()
        episode_data, final_reward, episode_length, mcts_local_data = self_play_episode(
            game, net, local_value_net, mcts_simulations=50, temperature=1.0, plotting=True, max_moves=1000)
        training_data.extend(episode_data)
        local_training_data.extend(mcts_local_data)
        if len(training_data) > 10000:
            print(len(training_data))
            training_data = training_data[-10000:]
        writer.add_scalar("SelfPlay/EpisodeReward", final_reward, iteration)
        writer.add_scalar("SelfPlay/EpisodeLength", episode_length, iteration)
        writer.add_scalar("SelfPlay/TrainingDataSize", len(training_data), iteration)
        avg_loss = train_deepnet(net, optimizer, training_data, batch_size=32, epochs=5)
        writer.add_scalar("Train/GlobalAverageLoss", avg_loss, iteration)
        avg_local_loss = train_local_value_net(local_value_net, optimizer_local, local_training_data, batch_size=32, epochs=5)
        writer.add_scalar("Train/LocalAverageLoss", avg_local_loss, iteration)
        # Optionally, clear local training data to train fresh every iteration.
        local_training_data = []
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f"deepcatan_net_iter{iteration+1}.pth")
            torch.save(local_value_net.state_dict(), f"local_value_net_iter{iteration+1}.pth")
            writer.add_text("ModelCheckpoint", f"Saved at iteration {iteration+1}", iteration)
    
    writer.close()

if __name__ == "__main__":
    main_training_loop()
