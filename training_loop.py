import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import queue
import os
import time

# ===========================================================
# 1. Deep Network Architecture (DeepCatanNet)
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
    Deep network for Catan that takes two inputs:
      - A board representation as a 2D tensor with 29 channels (23x13 image)
      - A game_info vector with additional game state features.
    The network processes the board via convolution and residual blocks and then
    fuses it with the game info before outputting a policy (log-probabilities over actions)
    and a value prediction (softmax over 4 players' win probabilities).
    """
    def __init__(self, board_channels=29, board_height=23, board_width=13, game_info_dim=200, action_size=10, num_res_blocks=5):
        super(DeepCatanNet, self).__init__()
        # Initial convolution on board input using the 5x3 brick kernel.
        self.board_conv = nn.Conv2d(board_channels, 128, kernel_size=(5, 3), padding=(2, 1))
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_res_blocks)])
        # Flatten board representation.
        self.board_fc = nn.Linear(128 * board_height * board_width, 256)
        # Process game information vector.
        self.game_info_fc = nn.Linear(game_info_dim, 256)
        # Combine board and game info.
        self.combined_fc = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        # Policy head: outputs log-probabilities for each action.
        self.policy_head = nn.Linear(256, action_size)
        # Value head: outputs 4 numbers (one per player) and applies softmax.
        self.value_head = nn.Linear(256, 4)
    
    def forward(self, board, game_info):
        # board: (batch, 29, 23, 13)
        x = F.relu(self.board_conv(board))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        board_repr = F.relu(self.board_fc(x))
        game_info_repr = F.relu(self.game_info_fc(game_info))
        combined = torch.cat([board_repr, game_info_repr], dim=1)
        combined = F.relu(self.combined_fc(combined))
        combined = self.dropout(combined)
        policy_logits = self.policy_head(combined)
        policy = F.log_softmax(policy_logits, dim=1)
        value_logits = self.value_head(combined)
        value = F.softmax(value_logits, dim=1)  # win probabilities for 4 players
        return policy, value

# ===========================================================
# 2. Catan Environment Wrapper (Simplified)
# ===========================================================
# This class wraps your existing Catan game logic.
# For illustration, get_initial_state returns a dummy state comprising:
#   - board: a tensor of shape (29, 23, 13)
#   - game_info: a vector (e.g., length 200)
# You must adapt these functions to encode your board and game information.

class CatanGame:
    def __init__(self, names, ai_or_not):
        # Create the board using your existing Board class.
        self.board = Board(names, ai_or_not)
        self.num_players = self.board.num_players
        self.current_player = self.board.players[0]
        # Set state and action dimensions.
        self.board_channels = 29
        self.board_height = 23
        self.board_width = 13
        self.game_info_dim = 200  # Adjust based on your state encoding details.
        self.action_size = 10     # Define your action space accordingly.
    
    def get_initial_state(self):
        """
        Returns an initial state as a tuple (board, game_info):
          - board: a float tensor with shape (29, 23, 13)
          - game_info: a float tensor with shape (game_info_dim,)
        For demonstration purposes, these are initialized with zeros.
        """
        board = np.zeros((self.board_channels, self.board_height, self.board_width), dtype=np.float32)
        game_info = np.zeros((self.game_info_dim,), dtype=np.float32)
        return (board, game_info)
    
    def legal_actions(self):
        # Return a list of legal action indices.
        # In a complete implementation, compute legal moves based on the board state.
        return list(range(self.action_size))
    
    def step(self, action):
        """
        Apply an action to the environment.
        This should update the board state and change the current player.
        Returns: next_state (tuple), reward, done flag, and an info dict.
        This implementation is a stub.
        """
        # Here you should integrate the action (e.g., build settlement, road, upgrade, trade, etc.)
        # For illustration, we simulate a random state transition.
        next_board = np.random.rand(self.board_channels, self.board_height, self.board_width).astype(np.float32)
        next_game_info = np.random.rand(self.game_info_dim).astype(np.float32)
        next_state = (next_board, next_game_info)
        done = False
        reward = 0.0
        # Optionally, use self.board.check_winner() to decide terminal state.
        if self.board.check_winner() is not None:
            done = True
            reward = 1.0  # Adjust the reward as needed.
        # Advance turn.
        self.current_player = self.board.next_turn()
        return next_state, reward, done, {}
    
    def is_terminal(self):
        # For simplicity, we check if a winner exists.
        return self.board.check_winner() is not None
    
    def current_player_index(self):
        return self.current_player.number

# ===========================================================
# 3. Monte Carlo Tree Search (MCTS) Components
# ===========================================================

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # state is a tuple: (board, game_info)
        self.parent = parent
        self.children = {}  # action -> child MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = 0.0  # Prior probability from the neural network
    
    def q_value(self):
        return 0 if self.visit_count == 0 else self.total_value / self.visit_count

class MCTS:
    def __init__(self, game, net, simulations=50, c_puct=1.0):
        self.game = game
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
    
    def search(self, state):
        # Create the root node.
        root = MCTSNode(state)
        board, game_info = state
        state_tensor = torch.FloatTensor(board).unsqueeze(0)
        game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0)
        with torch.no_grad():
            log_policy, value = self.net(state_tensor, game_info_tensor)
        policy = torch.exp(log_policy).squeeze(0).numpy()
        legal = self.game.legal_actions()
        # Initialize root children with priors.
        for action in legal:
            child_state = deepcopy(state)  # In practice, simulate action to obtain next state.
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
                ucb = child.q_value() + self.c_puct * child.prior * np.sqrt(current.visit_count + 1) / (1 + child.visit_count)
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
        leaf_value = value.mean().item()  # Use average win probability.
        # Expand node if not terminal.
        if not self.game.is_terminal():
            policy = torch.exp(log_policy).squeeze(0).numpy()
            legal = self.game.legal_actions()
            for action in legal:
                if action not in current.children:
                    child_state = deepcopy(current.state)
                    child_node = MCTSNode(child_state, parent=current)
                    child_node.prior = policy[action]
                    current.children[action] = child_node
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
# 4. Self-Play and Training Loop with Metrics Logging
# ===========================================================

def self_play_episode(game, net, mcts_simulations=50, temperature=1.0):
    """
    Plays one episode using MCTS guided by the current network.
    Returns a list of training examples: (state, mcts_probs, reward)
    where state is a tuple (board, game_info).
    """
    examples = []
    state = game.get_initial_state()
    done = False
    episode_length = 0
    mcts = MCTS(game, net, simulations=mcts_simulations)
    while not done:
        root = mcts.search(state)
        action_probs = mcts.get_action_probabilities(root, temperature)
        examples.append((state, action_probs))
        action = np.random.choice(range(game.action_size), p=action_probs)
        state, reward, done, _ = game.step(action)
        episode_length += 1
    return examples, reward, episode_length

def train_deepnet(net, optimizer, training_data, batch_size=32, epochs=5):
    net.train()
    epoch_losses = []
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            # Unpack state which is a tuple (board, game_info)
            boards = torch.FloatTensor([x[0][0] for x in batch])
            game_infos = torch.FloatTensor([x[0][1] for x in batch])
            target_policies = torch.FloatTensor([x[1] for x in batch])
            target_values = torch.FloatTensor([[x[2]] for x in batch])
            optimizer.zero_grad()
            out_policies, out_values = net(boards, game_infos)
            loss_policy = -torch.mean(torch.sum(target_policies * out_policies, dim=1))
            loss_value = F.mse_loss(out_values.mean(dim=1, keepdim=True), target_values)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.4f}")
    return np.mean(epoch_losses)

def main_training_loop():
    # Create a logs directory with timestamp.
    log_dir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    # Hyperparameters and initialization.
    names = ["Alice", "Bob", "Carol", "Dave"]
    ai_or_not = [False, False, True, True]
    game = CatanGame(names, ai_or_not)
    net = DeepCatanNet(board_channels=game.board_channels,
                       board_height=game.board_height,
                       board_width=game.board_width,
                       game_info_dim=game.game_info_dim,
                       action_size=game.action_size)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    training_data = []
    num_iterations = 1000
    for iteration in range(num_iterations):
        print(f"Self-play iteration {iteration+1}")
        episode_examples, final_reward, episode_length = self_play_episode(game, net, mcts_simulations=50, temperature=1.0)
        # For simplicity, we assign the final reward to every state in the episode.
        episode_data = [(s, p, final_reward) for s, p in episode_examples]
        training_data.extend(episode_data)
        # Limit dataset size.
        if len(training_data) > 10000:
            training_data = training_data[-10000:]
        
        # Log self-play metrics.
        writer.add_scalar("SelfPlay/EpisodeReward", final_reward, iteration)
        writer.add_scalar("SelfPlay/EpisodeLength", episode_length, iteration)
        writer.add_scalar("SelfPlay/TrainingDataSize", len(training_data), iteration)
        
        # Train network on the collected data.
        avg_loss = train_deepnet(net, optimizer, training_data, batch_size=32, epochs=5)
        writer.add_scalar("Train/AverageLoss", avg_loss, iteration)
        
        # Save the network periodically.
        if (iteration + 1) % 100 == 0:
            torch.save(net.state_dict(), f"deepcatan_net_iter{iteration+1}.pth")
            writer.add_text("ModelCheckpoint", f"Saved at iteration {iteration+1}", iteration)
    
    writer.close()

if __name__ == "__main__":
    main_training_loop()
