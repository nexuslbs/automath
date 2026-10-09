import time
from collections import deque
from typing import Optional
import numpy as np
from numpy.random import MT19937, Generator
import torch
from torch import nn
import torch.nn.functional as F

from env import core
from env.action import RawAction
from env.base_agent import BaseAgent
from env.full_state import FullState
from env.node_data import NodeData

def random_generator(seed: int | None = None) -> Generator:
    bg = MT19937(seed)
    rg = Generator(bg)
    return rg

def _get_inner_layers(
    hidden_dim: int,
    hidden_amount: int,
    dropout_rate: float = 0.2,
) -> tuple[nn.Module, ...]:
    """
    Create a tuple of inner layers with dropout for better regularization.

    Args:
        hidden_dim: Size of hidden layers
        hidden_amount: Number of hidden layers to create
        dropout_rate: Probability of dropout (default: 0.2)

    Returns:
        A tuple of layers: (Linear, ReLU, Dropout) repeated hidden_amount times
    """
    layers: list[nn.Module] = []
    for _ in range(hidden_amount):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

    inner_layers: tuple[nn.Module, ...] = tuple(layers)
    return inner_layers

class NodeFeatureExtractor(nn.Module):
    """
    Neural network module that processes the node data array and extracts features.
    This network handles a varying number of nodes by using a node-wise processing approach
    followed by global pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_amount: int,
        output_dim: int,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the feature extractor network.

        Args:
            input_dim: Dimension of each node feature vector
            hidden_dim: Size of hidden layers
            hidden_amount: Number of hidden layers
            output_dim: Size of output feature vector
        """
        super().__init__()

        # Node-wise feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
        )

        # Global feature extraction
        self.global_encoder = nn.Sequential(
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of node data.

        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, input_dim)
              where num_nodes can vary between samples

        Returns:
            Global feature vector of shape (batch_size, output_dim)
        """
        # Process each node independently
        # Shape: (batch_size, num_nodes, hidden_dim)
        node_features: torch.Tensor = self.node_encoder(x)

        # Global pooling (mean across nodes)
        # Shape: (batch_size, hidden_dim)
        global_features = node_features.mean(dim=1)

        # Final feature extraction
        # Shape: (batch_size, output_dim)
        output = self.global_encoder(global_features)

        return output


class ActionPredictor(nn.Module):
    """
    Neural network module that predicts action indices and arguments.
    Arguments prediction is conditioned on the action logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_amount: int,
        action_space_size: int,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the action predictor network.

        Args:
            input_dim: Dimension of input feature vector
            hidden_dim: Size of hidden layers
            hidden_amount: Number of hidden layers
            action_space_size: Number of possible action indices
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.action_space_size = action_space_size
        self.dropout = nn.Dropout(dropout_rate)  # Create dropout layer

        # Common feature processing
        self.common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
        )

        # Action index head
        self.action_head = nn.Sequential(
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Argument heads now take both common features and action logits as input
        self.arg_common = nn.Sequential(
            nn.Linear(hidden_dim + action_space_size, hidden_dim),
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
        )

        # Separate output heads for each argument
        self.arg1_head = nn.Sequential(
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.arg2_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            self.dropout,
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.arg3_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            self.dropout,
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        # Add a joint Q-value head that takes action and all arguments into account
        self.joint_q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_space_size + 3, hidden_dim),
            nn.ReLU(),
            self.dropout,
            *_get_inner_layers(
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, action_space_size)
        )

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Predict action index and arguments.

        Args:
            x: Input feature vector of shape (batch_size, input_dim)

        Returns:
            Tuple of (action_logits, arg1, arg2, arg3, joint_q_values) where:
                action_logits: Action probabilities of shape (batch_size, action_space_size)
                arg1, arg2, arg3: Argument values of shape (batch_size, 1)
                joint_q_values: Q-values considering action and args (batch_size, action_space_size)
        """
        common_features = self.common(x)

        # Predict action logits
        action_logits = self.action_head(common_features)

        # Concatenate common features with action logits
        # pylint: disable=no-member
        combined_features = torch.cat([common_features, action_logits], dim=1)

        # Process combined features for arguments
        arg_features = self.arg_common(combined_features)

        # Generate arguments
        # pylint: disable=not-callable
        arg1 = F.softplus(self.arg1_head(arg_features))
        arg2_features = torch.cat([arg_features, arg1], dim=1)
        arg2 = F.softplus(self.arg2_head(arg2_features))
        arg3_features = torch.cat([arg_features, arg1, arg2], dim=1)
        arg3 = F.softplus(self.arg3_head(arg3_features))

        # Calculate joint Q-values that consider both action and arguments
        # pylint: disable=no-member
        joint_features = torch.cat([common_features, action_logits, arg1, arg2, arg3], dim=1)
        joint_q_values = self.joint_q_head(joint_features)

        return action_logits, arg1, arg2, arg3, joint_q_values

class DQN(nn.Module):
    """
    Deep Q-Network for the agent.
    Combines the feature extractor and action predictor.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        hidden_amount: int,
        action_space_size: int,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the DQN.

        Args:
            input_dim: Dimension of each node feature vector
            feature_dim: Dimension of feature vector produced by node feature extractor
            hidden_dim: Size of hidden layers
            hidden_amount: Number of hidden layers
            action_space_size: Number of possible action indices
            dropout_rate: Probability of dropout for regularization
        """
        super().__init__()

        self.feature_extractor = NodeFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_amount=hidden_amount,
            output_dim=feature_dim,
            dropout_rate=dropout_rate,
        )

        self.action_predictor = ActionPredictor(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            hidden_amount=hidden_amount,
            action_space_size=action_space_size,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Process node data and predict action.

        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, input_dim)
            training: Whether the model is in training mode (affects dropout)

        Returns:
            Tuple of (action_logits, arg1, arg2, arg3, joint_q_values)
        """
        # Set the training mode correctly
        self.train(training)

        features = self.feature_extractor(x)
        return self.action_predictor(features)


class ExperienceReplayBuffer:
    """
    Buffer to store experience tuples for experience replay in DQN.
    """

    def __init__(self, capacity: int = 10000, seed: int | None = None):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: deque[tuple] = deque(maxlen=capacity)
        self.rg = random_generator(seed)

    def add(
        self,
        state: np.ndarray,
        action: list[int],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of experiences
        """
        buffer_len = len(self.buffer)
        size = min(buffer_len, batch_size)

        # Use Generator.choice to randomly select indices without replacement
        indices = self.rg.choice(buffer_len, size=size, replace=False)

        # Return the selected experiences
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)

class SmartAgent(BaseAgent):
    """
    A reinforcement learning agent that uses a DQN to select actions.
    """

    def __init__(
        self,
        action_space_size: int,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        hidden_amount: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        replay_buffer_capacity: int,
        batch_size: int,
        target_update_frequency: int,
        device: Optional[torch.device],
        dropout_rate: float,
        seed: int | None = None,
    ):
        """
        Initialize the SmartAgent.

        Args:
            action_space_size: Number of possible action indices
            input_dim: Dimension of each node feature vector
            feature_dim: Dimension of feature vector produced by node feature extractor
            hidden_dim: Size of hidden layers
            hidden_amount: Number of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Starting value for epsilon (exploration probability)
            epsilon_end: Minimum value for epsilon
            epsilon_decay: Decay rate for epsilon
            replay_buffer_capacity: Capacity of experience replay buffer
            batch_size: Batch size for training
            target_update_frequency: How often to update target network
            device: Device to use for tensor operations
            dropout_rate: Probability of dropout for regularization
            seed: Random seed for reproducibility
        """
        self.action_space_size = action_space_size
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_amount = hidden_amount
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.rg = random_generator(seed)

        # Use GPU if available
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Create policy and target networks
        self.policy_net = DQN(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            hidden_amount=hidden_amount,
            action_space_size=action_space_size,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.target_net = DQN(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            hidden_amount=hidden_amount,
            action_space_size=action_space_size,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Create experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_capacity)

        # Training stats
        self.steps_done = 0

    def _preprocess_state(self, state: FullState) -> torch.Tensor:
        """
        Convert a FullState to a tensor for the neural network.

        Args:
            state: The current environment state

        Returns:
            Tensor representation of the state
        """
        # Convert tree structure to 2D array using NodeData
        node_data = NodeData(node=state, node_types=state.node_types())
        data_array = node_data.to_data_array()
        features = data_array.astype(np.float32)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        return tensor

    def select_action(self, state: FullState) -> RawAction:
        """
        Select an action based on the current state using ε-greedy policy.

        Args:
            state: Current environment state

        Returns:
            Selected action to perform
        """
        # Preprocess state
        state_tensor = self._preprocess_state(state)

        # Epsilon-greedy action selection
        if self.rg.random() > self.epsilon:
            start = time.time()
            # Exploit: use the model
            with torch.no_grad():
                tensors: tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ] = self.policy_net(state_tensor, training=False)
                action_logits, arg1, arg2, arg3, _ = tensors
                # Use joint Q-values instead of action_logits for action selection
                action_idx = int(action_logits.argmax(dim=1).item()) + 1
                arg1_val = int(round(arg1.item()))
                arg2_val = int(round(arg2.item()))
                arg3_val = int(round(arg3.item()))
            end = time.time()
            print(
                time.strftime('%H:%M:%S'),
                f"> [Exploit - Time: {end - start:.4f}s]",
                f"Action: {action_idx}, (Args: {arg1_val}, {arg2_val}, {arg3_val})",
                f"State size: {state_tensor.size()}",
            )
        else:
            # Explore: random action with enhanced exploration using normal distribution
            action_idx = self.rg.integers(1, self.action_space_size + 1, dtype=int)

            # Generate arguments using normal distribution centered at 1 with std=10
            # Round to nearest int and ensure non-negative
            arg1_val = max(0, int(round(self.rg.normal(loc=1.0, scale=10.0))))
            arg2_val = max(0, int(round(self.rg.normal(loc=1.0, scale=10.0))))
            arg3_val = max(0, int(round(self.rg.normal(loc=1.0, scale=10.0))))

            # For random exploration, still add a chance of using 0 for args 2 and 3
            # This helps the agent learn which actions don't need all arguments
            if self.rg.random() < 0.3:
                arg3_val = 0
                if self.rg.random() < 0.3:
                    arg2_val = 0
            print(
                time.strftime('%H:%M:%S'),
                f"> [Explore] Action: {action_idx}, Args: ({arg1_val}, {arg2_val}, {arg3_val})",
                f"State size: {state_tensor.size()}",
            )

        # Create and return RawAction
        raw_action = RawAction.with_raw_args(
            action_index=action_idx,
            arg1=arg1_val,
            arg2=arg2_val,
            arg3=arg3_val
        )

        return raw_action

    def train(
        self,
        state: FullState,
        action: RawAction,
        reward: float,
        next_state: FullState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Train the agent using an experience tuple.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            terminated: Whether episode ended naturally (e.g. goal achieved)
            truncated: Whether episode was cut short (e.g. max steps reached)
        """
        start = time.time()

        # Preprocess states
        state_tensor = self._preprocess_state(state)
        next_state_tensor = self._preprocess_state(next_state)

        # Extract action parameters
        # -1 because our model outputs 0-indexed actions
        action_idx = action.action_index.apply().real(core.IInt).as_int - 1
        arg1_val = action.arg1.apply().real(core.IInt).as_int
        arg2_val = action.arg2.apply().real(core.IInt).as_int
        arg3_val = action.arg3.apply().real(core.IInt).as_int

        # Convert state tensors to numpy arrays for the replay buffer
        state_np = state_tensor.cpu().numpy().squeeze(0)
        next_state_np = next_state_tensor.cpu().numpy().squeeze(0)

        # Store the feature dimension so we can ensure consistency
        feature_dim = state_np.shape[1]

        # Add experience to replay buffer
        self.replay_buffer.add(
            state_np,
            [action_idx, arg1_val, arg2_val, arg3_val],
            reward,
            next_state_np,
            terminated or truncated
        )

        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Extract batch components and convert to tensors
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # Verify that all states in the batch have the same feature dimension
        for b_state, b_action, b_reward, b_next_state, b_done in batch:
            assert b_state.shape[1] == feature_dim
            assert b_next_state.shape[1] == feature_dim
            batch_states.append(b_state)
            batch_actions.append(b_action)
            batch_rewards.append(b_reward)
            batch_next_states.append(b_next_state)
            batch_dones.append(b_done)

        # If we don't have enough valid samples after filtering, raise an error
        assert len(batch_states) >= 2

        # Handle variable-length sequences by padding
        # Find the maximum number of nodes in the batch
        max_nodes = max(len(s) for s in batch_states + batch_next_states)

        # Pad states and next_states to have the same number of nodes
        padded_states = []
        padded_next_states = []

        for s in batch_states:
            padded = np.zeros((max_nodes, s.shape[1]))
            padded[:len(s)] = s
            padded_states.append(padded)

        for s in batch_next_states:
            padded = np.zeros((max_nodes, s.shape[1]))
            padded[:len(s)] = s
            padded_next_states.append(padded)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(padded_states)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch_actions)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(padded_next_states)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch_dones)).to(self.device)

        start_policy = time.time()

        # Compute Q values
        action_logits, arg1, arg2, arg3, joint_q_values = self.policy_net(state_batch)

        end_policy = time.time()
        start_loss = end_policy

        # Get Q values for the actions taken from the joint_q_values
        # that include argument information
        q_values = joint_q_values.gather(1, action_batch[:, 0].unsqueeze(1)).squeeze(1)

        # Compute target Q values using joint Q-values from target network
        with torch.no_grad():
            # Get all outputs from target network
            _, _, _, _, next_joint_q_values = self.target_net(next_state_batch)
            # Use the joint Q-values that incorporate action and argument information
            next_q_values = next_joint_q_values.max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss for joint Q-values (our primary loss)
        joint_q_loss = F.smooth_l1_loss(q_values, target_q_values)

        # Compute additional losses for action logits and arguments (auxiliary losses)
        action_loss = F.cross_entropy(action_logits, action_batch[:, 0])
        arg1_loss = F.smooth_l1_loss(arg1.squeeze(1), action_batch[:, 1].float())
        arg2_loss = F.smooth_l1_loss(arg2.squeeze(1), action_batch[:, 2].float())
        arg3_loss = F.smooth_l1_loss(arg3.squeeze(1), action_batch[:, 3].float())

        # Combine losses with higher weight for joint Q-values
        # The joint Q-values are most important as they'll be used for action selection
        total_loss = (
            (2.0 * joint_q_loss)
            + (0.5 * action_loss)
            + (0.5 * arg1_loss)
            + (0.5 * arg2_loss)
            + (0.5 * arg3_loss)
        )

        end_loss = time.time()
        start_optimize = end_loss

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            assert param.grad is not None
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        end_optimize = time.time()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        end = time.time()

        print(
            time.strftime('%H:%M:%S'),
            f"[Train - Time: {end - start:.4f}s]",
            f"Policy: {end_policy - start_policy:.4f}s",
            f"Loss: {end_loss - start_loss:.4f}s",
            f"Optimize: {end_optimize - start_optimize:.4f}s",
            f'Action Loss: {action_loss.item():.4f}',
            f'Arg1 Loss: {arg1_loss.item():.4f}',
            f'Arg2 Loss: {arg2_loss.item():.4f}',
            f'Arg3 Loss: {arg3_loss.item():.4f}',
            f'Joint Q Loss: {joint_q_loss.item():.4f}',
            f'Total Loss: {total_loss.item():.4f}',
        )

    def reset(self) -> None:
        """
        Reset the agent's episode-specific variables.
        """
        # Nothing to reset for now, epsilon decay continues between episodes

    def save(self, path: str) -> None:
        """
        Save the agent's model to disk.

        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        """
        Load the agent's model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
