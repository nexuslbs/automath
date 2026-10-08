# MDP Settings
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100

# Bounded-state setting (design unit U3a).
# Maximum number of HistoryNode entries retained in FullState.history for one
# episode. When the caller of GoalEnv does not pass max_history_state_size,
# this value is used. The unbounded arithmetic run grew to ~1.80 GB RSS; 128
# retained entries keep one episode's history bounded (O(128) states) while
# leaving plenty of recent context for an agent.
DEFAULT_MAX_HISTORY_STATE_SIZE = 128

# Agent Hyperparameters
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# Logging Settings
LOGGER = "automath"
LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
LOG_FILE = "agent.log"
