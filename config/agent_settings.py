import typing

INPUT_DIM: int = 8
FEATURE_DIM: int = 2**7
HIDDEN_DIM: int = 2**6
HIDDEN_AMOUNT: int = 2**2
LEARNING_RATE: float = 0.003
GAMMA: float = 0.99
EPSILON_START: float = 0.9
EPSILON_END: float = 0.005
EPSILON_DECAY: float = 0.9999
REPLAY_BUFFER_CAPACITY: int = 2**7
BATCH_SIZE: int = 2**5
TARGET_UPDATE_FREQUENCY: int = 100
EPISODES_PER_STATE: int = 1
MAX_STEPS_PER_EPISODE: int = 100
MAX_STATE_SIZE: int = 500000  # Max state cost before truncation
DROPOUT_RATE: float = 0.1
FORCE_EXPLORATION_INTERVAL: int = 50
SAVE_INTERVAL: int = 1
DEVICE: typing.Optional[typing.Any] = None
SEED: int | None = 1
AGENT_TYPE: str = "smart"
AGENT_NAME: str = f"{AGENT_TYPE}_{FEATURE_DIM}_{HIDDEN_DIM}_{SEED or 0}"
MODEL_PATH: str = (
    "tmp/trained_model.pt"
    if AGENT_TYPE == "smart"
    else f"tmp/trained_model_{AGENT_NAME}.pt"
)
