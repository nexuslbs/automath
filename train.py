#!/usr/bin/env python
"""
Main training script that trains the SmartAgent using settings from the config directory.
"""
import os
import time

from config import agent_settings
from utils.env_logger import env_logger
from env.goal_env import GoalEnv
from env.node_types import HaveScratch
from env.base_agent import BaseAgent
from env import core
from agent.train import train_agent
from agent.simple_agent import SimpleAgent
from agent.smart_agent import SmartAgent
from env.macro_action import MacroActionEnv

def get_action_space_size() -> int:
    # The DQN selects MACRO-actions (design unit U3 fix 2): the action space is
    # the macro-action catalogue, not the 22-34 primitive typed actions.
    tmp_env = GoalEnv(goal=HaveScratch.with_goal(core.Void()))
    return MacroActionEnv(tmp_env).action_space_size()


def main() -> None:
    """
    Main function to run the agent training process.

    Loads settings from config/agent_settings.py and passes them to the
    train_smart_agent function in agent/train.py.
    """
    start_time = time.time()
    env_logger.info("Starting training using settings from config/agent_settings.py")

    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(agent_settings.MODEL_PATH), exist_ok=True)

    # Train the agent
    try:
        action_space_size = get_action_space_size()
        input_dim = agent_settings.INPUT_DIM
        feature_dim = agent_settings.FEATURE_DIM
        hidden_dim = agent_settings.HIDDEN_DIM
        hidden_amount = agent_settings.HIDDEN_AMOUNT
        learning_rate = agent_settings.LEARNING_RATE
        gamma = agent_settings.GAMMA
        epsilon_start = agent_settings.EPSILON_START
        epsilon_end = agent_settings.EPSILON_END
        epsilon_decay = agent_settings.EPSILON_DECAY
        replay_buffer_capacity = agent_settings.REPLAY_BUFFER_CAPACITY
        batch_size = agent_settings.BATCH_SIZE
        target_update_frequency = agent_settings.TARGET_UPDATE_FREQUENCY
        episodes_per_state = agent_settings.EPISODES_PER_STATE
        max_steps_per_episode = agent_settings.MAX_STEPS_PER_EPISODE
        dropout_rate = agent_settings.DROPOUT_RATE
        save_interval = agent_settings.SAVE_INTERVAL
        agent_type = agent_settings.AGENT_TYPE
        model_path = agent_settings.MODEL_PATH
        device = agent_settings.DEVICE
        seed = agent_settings.SEED

        if agent_type == "simple":
            agent: BaseAgent = SimpleAgent(
                action_space_size=action_space_size,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                max_arg_value=1000,
                seed=seed,
            )
        elif agent_type == "smart":
            agent = SmartAgent(
                action_space_size=action_space_size,
                input_dim=input_dim,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                hidden_amount=hidden_amount,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                replay_buffer_capacity=replay_buffer_capacity,
                batch_size=batch_size,
                target_update_frequency=target_update_frequency,
                device=device,
                dropout_rate=dropout_rate,
                seed=seed,
            )
            # Marker read by agent/train.py: train over the macro-action space.
            agent.action_space_is_macro = True
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        train_agent(
            agent=agent,
            model_path=model_path,
            episodes_per_state=episodes_per_state,
            max_steps_per_episode=max_steps_per_episode,
            save_interval=save_interval,
        )

        env_logger.info(
            f"Training completed successfully in {time.time() - start_time:.2f} seconds"
        )
    except Exception as e:
        env_logger.error(f"Training failed: {e}")
        raise e


if __name__ == "__main__":
    main()
