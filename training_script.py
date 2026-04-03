import torch
import numpy as np
import os

from dqn_agent import DQNAgent
from environment import create_env
from utils import load_config, plot_results, deploy, log_progress, save_checkpoint


# ===================================================================
# --- Warmup: Fill Replay Buffer ---
# ===================================================================
def warmup(env, agent):
    """
    Fills the replay buffer with random transitions before training begins.

    Uses purely random actions (no network forward passes) to collect
    diverse initial experiences. This is more efficient than checking
    the buffer size on every learn() call during training.

    For Rainbow DQN, the agent's store_transition() handles N-step
    accumulation internally. The while loop accounts for the delayed
    buffer filling (transitions only reach the replay buffer after
    N steps are accumulated).

    Args:
        env (gym.Env): The preprocessed Atari environment.
        agent (DQNAgent): The Rainbow DQN agent whose buffer will be filled.

    Returns:
        int: The number of environment steps taken during warmup.
    """
    print(f"Warming up: collecting {agent.learning_starts} random transitions...")
    state, info = env.reset()
    steps = 0
    while len(agent.replay_buffer) < agent.learning_starts:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        terminal = done or truncated
        agent.store_transition(np.array(state), action, reward, np.array(next_state), terminal)
        if terminal:
            state, info = env.reset()
        else:
            state = next_state
        steps += 1
    # Clear any partial N-step transitions from warmup so they don't
    # leak into the first training episode.
    agent.n_step_buffer.clear()
    print(f"Warmup complete. Buffer size: {len(agent.replay_buffer)}, Steps: {steps}")
    return steps


# ===================================================================
# --- The Training Function ---
# ===================================================================
def train(env, agent, num_steps, all_rewards, all_losses,
          print_every, moving_avg_window, target_reward,
          checkpoint_every, model_filepath, history_filepath, plot_filepath,
          is_new_run=True):
    """
    Executes the main step-based training loop for Rainbow DQN.

    Handles warmup (filling the replay buffer) internally before starting
    the training loop. The outer loop counts total agent steps (not episodes).
    Episodes start and end naturally within the loop, matching the original
    DeepMind DQN approach. Rainbow uses NoisyLinear layers for exploration
    instead of epsilon-greedy, and stores N-step transitions in a PER buffer.

    Args:
        env (gym.Env): The preprocessed Atari environment.
        agent (DQNAgent): The Rainbow DQN agent.
        num_steps (int): Total number of agent steps to train for.
        all_rewards (list): List to store reward history (per episode).
        all_losses (list): List to store loss history (per episode).
        print_every (int): Frequency (in steps) to log progress to console.
        moving_avg_window (int): Window size for reward logging.
        target_reward (float): Average reward threshold for early stopping.
        checkpoint_every (int): Frequency (in steps) to save model checkpoints. 0 to disable.
        model_filepath (str): Path to save model weights.
        history_filepath (str): Path to save training history.
        plot_filepath (str): Path to save the training plot.
        is_new_run (bool): If True, warmup steps count toward total_steps and
            exploration state is synced. False when resuming from a checkpoint.

    Returns:
        tuple: (all_rewards, all_losses)
    """
    # Warmup: fill the replay buffer with random transitions if needed.
    # The buffer is not persisted across runs, so this runs for both
    # new and resumed training.
    if len(agent.replay_buffer) < agent.learning_starts:
        warmup_steps = warmup(env, agent)
        if is_new_run:
            agent.total_steps = warmup_steps
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()
            if hasattr(agent, "_update_beta"):
                agent._update_beta()

    print(f"Starting training from step {agent.total_steps} (target: {num_steps} steps)...")

    state, info = env.reset()
    episode_reward = 0.0     # Accumulates reward for the current episode
    episode_losses = []      # Collects per-step losses for the current episode

    # Track step counts for periodic logging and checkpointing.
    # Initialized to agent.total_steps so that resumed runs don't
    # immediately trigger a log/checkpoint on the first step.
    last_print_step = agent.total_steps
    last_checkpoint_step = agent.total_steps

    # Main training loop: runs until the agent has taken num_steps total actions.
    # Episodes start and end naturally within this loop — there is no outer
    # episode loop, matching the original DeepMind DQN training procedure.
    while agent.total_steps < num_steps:
        # --- [RAINBOW: NOISY] 1. Choose action — no epsilon-greedy needed ---
        # Vanilla DQN: action = agent.choose_action(state) with epsilon-greedy
        # NoisyLinear layers handle exploration; increments total_steps and updates beta.
        action = agent.choose_action(np.array(state))

        # 2. Step in the environment.
        #    done = True if the game ended naturally (e.g., lost all lives)
        #    truncated = True if the episode hit a time limit
        next_state, reward, done, truncated, info = env.step(action)
        terminal = done or truncated

        # --- [RAINBOW: N-STEP+PER] 3. Store transition ---
        # Vanilla DQN: agent.store_transition(state, action, reward, next_state, done) — single-step, uniform buffer
        # Rainbow: accumulates N transitions, computes N-step return, stores in PER buffer.
        agent.store_transition(np.array(state), action, reward, np.array(next_state), terminal)

        # --- [RAINBOW: PER+C51] 4. Learn from a prioritized mini-batch ---
        # Vanilla DQN: learns from uniform mini-batch with MSE loss
        loss = agent.learn()
        episode_losses.append(loss)

        # 5. Move to next state
        state = next_state
        episode_reward += reward

        # --- Episode ended ---
        # When a life is lost (terminal_on_life_loss=True), the episode ends.
        # Record the episode's total reward and average loss, then reset.
        if terminal:
            all_rewards.append(episode_reward)
            all_losses.append(np.mean(episode_losses) if episode_losses else np.nan)

            # Reset environment and accumulators for the next episode
            state, info = env.reset()
            episode_reward = 0.0
            episode_losses = []

        # --- Periodic logging (step-based) ---
        if agent.total_steps - last_print_step >= print_every:
            last_print_step = agent.total_steps
            should_stop = log_progress(agent, num_steps, all_rewards,
                                       moving_avg_window, target_reward)
            if should_stop:
                break

        # --- Periodic checkpoint (step-based) ---
        if checkpoint_every > 0 and agent.total_steps - last_checkpoint_step >= checkpoint_every:
            last_checkpoint_step = agent.total_steps
            save_checkpoint(agent, all_rewards, all_losses,
                            model_filepath, history_filepath, plot_filepath,
                            moving_avg_window)

    print("Training finished.")
    return all_rewards, all_losses


# ===================================================================
# --- Main Execution ---
# ===================================================================
def main():
    """
    Entry point for the Rainbow DQN training and evaluation script.
    """
    print("--- Starting Rainbow DQN: Breakout ---")

    # --- 1. Load Configuration ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(SCRIPT_DIR, "config.yaml"))

    # Environment
    env_name = cfg["environment"]["name"]
    seed = cfg["seed"]

    # Training
    train_cfg = cfg["training"]
    mode = train_cfg["mode"]
    num_steps = train_cfg["num_steps"]
    target_reward = train_cfg["target_reward"]
    print_every = train_cfg["print_every"]
    checkpoint_every = train_cfg["checkpoint_every"]
    moving_avg_window = train_cfg["plot_window"]
    deploy_trials = train_cfg["deploy_trials"]

    # Agent
    agent_cfg = cfg["agent"]
    optimizer = agent_cfg["optimizer"]
    learning_rate = agent_cfg["learning_rate"]
    gamma = agent_cfg["gamma"]
    grad_clip_norm = agent_cfg["grad_clip_norm"]
    clip_rewards = agent_cfg["clip_rewards"]
    sigma_init = agent_cfg["sigma_init"]

    # C51 Distributional
    c51_cfg = cfg["c51"]
    num_atoms = c51_cfg["num_atoms"]
    v_min = c51_cfg["v_min"]
    v_max = c51_cfg["v_max"]

    # N-Step Returns
    n_steps = cfg["n_step"]["n"]

    # Replay Buffer
    buf_cfg = cfg["replay_buffer"]
    buffer_capacity = buf_cfg["capacity"]
    batch_size = buf_cfg["batch_size"]
    learning_starts = buf_cfg["learning_starts"]

    # Prioritized Experience Replay
    per_cfg = cfg["prioritized_replay"]
    per_alpha = per_cfg["alpha"]
    per_beta_start = per_cfg["beta_start"]
    per_beta_frames = per_cfg["beta_frames"]
    priority_epsilon = per_cfg["priority_epsilon"]

    # Target Network
    target_update_freq = cfg["target_network"]["update_freq"]

    # Paths
    paths_cfg = cfg["paths"]
    results_dir = os.path.join(SCRIPT_DIR, paths_cfg["results_dir"])
    model_filepath = os.path.join(results_dir, paths_cfg["model_filename"])
    history_filepath = os.path.join(results_dir, paths_cfg["history_filename"])
    plot_filepath = os.path.join(results_dir, paths_cfg["plot_filename"])

    # --- 2. Set Seed for Reproducibility ---
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- 3. Set Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 4. Save Location Setup ---
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- 5. Initialization ---
    env, state_shape, action_size = create_env(env_name)

    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        device=device,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        learning_starts=learning_starts,
        target_update_freq=target_update_freq,
        optimizer=optimizer,
        grad_clip_norm=grad_clip_norm,
        clip_rewards=clip_rewards,
        sigma_init=sigma_init,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        n_steps=n_steps,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_frames=per_beta_frames,
        priority_epsilon=priority_epsilon,
    )

    # --- 6. Run ---
    episode_rewards = []
    episode_losses = []

    if mode == 'new':
        # --- Start fresh training (warmup handled inside train()) ---
        try:
            episode_rewards, episode_losses = train(
                env=env,
                agent=agent,
                num_steps=num_steps,
                all_rewards=episode_rewards,
                all_losses=episode_losses,
                print_every=print_every,
                moving_avg_window=moving_avg_window,
                target_reward=target_reward,
                checkpoint_every=checkpoint_every,
                model_filepath=model_filepath,
                history_filepath=history_filepath,
                plot_filepath=plot_filepath,
                is_new_run=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print(f"Saving model and history to {results_dir}...")
            save_checkpoint(agent, episode_rewards, episode_losses,
                            model_filepath, history_filepath, plot_filepath,
                            moving_avg_window)

    elif mode == 'resume':
        # --- Resume training from checkpoint ---
        checkpoint_loaded = False
        if os.path.exists(model_filepath):
            print(f"Loading model from {model_filepath}...")
            agent.load_model(model_filepath)
            checkpoint_loaded = True
            if os.path.exists(history_filepath):
                print(f"Loading history from {history_filepath}...")
                with np.load(history_filepath) as data:
                    episode_rewards = data['rewards'].tolist()
                    episode_losses = data['losses'].tolist()
                    if 'total_steps' in data:
                        agent.total_steps = int(data['total_steps'])
                # Sync exploration state to the restored step count
                if hasattr(agent, "update_epsilon"):
                    agent.update_epsilon()
                if hasattr(agent, "_update_beta"):
                    agent._update_beta()
                print(f"Resuming from Step {agent.total_steps}, Episode {len(episode_rewards)}")
            else:
                print("Warning: Model found but no history file.")
        else:
            print(f"No model found at {model_filepath}. Starting from scratch.")

        try:
            episode_rewards, episode_losses = train(
                env=env,
                agent=agent,
                num_steps=num_steps,
                all_rewards=episode_rewards,
                all_losses=episode_losses,
                print_every=print_every,
                moving_avg_window=moving_avg_window,
                target_reward=target_reward,
                checkpoint_every=checkpoint_every,
                model_filepath=model_filepath,
                history_filepath=history_filepath,
                plot_filepath=plot_filepath,
                is_new_run=not checkpoint_loaded
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print(f"Saving model and history to {results_dir}...")
            save_checkpoint(agent, episode_rewards, episode_losses,
                            model_filepath, history_filepath, plot_filepath,
                            moving_avg_window)

    else:  # deploy
        # --- Deploy trained agent ---
        if os.path.exists(model_filepath):
            print(f"Loading model from {model_filepath}...")
            agent.load_model(model_filepath)
            deploy(env_name, agent, num_trials=deploy_trials)
        else:
            print(f"No model found at {model_filepath}. Cannot deploy.")

    env.close()
    print("\n--- Program Finished ---")

if __name__ == "__main__":
    main()
