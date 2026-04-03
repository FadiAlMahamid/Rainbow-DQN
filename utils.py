import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from environment import create_env

# Try to use a backend that works well for animation
try:
    matplotlib.use('TkAgg')
except Exception:
    pass


# ===================================================================
# --- Helper Function: log_progress ---
# ===================================================================
def log_progress(agent, num_steps, all_rewards, moving_avg_window, target_reward):
    """
    Logs training progress and checks for early stopping.

    Dynamically includes exploration info (epsilon, beta) based on
    which attributes the agent has.

    Args:
        agent: The DQN agent.
        num_steps (int): Total training steps target.
        all_rewards (list): Accumulated episode rewards.
        moving_avg_window (int): Window size for reward averaging.
        target_reward (float): Average reward threshold for early stopping.

    Returns:
        bool: True if the target reward has been reached (early stop).
    """
    num_episodes = len(all_rewards)
    if num_episodes >= moving_avg_window:
        avg_reward = np.mean(all_rewards[-moving_avg_window:])
    elif num_episodes > 0:
        avg_reward = np.mean(all_rewards)
    else:
        avg_reward = 0.0

    parts = [
        f"Step {agent.total_steps}/{num_steps}",
        f"Frames: {agent.total_steps * 4}",
        f"Episodes: {num_episodes}",
    ]
    if hasattr(agent, 'epsilon'):
        parts.append(f"Epsilon: {agent.epsilon:.4f}")
    if hasattr(agent, 'per_beta'):
        parts.append(f"Beta: {agent.per_beta:.4f}")
    elif hasattr(agent, 'current_beta'):
        parts.append(f"Beta: {agent.current_beta:.4f}")
    parts.append(f"Buffer: {len(agent.replay_buffer)}")
    parts.append(f"Avg Reward (Last {moving_avg_window}): {avg_reward:.2f}")
    print(" | ".join(parts))

    if num_episodes >= moving_avg_window and avg_reward >= target_reward:
        print(f"\nTarget reward of {target_reward} reached! Stopping training early.")
        return True
    return False


# ===================================================================
# --- Helper Function: save_checkpoint ---
# ===================================================================
def save_checkpoint(agent, all_rewards, all_losses,
                    model_filepath, history_filepath, plot_filepath,
                    moving_avg_window):
    """
    Saves model weights, training history, and a plot to disk.

    Args:
        agent: The DQN agent to save.
        all_rewards (list): Accumulated episode rewards.
        all_losses (list): Accumulated episode losses.
        model_filepath (str): Path to save model weights.
        history_filepath (str): Path to save training history.
        plot_filepath (str): Path to save the training plot.
        moving_avg_window (int): Window size for the plot moving average.
    """
    print(f"  [Checkpoint] Saving model at step {agent.total_steps}...")
    agent.save_model(model_filepath)
    np.savez(
        history_filepath,
        rewards=np.array(all_rewards, dtype=np.float32),
        losses=np.array(all_losses, dtype=np.float32),
        total_steps=np.array(agent.total_steps, dtype=np.int64),
    )
    plot_results(all_rewards, all_losses, save_path=plot_filepath,
                 moving_avg_window=moving_avg_window)


# ===================================================================
# --- Helper Function: load_config ---
# ===================================================================
def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ===================================================================
# --- Helper Function: plot_results ---
# ===================================================================
def plot_results(episode_rewards, episode_losses, save_path, moving_avg_window=100):
    """
    Generates and saves a training progress plot showing rewards and losses.

    Args:
        episode_rewards (list): List of total rewards per episode.
        episode_losses (list): List of average losses per episode.
        save_path (str): The full path where the plot will be saved.
        moving_avg_window (int): The window size for calculating moving averages.
    """
    print(f"\nPlotting results to {save_path}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Plot 1: Rewards ---
    rewards = np.array(episode_rewards, dtype=np.float32)
    # Plot raw per-episode rewards as a subtle scatter to show variance
    # without flooding the chart — small dots at low opacity work better
    # than a dense line plot for thousands of episodes.
    ax1.scatter(np.arange(len(rewards)), rewards, s=1, alpha=0.15,
                color='steelblue', label='Per-Episode Reward')
    if len(rewards) >= moving_avg_window:
        # Compute a simple moving average using convolution with a uniform kernel.
        # mode='valid' only outputs where the full kernel overlaps the data,
        # so the moving average starts at episode (moving_avg_window - 1).
        moving_avg = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        ax1.plot(np.arange(moving_avg_window - 1, len(rewards)), moving_avg,
                 color='darkorange', linewidth=2, label=f'Moving Avg ({moving_avg_window})')

    ax1.set_title("Rainbow DQN: Training Rewards")
    ax1.set_ylabel("Total Reward")
    # Cap y-axis based on the 99th percentile to avoid outlier spikes
    # stretching the axis and squashing the actual learning curve.
    if len(rewards) > 0:
        y_cap = np.percentile(rewards, 99) * 1.2
        ax1.set_ylim(bottom=min(0, rewards.min()), top=max(y_cap, 1))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- Plot 2: Losses ---
    losses = np.array(episode_losses, dtype=np.float32)
    if len(losses) > 0:
        # Raw losses as subtle scatter, same approach as rewards
        ax2.scatter(np.arange(len(losses)), losses, s=1, alpha=0.15,
                    color='steelblue', label='Per-Episode Loss')
        if len(losses) >= moving_avg_window:
            # Add a moving average trend line for losses too
            loss_avg = np.convolve(losses, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
            ax2.plot(np.arange(moving_avg_window - 1, len(losses)), loss_avg,
                     color='darkorange', linewidth=2, label=f'Moving Avg ({moving_avg_window})')

    ax2.set_title("Rainbow DQN: Training Loss (Cross-Entropy)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    print("Plot saved successfully.")


# ===================================================================
# --- The Deployment / Visualization Function ---
# ===================================================================
def deploy(env_name, agent, num_trials=5, fps=30):
    """
    Visualizes a trained agent playing the environment using pure exploitation.

    Args:
        env_name (str): The name of the Atari environment.
        agent (DQNAgent): The trained Rainbow DQN agent.
        num_trials (int): Number of episodes to play for visualization.
        fps (int): Frames per second for the visualization window.
    """
    print("\nDeploying Rainbow DQN Agent (matplotlib)...")
    try:
        env, _, _ = create_env(env_name, render_mode="rgb_array")
    except Exception as e:
        print(f"Warning: Could not create rgb_array render env. {e}. Exiting deployment.")
        return

    # Enable interactive mode so the plot window updates in real-time
    # without blocking the game loop.
    plt.ion()
    fig, ax = plt.subplots()
    img = None  # Will hold the matplotlib image object (created on first frame)

    # Dynamically get action names from the environment
    ACTION_NAMES = dict(enumerate(env.unwrapped.get_action_meanings()))

    for trial in range(num_trials):
        state, info = env.reset()
        terminal = False
        total_reward = 0.0
        action_counts = {}
        print(f"--- Trial {trial + 1} ---")
        time.sleep(1.0)

        while not terminal:
            # Pure exploitation: eval mode disables noise for deterministic actions
            action = agent.choose_action(np.array(state), training=False)
            action_counts[action] = action_counts.get(action, 0) + 1

            next_state, reward, done, truncated, _ = env.step(action)
            terminal = done or truncated
            state = next_state
            total_reward += reward

            # Render the current frame as an RGB array from the unwrapped
            # (non-preprocessed) environment to show the full-resolution game.
            frame = env.unwrapped.render()
            if img is None:
                # First frame: create the image object
                img = ax.imshow(frame)
                ax.axis("off")
            else:
                # Subsequent frames: update pixel data in-place (faster than redrawing)
                img.set_data(frame)

            ax.set_title(f"Rainbow DQN | Trial {trial + 1} | Reward: {total_reward}")
            fig.canvas.draw()
            # plt.pause both updates the display and controls frame rate
            plt.pause(1.0 / fps)

            if not plt.fignum_exists(fig.number):
                print("Window closed by user.")
                env.close()
                return

        # Log action distribution for this trial
        total_actions = sum(action_counts.values())
        dist = ", ".join(
            f"{ACTION_NAMES.get(a, a)}: {c / total_actions * 100:.1f}%"
            for a, c in sorted(action_counts.items())
        )
        print(f"Trial {trial + 1} | Reward: {total_reward} | Actions: {{{dist}}}")

    env.close()
    plt.ioff()
    plt.close(fig)
