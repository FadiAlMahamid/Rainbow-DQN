import gymnasium as gym
import ale_py  # Provides the Atari ROM interface for Gymnasium

# Register all Atari environments (e.g., BreakoutNoFrameskip-v4) so they
# can be created with gym.make(). Must be called before any gym.make() call.
gym.register_envs(ale_py)


# ===================================================================
# --- FireResetEnv wrapper ---
# ===================================================================
class FireResetEnv(gym.Wrapper):
    """
    Wrapper that automatically takes a FIRE action after reset.
    Ensures the ball is launched so the agent can begin acting.
    """
    def __init__(self, env):
        """
        Initializes the FireResetEnv wrapper.

        Args:
            env (gym.Env): The original Gymnasium environment.
        """
        super().__init__(env)
        # Look up the integer action ID for "FIRE" from the environment's action table.
        # In Breakout, FIRE = 1, which launches the ball at the start of each life.
        meanings = env.unwrapped.get_action_meanings()
        self.fire_action = meanings.index("FIRE")

    def reset(self, **kwargs):
        """
        Resets the environment and immediately takes a FIRE action.

        Args:
            **kwargs: Arbitrary keyword arguments for the environment reset.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        obs, info = self.env.reset(**kwargs)
        # Press FIRE once to launch the ball
        obs, reward, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            # Extremely rare, but just in case: reset and re-fire
            obs, info = self.env.reset(**kwargs)
            obs, reward, terminated, truncated, info = self.env.step(self.fire_action)
        return obs, info


# ===================================================================
# --- Helper Function: create_env ---
# ===================================================================
def create_env(env_name, render_mode=None):
    """
    Creates the Atari environment with standard preprocessing and frame stacking.

    Args:
        env_name (str): The name of the Atari environment.
        render_mode (str, optional): The mode for rendering. Defaults to None.

    Returns:
        tuple: (env, state_shape, action_size)
    """
    # frameskip=1 disables the built-in ALE frame skip so that
    # AtariPreprocessing can handle it instead (with max-pooling over
    # the last 2 raw frames to remove flickering sprites).
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)

    # === Wrapper ordering matters ===
    #
    # AtariPreprocessing MUST wrap the base env first so that its no-op reset
    # (1-30 random NOOPs) happens BEFORE the ball is launched.
    # If FireResetEnv were inside AtariPreprocessing, the ball would launch first,
    # then drift uncontrolled during the no-op steps.
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,         # Repeat each action for 4 raw frames (agent sees every 4th)
        screen_size=84,       # Downscale 210x160 to 84x84 for the CNN input
        grayscale_obs=True,   # Convert RGB to grayscale (reduces input from 3 channels to 1)
        terminal_on_life_loss=True  # Treat losing a life as episode end (speeds up learning)
    )

    # FireResetEnv wraps AFTER preprocessing so FIRE happens after no-op reset.
    # This ensures the ball launches right when the agent gets control.
    env = FireResetEnv(env)

    # Stack the last 4 preprocessed frames along a new axis to give the CNN
    # temporal information (e.g., ball direction and velocity).
    # Resulting observation shape: (4, 84, 84).
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env, env.observation_space.shape, env.action_space.n
