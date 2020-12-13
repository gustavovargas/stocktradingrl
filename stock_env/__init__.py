from gym.envs.registration import register

register(
    id='StockEnv-v0',
    entry_point='stock_env.envs:CustomEnv',
    max_episode_steps=2000,
)