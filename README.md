# stocktradingrl

Stock Trading Model using Q Learning.

Code that follows the article [Reinforcement Learning for trading](https://quantdare.com/reinforcement-learning-for-trading/)

All you need to run experiments with this model is in `main.py`.

- Add the data you want to use to dir `data`. It just have to be the prices for a single stock. Change the `get_data` function in order to read your own dataset.
- Write how many episodes you want to train your q-agent in the `args.parse`. And just run the script!

Every `interval` episodes the code is going to save the model and calculate the performance in the evaluation data. If you want to resume the training, just write it in the `args.parse`.

Things that you could try:
- Change details in the gym environment. There you can change the `_take_action` function, in order to change the decision process of the agent, or even the reward system.
- You could use some pre-wrapped agents with Stable Baselines or TF-Agents or even Ray Rllib Agents. It's more tricky but definetively possible. Next article in Quantdare blog will be about this point.
- Try to use more data! Hourly data, for example, could improve the performance. But here you should take into account the cost of the Stock Trade fee.

Well, thank for reading. Any doubts you could contact me! ge.vargasn@gmail.com