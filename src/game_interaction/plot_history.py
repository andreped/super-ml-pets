from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rc

if __name__ == "__main__":
	history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history/sb3_log_180822/")

	data = pd.read_csv(os.path.join(history_path, "progress.csv"))
	print(data.head())
	print(list(data.keys()))

	# set plot config
	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}) # , 'size': 16})

	# load history and make plot
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(list(range(len(data["rollout/ep_len_mean"]))), data["rollout/ep_len_mean"])
	ax[0].set_title("Number of actions before game is done")
	ax[0].grid("on")
	ax[1].plot(list(range(len(data["rollout/ep_rew_mean"]))), data["rollout/ep_rew_mean"])
	ax[1].set_title("Number of wins in 100 games")
	ax[1].grid("on")
	plt.tight_layout()
	plt.show()
