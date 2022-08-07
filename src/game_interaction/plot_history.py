from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == "__main__":
	history_path = "./history/sb3_log_070822/"

	data = pd.read_csv(os.path.join(history_path, "progress.csv"))
	print(data.head())
	print(list(data.keys()))

	# load history and make plot
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(data["time/total_timesteps"], data["rollout/ep_len_mean"])
	ax[0].set_title("Number of rolls before game is done")
	ax[0].grid("on")
	ax[1].plot(data["time/total_timesteps"], data["rollout/ep_rew_mean"])
	ax[1].set_title("Number of wins in 100 games")
	ax[1].grid("on")
	plt.tight_layout()
	plt.show()
