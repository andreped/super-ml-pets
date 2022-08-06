"""
Utility classes and functions relevant for training and evaluating RL SAP models
"""

class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]

    total_wins = 0
    total_losses = 0
    total_draws = 0

    logs = []

def save_logs(data):
    with open('past_teams', 'w', newline='') as f:
        a = csv.writer(f)
        a.writerows(data.past_teams)

    with open('past_teams_bin', 'wb') as f:
        pickle.dump(data.past_teams, f)

    with open('logs', 'w', newline='') as f:
        a = csv.writer(f)
        for l in data.logs:
            a.writerow([str(l)])
    data.logs = []

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]), verbose=0)[0]
