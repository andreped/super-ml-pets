"""
Super Auto Pets AI using a feed-forward neural network.
"""

from __future__ import print_function

import multiprocessing as mp
import os
import pickle
import csv
import sys

import neat

import sap
import visualize


runs_per_net = 5
num_generations = 2000

class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]

    total_wins = 0
    total_losses = 0
    total_draws = 0

    logs = []


data = Data()

class TeamReplacer(neat.reporting.BaseReporter):
    """Replaces part of the past teams with every generation"""
    def __init__(self):    
        pass

    def start_generation(self, generation):
        # data.past_teams = [data.past_teams[i][len(data.past_teams[i])//5:]
        #             for i in range(len(data.past_teams))]

        save_logs()
        data.logs = []


        print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)


def eval_genome(genome, config):
    # Use the NN network phenotype.
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    global data

    for runs in range(runs_per_net):
        sim = sap.SAP(data)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while not sim.isGameOver():
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated sap game
            sim.step(action)

            fitness = sim.score

        data.total_wins += sim.wins
        data.total_losses += sim.losses
        data.total_draws += sim.draws

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def save_logs():
    with open('past_teams', 'w', newline='') as f:
        a = csv.writer(f)
        a.writerows(data.past_teams)

    with open('past_teams_bin', 'wb') as f:
        pickle.dump(data.past_teams, f)

    with open('logs', 'w', newline='') as f:
        a = csv.writer(f)
        for l in data.logs:
            a.writerow([str(l)])

def run():                    
    if True:
        population = neat.Checkpointer.restore_checkpoint('ckpt/ckpt-9992')
        data.total_wins = 96920
        data.total_draws = 127698
        data.total_losses = 90426
        with open('past_teams_bin', 'rb') as f:
            data.past_teams = pickle.load(f)

        print("loaded")

        # species # 531, id 2479534, Total extinctions: 435
    else:
        # Load the config file, which is assumed to live in
        # the same directory as this script.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
        population = neat.Population(config)

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(10, filename_prefix='ckpt/ckpt-'))
    population.add_reporter(TeamReplacer())

    # so basically just alt-f4 to stop the program :)
    # pe = neat.ParallelEvaluator(mp.cpu_count()-4, eval_genome)
    pe = neat.ThreadedEvaluator(1, eval_genome)
    # winner = population.run(eval_genomes, num_generations)

    try:
        winner = population.run(pe.evaluate, num_generations)
    except KeyboardInterrupt:
        print('Interrupted')
        save_logs()
        print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    save_logs()
    
    print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)


if __name__ == "__main__":
    run()