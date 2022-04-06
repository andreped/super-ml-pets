"""
Super Auto Pets AI using a feed-forward neural network.
"""

from __future__ import print_function

import multiprocessing as mp
import os
import pickle
import csv

import neat

import sap
import visualize


runs_per_net = 5
num_generations = 10000

class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]

    total_wins = 0
    total_losses = 0
    total_draws = 0


data = Data()

class TeamReplacer(neat.reporting.BaseReporter):
    """Replaces part of the past teams with every generation"""
    def __init__(self):    
        pass

    def start_generation(self, generation):
        data.past_teams = [data.past_teams[i][len(data.past_teams[i])//5:]
                    for i in range(len(data.past_teams))]


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

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

    sim = sap.SAP(data)
    end_turn = [0]*69
    end_turn[68] = 1
    sim.step(end_turn)
    data.total_wins += sim.wins
    data.total_losses += sim.losses
    data.total_draws += sim.draws
    print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)
    return
                        
    if False:
        population = neat.Checkpointer.restore_checkpoint('ckpt/ckpt-1289')
    else:
        population = neat.Population(config)

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(10, filename_prefix='ckpt/ckpt-'))
    population.add_reporter(TeamReplacer())

    # so basically just alt-f4 to stop the program :)
    # pe = neat.ParallelEvaluator(mp.cpu_count()-4, eval_genome)
    pe = neat.ThreadedEvaluator(1, eval_genome)
    winner = population.run(pe.evaluate, num_generations)
    # winner = population.run(eval_genomes, num_generations)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    with open('past_teams', 'w', newline='') as f:
        a = csv.writer(f)
        a.writerows(data.past_teams)

    
    print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)


if __name__ == "__main__":
    run()