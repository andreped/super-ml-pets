"""
Super Auto Pets AI using a feed-forward neural network.
"""

from __future__ import print_function

import multiprocessing
import os
import pickle

import neat

import sap
import visualize


runs_per_net = 5
simulation_turns = 30
num_generations = 1000

population = None

# Save the teams from every level, refresh every generation, fight against these
current_generation = 0
past_teams = []


def eval_genome(genome, config):
    # Use the NN network phenotype.
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    global population

    if population.generation > current_generation:
        current_generation += 1
        past_teams = [past_teams[i][min(len(past_teams[i]), 10):]
                      for i in range(len(past_teams))]

    fitnesses = []

    for runs in range(runs_per_net):
        sim = sap.SAP()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.turns < simulation_turns:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated sap game
            sim.step(action)

            # Stop if the network fails to end the game in 30 turns or does 50 actionstakenthisturn without ending turn
            # The per-run fitness is calculated using (wins*50 + draws*20 - losses*10 - max(actionstakenthisturn-20, 0)*.1)
            if sim.actions_taken_this_turn >= 50:
                break

            fitness = sim.score

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


""" Not Used?
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
"""

def run():
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

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(pe.evaluate, num_generations)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True,
                         filename="feedforward-fitness.svg")
    visualize.plot_species(
        stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
