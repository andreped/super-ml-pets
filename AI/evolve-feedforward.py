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

total_wins = 0
total_losses = 0
total_draws = 0

class TeamReplacer(neat.StdOutReporter):
    """Replaces part of the past teams with every generation"""
    def start_generation(self, generation):
        sap.past_teams = [sap.past_teams[i][len(sap.past_teams[i])/5:]
                    for i in range(len(sap.past_teams))]


def eval_genome(genome, config):
    # Use the NN network phenotype.
    net = neat.nn.FeedForwardNetwork.create(genome, config)

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

        total_wins += sim.wins
        total_losses += sim.losses
        total_draws += sim.draws

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
    population.add_reporter(neat.Checkpointer(1))
    population.add_reporter(TeamReplacer())

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

if __name__ == "__main__":
    run()