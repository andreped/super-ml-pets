"""
Super Auto Pets AI using a feed-forward neural network.
"""

from __future__ import print_function

import multiprocessing as mp
import os
import pickle
import csv
import sys
import time

import neat
import sapai

import sap


runs_per_net = 5
num_generations = 10000


class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]
    preset_teams = []

    total_wins = 0
    total_losses = 0
    total_draws = 0

    extinctions = 0

    logs = []


data = Data()


class TeamReplacer(neat.reporting.BaseReporter):
    """Replaces part of the past teams with every generation"""

    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        with open("terminal_output", 'a') as f:
            f.write(
                '\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

        # data.past_teams = [data.past_teams[i][len(data.past_teams[i])//5:]
        #             for i in range(len(data.past_teams))]

        save_logs()
        data.logs = []
        data.preset_teams = []
        
        print("stats: {0} / {1} / {2}".format(data.total_wins,
              data.total_draws, data.total_losses))

        with open("terminal_output", 'a') as f:
            f.write("stats: {0} / {1} / {2}".format(data.total_wins,
                    data.total_draws, data.total_losses))


    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        with open("terminal_output", 'a') as f:
            if self.show_species_detail:  
                f.write('Population of {0:d} members in {1:d} species:'.format(ng, ns))
                f.write("   ID   age  size   fitness   adj fit  stag")
                f.write("  ====  ===  ====  =========  =======  ====")
                for sid in sorted(species_set.species):
                    s = species_set.species[sid]
                    a = self.generation - s.created
                    n = len(s.members)
                    f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                    af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                    st = self.generation - s.last_improved
                    f.write(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
            else:
                f.write(
                    'Population of {0:d} members in {1:d} species'.format(ng, ns))

            elapsed = time.time() - self.generation_start_time
            self.generation_times.append(elapsed)
            self.generation_times = self.generation_times[-10:]
            average = sum(self.generation_times) / len(self.generation_times)
            f.write('Total extinctions: {0:d}'.format(self.num_extinctions))
            if len(self.generation_times) > 1:
                f.write("Generation time: {0:.3f} sec ({1:.3f} average)".format(
                    elapsed, average))
            else:
                f.write("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        with open("terminal_output", 'a') as f:
            f.write('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(
                fit_mean, fit_std))
            f.write(
                'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                    best_genome.size(),
                                                                                    best_species_id,
                                                                                    best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        data.extinctions += 1
        with open("terminal_output", 'a') as f:
            f.write('All species extinct.')

    def found_solution(self, config, generation, best):
        with open("terminal_output", 'a') as f:
            f.write('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
                self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            with open("terminal_output", 'a') as f:
                f.write("\nSpecies {0} with {1} members is stagnated: removing it".format(
                    sid, len(species.members)))

    def info(self, msg):
        with open("terminal_output", 'a') as f:
            f.write(msg)


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

    with open('logs', 'w', newline='') as f:
        a = csv.writer(f)
        for l in data.logs:
            a.writerow([str(l)])

    with open('gen_teams', 'w', newline='') as f:
        a = csv.writer(f)
        for l in data.preset_teams:
            a.writerow([str(l)])

    with open('metadata', 'w', newline='') as f:
        a = csv.writer(f)
        a.writerow([str(data.total_wins), str(
            data.total_draws), str(data.total_losses)])
        a.writerow([str(data.extinctions)])


def run():
    if False:
        population = neat.Checkpointer.restore_checkpoint('ckpt/ckpt-9992')
        data.total_wins = 96920
        data.total_draws = 127698
        data.total_losses = 90426

        data.extinctions = 0

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
    population.add_reporter(neat.Checkpointer(
        20, filename_prefix='ckpt/ckpt-'))
    population.add_reporter(TeamReplacer())

    with open('gen-enemy', 'r') as f:
        read = csv.reader(f)
        for row in read:
            team = []
            for pet in row:
                pet = pet.strip().split(" ")
                if len(pet) == 1:
                    continue
                id = pet[0]
                atk = int(pet[1])
                hlt = int(pet[2])
                lvl = int(pet[3])

                if lvl == 5:
                    exp = 0
                    lvl = 3
                elif lvl >= 2:
                    exp -= 2
                    lvl = 2
                else:
                    exp = lvl
                    lvl = 1

                if len(pet) == 5:
                    status = pet[4]
                else:
                    status = "none"

                spet = sapai.Pet(id)
                spet._attack = atk
                spet._health = hlt
                spet.experience = exp
                spet.level = lvl
                spet.status = status
                team.append(spet)
            data.preset_teams.append(team)

    # so basically just alt-f4 to stop the program :)
    # pe = neat.ParallelEvaluator(mp.cpu_count()-4, eval_genome)
    # pe = neat.ThreadedEvaluator(1, eval_genome)

    try:
        # winner = population.run(pe.evaluate, num_generations)
        winner = population.run(eval_genomes, num_generations)
    except KeyboardInterrupt:
        print('Interrupted')
        save_logs()
        print("stats: ", data.total_wins, "/",
              data.total_draws, "/", data.total_losses)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    save_logs()

    print("stats: ", data.total_wins, "/",
          data.total_draws, "/", data.total_losses)


if __name__ == "__main__":
    run()
