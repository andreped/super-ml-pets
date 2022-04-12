p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
winner = p.run(eval_genomes, 1)
