## v0 - 11/06/2017
## Run NEAT with RobotMaze world

from RobotMaze import robot_maze

import numpy as np

## Number of generations
Ngen=10000

## World Parameters
param={'population_size':1000,
       'max_species':100,
       'activation_func':'sigmoid',
       'viability':'fitness',
       #'mutation_rate':0.05,
       #'speciation_dist':1,
       'world_count':4,
       'world_type':'random',
       'world_pos_type':'corner',
       'world_wall_nb':20,
       'world_size':256,
       'max_step':5000,
       'reset_innovation_rate':1,
       'specie_delta':2.0,
       'c1_excess':2.0,
       'c2_disjoint':2.0,
       'c3_weight':1.0,
       'keep_prct':0.2,
       'max_weight_perturb':1.3,
       'disable_gene_prob':0.75,
       'crossover_prob':0.75,
       'different_specie_crossover_prob':0.001,
       'weight_mut_prob':0.8,
       'weight_mut_is_uniform_prob':0.9,
       'add_node_prob':0.03,
       'add_connection_prob':0.05,
       'change_activation_prob':0.00,
       'work_dir':'test',
       'verbose':True,
       'debug':False
       }


## Create the world
world=robot_maze(param)

## Populate the world (mazes and population)
world.populate(load_maze=True,dir=(param['work_dir']+'mazes\\'))

# Init generation counter
igen=1

# Define boolean for ending simulation
ReachedGoal=False

while ( (igen<Ngen) and (not ReachedGoal) ):

    print('#####################')
    print('##### GENERATION ',igen)
    print('#####################')

    ## Evaluate new population
    ReachedGoal=world.eval_pop(param['max_step'])

    ## Reproduce population
    world.reproduce_pop()

    ## Visualize world
    world.visualize(igen)

    ## Reset innovation at given rate
    if (np.mod(igen,param['reset_innovation_rate'])==0):
        world.reset_innovation()

    ## Evolve world ?
    #world.evolve()

    # Increment generation counter
    igen+=1
