## v0 - 11/06/2017
## RobotMaze world

from Maze import maze
from Robot import robot
from RobotPhenotype import robot_phenotype
from Innovation import innovation
from Activations import activations
from Specie import specie
import Reproduction

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import sys

## Define RobotMaze class
class robot_maze:


    # Create world
    def __init__(self,param):

        # Global parameters
        self.population_size=param['population_size']
        self.maze_count=param['world_count']
        self.max_species=param['max_species']
        self.wall_nb=param['world_wall_nb']
        self.pos_type=param['world_pos_type']
        self.maze_type=param['world_type']
        self.maze_size=param['world_size']
        self.viability=param['viability']
        self.activation_name=param['activation_func']
        self.verbose=param['verbose']
        self.debug=param['debug']
        self.work_dir=param['work_dir']
        self.debug=param['debug']
        
        # Reproduction probabilities parameters
        self.reproduction_dic={'keep_prct':param['keep_prct'],
                               'disable_gene_prob':param['disable_gene_prob'],
                               'crossover_prob':param['crossover_prob'],
                               'different_specie_crossover_prob':param['different_specie_crossover_prob'],
                               'weight_mut_prob':param['weight_mut_prob'],
                               'weight_mut_is_uniform_prob':param['weight_mut_is_uniform_prob'],
                               'add_node_prob':param['add_node_prob'],
                               'add_connection_prob':param['add_connection_prob'],
                               'change_activation_prob':param['change_activation_prob'],
                               'max_weight_perturb':param['max_weight_perturb']
                               }
        
        # Activation functions
        self.activation_func=self.activation_name

        # Activation list
        self.activation_list=activations(self.activation_func)

        # Maze
        self.mazes=list()

        # Robots 
        self.R_parents=list()
        self.R_children=list()

        # Robot phenotype
        self.Rphenotype_parents=robot_phenotype()
        self.Rphenotype_children=robot_phenotype()

        # Gene and innovation trackers (as list to be able to update them in sub routines)
        self.cur_innovation=[-1]
        self.cur_node=[-1]

        # Innovation list to avoid duplicated mutations in a generation
        self.innovation_list=innovation()

        # Species
        self.specie=specie(param['specie_delta'],param['c1_excess'],param['c2_disjoint'],param['c3_weight'])
        
    # Populate world
    def populate(self,load_maze=False,dir=-1):
        
        print('# Populate world')

        # Load Mazes if load_maze==True
        if (load_maze) :

            for i_maze in range(self.maze_count):

                file = open(dir+'maze_'+str(i_maze)+'.p', "rb")
                self.mazes.append(pickle.load(file))
                file.close()

        # Otherwise create mazes
        else :
            
            for i_maze in range(self.maze_count):

                print('### Maze {}'.format(i_maze))

                # Append new maze
                if (self.wall_nb==-1):
                    self.mazes.append(maze(self.maze_size,self.maze_type,i_maze,position=self.pos_type))
                else :
                    self.mazes.append(maze(self.maze_size,self.maze_type,self.wall_nb,position=self.pos_type))                    
                
                # Save maze
                if (dir == -1):
                    file = open('./maze_'+str(i_maze)+'.p', "wb")
                else:
                    file = open(dir+'maze_'+str(i_maze)+'.p', "wb")

                pickle.dump(self.mazes[-1],file)
                file.close()

        # Robots 
        for i_robot in range(self.population_size):
            self.R_parents.append(robot(self.maze_count,self.mazes,self.cur_node,self.activation_func))

        # Robots phenotypes
        for i_robot in range(self.population_size):
            self.Rphenotype_parents.add_robot(self.R_parents,i_robot,self.maze_count)

        # Robots species
        for i_robot in range(self.population_size):
            self.specie.compatibility(self.R_parents[i_robot].genome)

        # Species count
        self.specie.count(self.R_parents)


    #  Reproduce population
    def reproduce_pop(self):

        # Reproduce parents
        self.R_children=Reproduction.reproduce(self.R_parents,self.Rphenotype_parents,self.specie,self.max_species,self.population_size,self.maze_count,self.mazes,self.cur_node,self.cur_innovation,self.innovation_list,self.activation_func,self.reproduction_dic,self.debug)
                    
        # Children phenotypes
        self.Rphenotype_children=robot_phenotype()
        for i_robot in range(self.population_size):
            self.Rphenotype_children.add_robot(self.R_children,i_robot,self.maze_count)

        # Replace parents
        del self.R_parents
        del self.Rphenotype_parents
        
        self.R_parents=self.R_children
        self.Rphenotype_parents=self.Rphenotype_children
        

    # Reset innovation
    def reset_innovation(self):

        self.innovation_list=innovation()

    # Evaluate population
    def eval_pop(self,max_step):

        # Evaluate population
        for i_robot in range(self.population_size):

            # Compute time
            t0=time.time()

            # Variables initialization
            t_move=0.0
            t_fit=0.0
            step=0
            IsMoving=True
            ReachedGoal=False

            # Run only if not evaluated already
            if (not self.R_parents[i_robot].IsEvaluated):

                # Time
                t0b=time.time()

                # while robot moving, not reached goals on all mazes and not reached max step
                while ( (step<max_step) and (not ReachedGoal) and (IsMoving)):

                    # Move robot
                    IsMoving=self.Rphenotype_parents.move_robot(self.R_parents,self.mazes,i_robot,self.maze_count,self.activation_list,time_step=1.0)
                    t0a=time.time()
                    t_move+=t0a-t0b

                    # Compute fitness every 50 steps
                    if (np.mod(step,50)==0):
                        ReachedGoal=self.Rphenotype_parents.compute_fitness(self.R_parents,self.mazes,i_robot,self.maze_count,max_dist=10)
                    t0b=time.time()
                    t_fit+=t0b-t0a

                    # Increment step
                    step+=1

                #print(step,ReachedGoal,IsMoving)

                # Compute adjusted fitness
                self.Rphenotype_parents.adjust_fitness(self.R_parents,i_robot,self.specie)

                # Switch flag for evaulation to True
                self.R_parents[i_robot].IsEvaluated=True

                # If goal reached stop and return
                if (ReachedGoal):
                    print('Robot {} reached Goal !!!'.format(i_robot))
                    return True

            # Time
            t1=time.time()

            # Print timing
            sys.stdout.write('Robot {0:6d} took {1:4.2f} s to evaluate, move: {2:4.4f}, fit: {3:4.4f}, avg fitness: {4:4.4f} \r'.format(i_robot,t1-t0,t_move,t_fit,self.R_parents[i_robot].fitness_avg) )
            sys.stdout.flush()
            #print('Robot {0:6d} took {1:4.2f} s to evaluate, move: {2:4.4f}, fit: {3:4.4f}, avg fitness: {4:4.4f}'.format(i_robot,t1-t0,t_move,t_fit,self.R_parents[i_robot].fitness_avg))

        return False

    # Visualize world
    def visualize(self,igen):

        print('Visuliase world')

        # Get 2 mazes max
        n_maze = min(self.maze_count,4)
        
        # Get 4 robots
        n_robot = 4

        # number column display
        n_col = n_maze

        # number rows display
        n_row = n_robot

        # Find the best 4
        fitness_robot=np.zeros(len(self.R_parents))
        for i_robot in range(len(self.R_parents)):
            fitness_robot[i_robot]=self.R_parents[i_robot].fitness_avg
        i_sorted_robot=sorted(range(len(fitness_robot)), key=lambda k: fitness_robot[k])
        i_best_robot=i_sorted_robot[-4:]

        print(i_best_robot)

        print('# Best robot genome')
        self.R_parents[i_best_robot[3]].genome.print_genome()

        print('# Fitness best robot: {}'.format(self.R_parents[i_best_robot[3]].fitness_avg))
        
        i_subplot=1

        # For each robot
        for i_robot in i_best_robot:
            # For each maze
            for i_maze in range(n_maze):
                #Compute vizualisation
                viz=self.R_parents[i_robot].visualize_path(i_maze,self.mazes)
                plt.subplot(n_row,n_col,i_subplot)
                plt.imshow(viz,cmap='gray')
                i_subplot+=1

        # Save plot with generation number
        plt.savefig(self.work_dir+'images\\'+str(igen)+'.png', dpi=400, bbox_inches='tight')

        # Number of species
        print('Species count: {}',format(self.specie.specie_count))
        print('Species delta: {}',format(self.specie.delta))
        print('Number of species : {}'.format(len(self.specie.specie_count)))

    # Evolve maze (TODO)
    #def evolve(self):

        # Evolve maze
        #self.mazes.evolve_maze()

        # Update robots accordingly
        #for irobot in range(len(self.R_parents)):
        #    robot.update_maze(self.maze)
