## v0 - 17/06/2017
## robot_phenotype

import time
import numpy as np

from RobotGenome import robot_genome
from Activations import activations
from Specie import specie

## Define robot phenotype
class robot_phenotype:


    # Create robot phenotype
    def __init__(self):
    
        # Current and next state for each robot
        self.node_gene_cur_state=[]
        self.node_gene_next_state=[]

        # Dictionnaries for each robot
        self.gene_nb_TO_cur_state=[]
        self.gene_nb_TO_next_state=[]
        self.gene_nb_TO_index=[]

        # Dictionnary for activation functions (name:operator)
        #self.ac_name_TO_ac_op={i:j for i,j, in zip(activation_list.list,activation_list.op)}

    # Add a robot phenotype from its genome
    def add_robot(self,robots,i_robot,maze_count):

        G=robots[i_robot].genome

        # Init current and next states to 0 for all nodes
        # each node as as many state entries as there are mazes
        self.node_gene_cur_state.append([])
        self.node_gene_next_state.append([])
        for i_gene in range(len(G.node_gene_number)):
            self.node_gene_cur_state[-1].append(np.zeros(maze_count))
            self.node_gene_next_state[-1].append(np.zeros(maze_count))

        # Create dictionnaries
        self.gene_nb_TO_cur_state.append([])
        self.gene_nb_TO_cur_state[-1]={i:j for i,j, in zip(G.node_gene_number,self.node_gene_cur_state[-1])}

        self.gene_nb_TO_next_state.append([])
        self.gene_nb_TO_next_state[-1]={i:j for i,j, in zip(G.node_gene_number,self.node_gene_next_state[-1])}

        self.gene_nb_TO_index.append([])
        self.gene_nb_TO_index[-1]={j:i for i,j, in enumerate(G.node_gene_number)}


    # Print phenotype for a given robot and maze
    def print_phenotype(self,robots,i_robot,i_maze,print_connection=False):
        
        G=robots[i_robot].genome

        # Print nodes
        print('## NODES :')
        print('## Number \tType \t Activation \t State')
        for i_node in range(len(G.node_gene_number)):
            print('## {}\t {}\t {}\t {}'.format(G.node_gene_number[i_node],G.node_gene_type[i_node],G.node_gene_activation[i_node],self.node_gene_cur_state[i_robot][i_node][i_maze]))

        # Print connections
        if (print_connection):
            print('## CONNECTIONS :')
            print('## Input \tOutput \tWeight \tIsEnabled \tInnovation')
            for i_connec in range(len(G.connection_gene_input)):
                print('## {}\t {}\t {}\t {}\t {}'.format(G.connection_gene_input[i_connec],G.connection_gene_output[i_connec],G.connection_gene_weight[i_connec],G.connection_gene_IsEnabled[i_connec],G.connection_gene_innovation[i_connec]))


    # Update sensor (input) values in the current state for a given maze and robot
    def update_sensors(self,robots,mazes,i_robot,i_maze):

        G=robots[i_robot].genome

        robots[i_robot].get_sensors(mazes,i_maze)
        sensors=robots[i_robot].sensor

        n_input=G.n_input
        n_sensor=len(sensors)

        # Check there are as many sensors as input
        if (n_input!=n_sensor):
            raise ValueError('there should be as many sensors as input in the genome')

        # Update inputs
        for i_sensor in range(n_sensor):
            self.node_gene_cur_state[i_robot][i_sensor][i_maze]=sensors[i_sensor]
        #self.node_gene_cur_state[i_robot][:][i_maze]=sensors

    # Update bias vlues in the current state for a given maze and robot
    def update_bias(self,robots,value,i_robot,i_maze):

        G=robots[i_robot].genome

        n_input=G.n_input
        n_bias=G.n_bias

        # Update bias
        for i_bias in range(n_bias):
            self.node_gene_cur_state[i_robot][n_input+i_bias][i_maze]=value
                    

    # Get controllers (output) values from the current state for a given maze and robot
    def get_controllers(self,robots,i_robot,i_maze):

        G=robots[i_robot].genome

        n_input=G.n_input
        n_bias=G.n_bias
        n_output=G.n_output
        
        controllers=list()
        
        # Get output
        for i_output in range(n_output):
            controllers.append(self.node_gene_cur_state[i_robot][n_input+n_bias+i_output][i_maze])

        return controllers

    
    # Compute next state for a robot
    def compute_next_state(self,robots,mazes,i_robot,maze_count,activation_list):

        #t0=time.time()
        #print('###COMPUTE NEXT STATE#')
        
        # Get sensors
        for i_maze in range(maze_count):
            self.update_sensors(robots,mazes,i_robot,i_maze)

        #t1=time.time()
        #print('time to get sensors : {}',format(t1-t0))
        
        # Update bias
        for i_maze in range(maze_count):
            self.update_bias(robots,1.0,i_robot,i_maze)
        
        #t2=time.time()
        #print('time to udpdate bias : {}',format(t2-t1))

        G=robots[i_robot].genome

        # Compute next state entries (before activation function)
        for i_connec in range(len(G.connection_gene_input)):
            if (G.connection_gene_IsEnabled[i_connec]):
                #print('debug')
                #print(G.connection_gene_output[i_connec])
                #print(self.gene_nb_TO_next_state[i_robot])
                #print([G.connection_gene_input[i_connec]])
                #print(self.gene_nb_TO_cur_state[i_robot])
                #print(G.connection_gene_weight[i_connec])
                self.gene_nb_TO_next_state[i_robot][G.connection_gene_output[i_connec]]+=self.gene_nb_TO_cur_state[i_robot][G.connection_gene_input[i_connec]]*G.connection_gene_weight[i_connec]

        #t3=time.time()
        #print('compute next state before act : {}',format(t3-t2))

        # Prepare vector to pack all values that need the same activation function
        vector=list()
        for i_ac in range(len(activation_list.list)):
            vector.append([])
            
        #t4=time.time()
        #print('time to prepare vector : {}',format(t4-t3))

        # Pack values in vector
        for i_node in range(len(G.node_gene_number)):
            if (G.node_gene_activation[i_node]!='NA'):
                i_ac=activation_list.list.index(G.node_gene_activation[i_node])
                vector[i_ac].append(self.gene_nb_TO_next_state[i_robot][G.node_gene_number[i_node]])
            
        #t5=time.time()
        #print('time to pack in vector : {}',format(t5-t4))

        # Apply activation function
        for i_ac in range(len(activation_list.list)):
            if (len(vector[i_ac])>0):
                vector[i_ac]=activation_list.op[i_ac](np.asarray(vector[i_ac])).tolist()

        #t6=time.time()
        #print('time to apply activation function : {}',format(t6-t5))

        # Unpack vector in next state
        for i_node in reversed(range(len(G.node_gene_number))):
            if (G.node_gene_activation[i_node]!='NA'):
                i_ac=activation_list.list.index(G.node_gene_activation[i_node])
                self.gene_nb_TO_next_state[i_robot][G.node_gene_number[i_node]]=np.asarray(vector[i_ac].pop())
            else:
                self.gene_nb_TO_next_state[i_robot][G.node_gene_number[i_node]]=self.gene_nb_TO_cur_state[i_robot][G.node_gene_number[i_node]]
                
        #t7=time.time()
        #print('time to unpack vector : {}',format(t7-t6))

        # Update current state
        for node in G.node_gene_number:
            self.node_gene_cur_state[i_robot][self.gene_nb_TO_index[i_robot][node]]=self.gene_nb_TO_next_state[i_robot][node]

        #t8=time.time()
        #print('time to update state : {}',format(t8-t7))

        # Reset next state
        for i_node in range(len(self.node_gene_next_state[i_robot])):
            self.node_gene_next_state[i_robot][i_node]=np.zeros(maze_count)

        #t9=time.time()
        #print('time to reset next state : {}',format(t9-t8))

        # Update dictionnaries
        self.gene_nb_TO_cur_state[i_robot]={i:j for i,j, in zip(G.node_gene_number,self.node_gene_cur_state[i_robot])}
        self.gene_nb_TO_next_state[i_robot]={i:j for i,j, in zip(G.node_gene_number,self.node_gene_next_state[i_robot])}

        #t10=time.time()
        #print('time to udpdate dictionnaries : {}',format(t10-t9))


    # Move robot in each maze
    # return True is the robot is moving, False if stuck
    def move_robot(self,robots,mazes,i_robot,maze_count,activation_list,time_step):

        # Boolean to check if robot is moving in at at least one maze
        IsMoving = True

        #t0=time.time()
        #print('#MOVE ROBOT#')

        # Compute next neural state
        self.compute_next_state(robots,mazes,i_robot,maze_count,activation_list)

        #t1=time.time()
        #print('time to compute neural space : {}',format(t1-t0))
        
        # Init controllers
        controllers=[]

        #print('maze count: {}'.format(maze_count))

        # For each maze
        for i_maze in range(maze_count):

            #print ('move maze {}'.format(i_maze))

            #t2=time.time()
            #print('### MAZE {}#'.format(i_maze))

            # Get controllers
            controllers.append(self.get_controllers(robots,i_robot,i_maze))

            #t3=time.time()
            #print('time to get controllers : {}',format(t3-t2))

            # Clip to [0,1]
            for i_cont in range(len(controllers[-1])):
                controllers[-1][i_cont]=np.clip(controllers[-1][i_cont],0,1)

            #t4=time.time()
            #print('time to clip : {}',format(t4-t3))

            # Compute new heading (-18 deg for 0 and +18 deg for 1)
            heading=robots[i_robot].heading[i_maze]+(controllers[-1][1]-0.5)*18

            #t5=time.time()
            #print('time to compute heading : {}',format(t5-t4))

            # Take modulo and save in robot
            robots[i_robot].heading[i_maze]=np.mod(heading,360)
        
            #t6=time.time()
            #print('time to take modulo and save to robot : {}',format(t6-t5))

            # Compute velocity (-0,5 m/s for 0 and 0.5 m/s for 1)
            velocity=robots[i_robot].velocity[i_maze]+(controllers[-1][0]-0.5)*0.5

            #t7=time.time()
            #print('time to compute velocity : {}',format(t7-t6))

            # Clip to +/- 2m/s and save in robot
            robots[i_robot].velocity[i_maze]=np.clip(velocity,-2,2)

            #t8=time.time()
            #print('time to clip velocity : {}',format(t8-t7))

            # Compute distance
            distance=robots[i_robot].velocity[i_maze]*time_step
    
            #t9=time.time()
            #print('time to compute distance : {}',format(t9-t8))

            # Compute new position
            robots[i_robot].position[i_maze].append(mazes[i_maze].next_pos(robots[i_robot].position[i_maze][-1],distance,robots[i_robot].heading[i_maze]))

            #t10=time.time()
            #print('time to compute new position : {}',format(t10-t9))

            # Check if robot is moving
            n_pos = len(robots[i_robot].position[i_maze])
            if ( (n_pos >= 50) and (np.mod(n_pos,50)==0) ):
                IsMoving = False
                x_move = robots[i_robot].position[i_maze][-1][0] - robots[i_robot].position[i_maze][-50][0]
                y_move = robots[i_robot].position[i_maze][-1][1] - robots[i_robot].position[i_maze][-50][1]
                if ( (abs(x_move) > 0.1) and (abs(y_move) > 0.1) ):
                    IsMoving=True

            #t11=time.time()
            #print('time to check if robot is moving : {}',format(t11-t10))

        return IsMoving

    # Compute the robot fitness (inverse of the distance to the goal + penality if wall in between)
    # Also check if robot reached goal
    # -> max_dist is the largest distance to the goal under which we consider the robot reached the goal
    # returns true if all mazes achieved goal, false otherwise
    def compute_fitness(self,robots,mazes,i_robot,maze_count,max_dist):

        reached_goal = True

        offset = 1.5*mazes[0].size
        
        #print('#COMPUTE FITNESS#')

        # For each maze
        for i_maze in range(maze_count):

            #t0 = time.time()
            #print('### MAZE {}#'.format(i_maze))

            # Compute distance to goal and check if there is a wall
            dist_goal,Iswall=mazes[i_maze].dist_goal(robots[i_robot].position[i_maze][-1])

            #t1=time.time()
            #print('time to compute distance to goal : {}',format(t1-t0))

            # Is goal reached ? (distance under max_dist and no wall)
            # Record fitness
            if ( (dist_goal<max_dist) and (not Iswall) ):
                robots[i_robot].reached_goal[i_maze]=True
                robots[i_robot].fitness[i_maze]=max(offset-dist_goal,robots[i_robot].fitness[i_maze])
            elif ( Iswall ):
                robots[i_robot].reached_goal[i_maze]=robots[i_robot].reached_goal[i_maze] or False
                robots[i_robot].fitness[i_maze]=max(offset-(dist_goal+max_dist),robots[i_robot].fitness[i_maze])
            else :
                robots[i_robot].reached_goal[i_maze]=robots[i_robot].reached_goal[i_maze] or False
                robots[i_robot].fitness[i_maze]=max(offset-dist_goal,robots[i_robot].fitness[i_maze])
                
            reached_goal=reached_goal and robots[i_robot].reached_goal[i_maze]

            #t2=time.time()
            #print('time to record fitness : {}',format(t2-t1))

     
        # Compute min fitness of robot
        if (robots[i_robot].fitness_min==-1):
            robots[i_robot].fitness_min=np.min(robots[i_robot].fitness)
        else :
            robots[i_robot].fitness_min=min(np.min(robots[i_robot].fitness),robots[i_robot].fitness_min)

        # Compute max fitness of robot
        if (robots[i_robot].fitness_max==-1):
            robots[i_robot].fitness_max=np.max(robots[i_robot].fitness)
        else :
            robots[i_robot].fitness_max=max(np.max(robots[i_robot].fitness),robots[i_robot].fitness_max)
        
        # Compute avg fitness of robot
        robots[i_robot].fitness_avg=np.mean(robots[i_robot].fitness)

        #t3=time.time()
        #print('time to compute fitness : {}',format(t3-t2))

        return reached_goal

    # Adjust fitness given the number of robots per species
    def adjust_fitness(self,robots,i_robot,species):

        i_specie=robots[i_robot].genome.specie
        
        # Find the number of robots with the same specie as i_robot
        count=species.specie_count[species.specie_list.index(i_specie)]

        # Adjust the fitness
        robots[i_robot].adj_fitness_avg=robots[i_robot].fitness_avg/count
