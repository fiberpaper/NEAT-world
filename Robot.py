## v0 - 11/06/2017
## robot

import math
import numpy as np

from Activations import activations
from RobotGenome import robot_genome
from Innovation import innovation

## Define robot class
class robot:


    # Create robot
    def __init__(self,maze_count,mazes,cur_node,activation):

        # Input sensors
        # As defined in "Exploiting Open-Endedness to Solve Problems Through the Search for Novelty"
        self.sensor=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.sensor_name=['rangefinder_W','rangefinder_NW','rangefinder_N',
                            'rangefinder_NE','rangefinder_E','rangefinder_S',
                            'pieslice_detector_N','pieslice_detector_E',
                            'pieslice_detector_S','pieslice_detector_W']

        # Output controller
        # As defined in the same article
        self.controller=[0.0,0.0]
        self.controller_name=['left_right','forward_backward']

        # Position
        # [maze][time][x,y]
        # Initialize position from maze definition
        self.position=[]
        for imaze in range(maze_count):
            self.position.append([])
            self.position[-1].append(mazes[imaze].start_pos)

        # Heading/direction in which the robot is moving)
        # [maze][time][deg]
        # Initialize at north (0 degree)
        self.heading=[]
        for imaze in range(maze_count):
            self.heading.append([])
            self.heading[-1].append(0.0)        

        # velocityin which the robot is moving
        # [maze][time][deg]
        # Initialize at 0 m/s
        self.velocity=[]
        for imaze in range(maze_count):
            self.velocity.append([])
            self.velocity[-1].append(0.0)        

        # Possible activation functions
        # .list returns a list of all functions names
        # .op   returns a list of all functions directly
        self.activation=activations(activation)

        # Check if a robot has been evaluated already
        self.IsEvaluated = False

        # Fitness
        self.fitness=[]
        for imaze in range(maze_count):
            self.fitness.append(-1)

        # Fitness stats
        self.fitness_min = -1
        self.fitness_max = -1
        self.fitness_avg = -1

        # Adjusted average fitness (divide number robots in a specie)
        self.adj_fitness_avg = -1

        # Novelty
        self.novelty = -1

        # Goal reached?
        self.reached_goal=[]
        for imaze in range(maze_count):
            self.reached_goal.append(False)        

        # Genome
        self.n_input=len(self.sensor)
        self.n_output=len(self.controller)
        self.n_bias=1
        self.genome=robot_genome()
        output_act=self.activation.list[np.random.randint(len(self.activation.list))]
        cur_node[0]=self.genome.add_in_out_nodes(self.n_input,self.n_output,self.n_bias,output_act)
        

    def get_sensors(self,mazes,i_maze):

        # Get input sensors values for range finders
        self.sensor[0]=mazes[i_maze].dist_wall(self.position[i_maze][-1],270)
        self.sensor[1]=mazes[i_maze].dist_wall(self.position[i_maze][-1],315)
        self.sensor[2]=mazes[i_maze].dist_wall(self.position[i_maze][-1],0)
        self.sensor[3]=mazes[i_maze].dist_wall(self.position[i_maze][-1],45)
        self.sensor[4]=mazes[i_maze].dist_wall(self.position[i_maze][-1],90)
        self.sensor[5]=mazes[i_maze].dist_wall(self.position[i_maze][-1],180)
        
        # Get input sensors values for pie slice detector
        (self.sensor[6],self.sensor[7],self.sensor[8],self.sensor[9])=mazes[i_maze].goal_insight(self.position[i_maze][-1])

        #print(self.sensor)


    def mutate_add_connection(self,innov_list,innov_nb,magnitude=1.0):

        n_connec=len(self.genome.connection_gene_input)
        n_node=len(self.genome.node_gene_number)

        # If max connection already reached then we cannot add a connection
        if (n_connec==self.genome.MaxConnection()):
            return

        # Randomly choose input
        node_in=self.genome.node_gene_number[np.random.randint(n_node)]

        # Randomly choose output
        node_out=self.genome.node_gene_number[np.random.randint(self.n_input+self.n_bias,n_node)]

        # While a connection exists between the nodes, choose new nodes
        while self.genome.IsConnected(node_in,node_out):
            node_in=self.genome.node_gene_number[np.random.randint(n_node)]
            node_out=self.genome.node_gene_number[np.random.randint(self.n_input+self.n_bias,n_node)]

        # Get a random weight
        weight=np.random.rand()*2*magnitude-magnitude

        # Search if it is in the innovation list and return the number if so
        IsInnovation,innov=innov_list.IsInnovation(node_in,node_out)

        # If not increment the innovation number add to the innovation list
        if (IsInnovation):
            innov_nb[0]+=1
            innov_list.add(node_in,node_out,innov_nb[0],'connection')
            innov=innov_nb[0]
        
        # Mutate
        #print('==> Before Add Connection Mutation')
        #self.genome.print_genome()
        self.genome.add_connection(node_in,node_out,weight,innov) 
        #print('==> After Add Connection Mutation')
        #self.genome.print_genome()


    def mutate_add_node(self,innov_list,innov_nb,node_nb,magnitude=1.0):

        n_connec=len(self.genome.connection_gene_innovation)

        # if no connection stop
        if (n_connec==0):
            return

        # If all connections are disabled then do nothing
        if (not self.genome.IsOKdisable_connection() ):
            return

        # Randomly choose connection
        connec=np.random.randint(n_connec)

        # While a connection is disabled choose new one
        while (self.genome.connection_gene_IsEnabled[connec]==False):
            connec=np.random.randint(n_connec)

        # Get node_in and node_out
        node_in=self.genome.connection_gene_input[connec]
        node_out=self.genome.connection_gene_output[connec]
        
        # Get a random weight
        weight=np.random.rand()*2*magnitude-magnitude

        # Get a random activation function
        node_act=self.activation.list[np.random.randint(len(self.activation.list))]
        
        # Search if it is in the innovation list and return the numbers if so
        IsInnovation,innov1,innov2,node=innov_list.IsNodeInnovation(node_in,node_out)

        # If not increment the innovation number and node number and add to the innovation list
        if (IsInnovation):
            innov1=innov_nb[0]+1
            innov2=innov_nb[0]+2
            node=node_nb[0]+1
            innov_nb[0]+=2
            node_nb[0]+=1
            node=node_nb[0]
            innov_list.add(node_in,node_nb[0],innov1,'node')
            innov_list.add(node_nb[0],node_out,innov2,'node')
        # Otherwise check that the innovation does not already exists
        # If so do nothing
        else :
            if(self.genome.ConnectionExists(node_in,node_out,node,innov1,innov2)):
                return
        
        # Mutate
        #print('==> Before Add Node Mutation')
        #self.genome.print_genome()
        self.genome.add_node(connec,node_in,node_out,node_act,node,innov1,innov2,weight) 
        #print('==> After Add Node Mutation')
        #self.genome.print_genome()

    def replace_weight(self):

        n_connec = len(self.genome.connection_gene_innovation)

        # if no connection stop
        if (n_connec==0):
            return
        
        # If all connections are disabled then do nothing
        if (not self.genome.IsOKdisable_connection() ):
            return

        # Randomly choose connection
        connec=np.random.randint(n_connec)

        # While a connection is disabled choose new one
        while (self.genome.connection_gene_IsEnabled[connec]==False):
            connec=np.random.randint(n_connec)

        # Perturbation value can be negative or positive
        rnd=np.random.rand()-0.5

        # Replace
        #print('==> Before Weight Replacement')
        #self.genome.print_genome()
        self.genome.change_connection_weight(connec,rnd)  
        #print('==> After Weight replacement')
        #self.genome.print_genome()

    def mutate_weight(self,perturb_max):

        n_connec = len(self.genome.connection_gene_innovation)

        # if no connection stop
        if (n_connec==0):
            return
        
        # If all connections are disabled then do nothing
        if (not self.genome.IsOKdisable_connection() ):
            return

        # Randomly choose connection
        connec=np.random.randint(n_connec)

        # While a connection is disabled choose new one
        while (self.genome.connection_gene_IsEnabled[connec]==False):
            connec=np.random.randint(n_connec)

        # Perturbation value can be negative or positive
        rnd=np.random.rand()-0.5

        # if not zero we compute the perturbation as a gaussian with max perturb_max
        if (rnd>0.0):
            perturb=np.exp(-1.0 * (abs(rnd)**2))*perturb_max
        else:
            perturb=1/(np.exp(-1.0 * (abs(rnd)**2))*perturb_max)
            
        # Mutate
        #print('==> Before Weight Mutation')
        #self.genome.print_genome()
        self.genome.change_connection_weight(connec,perturb)  
        #print('==> After Weight Mutation')
        #self.genome.print_genome()


    def mutate_activation(self):

        n_node=len(self.genome.node_gene_number)

        n_connec=len(self.genome.connection_gene_innovation)

        # if no connection stop
        if (n_connec==0):
            return

        # If all connections are disabled then do nothing
        if (not self.genome.IsOKdisable_connection() ):
            return
        
        # Randomly choose node
        node=self.genome.node_gene_number[np.random.randint(self.n_input+self.n_bias,n_node)]

        # While a node has no connection choose new one
        while (not self.genome.IsActive(node)):
            node=self.genome.node_gene_number[np.random.randint(self.n_input+self.n_bias,n_node)]
 
        # Get a random activation function
        new_node_act=self.activation.list[np.random.randint(len(self.activation.list))]

        # While the new activation is the same as the old one change
        cur_node_act=self.genome.node_gene_activation[self.genome.node_gene_number.index(node)]
        while(new_node_act==cur_node_act):
            new_node_act=self.activation.list[np.random.randint(len(self.activation.list))]

        # Mutate
        #print('==> Before Activation Mutation')
        #self.genome.print_genome()
        self.genome.change_activation(node,new_node_act)
        #print('==> After Activation Mutation')
        #self.genome.print_genome()
        

    def mutate_disable(self):

        n_connec = len(self.genome.connection_gene_innovation)

        # If all connections are disabled then do nothing
        if (not self.genome.IsOKdisable_connection() ):
            return

        # Randomly choose connection
        connec=np.random.randint(n_connec)

        # While a connection is disabled choose new one
        while (self.genome.connection_gene_IsEnabled[connec]==False):
            connec=np.random.randint(n_connec)

        # Mutate
        #print('==> Before Disable Mutation')
        #self.genome.print_genome()
        self.genome.disable_connection(connec)  
        #print('==> After Disable Mutation')
        #self.genome.print_genome()
        #print('==> disabled connection {} ?'.format(connec))

        
    def mutate_enable(self):

        n_connec = len(self.genome.connection_gene_innovation)

        # If all connections are enabled then do nothing
        if (not self.genome.IsOKenable_connection()):
            return

        # Randomly choose connection
        connec=np.random.randint(n_connec)

        # While a connection is enabled choose new one
        while (self.genome.connection_gene_IsEnabled[connec]==True):
            connec=np.random.randint(n_connec)

        # Mutate
        #print('==> Before Enable Mutation')
        #self.genome.print_genome()
        self.genome.enable_connection(connec)  
        #print('==> After Enable Mutation')
        #self.genome.print_genome()
        #print('==> enabled connection {} ?'.format(connec))


    def visualize_path(self,i_maze,mazes):

        # Copy playground
        viz=mazes[i_maze].playground.copy().astype('int8')*255
        
        #print('size position list : {}'.format(len(self.position[i_maze])))
        #print('position list first: {}'.format(self.position[i_maze][0]))
        #print('position list last: {}'.format(self.position[i_maze][-1]))

        # for each time step
        for t in range(len(self.position[i_maze])):
            # Get position as integer
            pos=(int(np.round(self.position[i_maze][t][0])),int(np.round(self.position[i_maze][t][1])))
            # Add onto viw
            viz[pos]=200

        # Mark the goal with an X
        x=int(np.round(mazes[i_maze].goal_pos[0]))
        y=int(np.round(mazes[i_maze].goal_pos[1]))
        for i_x in range(x-1,x+2,1):
            for i_y in range(y-1,y+2,1):
                viz[i_x,i_y]=255

        return viz        
