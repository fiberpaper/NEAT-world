## v0 - 14/06/2017
## robot_genome

import numpy as np

## Define robot genome
class robot_genome:


    # Create robot genome
    # As defined in for original NeuroEvoluation of Augmented Topologies
    def __init__(self):
    
        self.n_input = 0
        self.n_bias = 0
        self.n_output = 0

        self.specie = -1

        self.node_gene_number=list()
        self.node_gene_type=list()
        self.node_gene_activation=list()

        self.connection_gene_input=list()
        self.connection_gene_output=list()
        self.connection_gene_weight=list()
        self.connection_gene_IsEnabled=list()
        self.connection_gene_innovation=list()


    # Get node type and activation given its number
    def get_node_details(self,node):

        if (node not in self.node_gene_number):
            raise ValueError('node not in genome')

        ind=self.node_gene_number.index(node)

        return self.node_gene_type[ind],self.node_gene_activation[ind]

    # Print genome
    def print_genome(self):

        # Print nodes
        print('## NODES :')
        print('## Number \tType \t Activation')
        for i_node in range(len(self.node_gene_number)):
            print('## {}\t {}\t {}'.format(self.node_gene_number[i_node],self.node_gene_type[i_node],self.node_gene_activation[i_node]))
        print('## CONNECTIONS :')
        print('## Input \tOutput \tWeight \tIsEnabled \tInnovation')
        for i_connec in range(len(self.connection_gene_input)):
            print('## {}\t {}\t {}\t {}\t {}'.format(self.connection_gene_input[i_connec],self.connection_gene_output[i_connec],self.connection_gene_weight[i_connec],self.connection_gene_IsEnabled[i_connec],self.connection_gene_innovation[i_connec]))
        print('## SPECIE : {}'.format(self.specie))

    # Count total number of possible connections
    def MaxConnection(self):

        # Find number of input, output and hidden nodes
        n_in=0
        n_out=0
        n_hid=0
        for i_node in range(len(self.node_gene_number)):
            if (self.node_gene_type[i_node]=='input'):
                n_in+=1
            elif (self.node_gene_type[i_node]=='output'):
                n_out+=1
            elif (self.node_gene_type[i_node]=='hidden'):
                n_hid+=1
            else :
                raise ValueError('type of node unknown : {}'.format(self.node_gene_type[i_node]))

        # Number of connection for input nodes is n_in*(n_hid+n_out) => can connect to any of the hidden and output nodes
        in_connec=n_in*(n_hid+n_out)
        # Number of connection for output nodes is n_out*(n_hid+1) => can connect to any hidden nodes and to itself
        out_connec=n_out*(n_hid+1)
        # Number of conenction for the hidden nodes is n_hid*(n_hid+n_out) => can connect to any hidden and output nodes
        hid_connec=n_hid*(n_hid+n_out)

        return in_connec+out_connec+hid_connec

    # Add input and output nodes
    def add_in_out_nodes(self,n_input=10,n_output=2,n_bias=1,output_act='sigmoid'): 

        # n_input qnd n_output should be > 0
        if ( n_input < 1 ):
            raise ValueError('number of input of the genome should be > 0. Found {}'.format(n_input) )
        if ( n_output < 1 ):
            raise ValueError('number of output of the genome should be > 0. Found {}'.format(n_output) )

        # n_bias should be 0 or 1
        if ( n_bias != 1 and n_bias != 0):
            raise ValueError('number of bias of the genome should be 0 or 1. Found {}'.format(n_output) )

        # initialize the node count
        cur_node=-1

        # Add input nodes
        self.n_input=n_input
        for i_input in range(n_input):
            cur_node+=1
            self.node_gene_number.append(cur_node)
            self.node_gene_type.append('input')
            self.node_gene_activation.append('NA')
            
        # Add bias nodes
        self.n_bias=n_bias
        if (n_bias==1):
            cur_node+=1
            self.node_gene_number.append(cur_node)
            self.node_gene_type.append('input')
            self.node_gene_activation.append('NA')

        # Add output nodes
        self.n_output=n_output
        for i_output in range(n_output):
            cur_node+=1
            self.node_gene_number.append(cur_node)
            self.node_gene_type.append('output')
            self.node_gene_activation.append(output_act)

        return cur_node


    # Add a connection to the genome
    # -> node_in is the input node number
    # -> node_out is the output node number
    # -> weigth is the weight of the connection (if zero it is randomly generated in [-0.5;0.5])
    # -> innov is the innovation number in the population
    def add_connection(self,node_in,node_out,weight,innov):

        # Make sure the in/out nodes exist
        if (node_in not in self.node_gene_number):
            raise ValueError('node in should be in the list of node numbers : {}'.format(node_in) )
        if (node_out not in self.node_gene_number):
            raise ValueError('node out should be in the list of node numbers : {}'.format(node_out) )
        
        # Add nodes to lists
        self.connection_gene_input.append(node_in)
        self.connection_gene_output.append(node_out)

        # Add weight
        self.connection_gene_weight.append(weight)

        # Enable new connection
        self.connection_gene_IsEnabled.append(True)

        # Give the new connection a unique innovation marker
        self.connection_gene_innovation.append(innov)



    # Change weight of an existing connection in the genome
    # -> pos is the position of the connection gene to change
    # -> perturn is the multiplier to change the current weight
    def change_connection_weight(self,pos,perturb):

        # If perturb is not zero change the weight by perturb
        if (perturb!=0):
            self.connection_gene_weight[pos] *= perturb
        # Otherwise raise an error
        else:
            raise ValueError('Amount to change connection should not be zero : {}'.format(amount) )
            # Enable new connection


    # Check if there is at least one connection enabled
    # returns True is so, False otherwise
    def IsOKdisable_connection(self):

        for gene_isEnabled in self.connection_gene_IsEnabled:
            if (gene_isEnabled):
                return True

        return False


    # Disable an existing connection in the genome
    # -> pos is the position of the connection gene to change
    def disable_connection(self,pos):

        # Disable the connection
        self.connection_gene_IsEnabled[pos] = False


    # Check if there is at least one connection disabled
    # returns True is so, False otherwise
    def IsOKenable_connection(self):

        for gene_isEnabled in self.connection_gene_IsEnabled:
            if (not gene_isEnabled):
                return True

        return False

    # Enable an existing connection in the genome
    # -> pos is the position of the connection gene to change
    def enable_connection(self,pos):

        # Disable the connection
        self.connection_gene_IsEnabled[pos] = True


    # Change node activation function in the genome
    # -> node is the node gene to change
    # -> new_node_act is the new activation function for the node
    def change_activation(self,node,new_node_act):

        self.node_gene_activation[self.node_gene_number.index(node)]=new_node_act


    # Is there a connection between two existing nodes
    # -> node_in is the input node number
    # -> node_out is the output node number
    # returns True if a connection exists or False otherwise
    def IsConnected(self,node_in,node_out):

        # If both nodes are output node, fake that there is a connection to prevent them to connect, unless both nodes are identical
        if (self.node_gene_type[self.node_gene_number.index(node_in)]=='output' and self.node_gene_type[self.node_gene_number.index(node_out)]=='output'):
            if (node_in!=node_out):
                return True
        
        # Find if connection existed between original nodes
        i_connec=[i for i, j, k, in zip(np.arange(len(self.connection_gene_input)),self.connection_gene_input,self.connection_gene_output) if (j==node_in and k==node_out)]
        if (len(i_connec)>1):
            raise IndexError('too many connections exist between the same nodes : {} {} {}'.format(node_in,node_out,i_connec))
        elif (len(i_connec)==1):
            return True
        else:
            return False

    # Is a connection defined by node_in, node_out, node, innov1, innov2 exists?
    # returns True is so, False otherwise
    def ConnectionExists(self,node_in,node_out,node,innov1,innov2):

        connec1=False
        connec2=False

        for i_connec in range(len(self.connection_gene_input)):
            if (self.connection_gene_input[i_connec]==node_in and self.connection_gene_output[i_connec]==node and self.connection_gene_innovation[i_connec]==innov1):
                connec1=True
            if (self.connection_gene_input[i_connec]==node and self.connection_gene_output[i_connec]==node_out and self.connection_gene_innovation[i_connec]==innov2):
                connec2=True

        return (connec1 and connec2)
    
    

    # Add a node in between two existing nodes
    # -> pos is the position of the connection where we insert the node
    # -> node_in is the original input node number
    # -> node_out is the original output node number
    # -> node_act is the activation function for the node
    # -> node is the node number
    # -> innov1 is the innovation number for the first new connection between the input and new node
    # -> innov2 is the innovation number for the second new connection between the new node and output node
    # -> weight is the weight of the connection
    def add_node(self,pos,node_in,node_out,node_act,node,innov1,innov2,weight):

        # Make sure the in/out nodes exist
        if (node_in not in self.node_gene_number):
            raise ValueError('node in should be in the list of node numbers : {}'.format(node_in) )
        if (node_out not in self.node_gene_number):
            raise ValueError('node out should be in the list of node numbers : {}'.format(node_out) )
        
        # Add node to list
        self.node_gene_number.append(node)
        self.node_gene_type.append('hidden')
        self.node_gene_activation.append(node_act)
        
        # Get the weigth from previous connection
        weight=self.connection_gene_weight[pos]

        # Add connection between original input and new node (weight=1)
        self.connection_gene_input.append(node_in)
        self.connection_gene_output.append(node)
        self.connection_gene_weight.append(1.0)
        self.connection_gene_IsEnabled.append(True)
        self.connection_gene_innovation.append(innov1)

        # Add connection between the new node and the original output (weight=original weight)
        self.connection_gene_input.append(node)
        self.connection_gene_output.append(node_out)
        self.connection_gene_weight.append(weight)
        self.connection_gene_IsEnabled.append(True)
        self.connection_gene_innovation.append(innov2)

        # Disable the old connection
        self.connection_gene_IsEnabled[pos]=False
        

    # Is there at least ont active connection from/to a node
    # -> node is the node to check
    # returns True if a connection exists or False otherwise
    def IsActive(self,node):

        # Find if connection existed between original nodes
        active=[i for i, j, k, in zip(self.connection_gene_IsEnabled,self.connection_gene_input,self.connection_gene_output) if ((i==True) and (j==node or k==node))]
        if (len(active)>=1):
            return True
        else:
            return False

