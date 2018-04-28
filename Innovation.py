## v0 - 16/06/2017
## Innovation

import numpy as np


## Define innovation class
class innovation:

    # Create an innovation list
    def __init__(self):

        self.node_in=list()
        self.node_out=list()
        self.innovation_nb=list()
        self.innovation_type=list()

    # Print innovation list
    def print_innovation(self):

        print('## INNOVATION LIST:')
        print('## In \tOut \t Innovation \t Type')
        for i_innov in range(len(self.node_in)):
            print('## {}\t {}\t {}\t {}'.format(self.node_in[i_innov],self.node_out[i_innov],self.innovation_nb[i_innov],self.innovation_type[i_innov]))
    
    # Add an innovation
    # -> node_in is the input node number
    # -> node_out is the output node number
    # -> innovation_number is the innovation number
    # -> innovation_type is the innovation type (connection or node)
    def add(self,node_in,node_out,innovation_number,innovation_type):

        self.node_in.append(node_in)
        self.node_out.append(node_out)
        self.innovation_nb.append(innovation_number)

        # check it is 'node' or 'connection'
        if (innovation_type!='node' and innovation_type!='connection'):
            raise ValueError('Innovation type should be node or connection')

        self.innovation_type.append(innovation_type)


    # Is a connection between two nodes an innovation ?
    # -> node_in is the input node number
    # -> node_out is the output node number
    # returns (bool,int)
    # True if it is an innovation, False otherwise
    # [] if it is an innvoation, innov number otherwise
    def IsInnovation(self,node_in,node_out):

        innov=[i for i, j, k, in zip(self.innovation_nb,self.node_in,self.node_out) if (j==node_in and k==node_out)]

        if (len(innov)==0):
            return True,[]
        elif (len(innov)==1):
            return False,innov[0]
        else:
            raise IndexError('too many innovations exist between the same nodes : {} {}'.format(node_in,node_out))


    # Is a node between two nodes an innovation ?
    # This means two consecutives innovations were created at some point and recorder in the list 
    # -> node_in is the original input node number
    # -> node_out is the original output node number
    # returns (bool,int,int,int)
    # True if it is an innovation, False otherwise
    # [],[],[] if it is an innvoation, innov1,innov2,node numbers otherwise
    def IsNodeInnovation(self,node_in,node_out):

        n_innov=len(self.innovation_nb)

        if (n_innov<2):
            return True,[],[],[]

        i_innov1=[i for i, j, k, l, m, n, in zip(np.arange(n_innov),self.node_in[:-1],self.node_out[:-1],self.node_in[1:],self.node_out[1:],self.innovation_type[:-1]) if (j==node_in and k==l and m==node_out and n=='node')]

        if (len(i_innov1)==0):
            return True,[],[],[]
        elif (len(i_innov1)==1):
            return False,self.innovation_nb[i_innov1[0]],self.innovation_nb[i_innov1[0]+1],self.node_out[i_innov1[0]]
        else:
            raise IndexError('too many innovations exist between the same nodes : {} {}'.format(node_in,node_out))

        
