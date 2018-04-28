## v0 - 11/06/2017
## maze_genome

import numpy as np

## Define maze genome
class maze_genome:


    # Create maze_genome
    # As defined in "Minimal Criterion Coevolution: A new Approach to Open-Ended Search"
    def __init__(self,weight=0.0,passage=0.0):
    
        # Weight and passage should be between 0 and 1
        if ( (weight<0) or (weight>1) ):
            raise ValueError('weight should be between 0 and 1 but found {}'.format(weight) )
        if ( (passage<0) or (passage>1) ):
            raise ValueError('passage should be between 0 and 1 but found {}'.format(passage) )
        
        self.gene_weight=list()
        self.gene_passage=list()
        self.Nwalls=1
        
        if (weight==0.0):
            w=(np.random.rand()-0.5)*0.8+0.5
            self.gene_weight.append(w)
        else:
            self.gene_weight.append(weight)

        if (passage==0.0):
            p=(np.random.rand()-0.5)*0.8+0.5
            self.gene_passage.append(p)
        else:
            self.gene_passage.append(passage)


    # Add wall
    def add(self,weight=0.0,passage=0.0):

        # Weight and passage should be between 0 and 1
        if ( (weight<0) or (weight>1) ):
            raise ValueError('weight should be between 0 and 1 but found {}'.format(weight) )
        if ( (passage<0) or (passage>1) ):
            raise ValueError('passage should be between 0 and 1 but found {}'.format(passage) )

        if (weight==0.0):
            w=(np.random.rand()-0.5)*0.8+0.5
            self.gene_weight.append(w)
        else:
            self.gene_weight.append(weight)

        if (passage==0.0):
            p=(np.random.rand()-0.5)*0.8+0.5
            self.gene_passage.append(p)
        else:
            self.gene_passage.append(passage)

        self.Nwalls+=1
