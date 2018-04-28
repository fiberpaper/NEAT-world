## v0 - 23/06/2017
## specie

import numpy as np

## Define specie
class specie:


    # Create specie
    # As defined in for original NeuroEvoluation of Augmented Topologies
    def __init__(self,delta,c1_excess,c2_disjoint,c3_weight):

        self.delta=delta

        self.c1_excess=c1_excess
        self.c2_disjoint=c2_disjoint
        self.c3_weight=c3_weight
        
        self.specie_list=[]
        self.specie_genome=[]
        self.specie_count=[]
        
        self.newest_specie=-1

    # Change delta
    def change_delta(self,delta_add):

        self.delta+=delta_add

    # Create specie list from previous generation
    # Should not be used !?
    def create_from_generation(self,robots):

        # Reset lists
        self.specie_list=[]
        self.specie_genome=[]
        self.specie_count=[]

        # Reset specie counter
        self.newest_specie=-1

        n_robot = len(robots)
        ind_robot = np.arange(n_robot)
        rand_ind_robot = np.random.permutation(ind_robot)

        for i_robot in range(n_robot):

            # Recompute compatibility
            self.compatibility(robots[rand_ind_robot[i_robot]].genome)

            # If the specie of i_robot is not in the list, add it
            if (robots[rand_ind_robot[i_robot]].genome.specie not in self.specie_list):
                #print(rand_ind_robot[i_robot])
                #print(robots[rand_ind_robot[i_robot]].genome.specie)
                self.specie_list.append(robots[rand_ind_robot[i_robot]].genome.specie)
                self.specie_genome.append(robots[rand_ind_robot[i_robot]].genome)
                self.specie_count.append(0)

        # Sort lists by increasing specie number
        n_specie = len(self.specie_list)
        ind=sorted(range(n_specie), key=lambda k: self.specie_list[k])
        specie_list_tmp=self.specie_list.copy()
        specie_genome_tmp=self.specie_genome.copy()
        specie_count_tmp=self.specie_count.copy()
        for i_specie in range(n_specie):
            self.specie_list[i_specie]=specie_list_tmp[ind[i_specie]]
            self.specie_genome[i_specie]=specie_genome_tmp[ind[i_specie]]
            self.specie_count[i_specie]=specie_count_tmp[ind[i_specie]]
           

    # Check compatibility and add specie number to genome
    def compatibility(self,genome):

        compatible=False

        n_specie=len(self.specie_genome)
        i_specie=0

        # Genome to check against list of species
        G_n_connec=len(genome.connection_gene_innovation)
        G_sort_ind=sorted(range(G_n_connec), key=lambda k: genome.connection_gene_innovation[k])

        # While compatibility is not found and there are still species
        while ( (i_specie<n_specie) and (not compatible)):

            i_delta=0
            i_disjoint=0
            i_excess=0
            i_weight=0
            n_weight=0

            # Representative genome for each specie
            Gi_n_connec=len(self.specie_genome[i_specie].connection_gene_innovation)
            Gi_sort_ind=sorted(range(Gi_n_connec), key=lambda k: self.specie_genome[i_specie].connection_gene_innovation[k])

            # Gene index
            Gi_j_connec=0
            G_j_connec=0

            # While there are still genes
            while( (G_j_connec<G_n_connec) and (Gi_j_connec<Gi_n_connec) ):

                # If both genes are identical
                if (self.specie_genome[i_specie].connection_gene_innovation[Gi_sort_ind[Gi_j_connec]]==genome.connection_gene_innovation[G_sort_ind[G_j_connec]]):

                    # Add the weight difference to the weight total and increment the counter
                    w=genome.connection_gene_weight[G_sort_ind[G_j_connec]]
                    w_i=self.specie_genome[i_specie].connection_gene_weight[Gi_sort_ind[Gi_j_connec]]
                    i_weight+=np.abs(w-w_i)
                    n_weight+=1
                    
                    # Increment gene index
                    G_j_connec+=1
                    Gi_j_connec+=1

                # If i_specie gene is smaller
                elif (self.specie_genome[i_specie].connection_gene_innovation[Gi_sort_ind[Gi_j_connec]]<genome.connection_gene_innovation[G_sort_ind[G_j_connec]]):

                    # Increment the disjoint counter
                    i_disjoint+=1

                    # Increment gene index
                    Gi_j_connec+=1
                    
                else :

                    # Increment the disjoint counter
                    i_disjoint+=1

                    # Increment gene index
                    G_j_connec+=1
                
            # If some genes left, add to excess counter
            if (G_j_connec<G_n_connec):

                i_excess+=G_n_connec-G_j_connec

            if (Gi_j_connec<Gi_n_connec):

                i_excess+=Gi_n_connec-Gi_j_connec

            # Normalize values and apply scalars
            N=max(G_n_connec,Gi_n_connec)
            if (N!=0 and N<20):
                N=1
            if (N!=0):
                i_excess=self.c1_excess*i_excess/N
                i_disjoint=self.c2_disjoint*i_disjoint/N
            if(n_weight!=0):
                i_weight=self.c3_weight*i_weight/n_weight

            # Conpute i_delta for i_specie
            i_delta=i_excess+i_disjoint+i_weight

            # If i_delta is smaller than delta of the class, then we the genomes are compatible
            if (i_delta<self.delta):
                compatible=True
                genome.specie=self.specie_list[i_specie]

            # Increment i_specie
            i_specie+=1
            
        # If not compatible create a new specie
        if (not compatible):

            self.newest_specie+=1
            self.specie_list.append(self.newest_specie)
            self.specie_genome.append(genome)
            self.specie_count.append(0)
            genome.specie=self.newest_specie

##            if (self.newest_specie==-1):
##                self.newest_specie+=1
##                self.specie_list.append(self.newest_specie)
##                self.specie_genome.append(genome)
##                self.specie_count.append(1)
##                genome.specie=self.newest_specie
##            else:
##                self.newest_specie+=1
##                self.specie_list.append(self.newest_specie)
##                self.specie_genome.append(genome)
##                self.specie_count.append(1)
##                genome.specie=self.newest_specie
##                    


    # Count robots per species
    def count(self,robots):

        n_specie = len(self.specie_list)
        n_robot = len(robots)
        
        # reset count
        for i_specie in range(n_specie):

            self.specie_count[i_specie]=0

        # Count
        for i_robot in range(n_robot):

            i_specie=robots[i_robot].genome.specie

            if (i_specie in self.specie_list):
                ind=self.specie_list.index(i_specie)
                self.specie_count[ind]+=1
            else:
                raise ValueError('Specie not recognized, cannot count')
            
