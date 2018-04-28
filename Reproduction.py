## v0 - 24/06/2017
## Reproduction functions

import numpy as np
import copy

from Robot import robot

# Trim generation to keep best only
# -> robots are the robots in the generation
# -> keep_prct is the percentage of robots to keep in each specie
# returns trimmed robot list
def trim_generation(robots,keep_prct,species):

    # Robots to keep
    robots_keep=list()

    n_specie = len(species.specie_list)
    n_robots = len(robots)

    # Keep list will have a list or robots index to keep for each specie
    keep_list = []
    for i_specie in range(n_specie):
        keep_list.append([])

    # Go through each robot and order them in the keep list
    for i_robot in range(n_robots):

        i_robot_specie=robots[i_robot].genome.specie
        i_robot_specie_ind=species.specie_list.index(i_robot_specie)
        n_robot_list=len(keep_list[i_robot_specie_ind])

        # If there's not robot in the list yet, just add it
        if (n_robot_list==0):
            keep_list[i_robot_specie_ind].append(i_robot)

        # Otherwise go through list to find correct position
        else:

            ind=0
            i_fitness=robots[i_robot].adj_fitness_avg
            j_fitness=robots[keep_list[i_robot_specie_ind][ind]].adj_fitness_avg

            # Find index in the list based on fitness values
            while ( (i_fitness < j_fitness) and (ind < n_robot_list) ):
                ind+=1
                if (ind < n_robot_list):
                    j_fitness=robots[keep_list[i_robot_specie_ind][ind]].adj_fitness_avg

            # Insert robot in list
            keep_list[i_robot_specie_ind].insert(ind,i_robot)

    # Keep only the best of each specie
    for i_specie in range(n_specie):

        # Number of robot in a given specie
        n_robot_list=len(keep_list[i_specie])
        
        # Number of robot in a given specie to keep (min 1)
        n_robot_keep=max(1,int(n_robot_list*keep_prct))
              
        # For each robot to keep
        for i_keep in range(n_robot_keep):

            # Add robot to keep robot list
            robots_keep.append(robots[keep_list[i_specie][i_keep]])

    return robots_keep

# Add connection to new robot
def add_connection_new_robot(genome,ind,new_robot):

    # Get genome details
    node_in=genome.connection_gene_input[ind]
    node_in_type,node_in_ac=genome.get_node_details(node_in)
    node_out=genome.connection_gene_output[ind]
    node_out_type,node_out_ac=genome.get_node_details(node_out)
    weight=genome.connection_gene_weight[ind]
    innov=genome.connection_gene_innovation[ind]

    # If input node is not in the new robot genome, add it
    if (node_in not in new_robot.genome.node_gene_number):

        new_robot.genome.node_gene_number.append(node_in)
        new_robot.genome.node_gene_type.append(node_in_type)
        new_robot.genome.node_gene_activation.append(node_in_ac)
        
    # If output node is not in the new robot genome, add it
    if (node_out not in new_robot.genome.node_gene_number):

        new_robot.genome.node_gene_number.append(node_out)
        new_robot.genome.node_gene_type.append(node_out_type)
        new_robot.genome.node_gene_activation.append(node_out_ac)

    # Add connection to new robot
    new_robot.genome.add_connection(node_in,node_out,weight,innov)



# crossover between 2 robots
# -> robot1 is the first robot
# -> robot2 is the second robot
# -> reproduction_dic is the dictionnary of probabilities for reproduction
# returns the crossover robot
def crossover(robot1,robot2,maze_count,mazes,cur_node,activation,reproduction_dic):

    # crossover genome
    new_robot=robot(maze_count,mazes,cur_node,activation)

    # genome from robot1/2
    genome1=robot1.genome
    genome2=robot2.genome
    
    # Sort genome1 connections by innovation number
    G1_n_connec=len(genome1.connection_gene_innovation)
    G1_sort_ind=sorted(range(G1_n_connec), key=lambda k: genome1.connection_gene_innovation[k])

    # Sort genome2 connections by innovation number
    G2_n_connec=len(genome2.connection_gene_innovation)
    G2_sort_ind=sorted(range(G2_n_connec), key=lambda k: genome2.connection_gene_innovation[k])

    #print('Genome 1 nb genes: ',G1_n_connec)
    #print('Genome 2 nb genes: ',G2_n_connec)

    # Gene index
    G1_j_connec=0
    G2_j_connec=0

    # Find fittest genome
    if (robot1.fitness_avg>robot2.fitness_avg):
        robot1_fitter=True
        robot2_fitter=False
    else:
        robot1_fitter=False
        robot2_fitter=True

    #print('Start to go through genes...')

    # While there are still genes
    while( (G1_j_connec<G1_n_connec) and (G2_j_connec<G2_n_connec) ):

        #print('Genome 1 gene: ',G1_j_connec)
        #print('Genome 2 gene: ',G2_j_connec)
        
        # If both genes are identical
        if (genome1.connection_gene_innovation[G1_sort_ind[G1_j_connec]]==genome2.connection_gene_innovation[G2_sort_ind[G2_j_connec]]):

            #print('Same genes')
            # randomly choose if the gene will be taken from genome1 or genome2 and add aconnection
            if (np.random.rand()>0.5):

                add_connection_new_robot(genome1,G1_sort_ind[G1_j_connec],new_robot)
                
            else:

                add_connection_new_robot(genome2,G2_sort_ind[G2_j_connec],new_robot)
            
            # If at least one parent genome has connection disabled, then chance to disable in offspring is 75%
            if ( (genome1.connection_gene_IsEnabled[G1_sort_ind[G1_j_connec]]==False) or (genome2.connection_gene_IsEnabled[G2_sort_ind[G2_j_connec]]==False) ):
                if (np.random.rand()<reproduction_dic['disable_gene_prob']):
                    new_robot.genome.connection_gene_IsEnabled[-1]=False
                                
            # Increment gene index
            G1_j_connec+=1
            G2_j_connec+=1

        # If genome1 gene is smaller and genome 1 is fitter
        elif (genome1.connection_gene_innovation[G1_sort_ind[G1_j_connec]]<genome2.connection_gene_innovation[G2_sort_ind[G2_j_connec]] and robot1_fitter):

            #print('Genome 1 gene smaller and genome 1 fitter')

            # Add conection to new robot
            add_connection_new_robot(genome1,G1_sort_ind[G1_j_connec],new_robot)

            # If genome1 has connection disabled, then chance to disable in offspring is 75%
            if ( genome1.connection_gene_IsEnabled[G1_sort_ind[G1_j_connec]]==False ):
                if (np.random.rand()<reproduction_dic['disable_gene_prob']):
                    new_robot.genome.connection_gene_IsEnabled[-1]=False

            # Increment gene index
            G1_j_connec+=1
            
        # If genome1 gene is smaller and genome 2 is fitter
        elif (genome1.connection_gene_innovation[G1_sort_ind[G1_j_connec]]<genome2.connection_gene_innovation[G2_sort_ind[G2_j_connec]] and robot2_fitter):

            #print('Genome 1 gene smaller and genome 2 fitter')

            # Increment gene index
            G1_j_connec+=1

        # If genome2 gene is smaller and genome 2 is fitter
        elif (genome1.connection_gene_innovation[G1_sort_ind[G1_j_connec]]>genome2.connection_gene_innovation[G2_sort_ind[G2_j_connec]] and robot2_fitter):

            #print('Genome 2 gene smaller and genome 2 fitter')

            # Add conection to new robot
            add_connection_new_robot(genome2,G2_sort_ind[G2_j_connec],new_robot)

            # If genome2 has connection disabled, then chance to disable in offspring is 75%
            if ( genome2.connection_gene_IsEnabled[G2_sort_ind[G2_j_connec]]==False ):
                if (np.random.rand()<reproduction_dic['disable_gene_prob']):
                    new_robot.genome.connection_gene_IsEnabled[-1]=False

            # Increment gene index
            G2_j_connec+=1

        # If genome2 gene is smaller and genome 1 is fitter
        elif (genome1.connection_gene_innovation[G1_sort_ind[G1_j_connec]]>genome2.connection_gene_innovation[G2_sort_ind[G2_j_connec]] and robot1_fitter):

            #print('Genome 2 gene smaller and genome 1 fitter')

            # Increment gene index
            G2_j_connec+=1
        
    # If some genes left, and genome1 is the fittest then add to new robot
    if ( (G1_j_connec<G1_n_connec) and robot1_fitter ):

        #print('Genome 1 excess genes')
        
        while (G1_j_connec<G1_n_connec):

            # Add conection to new robot
            add_connection_new_robot(genome1,G1_sort_ind[G1_j_connec],new_robot)

            # If genome1 has connection disabled, then chance to disable in offspring is 75%
            if ( genome1.connection_gene_IsEnabled[G1_sort_ind[G1_j_connec]]==False ):
                if (np.random.rand()<reproduction_dic['disable_gene_prob']):
                    new_robot.genome.connection_gene_IsEnabled[-1]=False

            # ncrement gene index
            G1_j_connec+=1

    # If some genes left, and genome2 is the fittest then add to new robot
    if ( (G2_j_connec<G2_n_connec) and robot2_fitter ):

        #print('Genome 2 excess genes')

        while (G2_j_connec<G2_n_connec):

            # Add conection to new robot
            add_connection_new_robot(genome2,G2_sort_ind[G2_j_connec],new_robot)

            # If genome1 has connection disabled, then chance to disable in offspring is 75%
            if ( genome2.connection_gene_IsEnabled[G2_sort_ind[G2_j_connec]]==False ):
                if (np.random.rand()<reproduction_dic['disable_gene_prob']):
                    new_robot.genome.connection_gene_IsEnabled[-1]=False

            # ncrement gene index
            G2_j_connec+=1

    return new_robot


# Reproduce
# -> parents are robots to reproduce from
# -> phenotypes is the list of phenotypes for the parents
# -> species are existing species
# -> max_species is the tentative maximum of species
# -> population_size is the number of robots in a generation
# -> maze_count is the total numner of mazes
# -> mazes is the list of mazes
# -> cur_node is the current node number
# -> cur_innovation is the current innovation number
# -> innov_list is the list of innovations
# -> activation is the activation parameter name
# -> reproduction_dic is the dictionnary of probabilities and parameters for reproduction
# -> debug is the flag to output debug info
# returns children robots
def reproduce(parents,phenotypes,species,max_species,population_size,maze_count,mazes,cur_node,cur_innovation,innov_list,activation,reproduction_dic,debug=False):
            
    # Compute number robots in parents
    n_robots = len(parents)

    #print('Count species')

    # Compute species and get number of species
    species.count(parents)
    n_species = len(species.specie_list)

    # sum of adjusted fitnesses
    sum_fitness = np.zeros(n_species)

    # number of offsprings per specie
    offspring_per_specie = np.zeros(n_species,dtype=int)
    
    #print('Compute adjusted fitness and its sum')

    # Compute adjusted fitness and sum of adjusted fitness
    for i_robot in range(n_robots):
        phenotypes.adjust_fitness(parents,i_robot,species)
        sum_fitness[parents[i_robot].genome.specie]+=parents[i_robot].adj_fitness_avg

    #print('Compute allocation per species')

    # Compute the allocation per species (can be a little over the population_size but not below)
    total_fitness = np.sum(sum_fitness)
    count_pop = 0
    for i_specie in range(n_species):
        offspring_per_specie[i_specie]=int(np.round(population_size*sum_fitness[i_specie]/total_fitness))
        if (offspring_per_specie[i_specie] == 0):
            offspring_per_specie[i_specie] = 1
        count_pop += offspring_per_specie[i_specie]

    #print('Check total allocation per species')

    # If total population is below expected, increase it for random species
    dec = count_pop - population_size
    if (dec<0) :
        for i in range(dec):
            i_specie = np.random.randint(n_species)
            while (offspring_per_specie[i_specie] == 0):
                i_specie = np.random.randint(n_species)
            offspring_per_specie[i_specie] += 1
    
    #print('offspring per specie: ',offspring_per_specie)
    #print('trim parents : ',len(parents))

    # Trim parents population to keep bests only and add to children
    parents = trim_generation(parents,reproduction_dic['keep_prct'],species)    
    n_parents = len(parents)
    children = copy.deepcopy(parents)

    #print('trimmed parents : ',len(parents))

    #print('Make dictionnary of specie:robots')

    # Make dicitonnary of robots in each specie
    specie_robot_dic=dict()
    for i_robot in range(n_parents):
        if (parents[i_robot].genome.specie not in specie_robot_dic):
            specie_robot_dic[parents[i_robot].genome.specie]=[]
        specie_robot_dic[parents[i_robot].genome.specie].append(i_robot)

    # Compute number of active species
    n_active_species = len(specie_robot_dic)
    
    # population counter (init with the number of children already added)
    n_pop = len(children)

    # recompute number species, now in children
    species.count(children)

    #print('Start reproduction')

    # Reproduce until we reach the max number of robots in a generation
    while (n_pop < population_size):

        # Choose robot 1
        i_robot1 = np.random.randint(n_parents)
        i_specie1 = parents[i_robot1].genome.specie
        
        #print('Robot1: ',i_robot1)
        #print('Specie1: ',i_specie1)
        #print('Specie:robot dic: ',specie_robot_dic)
        
        # Choose if it is going to crossover
        if (np.random.rand()<reproduction_dic['crossover_prob']):
            DoCrossover = True
        else :
            DoCrossover = False

        #print('crossover?: ',DoCrossover)

        # If crossover choose robot 2
        if (DoCrossover):

            # choose from the same specie ?
            if (np.random.rand()<reproduction_dic['different_specie_crossover_prob']):
                i_robot2 = np.random.randint(n_parents)
                while ( (n_active_species > 1) and  (i_robot2 in specie_robot_dic[i_specie1]) ):
                    i_robot2 = np.random.randint(n_parents)
                #print('Different specie Robot 2: ',i_robot2)
            else :
                i_robot2 = specie_robot_dic[i_specie1][np.random.randint(len(specie_robot_dic[i_specie1]))]
                #print('Same specie Robot 2: ',i_robot2)
                    
            #print('Crossover...')

            # Crossover
            child = crossover(parents[i_robot1],parents[i_robot2],maze_count,mazes,cur_node,activation,reproduction_dic)

            if debug:
                print('### Parent 1 fitness: ',parents[i_robot1].fitness_avg)
                parents[i_robot1].genome.print_genome()
                print('### Parent 2 fitness: ',parents[i_robot2].fitness_avg)
                parents[i_robot2].genome.print_genome()
                print('### Child: ')
                child.genome.print_genome()

        # If no crossover pass the child directly
        else :

            child = crossover(parents[i_robot1],parents[i_robot1],maze_count,mazes,cur_node,activation,reproduction_dic)
            
        #print('Mutations...')
        # Mutate weights
        if (np.random.rand()<reproduction_dic['weight_mut_prob']):

            #print('Mutate weight')
            # Mutate as many times as we have weight (random weight chosen, contrary to NEAT paper)
            for i_weight in range(len(child.genome.connection_gene_innovation)):

                # Uniform perturbation
                if (np.random.rand()<reproduction_dic['weight_mut_is_uniform_prob']):
                    child.mutate_weight(reproduction_dic['max_weight_perturb'])
                # random weight assignment
                else :
                    child.replace_weight()

        # Mutate add node
        if (np.random.rand()<reproduction_dic['add_node_prob']):
            #print('Mutate add node')
            child.mutate_add_node(innov_list,cur_innovation,cur_innovation,magnitude=1.0)
        
        # Mutate add connection
        if (np.random.rand()<reproduction_dic['add_connection_prob']):
            #print('Mutate add connection')
            child.mutate_add_connection(innov_list,cur_innovation,magnitude=1.0)

        # Mutate change activation
        if (np.random.rand()<reproduction_dic['change_activation_prob']):
            #print('Mutate activation')
            child.mutate_activation()
            
        if debug:
            print('### Mutated Child: ')
            child.genome.print_genome()

        #print('Compute specie')

        # Compute specie
        species.compatibility(child.genome)
        child_specie = child.genome.specie
        ind_child_specie = species.specie_list.index(child_specie)
        
        #print('Child specie: ',child_specie)
        #print('Child specie index: ',ind_child_specie)
        #print('Space left: ',offspring_per_specie)
        
        # Add to specie and children if new specie
        if (ind_child_specie>(len(offspring_per_specie)-1)):
            #print('New Specie!')
            children.append(child)
            #child.genome.print_genome()
            n_pop+=1
        # Add to specie and children if there is space
        else:
            if (offspring_per_specie[ind_child_specie]>0):
                #print('Add to specie')
                offspring_per_specie[ind_child_specie]-=1
                #print('Space left in specie : ',offspring_per_specie[ind_child_specie])
                children.append(child)
                #child.genome.print_genome()
                n_pop+=1

    #print('Recompute species from full children generation')
    
    # Recompute species using children only
    species.create_from_generation(children)
    species.count(children)

    # Change specie delta if n_species is far from max_specie

    # If number of species is larger than expected, increase the delta
    if ( len(species.specie_list) > max_species ):
        species.change_delta(0.3)
        species.create_from_generation(children)
        species.count(children)

    # If number of species is smaller than expected, decrease the delta
    if ( (len(species.specie_list) < max_species) and (species.delta>1) ):
        species.change_delta(-0.3)
        species.create_from_generation(children)
        species.count(children)
    
                
    return children
