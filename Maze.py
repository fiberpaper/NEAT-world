## v0 - 11/06/2017
## maze

import numpy as np
import math
from MazeGenome import maze_genome

## Define maze class
class maze:


    # Create maze
    # -> size : the length of one side of the square defining the maze area
    # -> maze_type : can be predifined (simple,complex) or random
    # -> maze_wall_nb (optional) : number of wall in the maze, random if not defined (up to 20)
    # -> position (optional) : start and goal position for the robot
    #                          'standard' : upper left corner for robot and bottom right corner for goal
    #                          else : random
    def __init__(self,size,maze_type,maze_wall_nb=-1,position='standard'):

        # Main variables
        # -> size : maze is a square size*size
        # -> playground : matrix of boolean (size*size) with True for walls and False for free space
        # -> passage_size : size of passage in walls
        # -> genome : genome of the maze
        # -> start_pos : position for the robot to start
        # -> goal_pos : goal position for the robot to reach
        # -> subspace : list of subspaces (divided by walls)
        
        # Maze is a square of size x size
        # Defined as a matrix (playground) 
        self.size=size
        self.playground=np.full((self.size,self.size),False,dtype=bool)

        # Put main walls
        self.playground[0,:]=True
        self.playground[-1,:]=True
        self.playground[:,0]=True
        self.playground[:,-1]=True

        # Define size holes in walls
        self.passage_size=20
            
        # Create genome
        if (maze_type=='simple'):
            self.genome=maze_genome(0.3,0.3)
            self.genome.add(0.4,0.3)
        elif (maze_type=='complex'):
            self.genome=maze_genome(0.5,0.3)
            self.genome.add(0.3,0.7)
            self.genome.add(0.7,0.2)
            self.genome.add(0.5,0.5)
            self.genome.add(0.2,0.6)
            self.genome.add(0.6,0.2)
        elif (maze_type=='random'):
            self.genome=maze_genome()
            if (maze_wall_nb==-1):
                Nwall_add=np.random.randint(20)
            else:
                Nwall_add=maze_wall_nb
            for iwall in range(Nwall_add):
               self.genome.add()
        else:
            raise ValueError('maze_type unknown : {}'.format(maze_type))

        # Add walls to playground
        # subspace keeps track of each subspace and add walls in the biggest
        # [minx,maxx,miny,maxy]
        self.subspace=list()
        self.subspace.append([0,size-1,0,size-1])

        for iwall in range(self.genome.Nwalls):

            self.add_wall(iwall)
            
        # Create starting and goal position (check after adding walls that we have no problem)
        if (position=='standard'):

            x_start=10
            y_start=10
            while (self.playground[x_start,y_start]):
                x_start+=1
                y_start+=1
            self.start_pos=(float(x_start),float(y_start))

            x_goal=size-10
            y_goal=size-10
            while (self.playground[x_goal,y_goal]):
                x_goal-=1
                y_goal-=1
            self.goal_pos=(float(x_goal),float(y_goal))
            
        elif (position=='corner'):

            corner=np.random.randint(4)

            if (corner==0):
                x_start=10
                y_start=10
                x_goal=size-10
                y_goal=size-10
            elif (corner==1):
                x_start=10
                y_start=size-10
                x_goal=size-10
                y_goal=10
            elif (corner==2):
                x_start=size-10
                y_start=size-10
                x_goal=10
                y_goal=10
            elif (corner==3):
                x_start=size-10
                y_start=10
                x_goal=10
                y_goal=size-10

            while (self.playground[x_start,y_start]):
                x_start+=1
                y_start+=1
            self.start_pos=(float(x_start),float(y_start))

            while (self.playground[x_goal,y_goal]):
                x_goal-=1
                y_goal-=1
            self.goal_pos=(float(x_goal),float(y_goal))
               
        else:
            
            x_start=np.random.randint(size)
            y_start=np.random.randint(size)
            while (self.playground[x_start,y_start]):
                if (x_start<(size/2)):
                    x_start+=1
                else:
                    x_start-=1
                if (y_start<(size/2)):
                    y_start+=1
                else:
                    y_start-=1
            self.start_pos=(float(x_start),float(y_start))

            x_goal=np.random.randint(size)
            y_goal=np.random.randint(size)

            # Compute distance to goal
            dist=np.sqrt((x_goal-x_start)**2+(y_goal-y_start)**2)

            # While the goal and start position are close, change goal position
            while (dist<(self.size/2)):

                x_goal=np.random.randint(size)
                y_goal=np.random.randint(size)
                dist=np.sqrt((x_goal-x_start)**2+(y_goal-y_start)**2)

            while (self.playground[x_goal,y_goal]):
                if (x_goal<(size/2)):
                    x_goal+=1
                else:
                    x_goal-=1
                if (y_goal<(size/2)):
                    y_goal+=1
                else:
                    y_goal-=1
            self.goal_pos=(float(x_goal),float(y_goal))

        # Goal insight map
        print('Compute insight map')
        self.goal_insight_map=self.goal_insight_map_compute()
        
        # Distance goal map
        print('Compute goal distance map')
        self.dist_goal_map=self.dist_goal_map_compute()

        # Distances to wall
        print('Compute wall distance maps')
        self.dist_wall_d270_map=self.dist_wall_map_compute(270)
        self.dist_wall_d315_map=self.dist_wall_map_compute(315)
        self.dist_wall_d0_map=self.dist_wall_map_compute(0)
        self.dist_wall_d45_map=self.dist_wall_map_compute(45)
        self.dist_wall_d90_map=self.dist_wall_map_compute(90)
        self.dist_wall_d180_map=self.dist_wall_map_compute(180)
        


    # Add wall from genome position [iwall]
    def add_wall(self,iwall):

        # passage size
        hs=self.passage_size
        hhs=int(self.passage_size/2)

        # add wall along the smallest direction of the largest subspace
        minx=self.subspace[0][0]
        maxx=self.subspace[0][1]
        miny=self.subspace[0][2]
        maxy=self.subspace[0][3]
                    
        length=maxx-minx
        width=maxy-miny

        # Case 1 : length<=width
        if(length<=width):
            # Find position of the wall
            pos_wall=int(miny+(maxy-miny)*self.genome.gene_weight[iwall])
            # Move the wall is on top of a passage
            if (not self.playground[minx,pos_wall] or not self.playground[maxx,pos_wall]):
                if (self.genome.gene_weight[iwall]>0.5):
                    while (not self.playground[minx,pos_wall] or not self.playground[maxx,pos_wall]):
                        pos_wall-=1
                else:
                    while (not self.playground[minx,pos_wall] or not self.playground[maxx,pos_wall]):
                        pos_wall+=1
                    
            # Find position passage
            pos_passage=int(minx+(maxx-minx)*self.genome.gene_passage[iwall])

            # Create full wall
            self.playground[minx:maxx,pos_wall]=True

            # Find where to place the passage
            if ( (pos_passage-hhs)<=minx ):
                pos_passage_min=minx+1
            else:
                pos_passage_min=pos_passage-hhs
            if ( (pos_passage_min+hs)>=maxx ):
                pos_passage_max=maxx-1
            else:
                pos_passage_max=pos_passage_min+hs
                
            # Create the passage
            self.playground[pos_passage_min:pos_passage_max,pos_wall]=False

            # Update the subspaces
            area_newsub1=((maxx-minx)*(pos_wall-miny))
            area_newsub2=((maxx-minx)*(maxy-pos_wall))
            
            isub=0
            sub1_notplaced=True
            sub2_notplaced=True
            
            while ( (isub<len(self.subspace)) and (sub1_notplaced or sub2_notplaced) ):
                iminx=self.subspace[isub][0]
                imaxx=self.subspace[isub][1]
                iminy=self.subspace[isub][2]
                imaxy=self.subspace[isub][3]

                area_subspace=(imaxx-iminx)*(imaxy-iminy)

                if ( (area_subspace<area_newsub1) and sub1_notplaced):
                    self.subspace.insert(isub,[minx,maxx,miny,pos_wall])
                    sub1_notplaced=False
                    if ( (area_newsub1<area_newsub2) and sub2_notplaced):
                        self.subspace.insert(isub,[minx,maxx,pos_wall,maxy])
                        sub2_notplaced=False

                if ( (area_subspace<area_newsub2) and sub2_notplaced):
                    self.subspace.insert(isub,[minx,maxx,pos_wall,maxy])
                    sub2_notplaced=False

                isub+=1

            if (sub1_notplaced and sub2_notplaced):
                if (area_newsub1>area_newsub2):
                    self.subspace.append([minx,maxx,miny,pos_wall])
                    self.subspace.append([minx,maxx,pos_wall,maxy])
                else:
                    self.subspace.append([minx,maxx,pos_wall,maxy])
                    self.subspace.append([minx,maxx,miny,pos_wall])
            elif (sub1_notplaced):
                self.subspace.append([minx,maxx,miny,pos_wall])
            elif (sub2_notplaced):
                self.subspace.append([minx,maxx,pos_wall,maxy])
                                

            # Remove the subspace splitted in 2
            del self.subspace[0]
            
        # Case 2 : length>width
        else:
            # Find position of the wall
            pos_wall=int(minx+(maxx-minx)*self.genome.gene_weight[iwall])
            # Move the wall is on top of a passage
            if (not self.playground[pos_wall,miny] or not self.playground[pos_wall,maxy]):
                if (self.genome.gene_weight[iwall]>0.5):
                    while (not self.playground[pos_wall,miny] or not self.playground[pos_wall,maxy]):
                        pos_wall-=1
                else:
                    while (not self.playground[pos_wall,miny] or not self.playground[pos_wall,maxy]):
                        pos_wall+=1
                    
            # Find position passage
            pos_passage=int(miny+(maxy-miny)*self.genome.gene_passage[iwall])

            # Create full wall
            self.playground[pos_wall,miny:maxy]=True

            # Find where to place the passage
            if ( (pos_passage-hhs)<=miny ):
                pos_passage_min=miny+1
            else:
                pos_passage_min=pos_passage-hhs
            if ( (pos_passage_min+hs)>=maxy ):
                pos_passage_max=maxy-1
            else:
                pos_passage_max=pos_passage_min+hs
                
            # Create the passage
            self.playground[pos_wall,pos_passage_min:pos_passage_max]=False

            # Update the subspaces
            area_newsub1=((pos_wall-minx)*(maxy-miny))
            area_newsub2=((maxx-pos_wall)*(maxy-miny))
            
            isub=0
            sub1_notplaced=True
            sub2_notplaced=True
            
            while ( (isub<len(self.subspace)) and (sub1_notplaced or sub2_notplaced) ):
                iminx=self.subspace[isub][0]
                imaxx=self.subspace[isub][1]
                iminy=self.subspace[isub][2]
                imaxy=self.subspace[isub][3]

                area_subspace=(imaxx-iminx)*(imaxy-iminy)

                if ( (area_subspace<area_newsub1) and sub1_notplaced):
                    self.subspace.insert(isub,[minx,pos_wall,miny,maxy])
                    sub1_notplaced=False
                    if ( (area_newsub1<area_newsub2) and sub2_notplaced):
                        self.subspace.insert(isub,[pos_wall,maxx,miny,maxy])
                        sub2_notplaced=False

                if ( (area_subspace<area_newsub2) and sub2_notplaced):
                    self.subspace.insert(isub,[pos_wall,maxx,miny,maxy])
                    sub2_notplaced=False

                isub+=1                            

            if (sub1_notplaced and sub2_notplaced):
                if (area_newsub1>area_newsub2):
                    self.subspace.append([minx,pos_wall,miny,maxy])
                    self.subspace.append([pos_wall,maxx,miny,maxy])
                else:
                    self.subspace.append([pos_wall,maxx,miny,maxy])
                    self.subspace.append([minx,pos_wall,miny,maxy])
            elif (sub1_notplaced):
                self.subspace.append([minx,pos_wall,miny,maxy])
            elif (sub2_notplaced):
                self.subspace.append([pos_wall,maxx,miny,maxy])

            # Remove the subspace splitted in 2
            del self.subspace[0]


    # Find distance to wall from current position and in certain direction (0degree at North)
    # -> pos : tuple of the position
    # -> direction : direction in degrees from North
    # returns distance (float)
    def dist_wall_compute(self,pos,direction):

        # direction needs to be between 0 and 360 degree and needs to be an integer
        direction=int(direction)
        if ( (direction<0) or (direction>360) ):
            raise ValueError('direction should be between 0 and 360 but is : {}'.format(direction))

        pos_int=np.round(np.asarray(pos)).astype(int)
        ray=np.round(np.asarray(pos)).astype(int)
        istep=0
        iswall=self.playground[ray[0],ray[1]]

        #debug
        #print(self.playground[ray[0],ray[1]])
        #test=self.playground.copy()
        
        while (not iswall):
            istep+=1
            # move one step in main direction and compute other direction
            if (direction>45 and direction<=135):
                ray[1]=pos_int[1]+istep
                ray[0]=pos_int[0]-int(np.round(istep*math.tan((90-direction)*math.pi/180)))
            elif (direction>225 and direction<=315) :
                ray[1]=pos_int[1]-istep
                ray[0]=pos_int[0]+int(np.round(istep*math.tan((270-direction)*math.pi/180)))
            elif (direction>135 and direction<=225) :
                ray[0]=pos_int[0]+istep
                ray[1]=pos_int[1]+int(np.round(istep*math.tan((180-direction)*math.pi/180)))
            else :
                ray[0]=pos_int[0]-istep
                ray[1]=pos_int[1]-int(np.round(istep*math.tan((-direction)*math.pi/180)))

            #debug
            #print(ray)
            #print(self.playground[ray[0],ray[1]])
            #test[ray[0],ray[1]]=True
            
            iswall=self.playground[ray[0],ray[1]]

        #debug
        #return math.sqrt((ray[0]-pos[0])**2+(ray[1]-pos[1])**2),test

        return math.sqrt((float(ray[0])-pos[0])**2+(float(ray[1])-pos[1])**2)

    # Compute map distance to wall in certain direction (0degree at North)
    # -> direction : direction in degrees from North
    def dist_wall_map_compute(self,direction):

        dist_wall_map=np.full((self.size,self.size),0.0,dtype='float32')

        for x in range(self.size):
            for y in range(self.size):
                dist_wall_map[x,y]=self.dist_wall_compute((x,y),direction)

        return dist_wall_map

    # Find distance to wall from current position and in certain direction (0degree at North)
    # -> pos : tuple of the position
    # -> direction : direction in degrees from North
    # returns distance (float)
    def dist_wall(self,pos,direction):

        pos_int=np.round(np.asarray(pos)).astype(int)

        if (direction == 270):
            return self.dist_wall_d270_map[pos_int[0],pos_int[1]]

        if (direction == 315):
            return self.dist_wall_d315_map[pos_int[0],pos_int[1]]

        if (direction == 0):
            return self.dist_wall_d0_map[pos_int[0],pos_int[1]]

        if (direction == 45):
            return self.dist_wall_d45_map[pos_int[0],pos_int[1]]

        if (direction == 90):
            return self.dist_wall_d90_map[pos_int[0],pos_int[1]]

        if (direction == 180):
            return self.dist_wall_d180_map[pos_int[0],pos_int[1]]

        raise ValueError('direction {} degree not found'.format(direction))
    

    # Find is the goal point is visible (no wall) from current position
    # -> pos : tuple of the position
    # returns 4 pie slice detectors value (1 if visible, 0 otherwise) as a tuple [x,x,x,x] for [North,East,South,West]
    def goal_insight_compute(self,pos):

        if (pos==self.goal_pos):
            return (1,1,1,1)

        # determine direction
        if (self.goal_pos[0]-pos[0]<0):
            direction=np.mod(-math.atan( (self.goal_pos[1]-pos[1]) / (self.goal_pos[0]-pos[0]) )*180/math.pi,360)
        elif (self.goal_pos[0]-pos[0]>0):
            direction=np.mod(-math.atan( (self.goal_pos[1]-pos[1]) / (self.goal_pos[0]-pos[0]) )*180/math.pi+180,360)
        elif ( (self.goal_pos[0]-pos[0]==0) and (self.goal_pos[1]-pos[1])>0 ):
            direction=90
        else:
            direction=270

        #print(direction)

        # find if there is a wall blocking the view of the goal
        pos_int=np.round(np.asarray(pos)).astype(int)
        ray=np.round(np.asarray(pos)).astype(int)
        istep=0
        wall_fnd=self.playground[ray[0],ray[1]]
        goal_fnd=False
        
        while ( (not goal_fnd) and (not wall_fnd) ):
            istep+=1
            # move one step in main direction and compute other direction
            if (direction>45 and direction<=135):
                ray[1]=pos_int[1]+istep
                ray[0]=pos_int[0]-int(np.round(istep*math.tan((90-direction)*math.pi/180)))
            elif (direction>225 and direction<=315) :
                ray[1]=pos_int[1]-istep
                ray[0]=pos_int[0]+int(np.round(istep*math.tan((270-direction)*math.pi/180)))
            elif (direction>135 and direction<=225) :
                ray[0]=pos_int[0]+istep
                ray[1]=pos_int[1]+int(np.round(istep*math.tan((180-direction)*math.pi/180)))
            else :
                ray[0]=pos_int[0]-istep
                ray[1]=pos_int[1]-int(np.round(istep*math.tan((-direction)*math.pi/180)))
            
            wall_fnd=self.playground[ray[0],ray[1]]

            if (ray[0]==self.goal_pos[0]) and (ray[1]==self.goal_pos[1]):
                goal_fnd=True

        if (goal_fnd):
            if (direction>45 and direction<=135):
                return (0,1,0,0)
            elif (direction>135 and direction<=225) :
                return (0,0,1,0)
            elif (direction>225 and direction<=315) :
                return (0,0,0,1)
            else :
                return (1,0,0,0)
        else:
            return (0,0,0,0)

    # Compute maps of goal insight for the maze
    def goal_insight_map_compute(self):

        goal_insight_map=np.full((self.size,self.size,4),0,dtype='int8')

        for x in range(self.size):
            for y in range(self.size):
                res=self.goal_insight_compute((x,y))
                for i in range(4):
                    goal_insight_map[x,y,i]=res[i]

        return goal_insight_map

    # Find is the goal point is visible (no wall) from current position
    # -> pos : tuple of the position
    # returns 4 pie slice detectors value (1 if visible, 0 otherwise) as a tuple [x,x,x,x] for [North,East,South,West]
    def goal_insight(self,pos):

        pos_int=np.round(np.asarray(pos)).astype(int)

        res=self.goal_insight_map[pos_int[0],pos_int[1],:]

        return (res[0],res[1],res[2],res[3])

    # Find the distance to the goal from position pos and if there is a wall in between the position and the goal
    # -> pos : tuple of the position
    # returns float,bool : distance to goal, presence of a wall between pos and goal
    def dist_goal_compute(self,pos):

        # Compute distance to goal
        distance=math.sqrt((float(self.goal_pos[0])-pos[0])**2+(float(self.goal_pos[1])-pos[1])**2)

        return distance

    # Find the distance to the goal from position pos and if there is a wall in between the position and the goal
    # -> pos : tuple of the position
    # returns float,bool : distance to goal, presence of a wall between pos and goal
    def dist_goal_and_IsWall_compute(self,pos):

        if (pos==self.goal_pos):
            return 0.0, False

        # determine direction
        if (self.goal_pos[0]-pos[0]<0):
            direction=np.mod(-math.atan( (self.goal_pos[1]-pos[1]) / (self.goal_pos[0]-pos[0]) )*180/math.pi,360)
        elif (self.goal_pos[0]-pos[0]>0):
            direction=np.mod(-math.atan( (self.goal_pos[1]-pos[1]) / (self.goal_pos[0]-pos[0]) )*180/math.pi+180,360)
        elif ( (self.goal_pos[0]-pos[0]==0) and (self.goal_pos[1]-pos[1])>0 ):
            direction=90
        else:
            direction=270

        # find if there is a wall blocking the view of the goal
        pos_int=np.round(np.asarray(pos)).astype(int)
        ray=np.round(np.asarray(pos)).astype(int)
        istep=0
        wall_fnd=self.playground[ray[0],ray[1]]
        goal_fnd=False
        
        while ( (not goal_fnd) and (not wall_fnd) ):
            istep+=1
            # move one step in main direction and compute other direction
            if (direction>45 and direction<=135):
                ray[1]=pos_int[1]+istep
                ray[0]=pos_int[0]-int(np.round(istep*math.tan((90-direction)*math.pi/180)))
            elif (direction>225 and direction<=315) :
                ray[1]=pos_int[1]-istep
                ray[0]=pos_int[0]+int(np.round(istep*math.tan((270-direction)*math.pi/180)))
            elif (direction>135 and direction<=225) :
                ray[0]=pos_int[0]+istep
                ray[1]=pos_int[1]+int(np.round(istep*math.tan((180-direction)*math.pi/180)))
            else :
                ray[0]=pos_int[0]-istep
                ray[1]=pos_int[1]-int(np.round(istep*math.tan((-direction)*math.pi/180)))
            
            wall_fnd=self.playground[ray[0],ray[1]]

            if (ray[0]==self.goal_pos[0]) and (ray[1]==self.goal_pos[1]):
                goal_fnd=True

        # Compute distance to goal
        distance=math.sqrt((float(self.goal_pos[0])-pos[0])**2+(float(self.goal_pos[1])-pos[1])**2)

        return distance,goal_fnd

    # Compute maps of goal distance and goal reached for the maze
    def dist_goal_map_compute(self):

        dist_goal_map=np.full((self.size,self.size),0,dtype='float32')

        for x in range(self.size):
            for y in range(self.size):
                dist_goal_map[x,y]=self.dist_goal_compute((x,y))

        return dist_goal_map

    # Find the distance to the goal from position pos and if there is a wall in between the position and the goal
    # -> pos : tuple of the position
    # returns float,bool : distance to goal, presence of a wall between pos and goal
    def dist_goal(self,pos):

        pos_int=np.round(np.asarray(pos)).astype(int)

        goal_insight_pos=self.goal_insight_map[pos_int[0],pos_int[1],:]
        
        goal_fnd = goal_insight_pos[0] or goal_insight_pos[1] or goal_insight_pos[2] or goal_insight_pos[3]

        return (self.dist_goal_map[pos_int[0],pos_int[1]],goal_fnd)



    # Return next position, given a current position, step size and direction (0 degree is north)
    # -> pos : tuple of the current position
    # -> step : step size to move in the current position (can be negative)
    # -> direction : direction in degrees from North
    # returns next position tuple (x,y)
    def next_pos(self,pos,step,direction):

        # return same position is step is 0
        if (step==0.0):
            return pos

        # direction needs to be between 0 and 360 degree and needs to be an integer
        direction=int(direction)
        if ( (direction<0) or (direction>360) ):
            raise ValueError('direction should be between 0 and 360 but is : {}'.format(direction))

        inc=step/np.abs(step)
        
        pos_int=np.round(np.asarray(pos)).astype(int)
        ray=np.round(np.asarray(pos)).astype(int)
        next_pos=ray.copy()
        
        istep=0
        length=0
        iswall=self.playground[ray[0],ray[1]]

        no_move=True

        while ( (not iswall) and (length<=abs(step)) ):
            no_move=False
            # save position
            next_pos=ray.copy()
            # increment istep
            istep+=inc
            # move one step in main direction and compute other direction
            if (direction>45 and direction<=135):
                ray[1]=pos_int[1]+istep
                ray[0]=pos_int[0]-int(np.round(istep*math.tan((90-direction)*math.pi/180)))
            elif (direction>225 and direction<=315) :
                ray[1]=pos_int[1]-istep
                ray[0]=pos_int[0]+int(np.round(istep*math.tan((270-direction)*math.pi/180)))
            elif (direction>135 and direction<=225) :
                ray[0]=pos_int[0]+istep
                ray[1]=pos_int[1]+int(np.round(istep*math.tan((180-direction)*math.pi/180)))
            else :
                ray[0]=pos_int[0]-istep
                ray[1]=pos_int[1]-int(np.round(istep*math.tan((-direction)*math.pi/180)))
            
            # Is there a wall here ?
            iswall=self.playground[ray[0],ray[1]]

            # What is the real distance travelled
            length=math.sqrt((ray[0]-pos[0])**2+(ray[1]-pos[1])**2)

        # convert next_pos back to float
        next_pos=next_pos.astype(float)

        # If there was no wall compute the exact location
        if (not iswall):
            x=pos[0]-math.cos(direction*math.pi/180)*step
            y=pos[1]+math.sin(direction*math.pi/180)*step
            if (not self.playground[np.round(x).astype(int),np.round(y).astype(int)]):
                next_pos[0]=x
                next_pos[1]=y
        else :
            # If there was a wall at the first position, revert to originl pos
            if (no_move):
                next_pos[0]=pos[0]
                next_pos[1]=pos[1]

        return (next_pos[0],next_pos[1])
