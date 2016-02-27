from operator import itemgetter
import numpy as np

DROP1 = 1
DROP2 = 2
DROP3 = 3

X_POS = 0
Y_POS = X_POS + 1
X_VEL = Y_POS + 1
Y_VEL = X_VEL + 1
IS_VIS = Y_VEL + 1
OBJ_STATE = IS_VIS + 1

class MovingObject:
    """This class represents a moving entity in the game.
    It maintains the learning algorithm's current representation of
    this object, as well as information about the object's present
    state. """
    # z coordinates won't be needed for our examples, probably.
    # Position
    xPos = 0
    yPos = 0
    # Velocity
    xVel = 0
    yVel = 0
    # Tracks visibility
    isVisible = False

    #  Tracks existence state (eg. for ghost, edible or not)
    #  TODO this should be determined by some metric of the "difference" of
    # the object from one frame to the next
    objectState = 0
    # How many distince states observed so far for this object
    numGameStates = 1

    # Reference to the action at the most recent timestep
    playerAction = 0

    # Record of previous state changes with actions
    dictSPP = {}

    # Instantiate the object with some provided values
    def __init__(self, xStart, yStart, action):
        self.xPos = xStart
        self.yPos = yStart
        self.playerAction = action
        self.isVisible = True

    # Accessors (maybe not necessary?)
    def getPosition(self):
        return (self.xPos, self.yPos)

    def getVelocity(self):
        return (self.xVel, self.yVel)        

        
    # Returns a hashable representation of the object's present state
    def getFullState(self):
        return (self.xPos, self.yPos,
                self.xVel, self.yVel,
                self.isVisible, self.objectState)

    # Updates this object with its new state and the player action
    # that gave rise to it.
    # Presently this only considers the object's own state, not the state of
    # all other entities.  It should though...
    def updateState(self, newAction, newXPos, newYPos, newIsVis, newObjectState=0,
                    newXVel=0, newYVel=0):
        prevStateWithAct = self.getFullState() + (self.playerAction,)
        self.xPos = newXPos
        self.yPos = newYPos
        self.xVel = newXVel
        self.yVel = newYVel
        self.isVisible = newIsVis
        self.objectState = newObjectState
        self.dictSPP[prevStateWithAct] = self.getFullState()
        self.playerAction = newAction


    # Predict what this object's next state will be from past actions
    def predictNextState(self):
        currentState = getFullState + (newAction,) # comma to make it a tuple
        return predictFromState(currentState, self.dictSPP)

    def compress(self, k=5, alg=DROP3):
        """
        As per page 21 of "Reduction Techniques for Instance-Based Learning
        Algorithms", "DROP3 seemed to have the best mix of storage reduction
        and generalization accuracy", so we have chosen it as the default
        algorithm here.  You can specify DROP1 or DROP2 in the 'alg'
        argument if you want to use one of those algorithms instead.
        Compresses the internal instance-based knowledge representation. 
        """
        # Starting with DROP1
        neighbours = {}
        associates = {}
        for stateP in self.dictSPP:
            # build nearest-neighbours of P 
            neighbours[stateP] = updateNNearestNeighbours(
                stateP, k+1, [], associates, self.dictSPP)

        # Initialization complete.

        compressedSPP = self.dictSPP.copy()
        # If an instance's class differs from the result of the vote of
        # its k nearest neighbours, remove the instance
        if alg == DROP3:
            for stateP in compressedSPP.keys():
                pResult = compressedSPP[stateP]
                if len(neighbours[stateP]) == 0:
                    print "WARNING:", stateP, "has no neighbours!" 
                pPrediction = predictFromSubset(compressedSPP, neighbours[stateP])
                # TODO if results are very sparse, this could remove all nodes!
                if not pResult == pPrediction:
                    # We remove noisy entries from their neighbour's associate lists regardless of algorithm 
                    deleteState(stateP, compressedSPP, associates, neighbours, k)
            
        sppKeys = compressedSPP.keys()
        # Sort keys for removal in reverse order of distance to nearest enemy
        if alg > DROP1:
            enemy = {s: self._getNearestEnemy(s) for s in compressedSPP}
            sppKeys.sort(key=lambda x: _getStateDistance(x, enemy[x]),
                         reverse=True)

        # Compress the representation
        for stateP in sppKeys:
            print stateP
            SPPwoP = compressedSPP.copy()
            del SPPwoP[SPP]
            withP = 0
            withoutP = 0
            for asct in associates[stateP]:
                # if asct is predicted correctly with stateP
                if predictAWithoutA(asct, compressedSPP) == compressedSPP[asct]:
                    withP += 1
                if predictAWithoutA(asct, SPPwoP) == compressedSPP[asct]:
                    withoutP += 1
            if withoutP >= withP:
                deleteState(stateP, compressedSPP, associates, neighbours, k, alg)
                # # Remove P from the representation
                # del compressedSPP[stateP]
                # # Remove P from neighbours
                # for asct in associates[stateP]:
                #     neighbours[asct].remove(stateP)
                #     neighbours[asct] = self.updateNNearestNeighbours(
                #         asct, k+1, neighbours[asct])
                # del associates[stateP]
                # # remove P from associates
                # if alg == DROP1:
                #     for neighbour in neighbours[stateP]:
                #         if stateP in associates[neighbour]:
                #             associates[neighbour].remove(stateP)
                #     del neighbours[stateP]

        # Replace the old representation with the new compressed one
        self.dictSPP = compressedSPP
                    

    def _getFullStatePlusAction(self):
        return self.getFullState() + (playerAction,)

    def _getNearestEnemy(self, state):
        nearestEnemy = None
        result = self.dictSPP[state]
        for stateN, resultN in self.dictSPP.iteritems():
            if not result == resultN:
                if nearestEnemy == None:
                    nearestEnemy = stateN
                else:
                    dist = _getStateDistance(state, stateN)
                    if dist < _getStateDistance(state, nearestEnemy):
                        nearestEnemy = stateN
        return nearestEnemy
        # return nearest enemy

def deleteState(stateP, SPP, associates, neighbours, k, alg=DROP1):
    # Remove P from the representation
    del SPP[stateP]
    # Remove P from neighbours
    for asct in associates[stateP]:
        neighbours[asct].remove(stateP)
        neighbours[asct] = updateNNearestNeighbours(
            asct, k+1, neighbours[asct], associates, SPP)
        # if len(neighbours[asct]) == 0:
        #     del neighbours[asct]
    del associates[stateP]
    # remove P from associates
    if alg == DROP1:
        for neighbour in neighbours[stateP]:
            if stateP in associates[neighbour]:
                associates[neighbour].remove(stateP)
        del neighbours[stateP]

        
def updateNNearestNeighbours(stateP, N, pNeighbours, associates, SPP):
    for stateN in SPP:
        stateAdded = False
        if stateP is stateN:
            continue
        if (len(pNeighbours) < N) and (stateN not in pNeighbours):
            pNeighbours.append(stateN)
        else:
            dists = [_getStateDistance(stateP, s)
                     for s in pNeighbours]
            maxDist = max(enumerate(dists), key=itemgetter(1))
            nDist = _getStateDistance(stateP, stateN)
            if nDist < maxDist[1]:
                pNeighbours[maxDist[0]] = stateN
    updateAssociates(stateP, pNeighbours, associates)
    return pNeighbours
            
    
def updateAssociates(state, pNeighbours, associates):
    for n in pNeighbours:
        if n in associates:
            if state not in associates[n]:
                associates[n].append(state)
        else:
            associates[n] = [state]
    return associates


def predictFromSubset(SPP, subset):
    predCounts = {}
    if len(subset) == 0:
        return None
    for state in subset:
        result = SPP[state]
        if result not in predCounts:
            predCounts[result] = 0
        predCounts[result] += 1
    return max(enumerate(predCounts), key=itemgetter(1))[0]
    
    
def predictAWithoutA(stateA, dictSPP):
    woSPP = dictSPP.copy()
    del woSPP[stateA]
    predictFromState(stateA, woSPP)

    
def predictFromState(state, dictSPP):
    prediction = None;
    cosestDist = float("inf")
    # Return the state that followed from the closest state to this one
    for pastState, result in dictSPP.iteritems():
        dist = self._getStateDistance(pastState, state)
        if dist < clostestDist:
            prediction = result
    return prediction
        
        
def _getStateDistance(state1, state2):
    # TODO implement this in domain-appropriate way
    # consider the paper's technique 
    manhattan = abs(state1[X_POS] - state2[X_POS]) + abs(state1[Y_POS] - state2[Y_POS])
    vis = abs(state1[IS_VIS]*1 - state2[IS_VIS]*1)*np.inf
    return manhattan + vis

    
