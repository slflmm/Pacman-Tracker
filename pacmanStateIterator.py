import scipy.spatial.distance as distance
import numpy as np

NULL_CODE = 0
RCODE = 1
LCODE = 2
UCODE = 3
DCODE = 4

frameW = 683
frameH = 388

RUN_STATE = 'Run'
CHASE_STATE = 'Chase'
WAIT_STATE = 'Wait'
# stateStrs = ['Wait', 'Chase', 'Run', '']
# stateDict = { for stateStr, code in stateStrs}

class PacmanStateIterator:

    def __init__(self, filename):
        self.stateFile = open(filename, 'r')

    '''
    Returns a bundle of each entity's position + state at next time.
    Pacman is first, with only name and position (access by bundle[0][0-1])
    Ghosts are next, with name, position, and state (access by bundle[1-4][0-2])
    '''
    def getNextStateBundle(self):
        bundle = []
        for i in range(5):
            line = self.stateFile.readline()
            if not line:
                return None
            else:
                lineParts = line.split('     ')
                # name = lineParts[0]
                # posStr = lineParts[1]
                # stateStr = lineParts[2]
                # bundle[i] = GameState(name, posStr, stateStr)
                bundle.append(GameState(*lineParts))

        return bundle 

class GameState:
    def __init__(self, name, posStr, stateStr=None):
        self.name = name
        posCompStr = posStr.strip().strip('()').split(', ')
        nums = [float(posNumStr) for posNumStr in posCompStr]
        # ignore the Z position
        self.pos = (nums[0], frameH - nums[1])
        # print self.pos
        self.state = stateStr

    def getPixelPos(self):
        return int(self.pos[0]), int(self.pos[1])

    def getDistance(self, otherPos):
        otherArray = np.array(otherPos)
        npPos = np.array(self.pos)
        return distance.euclidean(npPos, otherPos)

    def isVisible(self):
        return self.state == WAIT_STATE
