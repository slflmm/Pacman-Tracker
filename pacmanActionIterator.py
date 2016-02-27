NULL_CODE = 0
RCODE = 1
LCODE = 2
UCODE = 3
DCODE = 4

class PacmanActionIterator:
    lastAction = NULL_CODE
    def __init__(self, filename):
        self.actionFile = open(filename, 'r')

    def getNextAction(self):
        line = self.actionFile.readline()
        if line == '':
            return None
            
        actionCode = int(line)
        # if actionCode == NULL_CODE:
        #     return self.lastAction
            
        # else:
        self.lastAction = actionCode
        return actionCode
            
