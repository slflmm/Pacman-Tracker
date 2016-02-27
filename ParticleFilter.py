# Based on the particle filter from the scipy Cookbook found at:
# http://wiki.scipy.org/Cookbook/ParticleFilter

# TODO: extend this to a pac-man specific filter that would look
# only at five particles: left, right, top, bottom, and current 

# Circle filter that looks at particles evenly distributed around a circle?

import numpy as np
from operator import itemgetter

# TODO Duplicated from tracking.py...would be much nicer to have it elsewhere and import it
X_COORD = 0
Y_COORD = 1


class ParticleFilter:

    # sensorModel is bound to a function that takes two arguments:
    # an image first, a position second
    def __init__(self, distMeasure, distThreshold, pos, stepSize, n=0):
        self.distMeasure = distMeasure
        self.distThreshold = distThreshold
        self.stepSize = stepSize
        self.particles = np.ones((n, len(pos))) * pos
        self.weights = np.ones(n).T/n

    # def reinit(self, pos, stepSize=None, n=None):
    #     if n == None:
    #         n = len(self.particles)
    #     self.particles = np.ones((n, len(pos))) * pos
    #     if stepSize != None:
    #         self.stepSize = stepSize

    def moveParticles(self, boundary):
        # for each particle, move it uniformly inside the boundary by stepsize
        
        for i, p in enumerate(self.particles):
            xRandMin = max(p[X_COORD]-self.stepSize, boundary[0][X_COORD])
            xRandMax = min(p[X_COORD]+self.stepSize, boundary[1][X_COORD])
            yRandMin = max(p[Y_COORD]-self.stepSize, boundary[0][Y_COORD])
            yRandMax = min(p[Y_COORD]+self.stepSize, boundary[1][Y_COORD])

            xNew = np.random.uniform(xRandMin, xRandMax)
            yNew = np.random.uniform(yRandMin, yRandMax)
            newPoint = np.array([xNew, yNew])
            self.particles[i] = newPoint
        # self.particles += np.random.uniform(-self.stepSize,
        #                                     self.stepSize,
        #                                     self.particles.shape)

    def weightParticles(self, frame):
        if len(self.particles) == 0:
            return []
        dists = self.getDists(frame)
        weights = quasiGaussian(dists)
        weights /= sum(weights)
        self.weights = weights
        return self.weights
        
    def resampleParticles(self):
        n = len(self.particles)
        newParticleIndices = []
        weightSums = [0.] + [sum(self.weights[:i+1]) for i in range(n)]
        u0 = np.random.random()
        j = 0
        for u in [(u0+i)/n for i in range(n)]:
            while u > weightSums[j]:
                j += 1
            newParticleIndices.append(j-1)
            
        self.particles = self.particles[newParticleIndices,:]

    def getPositionEstimate(self):
        return np.mean(self.particles, axis=0) 
        
    def updateParticles(self, frame, boundary=None):
        self.moveParticles(boundary)
        self.weightParticles(frame)
        # # Only resample if a handful of particles have very high weights
        # if 1./sum(self.weights**2) < len(self.weights)/2.:
        self.resampleParticles()
        # bestPos, w = max(zip(self.particles, self.weights), key=itemgetter(1))
        # print bestPos, w
        return self.getPositionEstimate()

    def isLost(self, frame):
        return all(self.getDists(frame) > self.distThreshold)

    def getDists(self, frame):
        return np.array([self.distMeasure(frame, p) for p in self.particles])
        

def quasiGaussian(x, sigma=0.3, mu=0):
    return np.exp(-np.power(x-mu, 2.) / (2 * np.power(sigma, 2.)))

    
def isPointInsideBox(point, box):
    # box = np.array(box, dtype='float')
    xInside = not ((point[X_COORD] < box[0][X_COORD]) ==
                   (point[X_COORD] < box[1][X_COORD]))
    yInside = not ((point[Y_COORD] < box[0][Y_COORD]) ==
                   (point[Y_COORD] < box[1][Y_COORD]))
    # print point, box, xInside, yInside
    return xInside and yInside

        
    # def setNumParticles(self, n):
    #     if n == 0:
    #         self.particles = None
    #     elif n > len(self.particles):
    #         newParticles = 
