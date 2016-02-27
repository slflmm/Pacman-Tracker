
import cv2 

import sys
from scipy import *
from scipy.cluster import vq
import numpy as np 
from MovingObject import MovingObject
import pickle
from pacmanActionIterator import PacmanActionIterator
from ParticleFilter import ParticleFilter
from pacmanStateIterator import PacmanStateIterator

frameW = 895 
frameH = 537
FRAME_TIME = 1

top = 0
bottom = 1
left = 0
right = 1
X_COORD = 0
Y_COORD = 1

# Use RGB in colour histogram
HIST_CHANNELS = [0, 1, 2]
HIST_BINS = [32, 32, 32]
HIST_RANGE = [0, 256]
HIST_RANGES = HIST_RANGE + HIST_RANGE + HIST_RANGE
# The discretization of the monotonically increasing histogram kernel
HIST_KERN_STEP = 5

STEP_SIZE = 8 # 8 Seems to give best tracking results
N_PARTICLES = 15 # Kinda works?  Try 15, 20?  30 is slow.

# Threshold for saying the object has changed state
# Hellinger distance ranges from 0 to 1
HELLINGER_THRESHOLD = 0.4
LOST_THRESHOLD = 0.5 # TODO try lowering this value (tighter)

# Number of entities to use in PAC MAN
NUM_PACMAN_ENTITIES = 9 # 1 pacman + 4 normal ghosts + 4 running ghosts (consider these different types)

# Entity components
E_NAME = 0
E_COLOR = 1
E_SEEN = 2
E_POS = 3
E_BOX = 4
E_HIST = 5
E_SIZE = 6
E_FILTER = 7
E_ASSOCIATIONS = 8

SIZE_THRESHOLD = 5

LEARN_MODE = 0
PREDICT_MODE = 1

# File to write the pickled learned behaviours to
BEHAVIOUR_OUT_FN = "entityBehaviours.pkl"

# NOTE: None of this is well-structured yet. 
# I'm just making sure the tracking works reasonably well for now.

# NOTE: this method will NOT work on windows!!
# because opencv is a whiny, difficult little bitch, and so is windows
# and I wasted hours on this shit D:
def make_video(sequence_number):
	fps = 30
	capsize = (frameW, frameH)
	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	writer = cv2.VideoWriter() 
	writer.open('sequence%d.avi'%sequence_number, fourcc, fps, capsize, True)

	nFrames = 2423
	for i in range(nFrames):
		img = cv2.imread("Sequences/%d/Images/Frame%d.png"%(sequence_number,i+1))
		writer.write(img)

	cv2.destroyAllWindows()
	writer.release()
	writer = None 


def updateEntityPos(entity, frame, frame_count, target, box, groundTruthStates):
        entity[E_SEEN] = frame_count
        entity[E_BOX] = box
        if entity[E_FILTER]:
                estimate = entity[E_FILTER].updateParticles(frame, box)
                entity[E_POS] = tuple(estimate.astype(int))
        else:
                entity[E_POS] = target
                updateDistMeasure(entity, groundTruthStates)
                
                
# Possible problem with this: the boxes lag behind the objects,
# so we may capture a lot of empty space.  Thresholding to square should
# take care of that though.
"""
Computes a histogram of pixel colours linearly weighted by the distance
of the pixel from the image center.
"""
def getColorHist(colorFrame, center, width, height):
        # stick to just a square
        size = min(width, height)
        kernelIters = max( min(HIST_KERN_STEP, size / 2), 1)
        stepSize = size / (2*kernelIters)

        # Repeatedly compute the histogram with a linearly-expanding
        # mask - the center pixels get weighted most, those on the edge
        # get weighted least.
        hist = None
        entityMask = np.zeros((colorFrame.shape[0:2]), dtype=np.uint8)
        for i in range(kernelIters):
                # expand the mask
                if stepSize > 0:
                        iRange = stepSize * (i+1)
                        xLow = max(center[X_COORD] - iRange, 0)
                        yLow = max(center[Y_COORD] - iRange, 0)
                        xHigh = min(xLow + 2*iRange, colorFrame.shape[1])
                        yHigh = min(yLow + 2*iRange, colorFrame.shape[0])
                        # x is first and y is second in image coords
                        entityMask[yLow:yHigh, xLow:xHigh] = 1
                else:
                        # this should only happen in degenerate cases
                        entityMask[center[Y_COORD], center[X_COORD]] = 1
                # run calcHist on the histogram so far with the mask
                if hist == None:
                        hist = cv2.calcHist([colorFrame],
                                            HIST_CHANNELS,
                                            entityMask,
                                            HIST_BINS,
                                            HIST_RANGES)
                else:
                        hist = cv2.calcHist([colorFrame],
                                            HIST_CHANNELS,
                                            entityMask,
                                            HIST_BINS,
                                            HIST_RANGES,
                                            hist,
                                            accumulate=True)
        # normalize before returning
        return cv2.normalize(hist, alpha=1, norm_type=cv2.NORM_L1)

        
def getEntityHellingerDist(frame, pos, entity):
        posHist = getColorHist(frame, pos, entity[E_SIZE], entity[E_SIZE])
        return getHellingerDist(posHist, entity[E_HIST])

        
def getPosHellingerDist(frame, pos, size, targetHist):
        posHist = getColorHist(frame, pos, size, size)
        return getHellingerDist(posHist, targetHist)


def getHellingerDist(hist1, hist2):
        # openCV actually gives hellinger when you ask for bhattacharyya
        # unravel the histograms because cv2.compareHist doesn't like 3D histograms
        return cv2.compareHist(hist1.ravel(), hist2.ravel(), cv2.cv.CV_COMP_BHATTACHARYYA)

        
# With some wiggle room
def doBoxesOverlap(box1, box2):
        overlap = True
        if (box1[1][X_COORD] < box2[0][X_COORD]): overlap = False
        if (box1[0][X_COORD] > box2[1][X_COORD]): overlap = False
        if (box1[1][Y_COORD] < box2[0][Y_COORD]): overlap = False
        if (box1[0][Y_COORD] > box2[1][Y_COORD]): overlap = False
        # if (box1[1][X_COORD]*1.02 < box2[0][X_COORD]*0.98): overlap = False
        # if (box1[0][X_COORD]*0.98 > box2[1][X_COORD]*1.02): overlap = False
        # if (box1[1][Y_COORD]*1.02 < box2[0][Y_COORD]*0.98): overlap = False
        # if (box1[0][Y_COORD]*0.98 > box2[1][Y_COORD]*1.02): overlap = False
        return overlap
        
def merge_collided_bboxes( bbox_list ):
	# For every bbox...
	for this_bbox in bbox_list:
		
		# Collision detect every other bbox:
		for other_bbox in bbox_list:
			if this_bbox is other_bbox: continue  # Skip self
			
			# Assume a collision to start out with:
			has_collision = True
			
			# These coords are in screen coords, so > means 
			# "lower than" and "further right than".  And < 
			# means "higher than" and "further left than".
			
			# We also inflate the box size by 4% to deal with
			# fuzziness in the data.  (Without this, there are many times a bbox
			# is short of overlap by just one or two pixels.)
                        has_collision = doBoxesOverlap(this_bbox, other_bbox)
			# if (this_bbox[1][X_COORD]*1.02 < other_bbox[0][X_COORD]*0.98): has_collision = False
			# if (this_bbox[0][X_COORD]*.98 > other_bbox[1][X_COORD]*1.02): has_collision = False
			
			# if (this_bbox[1][Y_COORD]*1.02 < other_bbox[0][Y_COORD]*0.98): has_collision = False
			# if (this_bbox[0][Y_COORD]*0.98 > other_bbox[1][Y_COORD]*1.02): has_collision = False

			if has_collision:
				# merge these two bboxes into one, then start over:
				top_left_x = min( this_bbox[left][0], other_bbox[left][0] )
				top_left_y = min( this_bbox[left][1], other_bbox[left][1] )
				bottom_right_x = max( this_bbox[right][0], other_bbox[right][0] )
				bottom_right_y = max( this_bbox[right][1], other_bbox[right][1] )
				
				new_bbox = ( (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) )
				
				bbox_list.remove( this_bbox )
				bbox_list.remove( other_bbox )
				bbox_list.append( new_bbox )
				
				# Start over with the new list:
                                # Python doesn't optimize tail recursion - maybe loop instead?
				return merge_collided_bboxes( bbox_list )
	
	# When there are no collions between boxes, return that list:
	return bbox_list

        
def getGroundTruthMap(entities, groundTruthStates):
        # in some degenerate cases, a single ground-truth entity
        # may have multiple entity hypotheses associated with it.
        # in such cases, throw away all but one.
        matches = [getGroundTruthMatch(entity, groundTruthStates).name
                   for entity in entities]
        return dict(zip(matches, entities))

        
def getGroundTruthMatch(entity, groundTruthStates):
        return min(groundTruthStates, key=lambda x:x.getDistance(entity[E_POS]))


def updateDistMeasure(entity, groundTruthStates):
        if not entity[E_ASSOCIATIONS]:
                gtMatch = getGroundTruthMatch(entity, groundTruthStates)
                entity[E_ASSOCIATIONS] = {gtMatch.name: []}
        for gtName, distances in entity[E_ASSOCIATIONS].iteritems():
                gtState = filter(lambda gts: gts.name == gtName, groundTruthStates)[0]
                distances.append(gtState.getDistance(entity[E_POS]))
                

def printDistanceStats(entity):
        for gtName, distances in entity[E_ASSOCIATIONS].iteritems():
                output = ' '.join(["BLOB TRACKING: name:", gtName,
                                  "Duration (frames):", str(len(distances)),
                                  "Average error:", str(np.mean(distances))])
                print output
                # print ("BLOB TRACKING: name:" + gtName + "Duration (frames):" + str(len(distances)) + "Average error:" + str(np.mean(distances)))
        
        
def updateEntitiesNew(newTargets, targetBoxes, oldEntities, entityMemory,
                      frame, frame_count, entity_free, entity_colours,
                      groundTruthStates):
        # Associate new bounding boxes with previous bounding boxes
        oldToEntities = {entity[E_BOX]:[] for entity in oldEntities}
        oldToNew = {entity[E_BOX]:[] for entity in oldEntities}
        newToOld = {box:[] for box in targetBoxes}
        for entity in oldEntities:
                oldBox = entity[E_BOX]
                oldToEntities[oldBox].append(entity)
                for newBox in targetBoxes:
                        if doBoxesOverlap(oldBox, newBox):
                                if newBox not in oldToNew[oldBox]:
                                        oldToNew[oldBox].append(newBox)
                                if oldBox not in newToOld[newBox]:
                                        newToOld[newBox].append(oldBox)

        # Search for and handle merges
        for newBox, oldBoxes in newToOld.iteritems():
                if len(oldBoxes) > 1:
                        # Case 1: Merge.  Create any necessary particle filters.
                        entities = reduce(lambda x, y: x + y,
                                          [entities for entities in oldToEntities.values()])
                        associations = getGroundTruthMap(entities, groundTruthStates)
                        print associations.keys()
                        print frame_count, "merged:", associations.keys()
                        for e in entities:
                                # Set associations for keeping track of how well this is doing
                                if e[E_ASSOCIATIONS] and not e[E_FILTER]:
                                        printDistanceStats(e)
                                e[E_ASSOCIATIONS] = associations
                                if not e[E_FILTER]:
                                        # initialize particle filter 
                                        distMeasure = lambda pos, frame: getPosHellingerDist(pos,
                                                                                             frame,
                                                                                             e[E_SIZE],
                                                                                             e[E_HIST])
                                        e[E_FILTER] = ParticleFilter(distMeasure,
                                                                     LOST_THRESHOLD,
                                                                     e[E_POS],
                                                                     STEP_SIZE,
                                                                     N_PARTICLES)

        # Apportion entities to the new blobs
        newToEntities = {newBox:[] for newBox in targetBoxes}
        for oldBox, newBoxes in oldToNew.iteritems():
                # Case 5: Disappearance
                if len(newBoxes) == 0:
                        # The entity/s have vanished!  Put them in memory - avoid duplicates.
                        for entity in oldToEntities[oldBox]:
                                if entity[E_ASSOCIATIONS] and not entity[E_FILTER]:
                                        printDistanceStats(entity)
                                        entity[E_ASSOCIATIONS] = None
                                if entity not in entityMemory:
                                        entityMemory.append(entity)

                # Case 2: Persistence
                if len(newBoxes) == 1:
                        # Old box persists to one new box
                        newToEntities[newBoxes[0]] += oldToEntities[oldBox]
                # Case 3: Split
                elif len(newBoxes) > 1:
                        # Split occurred! Get list of entities in old box
                        oldBoxEnts = oldToEntities[oldBox]
                        while len(oldBoxEnts) > 0:
                                for newBox in newBoxes:
                                        target = newTargets[targetBoxes.index(newBox)]
                                        if len(oldBoxEnts) > 0:
                                                entDistDict = {}

                                                # Find the box containing the target with the closest center
                                                for entity in oldBoxEnts:
                                                        d_x = entity[E_POS][X_COORD] - target[X_COORD] 
                                                        d_y = entity[E_POS][Y_COORD] - target[Y_COORD]
                                                        sqrDist = pow(d_x,2) + pow(d_y,2)
                                                        entDistDict[sqrDist] = entity
                                                closestEnt = entDistDict[min(entDistDict)]
                                                oldBoxEnts.remove(closestEnt)
                                                # assign it the unassigned entity closest to it
                                                if newBox not in newToEntities:
                                                        newToEntities[newBox] = [closestEnt]
                                                else:
                                                        newToEntities[newBox].append(closestEnt)
                                        else: break # Stop if we run out of entities

        # Set entities on new blobs
        for newBox in newToEntities:
                entities = newToEntities[newBox]
                if len(entities) == 1 and entities[0][E_FILTER]:
                        # Continues case 3: Split
                        entity = entities[0]
                        if entity[E_ASSOCIATIONS]:
                                # Am I still closest to the ground truth I was originally associated with?
                                msgStr = "SPLIT EVENT:"
                                newGTMatch = getGroundTruthMatch(entity, groundTruthStates)
                                oldGTMatches = entity[E_ASSOCIATIONS]
                                if ((newGTMatch.name in oldGTMatches)
                                    and (oldGTMatches[newGTMatch.name] is entity)):
                                        msgStr += newGTMatch.name + ' tracked through merge'
                                else:
                                        msgStr += newGTMatch.name + ' not tracked'
                                print frame_count, msgStr
                                # particle filter no longer needed
                                entity[E_ASSOCIATIONS] = None
                        entity[E_FILTER] = None
                        
                # Case 4: New entity appears
                if len(entities) == 0:
                        # A new blob has appeared - see if it matches an entity in memory
                        width = abs(newBox[right][X_COORD] - newBox[left][X_COORD])
                        height = abs(newBox[right][Y_COORD] - newBox[left][Y_COORD])
                        # determine the colour profle
                        target = newTargets[targetBoxes.index(newBox)]
                        hist = getColorHist(frame, target, width, height)
                        entity = None
                        
                        # Check if this is the return of an entity we remember
                        if len(entityMemory) > 0:
                                bestMemory = min(entityMemory, key=lambda e: getHellingerDist(hist, e[E_HIST]))
                                if getHellingerDist(hist, bestMemory[E_HIST]) < HELLINGER_THRESHOLD:
                                        # print "Entity remembered!"
                                        entity = bestMemory
                                        entityMemory.remove(bestMemory)
                                        # entity[E_HIST] = state

                        # Create a brand-new entity for this target
                        if entity == None:
                                color = (random.randint(0,255),
                                         random.randint(0,255),
                                         random.randint(0,255))
                                # Use a preset color if any is available
                                for i, isFree in enumerate(entity_free):
                                       if isFree:
                                                entity_free[i] = False
                                                color = entity_colours[i]
                                                break
                                size = min(width, height)
                                name = str(frame_count) + " " + str(color)
                                entity = [ name, color, frame_count, target, newBox, hist, size,
                                           None, None ]

                        # Associate this entity to the blob
                        newToEntities[newBox] = [entity]
                        
        # update all entities that have been associated with a new blob
        for newBox, entities in newToEntities.iteritems():
                target = newTargets[targetBoxes.index(newBox)]
                for e in entities:
                        updateEntityPos(e, frame, frame_count, target, newBox, groundTruthStates)
                        
        # return surviving entities
        return reduce(lambda x, y: x + y, newToEntities.values())


# adapted from http://derek.simkowiak.net/motion-tracking-with-python/
def blob_track(sequence_number, behavioursFilename=None, mode=LEARN_MODE):
   
        frame_count = 0
	last_target_count = 5 # pacman starts with five entities
	codebook = []
	frame_entity_list = []
        all_entities = []

	entity_free = [True]*last_target_count 
	entity_colours = [(255, 174, 185), (125, 158, 192), (113, 198,113), (255, 236, 139), (64, 224, 208), (171, 130, 255)]

	# cam = cv2.VideoCapture('sequence%d.avi'%sequence_number)
	frame = cv2.imread("Sequences/%d/Images/Frame1.png"%sequence_number)
        GroundTruthIter = PacmanStateIterator("Sequences/%d/Positions.txt"%sequence_number)

        entity_memory = []
        # learnedBehaviours = {}
        previous_frame = np.zeros(frame.shape)
	avg = np.float32(frame)
        # if behavioursFilename:
        #         with open(behavioursFilename, 'r') as pickleFile:
        #                 entityBehaviours = pickle.load(pickleFile)
        #         entity_memory = [entity for entity, _ in entityBehaviours]
        #         learnedBehaviours = {entity[E_NAME]:behaviour for entity, behaviour in entityBehaviours}

        
	first = True

	nFrames = 2423
        previous_boxes = []

	for i in range(nFrames):
		original_frame = cv2.imread("Sequences/%d/Images/Frame%d.png"%(sequence_number,i+1))
                if original_frame == None:
                        break
                
		# since we know exactly where the score is, this will 
		# stop the algorithm from seeing score as an entity
		frame = original_frame.copy()
		frame[5:25,225:275] = 255

		# use a weighted average as background rather than 
		# some initial background subtraction
		# this fixes the problem of vanished tokens appearing once eaten
		# (we could also keep a list of backgrounds from a few frames in the past)
		# (but that would just consume more memory, not sure if worthwhile)
		if first:
			first = False
		else:
			cv2.accumulateWeighted(frame, avg, 0.3)

		# convertScaleAbs converts float avg to uint8
		# so that subsequent operations can be done
		savg = cv2.convertScaleAbs(avg)
		difference = cv2.absdiff(frame, savg)

		# get black and white
		# white = 'blob' in theory
		# Note that adaptive thresholding works a lot better!!
		# (Normal thresholding had a hard time detecting dark blue ghosts as well, 
			# so they were just a bunch of tiny specks -- not as good for tracking)
		greyscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
		# (_, black_white) = cv2.threshold(greyscale, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		black_white = cv2.adaptiveThreshold(greyscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
		white_areas = np.where(black_white > 3)
		white_areas = zip (white_areas[1], white_areas[0])

		# contours will be used to find bounding boxes for blobs
		temp = black_white.copy()
		contours, hierarchy =  cv2.findContours(temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# obtain bounding boxes
		# they will be used to estimate # of blobs on-screen
		bounding_boxes = []
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			p1 = (x,y)
			p2 = (x+w, y+h)
			bounding_boxes.append( (p1, p2) )

		# find the area of each box: 
		# boxes with a very small area should be removed
		box_areas = []
		for box in bounding_boxes:
			box_width = box[right][0] - box[left][0]
			box_height = box[bottom][0] - box[top][0]
			box_areas.append(box_width * box_height)

		# calculate average box area...
		average_box_area = 0.0
		if len(box_areas): 
			average_box_area = sum(box_areas)*1.0 / len(box_areas)

		big_boxes = []
		for box in bounding_boxes:
			box_width = box[right][X_COORD] - box[left][X_COORD]
			box_height = box[bottom][Y_COORD] - box[top][Y_COORD]
			if (box_width * box_height) > average_box_area*0.1 and box_width > SIZE_THRESHOLD and box_height > SIZE_THRESHOLD:
				big_boxes.append(box)

		big_boxes = merge_collided_bboxes( big_boxes )

		# for each bounding box, find points that are white
		# find their mean coordinate
		center_points = [] 
		for box in big_boxes:
			meanx = 0
			meany = 0
			nwhite = 0
			for i in range(box[0][0], box[1][0]):
				for j in range(box[0][1], box[1][1]):
					if black_white[j,i] > 3:
						nwhite += 1
						meanx += i
						meany += j
			if nwhite > 0:
				meanx /= nwhite 
				meany /= nwhite
				center_point = (int(meanx), int(meany))
				center_points.append(center_point)

		trimmed_center_points = center_points
		for cp in trimmed_center_points:
			cv2.circle(frame, cp, 20, (255,255,255), 2)

		last_target_count = len(trimmed_center_points)
                                
		# build up list of moving characters
                groundTruthStates = GroundTruthIter.getNextStateBundle()
                frame_entity_list = updateEntitiesNew(trimmed_center_points,
                                                      big_boxes,
                                                      frame_entity_list,
                                                      entity_memory,
                                                      frame,
                                                      frame_count,
                                                      entity_free,
                                                      entity_colours,
                                                      groundTruthStates)
                all_entities = frame_entity_list + entity_memory
                # remove entities with crappy particles
                # frame_entity_list = [e for e in frame_entity_list if not e[E_FILTER]
                #                      or not e[E_FILTER].isLost(frame)]
                
                # Learn behaviours
                # action = actionIter.getNextAction()
                # if mode == LEARN_MODE:
                #         for entity in frame_entity_list:
                #                 entityName = entity[E_NAME]
                #                 newPos = entity[E_POS]
                #                 if entityName not in learnedBehaviours:
                #                         learnedBehaviours[entityName] = MovingObject(
                #                                 newPos[0], newPos[1], action)
                #                 else:
                #                         isVisible = (entity[E_SEEN] == frame_count)
                #                         learnedBehaviours[entityName].updateState(
                #                                 action, newPos[0], newPos[1], isVisible)


		# draw the entities 
		for entity in frame_entity_list:
			center_point = entity[E_POS]
                        color = entity[E_COLOR]
			cv2.circle(original_frame, center_point, 20, cv2.cv.CV_RGB(color[0],color[1],color[2]), 2)
                        # filterPos = pFilter.getPositionEstimate()
                                                # compute dist from entity histogram to hist at current pos
                        selfDist = getEntityHellingerDist(frame, center_point, entity)
                        cv2.putText(original_frame, str(selfDist)[0:5], center_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    cv2.cv.CV_RGB(color[0],color[1],color[2]), 1)

                        # cv2.rectangle(original_frame,entity[E_BOX][0],entity[E_BOX][1],(255,255,255),2)
                        
                        # pFilter = entity[E_FILTER]
                        # if pFilter:
                        #         for p, w in zip(pFilter.particles, pFilter.weights):
                        #                 pPixel = tuple(p.astype('int'))
                        #                 cv2.circle(original_frame, pPixel, 1, cv2.cv.CV_RGB(color[0],color[1],color[2]), 2)
                if groundTruthStates:
                        for gts in groundTruthStates:
                                if not gts:
                                        break
                                pos = gts.getPixelPos()
                                cv2.circle(original_frame, pos, 10, cv2.cv.CV_RGB(0,255,0), 3)
                # Draw bounding boxes
                for box in big_boxes:
                        width = abs(box[0][X_COORD] - box[1][X_COORD])
                        height = abs(box[0][Y_COORD] - box[1][Y_COORD])
                        cv2.rectangle(original_frame,box[0],box[1],(255,255,255),2)
                        

		cv2.imshow('frame', original_frame)

		frame_count+=1
		previous_frame = original_frame 
                previous_boxes = big_boxes
                if cv2.waitKey(FRAME_TIME) != -1:
			break

	cv2.destroyAllWindows()

        # entityBehaviourPairs = []
        # for name, behaviour in learnedBehaviours.iteritems():
        #         behaviour.compress()
        #         matches = filter(lambda e : entity[E_NAME] == name, all_entities)
        #         if len(matches) > 0:
        #                 entity = matches[0]
        #                 entityBehaviourPairs.append((matches[0], behaviour))

        # with open(BEHAVIOUR_OUT_FN, 'w') as pickleFile:
        #         pickle.dump(entityBehaviourPairs, pickleFile)
                


def main():
	# make_video(0)
        # bvrFilename = None
        # if len(sys.argv) > 1:
        #         bvrFilename = sys.argv[1]
        # mode = LEARN_MODE
        # if len(sys.argv) > 2 and sys.argv[2].lower() == 'predict':
        #         mode = PREDICT_MODE
	# blob_track(0, bvrFilename, mode)
        if len(sys.argv) > 1:
                blob_track(int(sys.argv[1]))
        else:
                blob_track(2)

if __name__ == "__main__": main()
