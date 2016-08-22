import random
import math
from copy import copy

class kPoint:
    __slots__ = ["lat", "lng", "time", "group"]
    def __init__(self, lat=0.0, lng=0.0, time=0, group=0):
        self.lat, self.lng, self.time, self.group = lat, lng, time, group

class kPlusPlus:
	__slots__ = ["points","centers","scaler"]
	def __init__(self, data, nclusters, lat=0):
		self.points = []
		self.centers = [kPoint() for _ in xrange(nclusters)]
		self.scaler = math.cos(math.radians(lat))
		# convert data to a form we can use
		for dpoint in data:
			self.points.append(kPoint(dpoint['lat'], dpoint['lng'], dpoint['time']))
		self.initalise()

	def initalise(self):
		# set a seed so the clustering is stable
		random.seed(1)
		# initalise the kpp algorithm
		self.centers[0] = copy(random.choice(self.points))
		d = [0.0 for _ in xrange(len(self.points))]
		for i in xrange(1, len(self.centers)):
			sum = 0
			for j, p in enumerate(self.points):
				d[j] = nearest_cluster_center(self.centers[:i], p, self.scaler)[1]
				sum += d[j]
			sum *= random.random()
			for j, di in enumerate(d):
				sum -= di
				if sum > 0:
					continue
				self.centers[i] = copy(self.points[j])
				break
		for p in self.points:
			p.group = nearest_cluster_center(self.centers, p, self.scaler)[0]

	def iterate(self, numSteps=1):
		# set the allowed error of 1/1024
		lenpts10 = len(self.points) >> 10
		changed = 0
		doneSteps = 0
		# iterate over upto numsteps times
		while doneSteps < numSteps:
			# move each center to the mean of the points in its group
			for cc in self.centers:
				cc.lat = 0.0
				cc.lng = 0.0
				cc.group = 0
			for p in self.points:
				self.centers[p.group].group += 1
				self.centers[p.group].lat += p.lat
				self.centers[p.group].lng += p.lng
			for cc in self.centers:
				cc.lat /= cc.group
				cc.lng /= cc.group
			# change the group of each point, to the closest center
			changed = 0
			for p in self.points:
				min_i = nearest_cluster_center(self.centers, p, self.scaler)[0]
				if min_i != p.group:
					changed += 1
					p.group = min_i
			doneSteps += 1
			# if less than 1 in 1024 have changed, break
			if changed <= lenpts10:
				break
		for i, cc in enumerate(self.centers):
			cc.group = i
		return doneSteps

	def getPoints(self):
		return self.points

	def getGroups(self):
		out = []
		for i in xrange(len(self.centers)):
			out.append([])
		for P in self.points:
			out[P.group].append(P)
		return out

	def getCenters(self):
		return self.centers

	def getGroupSize(self):
		out = [0 for _ in xrange(len(self.centers))]
		for P in self.points:
			out[P.group] += 1
		return out

def nearest_cluster_center(centers, point, scale):
	def sqr_distance_2D(a, b, scale):
		latS = (a.lat - b.lat)
		lngS = (a.lng - b.lng) * scale
		return (latS ** 2) + (lngS ** 2)
	min_index = -1
	min_dist = 1E100
	for i, cc in enumerate(centers):
		d = sqr_distance_2D(cc, point, scale)
		if min_dist > d:
			min_dist = d
			min_index = i
	return (min_index, min_dist)