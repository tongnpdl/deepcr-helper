import numpy as np


class interp_shower2d:
	references = []
	def __init__(self,library=None):
		"""
		Initialize a callable interpolator object.
		The object lookup library of simulated cosmic ray showers for appropriate entries to be used for interpolation.
		"""
		if library is None: ## use example library
			pass ## for now


	def __call__(self,x,y,method='nearest_neighbor'):
		"""
		Call function return interpolated electric fields
		"""
		t,Ex,Ey,Ez = np.array([0,0,0,0],dtype=float)
		return t,Ex,Ey,Ez
