import numpy as np


class shower_interp:
	
	def __init__(self,table=None):
		"""
		Initialize a callable interpolator object.
		The object lookup library of simulated cosmic ray showers for appropriate entries to be used for interpolation.
		"""
		self.content_table = table ## for now


	def __call__(self,x,y,z,Energy,theta,phi,method='nearest_neighbor',**kwarg):
		"""
		Call function return interpolated electric fields
		"""
		t,Ex,Ey,Ez = np.array([0,0,0,0],dtype=float)
		return t,Ex,Ey,Ez
