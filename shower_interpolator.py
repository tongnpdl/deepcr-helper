import numpy as np
import deepcr_helper as crhp
import cr_pulse_interpolator.interpolation_fourier as interpF
from cr_pulse_interpolator import signal_interpolation_fourier as sigF

class summarized_content(dict):
	params = {}
	path = ""
	def __init__(self,file_path):
		self.path = file_path

class shower_interp:
	content_list = []
	def __init__(self,list_of_contents):
		"""
		Initialize a callable interpolator object.
		The object lookup library of simulated cosmic ray showers for appropriate entries to be used for interpolation.
		"""
		self.content_list = list_of_contents ## for now

	def get_content_list(self):
		return self.content_list
	def add_content(self,new_content):
		self.content_list.append(new_content)

	def lookup_content(self,E,T,P,num=1,method='nearest_neighbor'):
		"""
		Look for relevant content(s) form content list
		"""
		rel_contents = self.content_list[0:num]
		return rel_contents
	def load_content(content):
		pos,t,Ex,Ey,Ez = 0,0,0,0,
		return pos,t,Ex,Ey,Ez
	@classmethod
	def interpolator3D(x,y,z,efields,times):


	def __call__(self,x,y,z,Energy,Theta,Phi,lookup_method='nearest_neighbor',**kwarg):
		"""
		Call function 
		return interpolated electric fields
		"""

		# look up the list of CR simulations (Energy,Theta,Phi)
		relevant_contents = self.lookup_content(Energy,Theta,Phi,method=lookup_method)
		# determine "electric field" (Ex[t],Ey[t],Ez[t]) for each content from the list of relevant contents (CR simulations)
		efields = []
		for content in relevant_contents:
			_pos,_t,_Ex,_Ey,_Ez = load_content(content)
			this_interp = interpolator3D(_pos.T[0],_pos.T[1],_pos.T[2],[_Ex,_Ey,_Ez],_t)
			this_efield = this_interp([x,y,z])
			efields.append(this_efield)
		# find 'average field'
		average_field = efields[0] ## for now

		return average_field
