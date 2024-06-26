import numpy as np
import deepcr_helper as crhp
from cr_pulse_interpolator import interpolation_fourier as interp_fourier
from cr_pulse_interpolator import signal_interpolation_fourier as signal_interp
class shower_content:
	def __init__(self,file_location):
		self.path = file_location
		self.energy = None
		self.zenith = None
		self.azimuth = None
		self.primary = 14
		self.input = ""

class shower_interp: 
	def __init__(self,list_of_contents):
		"""
		Initialize a callable interpolator object.
		The object lookup library of simulated cosmic ray showers for appropriate entries to be used for interpolation.
		"""
		self.content_count = len(list_of_contents)
		self.content_list = list_of_contents ## for now
		if self.content_count > 0:
			self.summarize_content()
	def get_content_list(self):
		return self.content_list
	def add_content(self,new_content):
		self.content_list = np.append(self.content_list,new_content)
	def summarize_content(self):
		self.unique_enegy = {}
		self.unique_zenith = {}
		self.unique_azimuth = {}
		print("Number of shower(s): "+f"{self.content_count}")
	def lookup_content(self,E,T,P,num=1,method='nearest_neighbor'):
		content_index = [0]*num
		return content_index


	def __call__(self,x,y,z,Energy,Theta,Phi,method='nearest_neighbor',**kwarg):
		"""
		Call function return interpolated electric fields
		"""
		content_indices = self.lookup_content(Energy,Theta,Phi,method=method)
		relevent_contents = self.content_list[content_indices]

		t,Ex,Ey,Ez = np.array([0,0,0,0],dtype=float)
		return t,Ex,Ey,Ez
