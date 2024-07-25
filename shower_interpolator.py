import numpy as np
import json
import deepcr_helper as crhp
import cr_pulse_interpolator.interpolation_fourier as interpF
from cr_pulse_interpolator import signal_interpolation_fourier as sigF
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.utilities import units
from numpy import cos,sin,arccos
import matplotlib.pyplot as plt

import time


import csv
import pandas as pd
import os
class summarized_content(dict):
	params = {}
	path = ""
	def __init__(self,file_path):
		self.path = file_path
	# def __repr__(self):
	# 	return 'summarized content'
	# def __str__(self):
	# 	return self.params

class shower_interp3D:
	content_list = []
	config = {}
	def __init__(self,list_of_contents,config="sample_config.json"):
		"""
		Initialize a callable interpolator object.
		The object lookup library of simulated cosmic ray showers for appropriate entries to be used for interpolation.
		"""
		with open(config,"r") as config_json: ## read json into python dict
			self.config = json.load(config_json)  
		self.content_list = list_of_contents ## for now
		## after reading in the list, initialization starts 'processing' the list
		counter = 0
		for content in self.content_list:
			counter += 1
			if content['file_type'] == 'channels':
				reaslist = content['reaslist']
				channels = []
				perfstart = time.perf_counter()
				with open(reaslist) as reasfile:
					csv_reader = csv.reader(reasfile,delimiter=' ')
					for line in csv_reader:
						position = line[2:5]
						id = int(line[5].strip('ch'))
						this_channel = {}
						this_channel['id'] = id
						this_channel['position'] = np.array(position,dtype=float)
						this_channel['data'] = None
						channels.append(this_channel)
				content['channels'] = channels
				perfend = time.perf_counter()
				print(f'entry:{counter}\t reading reaslist takes: {(perfend-perfstart)/1e-3:.4e} miliseconds')
				## define layers
				positions = np.array([ ch['position'] for ch in channels])
				content['positions'] = positions
				unique_heights = np.unique(positions[:,2])
				content['num_layer'] = len(unique_heights)
				content['layers'] = []
				for i in range(content['num_layer']):
					this_layer = {}
					this_layer['height'] = unique_heights[i]
					this_layer['interpolator'] = None
					this_layer['channels'] = []
					content['layers'].append(this_layer)
				## layer assignment
				for ch in content['channels']:
					layer_index = self.identify_layer([ch['position']],content,self.config['height_tol'])
					ch['layer_index'] = layer_index[0]
					content['layers'][layer_index[0]]['channels'].append(ch['id'])

			elif content['file_type'] == 'events':
				print('Not implemented')
			else:
				raise Exception('Unknown "file_type": available options: channels, events ')


	class pulse_interp3D:
		def __init__(self, positions, amplitudes,times):
			self.positions = positions
			self.amplitudes = amplitudes
			self.times = times
		def __call__(self,targets,method):
			interp_efields = []
			if method == "nearest_neighbor":
				for target in targets:
					distances = np.linalg.norm(self.positions - target)
					argmin_distance = np.argmin(distances)
					amp = self.amplitudes[argmin_distance]
					t = self.times[argmin_distance]
					sampling_rate = 1./(t[1]-t[0])
					efield = ElectricField(0)
					efield.set_position(self.positions[argmin_distance])
					efield.set_trace(np.array(amp),sampling_rate)
					efield.set_trace_start_time(t[0])

					interp_efields.append(efield)
			return interp_efields

	def get_content_list(self):
		return self.content_list
	def add_content(self,new_content):
		self.content_list.append(new_content)

	def lookup_content(self,E,T,P,method):
		"""
		Look for relevant content(s) form content list
		"""
		#_eza = np.array([ [c['energy'],c['zenith'],c['azimuth']] for c in self.content_list ])
		energy_fil = []
		for c in self.content_list:
			energy = float(c['energy'])
			if np.abs(np.log10(energy) - np.log10(E)) <= 0.7: ## energy tolerance setting?? 
				energy_fil.append(c)
		## check if len(energy_fil) > 0 ??
		## calculate angular separation
		ang = []
		for c in energy_fil:
			target_t = T*units.deg 
			target_p = P*units.deg *0 ## only consider zenith separation for now
			content_t = c['zenith']*units.deg
			content_p = c['azimuth']*units.deg
			dist = arccos(sin(target_t)*sin(content_t) + cos(target_t)*cos(content_t)*cos(content_p - target_p))
			ang.append(dist)
		
		rel_contents = []
		if method == "nearest_neighbor": ## angular tolerance setting??
			argmin = np.argmin(ang)
			rel_contents.append(energy_fil[argmin])

		return rel_contents
	def identify_layer(self,positions,content,tol):
		layer_index = []
		layers = content['layers']
		layer_heights = np.array([ l['height'] for l in layers])
		for pos in positions:
			h = pos[2]
			diff = np.abs(layer_heights - h)
			index = np.argmin(diff)
			if diff[index] < tol:
				layer_index.append(index)
			else:
				layer_index.append(index) ## TO DO: fail case
				#updiff = layer_heights - h
				#layer_index.append()
		return layer_index
	def get_relevant_ch(self,positions,content,method):
		# discard duplicated positios 
		positions = np.unique(positions,axis=0) 
		ids = []
		if method == 'nearest_neighbor':
			## find nearest channel
			channel_positions = np.array(content['positions'])
			channels = content['channels']
			for pos in positions:
				pos = np.array(pos)
				argmin = np.argmin( np.linalg.norm(pos-channel_positions,axis=-1))
				ids = np.append(ids, channels[argmin]['id'] )
		elif method == 'simple':
			## identify layer(s) needed for this 'content'
			layer_indices = self.identify_layer(positions,content,self.config['height_tol'])
			unique_layers = np.unique(layer_indices)
			for layer_id in unique_layers:
				ids = np.append(ids,content['layers'][layer_id]['channels'])
		else:
			raise Exception('Unknown pulse method. Available: nearest_neighbor ')
		# discard duplicated channels
		ids = np.array(ids,dtype=int)
		ids = np.unique(ids)
		return ids
	def load_content(self,content,load_ch):
		pos = [] # n_ch x 3
		t = [] # n_ch x n_sample
		amplitudes = [] # n_ch x 3 x n_sample 
		content_e = float(content['energy'])
		content_a = content['azimuth']
		content_z = content['zenith']
		# num_line = []
		# time_took = []
		if content['file_type'] == 'channels':
			channels = content['channels']
			simdir = content['dir']
			loadstart = time.perf_counter()
			for ch in load_ch:
				if channels[ch]['data'] == None:
					filepath = os.path.join(simdir,f'Coreas/raw_ch{ch}.dat')
					# start = time.perf_counter()
					with open(filepath,'r') as infile:
						temp_data = pd.read_csv(infile, header=None,delimiter='\t')
						_t = np.array(temp_data[0][:],dtype=float)
						_amp = np.array([temp_data[1][:],temp_data[2][:],temp_data[3][:]])
						_pos = channels[ch]['position']
						t.append( _t)
						amplitudes.append( _amp )
						pos.append(_pos)

					# end = time.perf_counter()
					# num_line.append(len(temp_data[0][:]))
					# time_took.append(end-start)
					channels[ch]['data'] = (t,amplitudes)
				else: ## data already loaded 
					_t,_amp = channels[ch]['data']
					_pos = channels[ch]['position']
					t.append(_t)
					amplitudes.append(_amp)
					pos.append(_pos)
			loadend = time.perf_counter()
			print(f'Loaded {len(load_ch)} channels from E: {content_e} zenith: {content_z} azimuth: {content_a}\n\t took {(loadend-loadstart)/1e-3:.3e} ms')
		else:
			raise Exception('Unknow load method / file_type ... available: channels ')
		# plt.scatter(num_line,time_took), plt.xlabel('num line'),plt.ylabel('time took')
		return pos,amplitudes,t
	def energy_scale(self,energy,target_energy):
		## Atarget = Ao*scale ... scale = Etarget/Eo
		scale = 10**(target_energy - np.array(energy))
		return scale


	def weighted_average(self,efields,contents,Energy,Theta,Phi,method):
		fields = np.array(efields)
		# num_content,num_field = np.array(efields).shape
		average_fields = []
		if method == 'energy_scaling':
			content_energy = np.array([ float(content['energy']) for content in contents])
			scale_factor = self.energy_scale(content_energy,Energy)

			# for ifield in range(num_field):
			# 	for icontent in range(num_content):
			# 		if icontent == 0:
			# 			_field = fields[icontent,ifield] * scale_factor[icontent]
			# 		else:
			# 			_field += fields[icontent,ifield]* scale_factor[icontent]
			average_fields = np.mean( (fields.T*scale_factor),axis=-1 )
		return average_fields

	def __call__(self,Energy,Theta,Phi,positions,**kwarg):
		"""
		Call function 
		return interpolated electric fields
		"""
		# look up the list of CR simulations (Energy,Theta,Phi)
		relevant_contents = self.lookup_content(Energy,Theta,Phi,method=self.config['event_method'])
		# determine "electric field" (Ex[t],Ey[t],Ez[t]) for each content from the list of relevant contents
		efields = [] ## expect n_content x n_pos 
		for content in relevant_contents:
			## determine which channels are needed
			load_ch = self.get_relevant_ch(positions,content,method=self.config['pulse_method']) ## should be the same as interpolation method
			## load simulated signals 
			_antenna_position, _amplitudes, _t = self.load_content(content,load_ch)
			## initialize single-event interpolator
			this_interp = self.pulse_interp3D(_antenna_position,_amplitudes,_t)
			## call single-event interpolator
			interp_efields = this_interp(positions,method=self.config['pulse_method'])

			##
			efields.append(interp_efields)
			## instead of appending to "list of efields", add to summary as another entry
			## this should help with "averaging" efield at the end 
			# content['efields'] = efields ## len(positions) x num_sample

		# find 'average field'
		# average_field = efields[0] ## for now
		average_fields = self.weighted_average(efields, relevant_contents,Energy,Theta,Phi,method=self.config["average_method"])

		return average_fields
