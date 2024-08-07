## Demonstration of signal interpolation for simulated deep cosmic ray signals. 
import numpy as np
import matplotlib.pyplot as plt

import shower_interpolator
from shower_interpolator import shower_interp3D
import deepcr_helper as helper
import NuRadioMC,NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.framework import electric_field
from NuRadioReco.detector import detector

import csv
import h5py
import pandas as pd
import json
import os 

import time
import datetime

# load configuration file 
demo_config = {}

dir_path = os.path.dirname(os.path.realpath(__file__))
config_filename = os.path.join(dir_path,"demo_config.json")
with open( config_filename , "r") as infile:
    demo_config = json.load(infile)
def print_config(config):
    print("config:")
    for k in config.keys(): print(f"\t{k}:",config[k])

# user-defined content list
content_list = []
primary = 'Proton'
energies = [ f"{e/1e18:.4f}" for e in [10**16.5] ] #energy in 10^18 eV unit
zeniths = [0,10] # degrees
azimuths = [0] # degrees
for e in energies:
    for z in zeniths:
        for a in azimuths:
            content= {}
            simdir = f'/data/user/npunsuebsay/sim/Rectangle_{primary}_{e}_{z}_{a}_1/' 
            content['dir'] = simdir
            
            content['zenith'] = z
            content['azimuth'] = a
            content['energy'] = e

            reaslist = os.path.join(simdir,'Parameters/SIM.list')
            content['reaslist'] = reaslist

            content['file_type'] =  'channels'
            content['channels'] = {}
            
            
            content_list.append(content) 

# perform single request (Energy,Theta,Phi) with mutiple positions (channels)
# ---------------------------------------------------------------------------

station_id = 11
# det = detector.Detector(source="rnog_mongo", always_query_entire_description=False,
#                         database_connection='RNOG_public', select_stations=station_id)
det = detector.Detector("RNO_single_station.json",source="json")

det.update(datetime.datetime(2024, 7, 1, 0, 0))

antenna_positions = []
for ch_id in det.get_channel_ids(station_id):
    antenna_type = det.get_antenna_type(station_id,ch_id)
    if antenna_type == "VPol" or antenna_type == "RNOG_vpol_4inch_center_n1.73":
        antenna_positions.append( det.get_relative_position(station_id,ch_id) )

# abs_pos = det.get_absolute_position( station_id )
# print("absolute position:",abs_pos)
del det

sim_surface_height = 3216.0 # meter
## move antenna relative positions to ice surface 
antenna_positions = np.array(antenna_positions) + np.array([0.,0.,sim_surface_height])

# initialize shower interpolator class
print("\nInitializing interpolator...\n")
ini_start = time.perf_counter()
shower_interp = shower_interp3D(content_list,config=config_filename)
ini_end = time.perf_counter()

Energy = 0.0316
Theta = 6. 
Phi = 0.

print(f"Initialization done. \n\tTime used:{(ini_end - ini_start):.3e} s")


# call interpolation at given Energy,Theta,Phi,and antenna positions
print("\nInterpolating...\n")
call_start = time.perf_counter()
interp_efields = shower_interp(Energy,Theta,Phi, antenna_positions)
call_end = time.perf_counter()

print(  f"Interpolated {len(antenna_positions)} channels.")
print_config(demo_config)
print(  f"\tTime used:{(call_end - call_start):.3e} s")

# turning nuradiomc efields into numpy array
print("Resulting (interpolated) efields", len(interp_efields))
np_efields = []
for field in interp_efields:
    times = field.get_times()
    efields = field.get_trace()
    block = np.vstack( (times,efields))
    np_efields.append(block)

np_efields = np.array(np_efields)
print("expect shape : num_channel x 4 x num_sample where 4 = len([t,Ex,Ey,Ez])")
print("numpy efields' shape :",np_efields.shape)