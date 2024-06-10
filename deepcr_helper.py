import numpy as np
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units
import NuRadioMC
import NuRadioReco
from NuRadioReco.detector import detector
import os
NuRadioMC_dirname = os.path.dirname(NuRadioMC.__file__)
NuRadioReco_dirname = os.path.dirname(NuRadioReco.__file__)
from datetime import datetime

def rot_mat(rot_angle):
    sine = np.sin(np.deg2rad(rot_angle))
    cosine = np.cos(np.deg2rad(rot_angle))
    rot = np.array([ [cosine, -sine,0] , [sine,cosine,0],[0,0,1] ])
    return rot 

def generate_star_grid(length,separation,zenith,azimuth,azimuth_offset=0,height=-100.,core_vertical=140000.):
    grid_positions = np.array([0,0,core_vertical+height])
    
    # num_antenna_per_tine = int(length/separation)
    
    num_tine = 8
    rot_angle = 360/num_tine
    
    master_tine = np.array([[x,0,core_vertical+height] for x in np.arange(0,length,separation)[1:]])
    
    for az in (azimuth_offset + np.arange(0,360,rot_angle)) :
        grid_positions = np.vstack( (grid_positions , np.matmul(master_tine,rot_mat(az).T) ) )
    return grid_positions
def tilt_star_grid(grid,zenith,azimuth):
    basis = np.array([[np.cos(np.deg2rad(azimuth)),np.sin(np.deg2rad(azimuth)),0],
                  [-np.sin(np.deg2rad(azimuth)),np.cos(np.deg2rad(azimuth)),0],[0,0,1]])
    new_basis = np.matmul(grid,basis.T)
    new_basis[:,0] = new_basis[:,0]/np.cos(np.deg2rad(zenith))
    tilted = np.matmul(new_basis,basis)
    return tilted
def generate_shell_grid(radius_array, zenith_array, azimuth_array):
    grid_positions = []
    for radius in radius_array:
        for zenith in zenith_array:
            for azimuth in azimuth_array:
                xyz = radius*np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth), np.cos(zenith)])
                if len(grid_positions) == 0:
                    grid_positions = xyz
                else:
                    grid_positions = np.vstack( (grid_positions,xyz) )
    return grid_positions

def generate_single_rnog(filename,station_id):

    det = detector.Detector(NuRadioReco_dirname+f'/detector/RNO_G/{filename}')
    det.update(datetime.now())
    positions = np.array([[0,0,0]])
    station_ids = det.get_station_ids()
    if station_id not in station_ids:
        print( "Error: detector file doesnot contain requested station_id")
        return
    for st_id in station_ids:
        if st_id != station_id: continue
        # n_channels = det.get_number_of_channels(st_id)
        ch_ids =  det.get_channel_ids(st_id)
        abs_pos = det.get_absolute_position( st_id )
        for ch_id in ch_ids:
            ## NuRadioMC X--> East // Y--> North
            _x,_y,_z = det.get_relative_position(st_id,ch_id)
            positions = np.vstack( (positions,[_y,-1*_x, abs_pos[2] + _z ]) )
    return positions[1:]

def generate_antenna_list(antenna_grid):
    ## Format: AntennaPosition = 0 -1000 273500 ch0
    with open('SIM.list', 'w') as f:
        for antenna_id,antenna_pos in enumerate(antenna_grid):
            f.write(f'AntennaPosition = '
            +f'{antenna_pos[0]/units.cm:>.1f}\t'
            +f'{antenna_pos[1]/units.cm:>.1f}\t'
            +f'{antenna_pos[2]/units.cm:.0f}\t'
            +f'ch{antenna_id}\n')

def generate_cosika_steering():
    pass       
## TODO: create atmosphere file
def generate_atmosphere():
    pass

if __name__ == "__main__":
    single_21 = generate_single_rnog('RNO_season_2021.json',21)
    generate_antenna_list(single_21)
    