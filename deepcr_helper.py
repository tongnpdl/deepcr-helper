import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
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
def read_antenna_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = np.array([ line.strip().split() for line in lines])
        positions = np.array(lines[:,2:5],dtype=float)
    return positions

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

def generate_hdf5(simdir):
    pass

def get_pulsetime(efield,passband=None):
    if passband is None:
        E = efield.get_trace()
    else:
        E = efield.get_filtered_trace(passband)
    H = hilbert(E,axis=-1)
    time = efield.get_times()
    #pulse_time = time[np.argmax( np.sum(np.abs(H)**2,axis=0) )]  
    pulse_time = time[np.argmax( np.abs(H)**2,axis=-1)] 
    return pulse_time

def get_fluence(efield,passband=None): ## Integrated power per unit area
    if passband is None:
        E = efield.get_trace()
        #pulse_times = get_pulsetime(efield)
    else:
        E = efield.get_filtered_trace(passband)
        #pulse_times = get_pulsetime(efield,passband)
    H = hilbert(E,axis=-1)
    time = efield.get_times()
    freq = efield.get_frequencies()
    #df = freq[1]-freq[0]
    dt = time[1]-time[0]
    
    #masks = np.array([[ True if ( (t<pulse_times[i]+5*units.ns) and (t>=pulse_times[i]-5*units.ns) ) else False for t in time] for i in range(3)])
    #pw = np.array([ dt* np.sum(np.abs(H[i][masks[i]])**2)for i in range(3)])
    pw = np.array([ dt* np.sum(np.abs(H[i])**2)for i in range(3)])  * (1/376.73 * units.farad / units.s) ## add ice dielectric constant
    return pw

def get_total_fluence(efield,passband=None):
    if passband is None:
        E = efield.get_trace()
    else:
        E = efield.get_filtered_trace(passband)
    H = hilbert(E,axis=-1)
    time = efield.get_times()
    dt = time[1]-time[0]
    H2 = np.abs(H)**2 
    pw = np.sum(H2) * dt * (1/376.73 * units.farad / units.s)
    return pw

def get_phase_constant(efield,passband=None):
    if passband is None:
        pulse_t = get_pulsetime(efield)
    else:
        pulse_t = get_pulsetime(efield,passband)
    wf_fft = efield.get_frequency_spectrum()
    ff = efield.get_frequencies()
    tt = efield.get_times()
    t0 = tt[0]
    shifted_fft = wf_fft * np.array([np.exp(2.j*np.pi*ff*(pt - t0)) for pt in pulse_t])
    #intitial_phase = (shifted_fft/np.abs(shifted_fft))[0]
    #sum_fft = np.sum(shifted_fft,axis=-1)
    #phase_constant = -np.angle(np.sum(shifted_fft,axis=-1))
    phase_constant = np.exp( 1.j*np.angle(np.sum(shifted_fft,axis=-1)))
    return phase_constant

if __name__ == "__main__":
    single_21 = generate_single_rnog('RNO_season_2021.json',21)
    generate_antenna_list(single_21)
    