import os
from glob import glob
import numpy as np
from scipy.constants import speed_of_light
from NuRadioReco.utilities import units

c = (speed_of_light * units.m/units.s)
environ = os.environ    
#input_dir = os.path.join(environ["SIM_DIR"],"original_geant4_inputfiles")
input_dir = '/data/user/npunsuebsay/sim/Proton_1e15_0_0_radius1m/original_geant4_inputfiles/'
#output_dir = os.path.join(environ["SIM_DIR"],"input_geant4")
output_dir = '/data/user/npunsuebsay/sim/Proton_1e15_0_0_mode5_10GeV_w10/input_geant4/'
particle_files = glob(os.path.join(input_dir,"*.txt"))

#################################################################   
# Units are: GeV cm ns
# Number of particles in original file (not weighted): 40043.0
# Format: part_id px(GeV) py(GeV) pz(GeV) x(cm) y(cm) z(cm) t(ns) weight

#################################################################
# This file contains a function creating a dictionary, maping the Corsika particle ID's to their masses.
# Unit of mass is GeV/c^2.

def get_particle_masses():
    particle_masses = {}
    particle_masses[1] = 0.
    particle_masses[2] = .000511
    particle_masses[3] = .000511
    particle_masses[5] = .105658
    particle_masses[6] = .105658
    particle_masses[7] = .134977
    particle_masses[8] = .139570
    particle_masses[9] = .139570
    particle_masses[10] = .49761
    particle_masses[11] = .493677
    particle_masses[12] = .493677
    particle_masses[13] = .939565
    particle_masses[14] = .938272
    particle_masses[15] = .938272
    particle_masses[16] = .49761
    particle_masses[17] = .547862
    particle_masses[18] = 1.11568
    particle_masses[19] = 1.18937
    particle_masses[20] = 1.192642
    particle_masses[21] = 1.197449
    particle_masses[22] = 1.31486
    particle_masses[23] = 1.32171
    particle_masses[24] = 1.67245
    particle_masses[25] = .939565
    particle_masses[26] = 1.115683
    particle_masses[27] = 1.18937
    particle_masses[28] = 1.192642
    particle_masses[29] = 1.197449
    particle_masses[30] = 1.31486
    particle_masses[31] = 1.32171
    particle_masses[32] = 1.67245
    particle_masses[201] = 1.87561 # This is deuteron (ID = 100 x A + Z)
    particle_masses[301] = 2.80892 # This is triton (ID = 100 x A + Z)
    particle_masses[302] = 2.80941 # This is helium-3 nucleus (ID = 100 x A + Z)
    particle_masses[402] = 3.7284 # This is helium nucleus (ID = 100 x A + Z)
    return particle_masses
particle_masses = get_particle_masses()

def greenland_density_profile(x):
    density = np.piecewise(x, [x <= 14.9*units.m, x > 14.9*units.m],
                           [lambda x : 0.917-0.594*np.exp(-1*x/30.8*units.m) ,
                            lambda x : 0.917-0.367*np.exp(-1*(x-14.9*units.m)/40.5*units.m)] )  
    return density *units.g/units.cm**3
def southpole_density_profile(x):
    density = 0.917 - (0.917-0.359)*np.exp(-1.9*x/(100*units.m))
    return density *units.g/units.cm**3
def greenland_integral(x):
    integral = np.piecewise(x, [x <= 14.9*units.m, x > 14.9*units.m],
                           [lambda x : (x*0.917*units.g/units.cm**3) + (30.8*units.m)*(0.594*units.g/units.cm**3)*(np.exp(-1*x/(30.8*units.m))-1.) ,
                            lambda x : (14.9*units.m*0.917*units.g/units.cm**3) + (30.8*units.m*0.594*units.g/units.cm**3)*(np.exp(-1*14.9*units.m/(30.8*units.m))-1.)
                            + (0.917*units.g/units.cm**3)*(x-14.9*units.m)+(40.5*units.m*0.367*units.g/units.cm**3)*(np.exp(-1*(x-14.9*units.m)/(40.5*units.m))-1.)] )
    return integral
def southpole_integral(x):
    integral =  (x*0.917*units.g/units.cm**3) + (100*units.m/1.9)*(0.917-0.359)*(units.g/units.cm**3)*(np.exp(-1.9*x/(100*units.m))-1.)
    return integral

def modify_particle_line(particle_line,delta = 1/10.,mode = 1,ice_model='southpole'):
    new_particle_line = ''

    particle_elements = particle_line.split()
    particle_id  = int(particle_elements[0])
    particle_px = float(particle_elements[1])
    particle_py = float(particle_elements[2])
    particle_pz = float(particle_elements[3])
    particle_x = float(particle_elements[4])
    particle_y = float(particle_elements[5])
    particle_z = float(particle_elements[6])
    particle_t = float(particle_elements[7])
    particle_w = float(particle_elements[8])

    particle_p = np.array([particle_px,particle_py,particle_pz])
    particle_E = np.sqrt(np.sum(particle_p**2) + particle_masses[particle_id]**2) ## in GeV

    dz = 1*units.cm
    z = np.arange(0,20*units.m,dz)
    if ice_model == 'greenland':
        density_profile = greenland_density_profile(z)
    elif ice_model == 'southpole':
        density_profile = southpole_density_profile(z)
    slant_depth = np.cumsum(density_profile)*dz


    if mode == 1:
        if particle_id in [1]:
            new_particle_line = particle_line           
    elif mode == 2:
        if particle_id in [1,2,3]:
            particle_p = np.array([particle_px, particle_py,particle_pz])
            particle_m = particle_masses[particle_id]
            particle_E2 = particle_m**2 + np.sum(particle_p**2)
            initial_position = np.array([particle_x,particle_y,particle_z])
            unit_p = particle_p/np.linalg.norm(particle_p)

            ##### changed routine from E -> delta E to P -> delta P
            # new_E2 = (delta**2) * particle_E2
            # new_p = np.sqrt(new_E2 - particle_m**2) * unit_p
            # new_w = particle_w / delta
            #####

            new_p = particle_p * delta
            new_E = np.sqrt(np.sum(new_p**2) + particle_m**2)
            fractionE = new_E / particle_E2**0.5 
            # new_w = particle_w / fractionE ## increaded weight according to changed energy caused stronger peak AMP
            new_w = particle_w / delta


            rho_ice = 0.323
            x0 = 36.08/rho_ice 
            # dx = x0*np.log(1/delta)
            dx = x0*np.log(1/fractionE)
            dt = dx / (speed_of_light*1e-7)
            
            new_t = particle_t + dt
            new_position = initial_position + dx*unit_p
            new_x,new_y,new_z = new_position
            new_particle_line = f"{particle_id} {new_p[0]:.9e} {new_p[1]:.9e} {new_p[2]:.9e} {new_x:.9e} {new_y:.9e} {new_z:.9e} {new_t:.9e} {new_w}\n"
    elif mode == 3:
        if (particle_E > 10) and particle_id in [1,2,3]:
            new_particle_line = particle_line
    elif mode == 4:
        ## energy cut with particle selection
        if (particle_E > 10) and (particle_id in [1,2,3]):
            new_E = delta* particle_E
            particle_m = particle_masses[particle_id]
            new_p2 = new_E**2 - particle_m**2
            new_p = particle_p / np.linalg.norm(particle_p) * np.sqrt(new_p2)
            new_w = particle_w / delta
            new_particle_line = f"{particle_id} {new_p[0]:.9e} {new_p[1]:.9e} {new_p[2]:.9e} {particle_x:.9e} {particle_y:.9e} {particle_z:.9e} {particle_t:.9e} {new_w}\n"
    elif mode == 5:
        ## energy cut with particle selection
        if (particle_E > 10) and (particle_id in [1,2,3]):
            new_E = delta* particle_E
            particle_m = particle_masses[particle_id]
            new_p2 = new_E**2 - particle_m**2
            unit_p = particle_p / np.linalg.norm(particle_p)
            cosine = np.abs(unit_p[1])
            new_p = unit_p * np.sqrt(new_p2)

            vector_0 = np.array([particle_x,particle_y,particle_z]) *units.cm
            x0 = 36.08*units.g/units.cm**2
            dSlant = x0*np.log(1./delta) *cosine
            
            # dDepth = z[np.argwhere(slant_depth > dSlant)[0]][0]
            dDepth = np.interp(dSlant,slant_depth,z)

            dLength = dDepth /cosine
            dVector = unit_p * dLength
            vector_1 = (vector_0 + dVector) /units.cm
            new_w = particle_w / delta
            delta_t = (dLength / c)
            new_t = particle_t + delta_t
            print(f'delta Slant:{dSlant/(units.g/units.cm**2):.9e} g/cm2 \t delta Length {dLength/units.cm:.9e} cm delta_t {delta_t:.9e}')
            new_particle_line = f"{particle_id} {new_p[0]:.9e} {new_p[1]:.9e} {new_p[2]:.9e} {vector_1[0]:.9e} {vector_1[1]:.9e} {vector_1[2]:.9e} {new_t:.9e} {    new_w}\n"

    return new_particle_line
for path2input in particle_files:
    filename = os.path.basename(path2input)
    path2output = os.path.join(output_dir,filename)
    # lines = []
    with open(path2input,'r') as file:
        lines = file.readlines()
        headers = []
        particle_lines = []
        for line in lines:
            if line[0] == '#': # header
                headers.append(line)
            else:
                modified_line = modify_particle_line(line,mode=5,delta=1./10.)
                if len(modified_line) > 0:
                    particle_lines.append(modified_line)
    if len(particle_lines) > 0:
        with open(path2output,'w') as outfile:
            for line in headers:
                outfile.write(line)
            for line in particle_lines:
                outfile.write(line)
    # break
             