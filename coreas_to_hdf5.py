from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import configparser
import sys
import io
import re
import argparse
import h5py
import math
from scipy import integrate

conversion_factor_integrated_signal = 2.65441729e-3 * 6.24150934e18  # to convert V**2/m**2 * s -> J/m**2 -> eV/m**2
conversion_fieldstrength_cgs_to_SI = 2.99792458e4

VERSION_MAJOR = 0
VERSION_MINOR = 5

def gaisser_hillas(X, N, X0, Xmax, p0, p1=0, p2=0):
    l = p0 + p1 * X + p2 * X ** 2
    power = (Xmax - X0) / l

    if np.sum(l < 0):
        # print("Some l are negative")
            return np.inf

    if np.sum(power < 0):
        # print("Some power are negative")
            return np.inf

    if np.sum(power > 100):
        return np.inf

    result = np.zeros_like(X)
    mask = (X - X0) >= 0
    result[mask] = N * ((X[mask] - X0) / (Xmax - X0)) ** (power[mask]) * np.exp((Xmax - X[mask]) / l[mask])
    result = np.nan_to_num(result)

    return result


def clean_corrupt_long_profiles(dE_data):
    """
    The longitudinal profile provided by CORSIKA (DAT*.long file)
    can have unphysical spikes. Reject any sample which is 20 % higher
    than the previous. This should not happening for a fine sampled profile
    """

    yy = dE_data[9][:-2]  # Skip two last samples. They can contain energy deposit in ground plane
    n = 0
    while True:
        yy_max = np.amax(yy)
        yy_max2 = np.amax(yy[yy < yy_max])

        if yy_max > 1.2 * yy_max2:
            mask = yy < yy_max
            dE_data = dE_data[:, np.append(mask, [True, True])]  # include two last bins
            yy = yy[mask]
            n += 1
        else:
            break

    if n:
        print("Reject {} samples in longitudinal profile".format(n))

    return dE_data


def fit_gaisser_hillas(xx, yy):
    # fit Gaisser Hillas to longitudinal profile anyway (fir in CORSIKA not accurate for high zenith angle)
    # print("Performing Gaisser-Hillas fit on longitudinal profile ...")

    popt, pcov = optimize.curve_fit(gaisser_hillas, xx, yy, p0=[yy.max(), 0, xx[yy.argmax()], 20], maxfev=3000)
    popt2, pcov = optimize.curve_fit(gaisser_hillas, xx, yy, p0=[popt[0], popt[1], popt[2], popt[3], 0, 0], maxfev=3000)

    return popt2, pcov


def read_input_file(hdf5_file, inp_file):

    if not isinstance(inp_file, io.IOBase):
        inp_file = open(inp_file, "r")

    inp_dict = {}
    for i, line in enumerate(inp_file.readlines()):
        elements = line.strip().split()
        inp_dict[elements[0]] = elements[1:]
    inp_file.close()

    f_h5_inputs = hdf5_file.create_group("inputs")

    # fill general attributes from inp file
    f_h5_inputs.attrs["RUNNR"] = int(inp_dict["RUNNR"][0])
    f_h5_inputs.attrs["EVTNR"] = int(inp_dict["EVTNR"][0])
    f_h5_inputs.attrs["PRMPAR"] = int(inp_dict["PRMPAR"][0])
    f_h5_inputs.attrs["ERANGE"] = np.array([float(inp_dict["ERANGE"][0]), float(inp_dict["ERANGE"][1])])
    f_h5_inputs.attrs["THETAP"] = np.array([float(inp_dict["THETAP"][0]), float(inp_dict["THETAP"][1])])
    f_h5_inputs.attrs["PHIP"] = np.array([float(inp_dict["PHIP"][0]), float(inp_dict["PHIP"][0])])
    f_h5_inputs.attrs["ECUTS"] = np.array([float(inp_dict["ECUTS"][0]), float(inp_dict["ECUTS"][1]), float(inp_dict["ECUTS"][2]), float(inp_dict["ECUTS"][3])])
    try:
        f_h5_inputs.attrs["THIN"] = np.array([float(inp_dict["THIN"][0]), float(inp_dict["THIN"][1]), float(inp_dict["THIN"][2])])
        f_h5_inputs.attrs["THINH"] = np.array([float(inp_dict["THINH"][0]), float(inp_dict["THINH"][1])])
    except KeyError:
        pass
    f_h5_inputs.attrs["OBSLEV"] = float(inp_dict["OBSLEV"][0])
    f_h5_inputs.attrs["MAGNET"] = np.array([float(inp_dict["MAGNET"][0]), float(inp_dict["MAGNET"][1])])

    try:
        f_h5_inputs.attrs["ATMOD"] = int(inp_dict["ATMOD"][0])
    except KeyError:
        f_h5_inputs.attrs["ATMOD"] = 1  # CORSIKA default, U.S standard by Linsley

    if "ATMFILE" in inp_dict:
        atm_entry = str(inp_dict["ATMFILE"][0])
        print("ATMFILE was set in *inp, store ATMOD = -1")
        f_h5_inputs.attrs["ATMFILE"] = atm_entry
        f_h5_inputs.attrs["ATMOD"] = -1


def read_reas_file(hdf5_file, reas_file):

    if not isinstance(reas_file, io.IOBase):
        reas_file = open(reas_file, "r")

    lreas = reas_file.readlines()
    tmp = u"[CoREAS]\n"
    for i, line in enumerate(lreas):
        lreas[i] = line.strip().split(";")[0].strip()
        tmp += lreas[i] + "\n"
    reas_file.close()

    configParser = configparser.RawConfigParser()
    configParser.optionxform = str
    tmp2 = io.StringIO(tmp)
    configParser.read_file(tmp2)
    items = configParser.items("CoREAS")

    f_h5_reas = hdf5_file.create_group("CoREAS")

    # store content of reas file as attributes
    for key, value in items:
        if key in ["CoreCoordinateNorth", "CoreCoordinateWest", "CoreCoordinateVertical",
                   "TimeResolution", "AutomaticTimeBoundaries", "TimeLowerBoundary",
                   "TimeUpperBoundary", "ResolutionReductionScale",
                   "GroundLevelRefractiveIndex", "EventNumber", "RunNumber", "GPSSecs",
                   "GPSNanoSecs", "CoreEastingOffline", "CoreNorthingOffline", "CoreVerticalOffline",
                   "RotationAngleForMagfieldDeclination", "ShowerZenithAngle", "ShowerAzimuthAngle",
                   "PrimaryParticleEnergy", "PrimaryParticleType", "DepthOfShowerMaximum", "DistanceOfShowerMaximum",
                   "MagneticFieldStrength", "MagneticFieldInclinationAngle"]:
            if key not in ["EventNumber", "RunNumber", "GPSSecs", "GPSNanoSecs", "PrimaryParticleType"]:
                f_h5_reas.attrs[key] = float(value)
            else:
                f_h5_reas.attrs[key] = int(value)
        else:
            f_h5_reas.attrs[key] = value


def read_longitudinal_profile(hdf5_file, long_file):

    if not isinstance(long_file, io.IOBase):
        long_file = io.open(long_file, "r", encoding="UTF-8")

    f_h5_long = hdf5_file.create_group("atmosphere")

    lines = long_file.readlines()
    n_steps = int(lines[0].rstrip().split()[3])

    def my_replace(m):
        return str(m.group(0)[0]) + " -"

    n_data_str = io.StringIO()
    n_data_str.writelines(lines[2:(n_steps + 2)])
    n_data_str.seek(0)

    n_data = np.genfromtxt(n_data_str)
    for i, line in enumerate(lines):
        lines[i] = re.sub("[0-9]-", my_replace, line)

    dE_data_str = io.StringIO()
    dE_data_str.writelines(lines[(n_steps + 4):(2 * n_steps + 4)])
    dE_data_str.seek(0)
    dE_data = np.genfromtxt(dE_data_str)
    data_set = f_h5_long.create_dataset("NumberOfParticles", n_data.shape, dtype="f")
    data_set[...] = n_data
    data_set.attrs['comment'] = "The collumns of the data set are: DEPTH, GAMMAS, POSITRONS, ELECTRONS, MU+, MU-, HADRONS, CHARGED, NUCLEI, CHERENKOV"

    data_set = f_h5_long.create_dataset("EnergyDeposit", dE_data.shape, dtype="f")
    data_set[...] = dE_data
    data_set.attrs['comment'] = "The collumns of the data set are: DEPTH, GAMMA, EM IONIZ, EM CUT, MU IONIZ, MU CUT, HADR IONIZ, HADR CUT, NEUTRINO, SUM"

    # read out hillas fit
    hillas_parameter = []
    for line in lines:
        if bool(re.search("PARAMETERS", line)):
            hillas_parameter = [float(x) for x in line.split()[2:]]  # strip aways 'PARAMETER', '='
    f_h5_long.attrs['Gaisser-Hillas-Fit'] = hillas_parameter

    long_file.close()


def read_atm_file(hdf5_file, atm_file_path):

    heights, refractive_index = np.genfromtxt(atm_file_path, unpack=True, skip_header=6)
    refractive_index_profile = np.array([heights, refractive_index]).T

    f_h5_atm = hdf5_file.create_group("atmosphere_model")

    data_set = f_h5_atm.create_dataset(
        "RefractiveIndexProfile", refractive_index_profile.shape, dtype="f")
    data_set[...] = refractive_index_profile


    atm_file = open(atm_file_path, "rb")
    lines = atm_file.readlines()

    # skip first entry (0), conversion cm -> m
    h = np.array(lines[1].strip(b"\n").split()[1:], dtype=float) / 100
    a = np.array(lines[2].strip(b"\n").split(), dtype=float) * 1e4
    b = np.array(lines[3].strip(b"\n").split(), dtype=float) * 1e4
    c = np.array(lines[4].strip(b"\n").split(), dtype=float) * 1e-2
    atm_file.close()

    f_h5_atm.attrs["h"] = h
    f_h5_atm.attrs["a"] = a
    f_h5_atm.attrs["b"] = b
    f_h5_atm.attrs["c"] = c


def read_antenna_data(hdf5_file, antenna_list, antenna_folder, skip_antenna_pattern=None, antenna_name_prefix="raw_", hdf5_group_name="observers"):


    if hdf5_group_name not in hdf5_file:
        observers = hdf5_file.create_group(hdf5_group_name)
    else:
        observers = hdf5_file[hdf5_group_name]

    for line in antenna_list:
        ll = line.strip().split()
        antenna_position = ll[2:5]
        antenna_label = ll[5]

        #skip_antenna_pattern = "\+"
        if skip_antenna_pattern is not None and \
		re.search(skip_antenna_pattern, antenna_label) is None:
            continue

        antenna_file = os.path.join(antenna_folder, f"{antenna_name_prefix}{antenna_label}.dat")

        data = np.genfromtxt(antenna_file)
        data_set = observers.create_dataset(antenna_label, data.shape, dtype=float)
        data_set[...] = data

        data_set.attrs['position'] = np.array(antenna_position, dtype=float)
        data_set.attrs['name'] = f"{antenna_name_prefix}{antenna_label}"

        # there seems to be a problem when storing a list of unicodes in an attribute (is the case for python3). followed the fix from:
        # https://github.com/h5py/h5py/issues/289
        if (len(ll) > 6):
            data_set.attrs['additional_arguments'] =  [a.encode('utf8') for a in ll[6:]]


def write_coreas_hdf5_file(reas_filename, output_filename, f_h5=None, atm_file=None, ignore_atm_file=False):

    # create hdf5 file
    if f_h5 is None:
        f_h5 = h5py.File(output_filename, "w", driver="core")  # "core": will only write file to disk ones it is closed
    f_h5.attrs["version_major"] = VERSION_MAJOR
    f_h5.attrs["version_minor"] = VERSION_MINOR

    # read all parameter from the SIM??????.reas file
    read_reas_file(f_h5, reas_filename)

    if args.store_input_file_in_hdf5:
        # print error message in case CorsikaParameterFile was not specified in the .reas file
        inp_filename = f_h5['CoREAS'].attrs['CorsikaParameterFile']
        if len(inp_filename) == 0:
            sys.exit("CorsikaParameterFile was not specified in the .reas file, aborting!")
        finp = open(os.path.join(os.path.dirname(reas_filename), inp_filename), "r")

        # read cards from .inp file and stores them into "inputs" group in f_h5 file
        read_input_file(f_h5, finp)

        if "ATMFILE" in f_h5['inputs'].attrs:
            print("Found following path for ATMFILE: %s" %
                f_h5['inputs'].attrs["ATMFILE"])
            atm_file_full_path = f_h5['inputs'].attrs["ATMFILE"]
            atm_file_name = os.path.basename(atm_file_full_path)

            if atm_file is not None:
                print("Use user specified ATMFILE: %s" % atm_file)
                read_atm_file(f_h5, atm_file)
            elif os.path.exists(atm_file_full_path):
                print("Read ATMFILE from: %s" % atm_file_full_path)
                read_atm_file(f_h5, atm_file_full_path)
            elif os.path.exists(atm_file_name):
                print("Read ATMFILE from: %s" % atm_file_name)
                read_atm_file(f_h5, atm_file_name)
            elif ignore_atm_file:
                print("Do not read any ATMFILE")
                pass
            else:
                sys.exit("Could not find the atm file in %s or %s. Full stop!" %
                        (atm_file_full_path, atm_file_name))

    if args.store_long_file_in_hdf5:
        # read in long file
        long_filename = "DAT" + os.path.splitext(os.path.basename(reas_filename))[0][-6:] + ".long"
        longinp = io.open(os.path.join(os.path.dirname(reas_filename), long_filename), "r", encoding="UTF-8")
        read_longitudinal_profile(f_h5, longinp)

    # read in antenna data
    listfile = open(os.path.splitext(reas_filename)[0] + ".list", "r")
    number = os.path.splitext(os.path.basename(reas_filename))[0][3:]
    with open(os.path.splitext(reas_filename)[0] + ".list", "r") as listfile:
        antennalist = [item for item in listfile.readlines() if item != "\n"]  # skip empty lines
        antennalist = np.array([item for item in antennalist if not item.startswith("#")])  # skip comments

    antenna_folder = os.path.join(os.path.dirname(reas_filename), "SIM%s_coreas" % number)
    read_antenna_data(f_h5['CoREAS'], antennalist, antenna_folder,antenna_name_prefix='raw_ch')

    if args.add_faerie_simulation:
        antenna_folder2 = os.path.join(os.path.dirname(reas_filename), "SIM%s_geant" % number)
        read_antenna_data(f_h5['CoREAS'], antennalist, antenna_folder2, hdf5_group_name="observers_geant",antenna_name_prefix='Antenna')

    return f_h5


def write_highlevel_attributes(f_h5_hl, f_h5):

    # compute high level quantities from longitudinal profile
    dE_data = f_h5["atmosphere"]["EnergyDeposit"]
    dE_data = np.array(dE_data).T
    for i in range(1, 10):  # convert units to eV
        dE_data[i] = dE_data[i] * 1e9

    Egamma = np.sum(dE_data[1])
    Eem_ion = np.sum(dE_data[2])
    Eem_cut = np.sum(dE_data[3])
    Emu_ion = np.sum(dE_data[4])
    Emu_cut = np.sum(dE_data[5])
    Ehad_ion = np.sum(dE_data[6])
    Ehad_cut = np.sum(dE_data[7])
    Eneutrino = np.sum(dE_data[8])
    Esum = np.sum(dE_data[9])
    Eem = Egamma + Eem_ion + Eem_cut
    Eem_atm = np.sum(dE_data[1][:-2]) + np.sum(dE_data[2][:-2]) + np.sum(dE_data[3][:-2])
    Einv = Emu_cut + Emu_ion + Eneutrino
    Ehad = Ehad_cut + Ehad_ion

    f_h5_hl.attrs["Eem"] = Eem
    f_h5_hl.attrs["Eem_atm"] = Eem_atm
    f_h5_hl.attrs["Einv"] = Einv
    f_h5_hl.attrs["Ehad"] = Ehad
    f_h5_hl.attrs["Egamma"] = Egamma
    f_h5_hl.attrs["Eem_ion"] = Eem_ion
    f_h5_hl.attrs["Eem_cut"] = Eem_cut
    f_h5_hl.attrs["Emu_ion"] = Emu_ion
    f_h5_hl.attrs["Emu_cut"] = Emu_cut
    f_h5_hl.attrs["Ehad_ion"] = Ehad_ion
    f_h5_hl.attrs["Ehad_cut"] = Ehad_cut
    f_h5_hl.attrs["Eneutrino"] = Eneutrino

    # From offline, calorimetric energy
    nrow_ground_plane = 2  # SLANT = 2, NonSLANT = 1
    muonFraction = 0.575
    hadronFraction = 0.  #0.261 < -- Tanguy thinks this should be zero
    Eelec = np.sum(dE_data[9, :-nrow_ground_plane]) - np.sum(dE_data[8, :-nrow_ground_plane]) - \
        muonFraction * np.sum(dE_data[5, :-nrow_ground_plane]) - \
        hadronFraction * np.sum(dE_data[7: -nrow_ground_plane])

    hadronGroundFraction = 0.390
    Eelec += np.sum((1 - hadronGroundFraction) * \
            dE_data[7, -nrow_ground_plane:] + dE_data[6, -nrow_ground_plane:] + \
            dE_data[4, -nrow_ground_plane:] + dE_data[2, -nrow_ground_plane:] + \
            dE_data[3, -nrow_ground_plane:] + dE_data[1, -nrow_ground_plane:])
    f_h5_hl.attrs["Eelec"] = Eelec  # calorimetric energy?!

    if 'Gaisser-Hillas-Fit' in f_h5["atmosphere"].attrs:
        f_h5_hl.attrs['Gaisser-Hillas-Fit'] = f_h5["atmosphere"].attrs['Gaisser-Hillas-Fit']

    if 1:
        # Skip two last samples. They can contain energy deposit in ground plane
        dE_data = clean_corrupt_long_profiles(dE_data)
        depths = dE_data[0, :-2]
        energy_deposit = dE_data[9, :-2]
        popt, pcov = fit_gaisser_hillas(depths, energy_deposit)
        f_h5_hl.attrs["gaisser_hillas_dEdX"] = popt


def write_coreas_highlevel_file(output_filename, f_h5, args, f_h5_sephl=None):

    # create file
    if f_h5_sephl is None:
        f_h5_sephl = h5py.File(output_filename, "w", driver="core")

    # Copy CoREAS information into highlevel file (only attributes)
    f_h5_hl_coreas = f_h5_sephl.create_group("CoREAS")
    for (attr, value) in f_h5['CoREAS'].attrs.items():
        f_h5_hl_coreas.attrs[attr] = value

    # Copy input information into highlevel file (only attributes)
    f_h5_hl_inputs = f_h5_sephl.create_group("inputs")
    for (attr, value) in f_h5['inputs'].attrs.items():
        f_h5_hl_inputs.attrs[attr] = value

    # Copy atmosphere_model information into highlevel file (only attributes)
    f_h5_hl_atmosphere_model = f_h5_sephl.create_group("atmosphere_model")
    for (attr, value) in f_h5['atmosphere_model'].attrs.items():
        f_h5_hl_atmosphere_model.attrs[attr] = value

    f_h5_hl = f_h5_sephl.create_group("highlevel")
    write_highlevel_attributes(f_h5_hl, f_h5)
    f_h5_inputs = f_h5['inputs']
    Bx, Bz = f_h5_inputs.attrs["MAGNET"]
    zenith = np.deg2rad(f_h5_inputs.attrs["THETAP"][0])
    azimuth = 3 * np.pi / 2. + np.deg2rad(f_h5_inputs.attrs["PHIP"][0])  # convert to auger cs
    azimuth = rdhelp.get_normalized_angle(azimuth)
    B_inclination = np.arctan2(Bz, Bx)
    B_declination = 0

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5
    magnetic_field_vector = rdhelp.spherical_to_cartesian(B_inclination + np.pi / 2, B_declination + np.pi * 0.5)  # in auger cooordinates north is + 90 deg

    ctrans = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)

    f_h5_hl.attrs['zenith'] = zenith
    f_h5_hl.attrs['azimuth'] = azimuth
    f_h5_hl.attrs['energy'] = f_h5_inputs.attrs["ERANGE"][0] * 1e9  # in eV
    f_h5_hl.attrs['magnetic_field_inclination'] = B_inclination
    f_h5_hl.attrs['magnetic_field_declination'] = B_declination
    f_h5_hl.attrs['magnetic_field_strength'] = B_strength
    # f_h5_hl.attrs['magnetic_field_vector'] = magnetic_field_vector


    # check for simulation of different observation planes
    f_h5_reas = f_h5["CoREAS"]
    observers = f_h5_reas["observers"]
    names = list(observers.keys())
    obs_values = list(observers.values())

    # if additional arguments were set use then to determine observeration planes
    # otherwise use names
    if "additional_arguments" in obs_values[0].attrs.keys():
        # concetinate
        planes = np.array(["_".join([str(xx).strip("b'").strip("'") for xx in x.attrs["additional_arguments"]]) for x in obs_values])
        planes_unique = np.unique(planes, axis=0)
    else:
        if len(names[0].split("_")) < 4:
            planes = np.array(["na_na"] * len(names))
        elif len(names[0].split("_")) == 4:
            planes = np.array([x.rstrip().split("_")[3] + "_na" for x in names])
        elif len(names[0].split("_")) >= 4:
            planes = np.array([x.rstrip().split("_")[3] + "_" + x.rstrip().split("_")[4] for x in names])
        planes_unique = np.unique(planes)

    if args.use_vB_vvB_polarization:
        print("Traces are stored and all relevent quantities are determined in vxB, vxvxB, and v polarization!")
    else:
        print("Traces are stored and all relevent quantities are determined in x, y, and z polarization!")

    index = np.array(range(len(observers.keys())))

    for plane in planes_unique:
        observation_height = f_h5_reas.attrs["CoreCoordinateVertical"] * 1e-2  # conversion to m
        try:
            observation_height = float(plane.split("_")[0])
        except:
            pass

        mask_plane = (planes == plane)
        nantennas = np.sum(mask_plane)
        print("\t%i antennas in observation plane %s" % (nantennas, plane))

        antenna_position = np.zeros([nantennas, 3])
        antenna_heights = np.zeros(nantennas)
        energy_fluence = np.zeros([nantennas, 3])
        peak_time = np.zeros([nantennas, 3])
        peak_amplitude = np.zeros([nantennas, 3])
        peak_amplitude_total = np.zeros(nantennas)
        polarization_vector = np.zeros([nantennas, 3])
        frequency_slope = np.zeros([nantennas, 3, 2])
        stokes_parameter = np.zeros([nantennas, 4])

        itnorm = None

        times_filtered = []
        traces_filtered = []
        slicing_boundaries = []
        slicing_method = ""
        names = []
        for i, j in enumerate(index[mask_plane]):
            # antenna_position[j] = (observers.values()[mask_plane][j].split(" ")[2:5])
            position = obs_values[j].attrs["position"]
            name = obs_values[j].attrs["name"]
            names.append(name)
            # name = lines[mask_plane][j].split(" ")[5]
            try:
                antenna_heights[i] = float(name.split("_")[4])  # hight might not be part of the antenna name, only relevant for simulations with observers at multiple heights
            except:
                pass

            # read slice
            if("additional_arguments" in obs_values[j].attrs.keys()):
                line_split = obs_values[j].attrs["additional_arguments"]
                if len(line_split) == 3:
                    slicing_method = line_split[0]
                    slicing_boundaries.append([float(line_split[1]), float(line_split[2])])
                elif len(line_split) == 6:
                    slicing_method = [line_split[0], line_split[3]]
                    slicing_boundaries.append([[float(line_split[1]), float(line_split[2])],
                                               [float(line_split[4]), float(line_split[5])]])
                else:
                    sys.exit("Length of additional arguments is wrong")

            # convert CORSIKA to AUGER coordinates (AUGER y = CORSIKA x, AUGER x = - CORSIKA y
            # TODO: add roation to correct north
            data = np.copy(obs_values[j])
            data[:, 1], data[:, 2] = -obs_values[j][:, 2], obs_values[j][:, 1]

            # convert to SI units
            data[:, 1] *= conversion_fieldstrength_cgs_to_SI
            data[:, 2] *= conversion_fieldstrength_cgs_to_SI
            data[:, 3] *= conversion_fieldstrength_cgs_to_SI

            # convert CORSIKA to AUGER coordinates (AUGER y = CORSIKA x, AUGER x = - CORSIKA y; cm to m
            antenna_position[i, 0], antenna_position[i, 1], antenna_position[i, 2] = -position[1] / 100., position[0] / 100., position[2] / 100.

            if np.sum(np.isnan(data[:, 1:4])):
                print("ERROR in antenna %j, time trace contains NaN" % j)
                import sys
                sys.exit(-1)

            if args.use_vB_vvB_polarization:
                # transform time traces into vxB-vxvxB frame
                data[:, 1:4] = ctrans.transform_to_vxB_vxvxB(data[:, 1:4])

            # needs to be done because it is more precise for stepsizes of the order of 1e-10
            dlength = data.shape[0]
            tstep = data[1, 0] - data[0, 0]
            if (f_h5_reas.attrs['ResolutionReductionScale'] == 0):
                tstep = f_h5_reas.attrs['TimeResolution']
                data[:, 0] = tstep * np.arange(dlength) + data[0, 0]

            # add zeros to beginning and end of the trace to increase the frequency resolution (this is not a resampling)
            #n_samples = int(np.round(2048e-9 * 5 / tstep))
            n_samples = int(np.round(1 / tstep / (args.frequencyResolution * 1e3)))
            #increase number of samples to a power of two for FFT performance reasons
            n_samples = int(2 ** math.ceil(math.log(n_samples, 2)))

            n_start = (n_samples - len(data[:, 0])) // 2
            padded_trace = np.zeros((n_samples, 3))
            padded_trace[n_start:(n_start + len(data[:, 0]))] = data[:, 1:4]

            # get frequency spectrum
            spec = np.fft.rfft(padded_trace, axis=-2)

            # get new time and frequency binning
            ff = np.fft.rfftfreq(n_samples, tstep)  # frequencies in Hz
            tt = tstep * np.arange(n_samples) + data[0, 0]
            tt *= 1e9  #  time in ns

            # determine actual frequency resolution
            actualFrequencyResolution = ff[1] - ff[0]
            # print("Frequency resolution used for filtering is %f kHz." % (actualFrequencyResolution * 1e-3))

            # apply bandwidth cut
            window = np.zeros(len(ff))
            window[(ff >= args.flow * 1e6) & (ff <= args.fhigh * 1e6)] = 1
            filtered_spec = np.array([spec[..., 0] * window,
                                      spec[..., 1] * window,
                                      spec[..., 2] * window])

            # get filtered time series
            filt = np.fft.irfft(filtered_spec, n_samples, axis=-1)

            if args.store_traces:
                # assume that simulated time resolution is higher than a time resolution of tstep_resampled -> resampling traces
                tstep_resampled = 1. / (args.sampling_frequency * 1e9)
                n_resampled = int(np.floor(tstep / tstep_resampled * n_samples))
                filt_short = np.fft.irfft(filtered_spec, n_resampled, axis=-1)

                # renormalizing amplitude
                filt_short *= float(n_resampled / n_samples)

                # trim downsampled time series
                n = filt_short.shape[1]
                maximum_bin = np.argmax(np.abs(filt_short[0]))

                ishift = 0
                if (args.number_of_samples < n):
                    istart = maximum_bin - args.number_of_samples // 2
                    if (istart < 0):
                        istart = 0

                    # to store the times relative corretly
                    if itnorm is None:
                        itnorm = istart  # and ishift = 0
                    else:
                        ishift += istart - itnorm

                    filt_short = filt_short[:, istart:(istart + args.number_of_samples)]

                if(args.samples_before_pulse is not None):
                    maximum_bin = np.argmax(np.abs(filt_short[0]))
                    ishift += args.samples_before_pulse - maximum_bin
                    filt_short = np.roll(filt_short, args.samples_before_pulse - maximum_bin, axis=-1)

            # calcualte energy fluence and other observables
            hilbenv = np.abs(hilbert(filt, axis=1))
            peak_time[i] = tt[np.argmax(hilbenv, axis=1)]
            peak_amplitude[i] = np.max(hilbenv, axis=1)
            mag_hilbert = np.sum(hilbenv ** 2, axis=0) ** 0.5
            peak_amplitude_total[i] = np.max(mag_hilbert)
            peak_time_sum = tt[np.argmax(mag_hilbert)]
            # calculate energy density (signal window = 100ns)
            signal_window = 100.  # in ns (tt is ns)
            mask_signal_window = (tt > (peak_time_sum - signal_window / 2.)) & (tt < (peak_time_sum + signal_window / 2.))
            mask_noise = ~mask_signal_window
            u_signal = np.sum(filt[..., mask_signal_window] ** 2, axis=-1) * conversion_factor_integrated_signal * tstep
            # u_noise = np.sum(filt[..., mask_noise] ** 2, axis=-1) * conversion_factor_integrated_signal * tstep * np.sum(mask_signal_window) / np.sum(mask_noise)
            energy_fluence[i] = u_signal  # - u_noises

            if args.calculate_stokes_parameter:
                mask_stokes_window = (tt > (peak_time_sum - args.stokes_window / 2.)) & (tt < (peak_time_sum + args.stokes_window / 2.))
                E_vxB, E_vxvxB, E_v = filt[:, mask_stokes_window]
                E_vxB_hil, E_vxvxB_hil, E_v_hil = np.imag(hilbert(filt[:, mask_stokes_window], axis=1))

                # calculation from https://arxiv.org/pdf/1406.1355.pdf
                I_stokes = 1 / np.sum(mask_stokes_window) * np.sum((E_vxB ** 2 + E_vxB_hil ** 2 + E_vxvxB ** 2 + E_vxvxB_hil ** 2))
                Q_stokes = 1 / np.sum(mask_stokes_window) * np.sum((E_vxB ** 2 + E_vxB_hil ** 2 - E_vxvxB ** 2 - E_vxvxB_hil ** 2))
                U_stokes = 2 / np.sum(mask_stokes_window) * np.sum((E_vxB * E_vxvxB + E_vxB_hil * E_vxvxB_hil))
                V_stokes = 2 / np.sum(mask_stokes_window) * np.sum((E_vxB_hil * E_vxvxB - E_vxB * E_vxvxB_hil))

                # conversion to eV/m2 (this just matters if one cares about the absolut scale of the parameter)
                stokes_parameter[i] = np.array([I_stokes, Q_stokes, U_stokes, V_stokes]) * conversion_factor_integrated_signal * tstep

            # compute frequency slope in all three polarizations
            ff = np.fft.rfftfreq(len(tt), tstep) * 1e-6
            mask = (ff > args.flow) & (ff < args.fhigh)

            # Loop over three polarizations
            for iPol in range(3):
                # Fit slope
                mask2 = filtered_spec[iPol][mask] > 0
                if np.sum(mask2):
                    xx = ff[mask][mask2]
                    yy = np.log10(np.abs(filtered_spec[iPol][mask][mask2]))
                    z = np.polyfit(xx, yy, 1)
                    frequency_slope[i][iPol] = z

            polarization_vector[i] = rdhelp.get_polarization_vector_FWHM(filt)

            if args.store_traces:
                times_filtered.append(tstep_resampled * np.arange(ishift, ishift + len(filt_short[0])) + data[0, 0])
                traces_filtered.append(filt_short.T)

        # shift antenna positions to core position of observation level
        # compute translation in x and y
        core = np.array([-1 * f_h5_reas.attrs["CoreCoordinateWest"], f_h5_reas.attrs["CoreCoordinateNorth"], f_h5_reas.attrs["CoreCoordinateVertical"]]) * 1e-2
        r = np.tan(zenith) * (observation_height - core[2])
        deltax = np.cos(azimuth) * r
        deltay = np.sin(azimuth) * r
        antenna_position[..., 0] -= deltax
        antenna_position[..., 1] -= deltay
        if r > 1:
            print("\tshifting antenna positions by x = %.1f, y =  %.1f" % (deltax, deltay))

        # calculate radiation enrergy
        core[2] = observation_height
        station_positions_transformed = ctrans.transform_to_vxB_vxvxB(antenna_position, core=core)
        xx = station_positions_transformed[..., 0]
        yy = station_positions_transformed[..., 1]
        distances = (xx ** 2 + yy ** 2) ** 0.5
        tot_power = np.sum(energy_fluence, axis=1)

        if args.use_vB_vvB_polarization:
            polarisation = "vB_vvB"
        else:
            polarisation = "x_y"

        obsplanename = "obsplane_%s_%s" % (plane, polarisation)
        f_h5_obsplane = f_h5_hl.create_group(obsplanename)


        az = rdhelp.get_normalized_angle(np.round(np.rad2deg(np.arctan2(yy, xx))), degree=True)
        mask = (az == 90)
        if np.sum(mask) > 5:  # check if simulation is star pattern simultion
            dd = yy[mask]
            sortmask = np.argsort(dd)  # for the numerical integration, the datapoints needs to be sorted by distance
            dd = dd[sortmask]
            yy_tot = tot_power[mask][sortmask]
            y_int_num = integrate.trapz(yy_tot * dd, dd) * 2 * np.pi
            f_h5_obsplane.attrs['radiation_energy_1D'] = y_int_num
            print("\tcalculating radiation energy via 1 integration: %.4e eV" % (y_int_num))

        if args.compute_radiation_energy:
            if (np.sum(np.isin(np.arange(0, 360, 45.), np.unique(az))) == 8):
                import scipy.interpolate as intp
                func = intp.Rbf(xx, yy, tot_power, smooth=0, function='quintic')

                x_range = xx.max()
                y_range = yy.max()

                def bounds_y(x):
                    return [-np.sqrt(np.abs(np.square(x_range) - np.square(x))),
                             np.sqrt(np.abs(np.square(y_range) - np.square(x)))]

                def bounds_x():
                    return [-x_range, y_range]

                opts = {'epsabs': 1., 'epsrel': 0.5e-3, 'limit': 1000}
                radiation_energy_2D = integrate.nquad(func, [bounds_y, bounds_x], opts=opts)
                f_h5_obsplane.attrs['radiation_energy'] = radiation_energy_2D[0]
                print("\tcalculating radiation energy via 2D integration: %.4e eV" % (radiation_energy_2D[0]))

        pol_string = 'vxB, vxvxB, v polarizations' if args.use_vB_vvB_polarization else 'N, W, vertical polarizations'
        f_h5_obsplane.attrs['comment'] = 'Trace-related quantities are in %s and bandpass filtered to %.1f - %.1f MHz with a resolution of %.1f kHz' % \
                                         (pol_string, args.flow, args.fhigh, actualFrequencyResolution * 1e-3)
        f_h5_obsplane.attrs['frequency_low'] = args.flow * 1e6
        f_h5_obsplane.attrs['frequency_high'] = args.fhigh * 1e6
        f_h5_obsplane.attrs['frequency_resolution'] = actualFrequencyResolution
        f_h5_obsplane.attrs['slicing_method'] = slicing_method

        f_h5_obsplane['antenna_position'] = antenna_position
        f_h5_obsplane['antenna_names'] = np.array([np.string_(x) for x in names])  # needed for python 2
        f_h5_obsplane['antenna_position_vBvvB'] = station_positions_transformed
        f_h5_obsplane['core'] = core
        f_h5_obsplane['polarization_vector'] = polarization_vector
        f_h5_obsplane['energy_fluence_vector'] = energy_fluence
        f_h5_obsplane['energy_fluence'] = np.sum(energy_fluence, axis=1)
        f_h5_obsplane['amplitude'] = peak_amplitude
        f_h5_obsplane['amplitude_total'] = peak_amplitude_total
        f_h5_obsplane['frequency_slope'] = frequency_slope

        if len(slicing_boundaries):
            f_h5_obsplane['slicing_boundaries'] = np.array(slicing_boundaries)

        if args.calculate_stokes_parameter:
            f_h5_obsplane['stokes_parameter'] = stokes_parameter

        if args.store_traces:
            f_h5_obsplane['times_filtered'] = times_filtered
            f_h5_obsplane['traces_filtered'] = traces_filtered

    return f_h5_sephl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coreas hdf5 converter')
    parser.add_argument('input_file', type=str, help='path to input reas file')

    parser.add_argument("-o", "--outputDirectory", dest='output_directory', type=str, default=None,
                        help='path to output hdf5 directory')
    parser.add_argument("-of", "--outputFileName", dest='output_filename', type=str, default=None,
                        help='output filename')

    # parameter regarding calculation of highlevel quantities
    parser.add_argument('-hl', '--highLevel', dest='compute_high_level', action='store_true',
                        help="compute radio high level quantities such as bandwithlimited time traces, "
                             "energy fluence, radiation energy etc.")
    parser.add_argument("--store_traces", action="store_true",
                        help="Stores rotated, filted, and resampled traces in highlevel file")
    parser.add_argument("--not_store_full_simulation", action="store_false",
                        dest="store_full_simulation_in_hdf5", help="If set, the full, converter simulation will not be"
                             " stored in a hdf5 file and only the highlevel file is written to disk.")

    parser.add_argument("--flow", type=float, default=30., help="low frequency cut in MHz")
    parser.add_argument("--fhigh", type=float, default=80., help="high frequency cut in MHz")

    parser.add_argument("--samplingFrequency", dest="sampling_frequency", type=float, default=1.,
                        help="Sampling frequency in Gsamples/second")
    parser.add_argument("--frequencyResolution", type=float, default=100.,
                        help="To increase precision, the frequency spectrum is padded prior appling a bandpass filter."
                             " Set maximum allowed frequency resolution in kHz (default: 100 kHz)")

    parser.add_argument("--NSamples", dest="number_of_samples", type=int, default=256,
                        help="the number of samples that should be kept for the downsampled trace")
    parser.add_argument("--NSamplesBeforePulse", dest="samples_before_pulse", type=int, default=None,
                        help="the number of samples before the pulse")

    parser.add_argument("--norad", action="store_false", dest="compute_radiation_energy",
                        help="Do not compute radiation energy?")
    parser.add_argument("--novB_vvB", action="store_false", dest="use_vB_vvB_polarization",
                        help="Return trace-related quantities in N, W, vertical instead of vxB, vxvxB and v polarizations")

    parser.add_argument("--stokes", action="store_true", dest="calculate_stokes_parameter",
                        help="Calculates Stokes' parameter, in eV/m2")
    parser.add_argument("--stokes_window", type=float, default=25.,
                        help="window around highest peak to calculate stokes parameter, in ns")

    parser.add_argument("--ignoreATMfile", action="store_true",
                        help="If a ATMFILE was used for the simulation but can"
                        " not be found choose to ignore it (Default: false).")

    parser.add_argument("--atmfile", type=str, default=None,
                        help='Path to a GDAS atm file. Only required if "ATMFILE" specified in *inp file.'
                        ' If not provided check for file and location specified in *inp file.')

    parser.add_argument("--not_store_input_file", action="store_false",
                        dest="store_input_file_in_hdf5", help="")

    parser.add_argument("--not_store_long_file", action="store_false",
                        dest="store_long_file_in_hdf5", help="")

    parser.add_argument("--add_faerie_simulation", action="store_true",
                        help="If True, add the electric fields generated by FAERIE (GEANT) to the hdf5 file. "
                        "The electriec fields are expected to be in the same format and units as CoREAS and in the folder `SIM??????_geant`")

    args = parser.parse_args()


    # Calculate highlevel quantities and convert simulation to hdf5 format (if input is a '.reas' file and store_simulation_in_hdf5_file is True)
    if args.compute_high_level:
        reas_filename = args.input_file
        output_filename = os.path.splitext(os.path.basename(reas_filename))[0] + ".hdf5"
        if(args.output_filename is not None):
            output_filename = args.output_filename
        if(args.output_directory is not None):
            output_filename = os.path.join(args.output_directory, output_filename)

        if os.path.splitext(args.input_file)[1] != ".hdf5":
            f_h5 = write_coreas_hdf5_file(
                reas_filename, output_filename, atm_file=args.atmfile, ignore_atm_file=args.ignoreATMfile)

        else:
            f_h5 = h5py.File(args.input_file, "r")

        try:
            from radiotools import helper as rdhelp
            from radiotools import coordinatesystems
        except ModuleNotFoundError as e:
            sys.exit("Could not find the radiotools module: '{}'\n"
                     "You can get this module from https://github.com/nu-radio/radiotools.\n"
                     "Make sure to add it to your enviourment, e.g., PYTHONPATH too. Stopping ...".format(e))
        from scipy.signal import hilbert
        from scipy import optimize

        output_filename_array = os.path.splitext(os.path.basename(output_filename))
        output_filename_hl = os.path.join(os.path.dirname(output_filename),
                                           output_filename_array[0] + "_highlevel" + output_filename_array[1])

        f_h5_sephl = write_coreas_highlevel_file(output_filename_hl, f_h5, args)
        f_h5_sephl.close()

        if not args.store_full_simulation_in_hdf5 and os.path.splitext(args.input_file)[1] != ".hdf5":
            # make file empty (so that writing to disc does not cost a lot of i/o), write it to disc and remove it.
            for key in f_h5.keys():
                del f_h5[key]
            oname = f_h5.filename
            f_h5.close()
            os.remove(oname)
        else:
            # in case it did not exists yet its get written to disc now
            f_h5.close()

    # only convert simulation to hdf5 format
    else:
        reas_filename = args.input_file
        output_filename = os.path.splitext(os.path.basename(reas_filename))[0] + ".hdf5"
        if(args.output_filename is not None):
            output_filename = args.output_filename
        if(args.output_directory is not None):
            output_filename = os.path.join(args.output_directory, output_filename)

        f_h5 = write_coreas_hdf5_file(
            reas_filename, output_filename, atm_file=args.atmfile, ignore_atm_file=args.ignoreATMfile)
        f_h5.close()
