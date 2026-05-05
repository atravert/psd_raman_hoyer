##########################################
# utilities for the analysis of MES data
# part of this file is temporary, to be used for testing and development before integration in scpy
#
# 1. signal to noise utilities for synthetic data (snr, make_noise, add_noise)
# 2. pure spectra generation
# 3. concentration evolution simulation for various reaction mechanisms
# 4. PSD utilities
# 5. CP decomposition
# 6. specific readers of publicated data
# 7. animated gif plotter
#
# %%
import numpy as np
import traitlets as tr
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import re
from datetime import datetime, timedelta
import os
import tensorly as tl

from typing import Union, Sequence

import spectrochempy as scp
from spectrochempy import NDDataset, Coord, CoordSet
from spectrochempy.analysis._base._analysisbase import AnalysisConfigurable

num = Union[int, float]


def reshape_to_3D(D, n_periods, n_spectra):
    '''reshape a 2D dataset to a 3D dataset with n_periods x n_spectra x n_wavenumbers'''

    times = D.y.data.reshape(n_periods, n_spectra)
    ave_rel_times = np.mean(times - np.expand_dims(times[:, 0], axis=1), axis=0)
    out = NDDataset(data=D.data.reshape(n_periods, n_spectra, D.shape[1]),
                    title=D.title,
                    units=D.units,
                    coordset=CoordSet(z = Coord(np.arange(1, n_periods + 1, 1), title='period', units=None),
                                      y = Coord(ave_rel_times, title='time', units='s'),
                                      x = D.x)
                    )
    return out


def despike(D, threshold)  :
    out = D.copy()
    diff = D[1:] - D[:-1]
    spectrum, spike = np.where(diff.data > threshold)
    for spec_idx, spike_idx in zip(spectrum, spike):
        out.data[spec_idx + 1, spike_idx] = 0.5 * (out.data[spec_idx + 2, spike_idx] + out.data[spec_idx, spike_idx])
    return out

#######################################################
# animated git plotter
#
# %%
#######################################################
# animated git plotter
#
#%%
def plot_gif(spectra, stdev=None, std_levels=1, interval=100, ylim=None, xlim=None, filename=None):
    """
    Plot a GIF of the evolution of spectra with their standard deviation.

    Parameters
    ----------
    spectra : NDDataset
        A NDDataset of spectra at a given time points.
    stdev : NDDataset, optional
         Standard deviations of tyhe spectra at the same time points.
    std_levels : int or list of int, optional
        The number of standard deviations to plot. Default is 1.
    interval : int, optional
        The duration of each frame in the GIF in milliseconds. Default is 100 ms.
    ylim : tuple of float, optional
        The y-axis limits for the plot. Default is None.
    filename : str, optional
        The file path where the GIF will be saved. Default is None.

    Returns
    -------
    None

    """

    def _yx(spec_index, ndev):
        x = spectra[spec_index].x.data
        y1 = _spectra[spec_index] + ndev * _stdev[spec_index]
        y2 = _spectra[spec_index] - ndev * _stdev[spec_index]

        return np.hstack([np.array([x, y1]), np.array([x, y2])[:,::-1]]).T

    if filename is None:
        filename = spectra.name
    else:
        # remove the extension
        filename = filename.split('.')[0]

    _spectra = spectra.data
    _stdev = stdev.data if stdev is not None else None

    if isinstance(std_levels, int):
        std_levels = [std_levels]

    fig, ax = plt.subplots()

    if stdev is not None:
        pathcollections = []
        for i, level in enumerate(std_levels):
            points = _yx(0, 1)
            polycollection = PatchCollection([Polygon(points, closed=False)], alpha=0.2/(i+1), edgecolor='w', facecolor='b')
            pathcollections.append(ax.add_collection(polycollection))

    line = ax.plot(spectra[0].x.data, spectra[0].data.squeeze())[0]
    if ylim is not None:
        ax.set(ylim = ylim)
    if xlim is not None:
        ax.set(xlim = xlim)

    ax.set_xlabel(f"{spectra.x.title} ({spectra.x.units})")
    ax.set_ylabel(f"{spectra.title} ({spectra.units})")


    def update(frame):
        line.set_ydata(_spectra[frame].squeeze())
        if stdev is not None:
            for i, level in enumerate(std_levels):
                pathcollections[i].get_paths()[0].vertices =  _yx(frame, level)


    A = FuncAnimation(fig=fig, func=update, frames=spectra.shape[0], interval=interval)
    A.save(filename=f"{filename}.gif", writer="pillow")


# %%
#############################################
# signal to noise ratio utilities
#############################################




def snr(signal, noise, mode='amplitude'):
    '''Calculate the signal-to-noise ratio (SNR) given a signal and a noise.''

    Parameters
    ----------
    signal : float or ndarray
        input signal or array of signal(s)
    noise : float or ndarray
        input noise or array of noise(s)
    mode: string
        'power', 'amplitude' or 'dB'

    Returns
    -------
    snr : float or ndarray
        corresponding values in amplitude ratio, power ration od decibels

    Examples
    --------
    >>> snr(10.0, 1.0)                                       # doctest: +SKIP
    10.0
    '''


    amplitude_ratio = np.linalg.norm(signal) / np.linalg.norm(noise)

    if mode=='amplitude':
        return amplitude_ratio
    elif mode=='power':
        return amplitude_ratio**2
    elif mode=='dB':
        return 10. * np.log10(amplitude_ratio**2)


def noise_reduction(D, k=1):
    '''analyze the noise level of MES time averaged spectra and its PSD tranform
    
    the data are subtracted from their linear trend'''

    out = {}
    # psd
    psd_ = psd(D, k=k, phi=np.arange(0, 91., 90.))
    # psd_.plot()
    out['direct']  = np.std(D.detrend())
    out['diff']  = np.std((D - scp.mean(D, dim=0)).detrend())  
    out['psd'] = np.std(psd_.detrend())
   
    return out['diff']/out['psd']




def read_extranormal(dir_path):

    dates = []
    timestamps = []
    data = []
    energies = []

    # read files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):

            data_ = []
            meta_ = {}

            with open(dir_path+'/'+filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('#'):
                        key_value = re.split(':|=', line[1:], 1  )
                        if len(key_value) == 2:
                            meta_[key_value[0].strip()] = key_value[1].strip()
                    else:
                        data_.append([float(x) for x in line.split()])

                hdf_file = meta_['Data from HDF File']
                date_start = datetime.strptime(meta_['Time at start'], '%Y-%m-%d %H:%M:%S.%f')

                date = date_start + timedelta(seconds=float(meta_['Time from start (seconds)']))
                dates.append(date)
                timestamps.append(date.timestamp())
                data_ = np.array(data_)
                data.append(data_[:,1])
                energies.append(data_[:,0])

    # check that all energies are the same
    energies = np.array(energies)
    if np.count_nonzero(energies - energies[0]):
        raise ValueError('Inconsistent energy values in the files')


    data = np.array(data)

    out = NDDataset(data)
    out.name = dir_path.split('/')[-2]
    out.title = 'absorbance'
    out.units = 'absorbance'
    out.x = Coord(energies[0], title='energy', units='eV')
    out.y = Coord(timestamps, title='timestamp', units='s')

    return out

def read_raman_raw(filename):
    times = []
    data = []

    with open(filename, 'r') as file:
        # return file.readline().strip()
        ramanshifts  = ([float(x) for x in file.readline().strip().split()])

        while True:
            line = file.readline()
            if line != '':
                items = line.strip().split()
                times.append(float(items[0]))
                data.append([float(x) for x in items[1:]])
            else:
                break

    data = np.array(data)
    times = np.array(times) * 24 * 3600
    out = NDDataset(data)
    out.name = filename.split('/')[-1]
    out.name = os.path.basename(filename)
    out.x = Coord(ramanshifts, title='Raman Shift', units='1/cm')
    out.y = Coord(times, title='timestamp', units='s')
    out.title = 'Raman intensity'
    out.units = 'a.u.'

    return out

def read_uv_raw(filename):
    wavelengths = []
    data = []

    with open(filename, 'r') as file:
        line_number = 0
        while True:
            line = file.readline()
            line_number += 1
            if line_number == 10:
                items = line.strip().split(";")
                times = [[float(x) for x in items[3:]]]
            elif line_number > 10:
                items = line.strip().split(";")
                if items[0] != '':
                    wavelengths.append(float(items[0].replace(",",".")))
                    data.append([float(x.replace(",",".")) for x in items[1:]])
                else:
                    break   
                             
    data = np.array(data).T
    times = np.concatenate((np.array([0,0]) ,np.array(times).squeeze()))/1e5
    out = NDDataset(data)
    out.name = filename.split('/')[-1]
    out.name = os.path.basename(filename)
    out.x = Coord(wavelengths, title='Wavelength', units='nm')
    out.y = Coord(times, title='timestamp', units='s')
    out.title = '-log(R)'
    out.units = 'a.u.'

    return out


    return out


def read_raman_averaged(filename):
    """
    Read and process a Raman averaged data file.

    Parameters
    ----------
    filename : str
        The path to the file to read.

    Returns
    -------
    NDDataset
        The processed Raman data.
    """

    data = []
    ramanshifts = []

    with open(filename, 'r') as file:
        # Skip the first line (header)
        file.readline()
        # Read the second line for time points
        times = [float(t) for t in file.readline().strip().split()[1:]]

        EOF = False
        while not EOF:
            line = file.readline()
            if line != '':
                # Split the line into items
                items = line.strip().split()
                # First item is the Raman shift
                ramanshifts.append(float(items[0]))
                # Remaining items are the data points
                data.append([float(x) for x in items[1:]])
            else:
                EOF = True

    # Convert data to a numpy array and transpose it
    data = np.array(data).T
    out = NDDataset(data)
    out.name = filename.split('/')[-1]
    out.y = Coord(times, title='time', units='s')
    out.x = Coord(ramanshifts, title='Raman Shift', units='1/cm')
    out.title = 'Raman intensity'
    out.units = 'a.u.'

    return out




