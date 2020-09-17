#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:11:20 2020

Show CPC contamination 
To make UHSAS flag: 1)sample_flow_rate is normal 2)larger than cpc concentration
@author: qingn
"""

import dask
import numpy as np
import numpy.ma as ma
import matplotlib.backends.backend_pdf
import scipy.stats as stats
#import matplotlib as mpl
#import astral
import pandas as pd
from matplotlib.dates import DateFormatter,date2num
import glob
import netCDF4
import os
from act.io.armfiles import read_netcdf
import datetime
import matplotlib
#matplotlib.use('Agg')
import sys
import matplotlib.dates as mdates
#import act
import matplotlib.pyplot as plt
#import mpl_toolkits
#import mpl_toolkits.basemap as bm
#from mpl_toolkits.basemap import Basemap, cm
import act
import seaborn as sns
#import module_ml
#from module_ml import machine_learning
import act.io.armfiles as arm
import act.plotting.plot as armplot
from sklearn.ensemble import RandomForestClassifier

import pathlib
FIGWIDTH = 6
FIGHEIGHT = 4 
FONTSIZE = 12
LABELSIZE = 12
plt.rcParams['figure.figsize'] = (FIGWIDTH, FIGHEIGHT)
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
params = {'legend.fontsize': 36,
          'legend.handlelength': 5}
def arm_read_netcdf(directory_filebase,time_resolution):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>
    : time_resolution needs to be a string'''
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
    file = file_ori.resample(time=time_resolution).nearest()
    return file

def arm_read_netcdf_ori(directory_filebase):
    '''Read a set of maraos files and append them together
    : param directory: The directory in which to scan for the .nc/.cdf files 
    relative to '/Users/qingn/Desktop/NQ'
    : param filebase: File specification potentially including wild cards
    : returns: A list of <xarray.Dataset>
    : time_resolution needs to be a string'''
    file_dir = str(directory_filebase)
    file_ori = arm.read_netcdf(file_dir)
    _, index1 = np.unique(file_ori['time'], return_index = True)
    file_ori = file_ori.isel(time = index1)
#    file = file_ori.resample(time='10s').mean()
#    file = file_ori.resample(time=time_resolution).nearest()
    return file_ori

# ---------------- #
# Data directories #
# ---------------- #
# home directory
home = os.path.expanduser("/Users/qingn/Desktop/NQ")
# main save directory

thesis_path = os.path.join(home,'personal','thesis')
# piclke file data (under NQ)
#pkl = os.path.join(os.getcwd(), "pkl")
#pkl = os.path.join(home, "pkl")
#/Users/qingn/Desktop/NQ/personal/thesis/IMG_5047.jpg

# figure save directory
figpath = os.path.join(thesis_path, "Figures")
if not os.path.exists(figpath):
    os.mkdir(figpath)
# close all figures
plt.close("all")
#%%
    
path_uhsas = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.201*.nc'
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/mar*.nc'
#path_uhsas_clean = '/Users/qingn/Desktop/NQ/maraosuhsas/maraosuhsasM1.a1.2017111[1]*.nc'
uhsas = arm_read_netcdf(path_uhsas,'10s')
con = uhsas['concentration'].sum(dim = 'bin_num')

cpc = arm_read_netcdf(path_cpc,'10s')
cn = cpc['concentration']
#con_ori = uhsas_ori['concentration'].sum(dim = 'bin_num')


#con_ori = uhsas_ori['concentration'].resample

#uhsas_clean = arm.read_netcdf(path_uhsas_clean)
#abnormal_flow = np.where(uhsas['sample_flow_rate']<50)[0]

#%% Figure Period shows how cpc looks like and how ARM QC perform
fig = plt.figure()
ax = fig.gca()
sta = 88000
end = 90000
qc_0 = np.where(cpc['qc_concentration'][sta:end]==0)[0]
ax.plot(cn.time[sta:end].values,cn[sta:end],color = 'grey',marker = '1',alpha = 0.25,label='before_ARM_QC')
ax.scatter(cn.time[sta:end][qc_0].values,cn[sta:end][qc_0],marker = '^',label = 'after_ARM_QC')
ax.set_yscale('log')
#ax.set_ylim((1))
fig.autofmt_xdate() 
plt.legend()
plt.ylabel('N10(#/cc)')
plt.xlabel('date')
fig.tight_layout()
fsave = "CPC_contamination_zoomin"
#fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#fig1.savefig(f"{os.path.join(figpath, fsave1)}.png", dpi=300, fmt="png")
plt.close(fig)
#%% whole MARCUS Period shows how cpc looks like and how ARM QC perform
fig = plt.figure()
plt.plot(cn.time,cn,color = 'grey',label ='before_ARM_QC')
plt.plot(cn[np.where(cpc['qc_concentration']==0)[0]].time,cn[np.where(cpc['qc_concentration']==0)[0]],color = 'blue',label = 'after_ARM_QC')
plt.axhline(y=10000,c='r',linestyle = '--')
fig.autofmt_xdate() 
plt.legend()
plt.ylabel('N10(#/cc)')
plt.xlabel('date')
fig.tight_layout()
fsave = "CPC_contamination_general"
#fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#fig1.savefig(f"{os.path.join(figpath, fsave1)}.png", dpi=300, fmt="png")
plt.close(fig)

#%% Figure show how uhsas compare with cpc
