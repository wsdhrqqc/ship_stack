#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:14:56 2020
Wind correction、 comparison to AAD and BBS sonde 
This will include the QC for BBS
@author: qingn
"""

import xarray as xr
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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

#import pathlib
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

#%% Read in AAd corrected wind netcdf port is left and starboard is right
path_met = '/Users/qingn/Desktop/NQ/maraadmet/*.nc'
met_aad = arm.read_netcdf(path_met)
# Understand the diff between port and starboard; their qc are all 0
#sp_p = met_aad['wind_speed_port']
sp_sta = met_aad['wind_speed_starboard']
#dir_p = met_aad['wind_direction_port']
dir_sta = met_aad['wind_direction_starboard']

lat = met_aad['lat'].resample(time='10T').nearest(tolerance = '10T')
lon = met_aad['lon'].resample(time='10T').nearest(tolerance = '10T')
#met_ori = arm.read_netcdf('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.2018020[1-9]*.nc')
#%% Skew-T figure at each launch, save all the launch figures
for filepath in sorted(list(glob.glob('/Users/qingn/Desktop/NQ/sounde/mar*')))[:3]:
    print(filepath)
    
#path_sonde_ds = '/Users/qingn/Desktop/NQ/sounde/marsondewnpnM1.b1.20171114.23*'
    sonde_ds = arm.read_netcdf(filepath)
    skewt = act.plotting.SkewTDisplay(sonde_ds)

    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    
    plt.show(skewt.fig)
    skewt.fig.suptitle('test title')
#    skewt.fig.tight_layout()
    fsave = "lnT_"+filepath[-19:-4]
#    ax.set_title(fsave)
    skewt.fig.savefig(f"{os.path.join(figpath,'Tlnp', fsave)}.png", dpi=300, fmt="png")
#%% Sonde QC
path_sonde = '/Users/qingn/Desktop/NQ/sounde/marsondewnpnM1.b1.201*'
sonde = arm.read_netcdf(path_sonde)
#wspd_qced = sonde['wspd'][np.where(sonde['qc_wspd']==0)[0]]

idx_surface = np.where(sonde['alt']<30)[0]

diff_idx = idx_surface[1:]-idx_surface[:-1]
wspd = sonde['wspd'][idx_surface]
deg = sonde['deg'][idx_surface]

wspd_uni = [] # initiate
wspd_uni_time = []
deg_uni = []
for i in range(np.size(diff_idx)):
    if diff_idx[i]>100:
        wspd_uni.append(wspd[i].values)
        wspd_uni_time.append(wspd[i].time.values)
        deg_uni.append(deg[i].values)
    else: 
        print(idx_surface[i])
    
        
        
abnormal_surface = np.where(np.array(wspd_uni)>40)[0]
# Hande two extreme value
wspd_uni[abnormal_surface[0]]=6.1
wspd_uni[abnormal_surface[1]]=10

wspd_uuni = xr.DataArray(wspd,coords = [wspd.time],dims = ['time'])
#249364:249407 BBS is not ascending quick enough to be higher than 30 meter
#here we only take the first value
#%% Read nav from met_aad(/1min) and read relative wind from aosmet(/1s)
#wspd_name='wind_speed';wdir_name='wind_direction'

met = arm_read_netcdf('/Users/qingn/Desktop/NQ/maraosmet/maraosmetM1.a1/maraosmetM1.a1.201*.nc','10s')
met_wspd = met['wind_speed'].resample(time = '10T').mean()
#met_wspd_std = met['wind_speed'].resample(time = '10T').std()
met_dir = np.deg2rad(met['wind_direction'].resample(time = '10T').mean())


#data_wind = {'date':met_wspd.time.values,'sp':met_wspd.values,'sp_std':met_wspd_std.values,'dir':met_dir.values}
#df_wind = pd.DataFrame(data_wind,columns = ['date','sp','sp_std','dir'])


#%% The process associated to .values is time and RAM consuming

rels1 = met_wspd.loc['2017-10-30':'2018-03-23']
time_10min = rels1.time
rels = rels1.values
reld = met_dir.loc['2017-10-30':'2018-03-23'].values
# Set variables to be used and convert to radians
#rels = met[wspd_name]
##unit_ut = met[wspd_name].units
#reld = np.deg2rad(met[wdir_name])
#reld_deg = met[wdir_name]
#%% The process associated to .values is time and RAM consuming
head = met_aad['heading'][np.where(met_aad['qc_heading']==0)[0]]
head = head.resample(time = '10T').mean().values

head = np.deg2rad(head)

sog = met_aad['speed_over_ground']
sog = sog.resample(time = '10T').mean().values
cog =  np.deg2rad(met_aad['course_over_ground'][np.where(met_aad['qc_course_over_ground']==0)[0]])
cog = cog.resample(time = '10T').mean().values
#%% After .values, this block becomes quick
# Calculate winds based on method in the document denoted above
relsn = rels * np.cos(head + reld)
relse = rels * np.sin(head + reld)

sogn = sog * np.cos(cog)
soge = sog * np.sin(cog)

un = relsn - sogn
ue = relse - soge
dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
ut = np.sqrt(un ** 2. + ue ** 2)

#%% We create a data set to include all the wind and lat lon info
data_wd = {'date':time_10min.values,'aos_tr_wind':ut,'aos_tr_dir':dirt,
           'aad_tr_wind':sp_sta.resample(time = '10T').nearest(tolerance ='10T').values*2,
           'aad_tr_dir':dir_sta.resample(time = '10T').nearest(tolerance ='10T').values,
           'aos_rel_wind':rels,'aos_rel_dir':reld,
           'lat':lat.values,'lon':lon.values}
df_wd = pd.DataFrame(data_wd,columns = ['date','aos_tr_wind','aos_tr_dir','aad_tr_wind','aad_tr_dir','aos_rel_wind','aos_rel_dir','lat','lon'])
#df_wd.to_csv('wind_corr_aosmet_marinemet_lat_lon_10min.csv')




#%% Read in my corrected wind csv
#wind_1_cpc = pd.read_csv('/Users/qingn/201711cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
#wind_2_cpc = pd.read_csv('/Users/qingn/201712cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
#wind_3_cpc = pd.read_csv('/Users/qingn/201801cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
#wind_4_cpc = pd.read_csv('/Users/qingn/201802cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
# Feb in 4
#wind_5_cpc = pd.read_csv('/Users/qingn/201803cpc_wind_lat_.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
#
#wind_full = pd.concat([wind_1_cpc,wind_2_cpc,wind_3_cpc,wind_4_cpc,wind_5_cpc])
#wind_full_10min = wind_full.resample('10T').mean()

wind_df = pd.read_csv(home+'/wind_corr_aosmet_marinemet_lat_lon_10min.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)
wind_df.columns
wind_df = wind_df.set_index('date')
wind_df.index = pd.to_datetime(wind_df.index)
wind_ds = wind_df.to_xarray()
#%% TWo kinds of figures: hist and time series
# Normalize histogram plot for wind direction
kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)

#% Normalize histogram plot for wind speed
plt.figure(figsize = (FIGWIDTH,FIGHEIGHT))
#plt.hist(ut, **kwargs, color = 'green', label = 'true wind speed')# rels
#plt.hist(met_aad['wind_speed_uncorr_starboard'], **kwargs, color='blue',label='relative wind speed_aad')
#plt.hist(rels, **kwargs, color='blue',label='relative wind speed_aos')
plt.hist(wspd_uni, **kwargs, color='yellow',label='radio wind speed')

plt.hist(wind_df[''], **kwargs, color='pink',label='aad true wind speed')

plt.gca().set(title='Probability Histogram of True Wind Speed', ylabel='Probability',xlabel='m/s')

plt.legend()
#axs[p].set_ylim(ylim)
#fig.tight_layout()
#%%
# Normalize histogram plot for wind direction
kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)

#% Normalize histogram plot for wind speed
fig = plt.figure(figsize = (FIGWIDTH,FIGHEIGHT))
#plt.hist(wind_df['aos_tr_wind'], **kwargs, color = 'green', label = 'aos true wind speed')# rels
plt.hist(wind_df['aos_rel_wind'], **kwargs, label = 'relative wind speed')# rels
plt.hist(wind_df['aad_tr_wind']/2*1.944, **kwargs, color='blue',label='true wind speed')
plt.hist(wspd_uni, **kwargs, color='yellow',label='radio wind speed')
#plt.hist(sp_sta, **kwargs, color='pink',label='aad wind speed')
plt.xlabel('wind speed(m/s)')
plt.gca().set(title='Probability Histogram of True Wind Speed', ylabel='Probability',xlabel='m/s')

plt.legend()
#axs[p].set_ylim(ylim)
#fig.tight_layout()
fig.tight_layout()
fsave = "aad_wind_correction_sonde"

fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
# Normalize histogram plot for wind direction
kwargs = dict(alpha=0.5, bins=20, density=True, stacked=True)


fig = plt.figure(figsize = (FIGWIDTH,FIGHEIGHT))
#plt.hist(wind_df['aad_tr_dir'], **kwargs, color='pink',label='aad true wind direction')
plt.hist(wind_df['aos_tr_dir'], **kwargs,label = 'true wind direction')
plt.hist(deg_uni, **kwargs,label = 'sonde wind direction')
#plt.title('histogram for true wind direction')
#plt.xlabel('degree')
#plt.ylabel('counts')
plt.hist(np.rad2deg(wind_df['aos_rel_dir']), **kwargs,label='relative wind direction')
#plt.hist(np.rad2deg(reld), **kwargs,label='relative wind direction')
plt.gca().set(title='Probability Histogram of True Wind Direction', ylabel='Probability',xlabel='degree')
#plt.hist(x3, **kwargs, color='r', label='Good')
#plt.gca().set(title='Frequency Histogram of Diamond Depths', ylabel='Frequency')

plt.legend()
fig.tight_layout()
fsave = "aad_wind_dir_correction_sonde"

fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%% There is some issue in finding alt <30(one sonde does not ascend as the others) And this would not effect the time seires, but will 
# effect the histgram. This is part of the sonde QC
fig = plt.figure()
plt.plot(sonde['alt'][249360:249500].time.values, sonde['alt'][249360:249500])
plt.yscale('log')
plt.ylim((0,50))
fig.autofmt_xdate()

#%% aad_tr_wind speed os actually better than the met aos_tr_wind speed
fig = plt.figure();
ax1 = plt.gca()
#plt.plot(wind_ds.aos_tr_wind.loc['2018-02-13':'2018-02-28'].date,wind_ds.aos_tr_wind.loc['2018-02-13':'2018-02-28'])
plt.plot(wind_ds.aad_tr_wind.loc['2018-02-13':'2018-02-28'].date,wind_ds.aad_tr_wind.loc['2018-02-13':'2018-02-28']/2*1.944,color = 'green',label='wind_corrected')
#plt.plot(wind_df['2018-02-13':'2018-02-28'].index,wind_df['aad_tr_wind']['2018-02-13':'2018-02-28'])
#plt.plot(wind_4_cpc['wind_speed']['2018-02-23':'2018-02-28'])
plt.plot(wspd.loc['2018-02-13':'2018-02-28'].time, wspd_uuni.loc['2018-02-13':'2018-02-28'].values,'r.',label = 'radiosonde')
plt.legend()
fig.autofmt_xdate()
plt.ylabel('wind speed(m/s)')
plt.xlabel('Date')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
fig.tight_layout()
fsave = "aad_wind_correction_sonde"

fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")

#%%
fig = plt.figure();
ax1 = plt.gca()
#plt.plot(wind_ds.aos_tr_wind.loc['2018-02-13':'2018-02-28'].date,wind_ds.aos_tr_wind.loc['2018-02-13':'2018-02-28'])
plt.plot(wind_ds.aad_tr_dir.loc['2018-02-13':'2018-02-28'].date,wind_ds.aad_tr_dir.loc['2018-02-13':'2018-02-28'],color = 'green',label='wind_relative_to_north')
#plt.plot(wind_df['2018-02-13':'2018-02-28'].index,wind_df['aad_tr_wind']['2018-02-13':'2018-02-28'])
#plt.plot(wind_4_cpc['wind_speed']['2018-02-23':'2018-02-28'])
plt.plot(deg.loc['2018-02-13':'2018-02-28'].time, deg.loc['2018-02-13':'2018-02-28'].values,'r.',label = 'radiosonde')
plt.legend()
fig.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('wind direction(degree)')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
fig.tight_layout()
fsave = "aad_wind_dir_correction_sonde"
#    ax.set_title(fsave)
fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
fig = plt.figure(figsize = (12,4));
plt.plot(wind_Feb_1min['wind_speed']['2018-02-01':'2018-02-13'],label = 'Mine')
plt.plot(sp_sta.loc['2018-02-01':'2018-02-13'].time,sp_sta.loc['2018-02-01':'2018-02-13'],label = 'ARM')
#plt.plot(wind_4_cpc['wind_speed']['2018-02-23':'2018-02-28'])
plt.plot(wspd.loc['2018-02-01':'2018-02-13'].time, wspd.loc['2018-02-01':'2018-02-13'],'r-',label = 'radiosonde')
plt.plot()
plt.legend()
plt.ylim((0,23))
fig.autofmt_xdate()
#%%
fig1 = plt.figure()
plt.plot(wind_Feb_1min['wind_speed']['2018-02-11 00:00':'2018-02-13 00:00']/met_aad['wind_speed_port'].loc['2018-02-11 00:00':'2018-02-13 00:00']/2)
fig1.autofmt_xdate()