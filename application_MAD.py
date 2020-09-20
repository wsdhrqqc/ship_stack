#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:47:42 2020
Application of MAD
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

#%% Read in data
# Read in original cn and co after custmized QC
df_cn_co_mad = pd.read_csv('/Users/qingn/Desktop/NQ/ori_cn_qced_co.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)


path_exhaust = '/Users/qingn/Desktop/NQ/exhaust_id/AAS_4292_ExhaustID_201718_AA_MARCUS.nc'
exhaust_id = netCDF4.Dataset(path_exhaust)
time_id = np.array(exhaust_id['time'])
time_id_date = pd.to_datetime(time_id, unit ='s', origin = pd.Timestamp('2017-10-18 23:45:06'))
exhaust_4mad02thresh = exhaust_id['exhaust_4mad02thresh']

exhaust_xr = xr.DataArray(exhaust_4mad02thresh, coords = [time_id_date],dims = ["date"])
exhaust_xr = exhaust_xr.loc['2017-10-29':'2018-03-24']
#
#exhaust_ds = xr.Dataset({'flag':('time',exhaust_4mad02thresh),'time':time_id_date})
#exhaust_ds_full = exhaust_ds.resample(time='1s').nearest(tolerance='10s')
##exhaust_ds_full_mean = exhaust_ds.resample(time='10s').mean()
#exhaust_ds_full_fill = exhaust_ds_full.fillna(1)
#flag = exhaust_ds_full_fill['flag']
#
# Resample the MAD flag
flag_10 = exhaust_xr.resample(date = '10T').sum() # 21168 same to df_cn_co
flag_std_10 = exhaust_xr.resample(date = '10T').std()
flag = flag_10>200 # around 70% data left

df_cn_co_mad['mad'] = flag

#%%
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/mar*.nc'
cpc = arm_read_netcdf(path_cpc,'10s')
cn = cpc['concentration']
qc_cn=cpc['qc_concentration']

cpc_con=cn.where((qc_cn==0)|(qc_cn==8)|(qc_cn==12))

values, counts = np.unique(qc_cn, return_counts=True)
cn_10 = cpc_con.resample(time = '10T').mean()
cn_std_10 = cpc_con.resample(time = '10T').std()

df_cn_co_mad['cn_qced'] = cn_10
df_cn_co_mad['cn_std_qced'] = cn_std_10

#Save df
df_cn_co_mad.to_csv('MAD_cn_co_10min.csv')
#%%
fig = plt.figure() 
ax = plt.gca()
#ax1 = ax.twinx()
#color ='tab:black'
lns2 = ax.plot(df_cn_co_mad['cn_qced']['2017-10-29':'2017-12-02'].index,df_cn_co_mad['cn_qced'].loc['2017-10-29':'2017-12-02'].values,'.b')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co_mad['cn_qced'][flag]['2017-10-29':'2017-12-02'].index ,df_cn_co_mad['cn_qced'][flag]['2017-10-29':'2017-12-02'].values,'.r',label = 'ship_stack')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
#ax1.plot(co_con_10.loc['2017-10-29':'2017-11-14'].time.values,co_con_10.loc['2017-10-29':'2017-11-14'].values,color = color)
#lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
#ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
#ax.set_ylim((1,100000))
#ax1.set_ylim((0.03,100))
ax.set_yscale('log')
#ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
fig.tight_layout()
fsave = "CPC_contamination_MAD_Hobart_Davis"
fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
fig = plt.figure() 
ax = plt.gca()
#ax1 = ax.twinx()
#color ='tab:black'
lns2 = ax.plot(df_cn_co_mad['cn_qced']['2017-12-13':'2018-01-10'].index,df_cn_co_mad['cn_qced'].loc['2017-12-13':'2018-01-10'].values,'.b')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co_mad['cn_qced'][flag]['2017-12-13':'2018-01-10'].index ,df_cn_co_mad['cn_qced'][flag]['2017-12-13':'2018-01-10'].values,'.r',label = 'ship_stack')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
#ax1.plot(co_con_10.loc['2017-10-29':'2017-11-14'].time.values,co_con_10.loc['2017-10-29':'2017-11-14'].values,color = color)
#lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
#ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
#ax.set_ylim((1,100000))
#ax1.set_ylim((0.03,100))
ax.set_yscale('log')
#ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
fig.tight_layout()
fsave = "CPC_contamination_MAD_Hobart_Casey"
fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
fig = plt.figure() 
ax = plt.gca()
#ax1 = ax.twinx()
#color ='tab:black'
lns2 = ax.plot(df_cn_co_mad['cn_qced']['2017-12-17':'2018-03-03'].index,df_cn_co_mad['cn_qced'].loc['2017-12-17':'2018-03-03'].values,'.b')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co_mad['cn_qced'][flag]['2017-12-17':'2018-03-03'].index ,df_cn_co_mad['cn_qced'][flag]['2017-12-17':'2018-03-03'].values,'.r',label = 'ship_stack')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
#ax1.plot(co_con_10.loc['2017-10-29':'2017-11-14'].time.values,co_con_10.loc['2017-10-29':'2017-11-14'].values,color = color)
#lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
#ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
#ax.set_ylim((1,100000))
#ax1.set_ylim((0.03,100))
ax.set_yscale('log')
#ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))
fig.tight_layout()
fsave = "CPC_contamination_MAD_Hobart_Mawson"
fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
fig = plt.figure() 
ax = plt.gca()
#ax1 = ax.twinx()
#color ='tab:black'
lns2 = ax.plot(df_cn_co_mad['cn_qced']['2018-03-09':'2018-03-24'].index,df_cn_co_mad['cn_qced'].loc['2018-03-09':'2018-03-24'].values,'.b')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co_mad['cn_qced'][flag]['2018-03-09':'2018-03-24'].index ,df_cn_co_mad['cn_qced'][flag]['2018-03-09':'2018-03-24'].values,'.r',label = 'ship_stack')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
#ax1.plot(co_con_10.loc['2017-10-29':'2017-11-14'].time.values,co_con_10.loc['2017-10-29':'2017-11-14'].values,color = color)
#lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
#ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
#ax.set_ylim((1,100000))
#ax1.set_ylim((0.03,100))
ax.set_yscale('log')
#ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
fig.tight_layout()
fsave = "CPC_contamination_MAD_Hobart_MacQuerie"
fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")

