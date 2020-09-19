#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Sep 17 16:12:59 2020
Comparing CO and CPC, This could be improved later with O3, which is similar in code as co. will carry out later
CO has been QCed with bit discription, and Hobar port has been removed. backed up at Github.
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
#%%
path_cpc = '/Users/qingn/Desktop/NQ/maraoscpc/mar*.nc'
path_co =  '/Users/qingn/Desktop/NQ/maraosco/mar*.nc'
#path_o3 =  '/Users/qingn/Desktop/NQ/maraoso3/mar*.nc'

cpc = arm_read_netcdf(path_cpc,'10s')
cn = cpc['concentration']
#cpc = arm_read_netcdf(path_cpc,'10s')


co = arm_read_netcdf(path_co,'10s')#
### DIY QC for CO
co_con = co['co_dry'].where((co['qc_co_dry']<16384)&(co['co_dry']>0.03)&(co['qc_co_dry']!=44)) # those does not be approved has been turned into nan
#%%
class Solution(object):
   def intersect(self, nums1, nums2):
      """
      :type nums1: List[int]
      :type nums2: List[int]
      :rtype: List[int]
      """
      m = {}
      if len(nums1)<len(nums2):
         nums1,nums2 = nums2,nums1
      for i in nums1:
         if i not in m:
            m[i] = 1
         else:
            m[i]+=1
      result = []
      for i in nums2:
         if i in m and m[i]:
            m[i]-=1
            result.append(i)
      return result
ob1 = Solution()

#%%# We take the date at Hobart port away
list_date_four = ['2017-10-02','2017-12-02','2017-12-13','2018-01-10','2018-01-16','2018-03-04','2018-03-09','2018-03-24']

# This is too much for dataframe and we are doing the mean for 10 min here, we also keep the std to plot error bar in the future
#%%## Resample with the xarray dataset and then put it in dataframe ~~ 21168 counts sample
co_con_10 = co_con.resample(time = '10T').mean()
co_std_10 = co_con.resample(time = '10T').std()
cn_10 = cn.resample(time = '10T').mean()
cn_std_10 = cn.resample(time = '10T').std()
print(np.size(co_con_10),np.size(co_std_10),np.size(cn_10),np.size(cn_std_10))
# We firstly make each measurement a df and save it for further use, as a backup to some extent
data_cn = {'date':cn_10.time.values,'cn_avg':cn_10.values,'cn_std':cn_std_10.values}
df_cpc = pd.DataFrame(data_cn,columns = ['date','cn_avg','cn_std'])
#df_cpc.to_csv('ori_cpc_10min.csv')
#%%
data_co = {'date':co_con_10.time.values,'co_10min_avg':co_con_10.values,'co_std':co_std_10.values}
df_co = pd.DataFrame(data_co,columns = ['date','co_10min_avg','co_std'])
#df_co.to_csv('diyqc_co_10min.csv')
#%% Combine cpc and co
df_cpc=df_cpc.set_index('date')
df_co=df_co.set_index('date')
df_cn_co = pd.concat([df_cpc,df_co],axis=1,join='outer')
df_cn_co['over_ocean'] = 0
df_cn_co.set_index('date')#, inplace = True
#df_cn_co.index = pd.to_datetime(df_cn_co.index)

df_cn_co['over_ocean'].loc['2017-10-29':'2017-12-02'] = 1
df_cn_co['over_ocean'].loc['2017-12-13':'2018-01-10'] = 1
df_cn_co['over_ocean'].loc['2018-01-16':'2018-03-04'] = 1
df_cn_co['over_ocean'].loc['2018-03-09':'2018-03-24'] = 1

plt.plot(df_cn_co['co_10min_avg'][df_cn_co['over_ocean']==0])
plt.plot(df_cn_co['co_10min_avg'][df_cn_co['over_ocean']==1])
#%%
abnormal_cn = np.where(df_cn_co['cn'][df_cn_co['over_ocean']==1]>2000)[0] #68050
abnormal_co = np.where(df_cn_co['co_10min_avg'][df_cn_co['over_ocean']==1]>.08)[0] #45027
intersect = ob1.intersect(abnormal_cn[abnormal_cn<2447], abnormal_co[abnormal_co<2447])
#%% Zoom in abnormal cpc and abnormal co
fig = plt.figure() 
ax = fig.gca()
sta = 88000
end = 90000
qc_0 = np.where(cpc['qc_concentration'][sta:end]==0)[0]
co_qc_0 = np.where(co['qc_co_dry'][sta:end]==0)[0]
ax.plot(cn.time[sta:end].values,cn[sta:end],color = 'grey',marker = '1',alpha = 0.25,label='before_ARM_QC')
ax.plot(co_con.time[sta:end].values,co_con[sta:end],color = 'green',marker = '*',alpha = 0.5,label='CO_after_diy_QC')
#ax.scatter(co_con.time[sta:end][co_qc_0].values,co_con[sta:end][co_qc_0],marker = '^',color = 'red',label = 'CO_after_ARM_QC')
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
#%% General figure of cpc and co

#plt.close(fig)
fig = plt.figure() 
ax = fig.gca()
sta = 88000
end = 90000
qc_0 = np.where(cn[sta:end]>2000)[0]
co_qc_0 = np.where(co_con[sta:end]>.08)[0]
ax1 = ax.twinx()
ax.plot(cn.time[sta:end].values,cn[sta:end],color = 'grey',marker = '1',alpha = 0.25,label='CN')
ax1.plot(co_con.time[sta:end].values,co_con[sta:end],color = 'orange',marker = '1',alpha = 0.2,label='CO')

ax.scatter(cn.time[sta:end][qc_0].values,cn[sta:end][qc_0],marker = '^',label = 'CN>2000',color = 'red')
ax1.scatter(co_con.time[sta:end][co_qc_0].values,co_con[sta:end][co_qc_0],marker = '^',color = 'blue',label = 'CO>0.08')
ax.set_yscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = 'orange')
#ax.set_ylim((1))
fig.autofmt_xdate() 

ax.legend(loc =9)
ax1.legend(loc = 8)
ax1.set_ylim((0.03,100))
ax.set_ylim((1,1000000))
ax.set_ylabel('N10(#/cc)')
ax.set_xlabel('date')
fig.tight_layout()
fsave = "CPC_co_contamination_zoomin"
#fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%% CO is actually abnormal at Hobart, since Hobart has a lot of human being activities going on
#General figure of cpc and co from Hobart to Davis
fig = plt.figure() 
ax = plt.gca()
ax1 = ax.twinx()
#color ='tab:black'
ax.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'].index,cn_10.loc['2017-10-29':'2017-11-14'].values,color = 'black')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
ax.plot(df_cn_co['cn'][abnormal_cn]['2017-10-29':'2017-11-14'].index ,df_cn_co['cn'][abnormal_cn]['2017-10-29':'2017-11-14'].values,'.r',label = 'CN>2000')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
ax1.plot(co_con_10.loc['2017-10-29':'2017-11-14'].time.values,co_con_10.loc['2017-10-29':'2017-11-14'].values,color = color)
ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-10-29':'2017-11-14'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
ax.set_ylim((1,100000))
ax1.set_ylim((0.03,100))
ax.set_yscale('log')
ax1.set_yscale('log')
fig.autofmt_xdate() 
ax.legend()
ax1.legend()
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax4.xaxis.set_minor_locator(ticker.MultipleLocator(1))
fig.tight_layout()
fsave = "CPC_contamination_CO_conta"
#fig.savefig(f"{os.path.join(figpath, fsave)}.png", dpi=300, fmt="png")
#%%
#General figure of cpc and co at Davis
fig = plt.figure() 
ax = plt.gca()
ax1 = ax.twinx()
#color ='tab:black'
ax.plot(df_cn_co['cn']['2017-11-14':'2017-11-21'].index,cn_10.loc['2017-11-14':'2017-11-21'].values,color = 'black')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co['cn'][abnormal_cn]['2017-11-14':'2017-11-21'].index ,df_cn_co['cn'][abnormal_cn]['2017-11-14':'2017-11-21'].values,'.r',label = 'CN>2000')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
ax1.plot(co_con_10.loc['2017-11-14':'2017-11-21'].time.values,co_con_10.loc['2017-11-14':'2017-11-21'].values,color = color)
lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-11-14':'2017-11-21'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-11-14':'2017-11-21'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
ax.set_ylim((1,100000))
ax1.set_ylim((0.03,100))
ax.set_yscale('log')
ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#%%
#General figure of cpc and co at Hobart Dec 2-13
fig = plt.figure() 
ax = plt.gca()
ax1 = ax.twinx()
#color ='tab:black'
ax.plot(df_cn_co['cn']['2017-12-02':'2017-12-13'].index,cn_10.loc['2017-12-02':'2017-12-13'].values,color = 'black')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co['cn'][abnormal_cn]['2017-12-02':'2017-12-13'].index ,df_cn_co['cn'][abnormal_cn]['2017-12-02':'2017-12-13'].values,'.r',label = 'CN>2000')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
ax1.plot(co_con_10.loc['2017-12-02':'2017-12-13'].time.values,co_con_10.loc['2017-12-02':'2017-12-13'].values,color = color)
lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2017-12-02':'2017-12-13'].index,df_cn_co['co_10min_avg'][abnormal_co]['2017-12-02':'2017-12-13'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
ax.set_ylim((1,100000))
ax1.set_ylim((0.03,100))
ax.set_yscale('log')
ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#%%
#General figure of cpc and co at Hobart Jan 11-16
fig = plt.figure() 
ax = plt.gca()
ax1 = ax.twinx()
#color ='tab:black'
ax.plot(df_cn_co['cn']['2018-01-11':'2018-01-16'].index,cn_10.loc['2018-01-11':'2018-01-16'].values,color = 'black')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co['cn'][abnormal_cn]['2018-01-11':'2018-01-16'].index ,df_cn_co['cn'][abnormal_cn]['2018-01-11':'2018-01-16'].values,'.r',label = 'CN>2000')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
ax1.plot(co_con_10.loc['2018-01-11':'2018-01-16'].time.values,co_con_10.loc['2018-01-11':'2018-01-16'].values,color = color)
lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2018-01-11':'2018-01-16'].index,df_cn_co['co_10min_avg'][abnormal_co]['2018-01-11':'2018-01-16'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
ax.set_ylim((1,100000))
ax1.set_ylim((0.03,100))
ax.set_yscale('log')
ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#%%
#General figure of cpc and co at Hobart Mar 4-9
fig = plt.figure() 
ax = plt.gca()
ax1 = ax.twinx()
#color ='tab:black'
ax.plot(df_cn_co['cn']['2018-03-04':'2018-03-09'].index,cn_10.loc['2018-03-04':'2018-03-09'].values,color = 'black')
#plt.scatter(cn_10[abnormal_cn].time.values,cn_10[abnormal_cn].values)
lns1 = ax.plot(df_cn_co['cn'][abnormal_cn]['2018-03-04':'2018-03-09'].index ,df_cn_co['cn'][abnormal_cn]['2018-03-04':'2018-03-09'].values,'.r',label = 'CN>2000')
#plt.plot(df_cn_co['cn']['2017-10-29':'2017-11-14'])
#plt.scatter(cn_10[abnormal_co].time.values,cn_10[abnormal_co].values)
color  = 'tab:orange'
ax1.plot(co_con_10.loc['2018-03-04':'2018-03-09'].time.values,co_con_10.loc['2018-03-04':'2018-03-09'].values,color = color)
lns2 = ax1.plot(df_cn_co['co_10min_avg'][abnormal_co]['2018-03-04':'2018-03-09'].index,df_cn_co['co_10min_avg'][abnormal_co]['2018-03-04':'2018-03-09'].values,'.b',label = 'CO>0.08')
ax.set_ylabel('N10(#/cc)')
ax1.set_ylabel('CO mixing ratio(ppmv)',color = color)
ax.set_xlabel('date')
ax.set_ylim((1,100000))
ax1.set_ylim((0.03,100))
ax.set_yscale('log')
ax1.set_yscale('log')
fig.autofmt_xdate() 
#ax.legend()
#ax1.legend()

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))



