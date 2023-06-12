"""
Calculating flow statistics for daily data

"""

import pandas as pd
import numpy as np
import scipy.stats as scistat
import Stats_functions as calc
from Stats_functions import meanORmed, scalardiv, seriesdiv, arraydiv, trimseries
import time

# ==================================================================================================================
# Calculating indices for average / low / high magnitudes
# Inputs qd, qm, qy, and par are generated in "CalcHITdaily" where this subroutine is called from.
def calcday_ma_ml_mh(qd,qm,qy,par):
    
    ma = pd.Series(index=range(1,46),dtype='float64')
    ml = pd.Series(index=range(1,23),dtype='float64')
    mh = pd.Series(index=range(1,28),dtype='float64')
    
    # percentile stats -----------------------------------------------------------------
    zeroflows = par['opt_nonzero']                                             # zero flow cutoff in ml per day                                       
    parray = np.arange(5,96,5)                                                 # set up percentile array
    qdq = qd['q'].to_numpy()
    qdy = qd['y'].to_numpy()
    qdm = qd['m'].to_numpy()
    qdd = qd['d'].to_numpy()
    
    # qmq = qm['avg'].to_numpy()
    # qmy = qm['y'].to_numpy()
    # qmm = qm['m'].to_numpy()
    
    qyq = qy['avg'].to_numpy()
    # qyy = qy['y'].to_numpy()
    
    #qd_nz,qdy_nz,qdm_nz,qdd_nz = calc.nonzeroflows(qdq,qdy,qdm,qdd,zeroflows)  # get a series of nonzero flows and matching dates
    qd_nz = qd[qd['q']>=zeroflows]['q'].to_numpy()
    pcflows = calc.flowpercentiles(qd_nz,parray)                               # cacluate percentiles of nonzero flows
    qdnz_log10 = np.where(qd_nz > 0, qd_nz, zeroflows)                         # Log10 of flows, exchange the zeros for a positive value
    qdnz_log10 = np.log10(qdnz_log10)
    pclogflows = calc.flowpercentiles(qdnz_log10,parray)                        # cacluate percentiles of log10 nonzero flows
    
    parray2 = np.array([1,10,25,50])                                           # set up percentile array for mh15-16-17
    pcflows2 = calc.flowpercentiles(qd_nz,parray2)                             # cacluate percentiles of nonzero flows

    #qm_nz,qmy_nz,qmm_nz,qmd_nz = calc.nonzeroflows(qmq,qmy,qmm,qmm,zeroflows)  # get a series of nonzero flows and matching dates
    qm_nz = qm[qm['avg']>=zeroflows]['avg'].to_numpy()
    pcflowsM = calc.flowpercentiles(qm_nz,parray)                              # cacluate percentiles of nonzero flows
    
    #qy_nz,qyy_nz,qym_nz,qyd_nz = calc.nonzeroflows(qyq,qyy,qyy,qyy,zeroflows)  # get a series of nonzero flows and matching dates
    qy_nz = qy[qy['avg']>=zeroflows]['avg'].to_numpy()
    pcflowsY = calc.flowpercentiles(qy_nz,parray)                              # cacluate percentiles of nonzero flows
    
    # annual stats ---------------------------------------------------------------------

    ma[1] = qd['q'].mean()                                                     # mean of all flows
    ma[2] = qd['q'].median()                                                   # median of all flows
    ma[3] = meanORmed(qy['std']*100/qy['avg'],par['opt_median'])               # mean or median of cv for each year
    ma[4] = np.std(pclogflows,ddof=1) * 100 / np.mean(pclogflows)              # std / mean for all percentiles
    ma[5] = scalardiv(ma[1] , ma[2])
    ma[6] = scalardiv(pcflows[1] , pcflows[17])                                # 10th / 90th percentile
    ma[7] = scalardiv(pcflows[3] , pcflows[15])                                # 20th / 80th percentile
    ma[8] = scalardiv(pcflows[4] , pcflows[14])                                # 25th / 75th percentile
    ma[9] = scalardiv((pclogflows[1]-pclogflows[17]) , pclogflows[9])          # 10th - 90th percentile of log flows
    ma[10] = scalardiv((pclogflows[3]-pclogflows[15]) , pclogflows[9])         # 20th - 80th percentile of log flows
    ma[11] = scalardiv((pclogflows[4]-pclogflows[14]) , pclogflows[9])         # 25th - 75th percentile of log flows
    
    ma[36] = scalardiv((qm['avg'].max() - qm['avg'].min()) , qm['avg'].median())
    ma[37] = scalardiv((pcflowsM[4] - pcflowsM[14]) , pcflowsM[9])             # (25th - 75th) / 50th
    ma[38] = scalardiv((pcflowsM[1] - pcflowsM[17]) , pcflowsM[9])             # (10th - 90th) / 50th
    ma[39] = qm['avg'].std(ddof=1) * 100 / qm['avg'].mean()
    ma[40] = scalardiv((qm['avg'].mean() - qm['avg'].median()) , qm['avg'].median())
    ma[41] = qy['avg'].mean() / par['catarea']
    ma[42] = scalardiv((qy['avg'].max() - qy['avg'].min()) , qy['avg'].median())
    ma[43] = scalardiv((pcflowsY[4] - pcflowsY[14]) , pcflowsY[9])             # (25th - 75th) / 50th
    ma[44] = scalardiv((pcflowsY[1] - pcflowsY[17]) , pcflowsY[9])             # (10th - 90th) / 50th
    ma[45] = scalardiv((qy['avg'].mean() - qy['avg'].median()) , qy['avg'].median())
    
    ml[13] = qm['min'].std(ddof=1) * 100 / qm['min'].mean()
    ml[14] = (seriesdiv(qy['min'] , qy['med'])).mean()
    ml[15] = (seriesdiv(qy['min'] , qy['avg'])).mean()
    ml[16] = (seriesdiv(qy['min'] , qy['med'])).median()
    
    # rolling stats ---------------------------------------------------------------------
    # min_7d,max_7d = calc.rolling_minmax(qdq,qdy,7)
    min_7d = qd['q'].rolling(7,center=True).mean().resample('AS').min()
    bfratio = arraydiv(min_7d[1:-1] , qyq[1:-1])                               # the [1:-1] slice strips out the first and last values
    
    ml[17] = meanORmed(bfratio,par['opt_median'])                              # average baseflow / annual flow
    ml[18] = scalardiv(np.std(bfratio,ddof=1) , np.mean(bfratio)) * 100        # cv of baseflow / annual flow
    
    ml[19] = meanORmed(qy['min']/qy['avg'],par['opt_median']) * 100

    # baseflow stats ---------------------------------------------------------------------
    q_5d = qd['q'].resample('5D').min()                                        # create 5 day blocks, get min per block
    q_5d.name = 't'
    q_5d = pd.DataFrame(q_5d)                                                  # change from series to dataframe so we can add columns
    q_5d['t+1'] = q_5d['t'].shift(1)                                           # new column = 1 block after
    q_5d['t-1'] = q_5d['t'].shift(-1)                                          # new column = 1 block before
    
    cond1 = q_5d['t']*0.9 < q_5d['t-1']                                        # if minflow*0.9 < block BEFORE
    cond2 = q_5d['t']*0.9 < q_5d['t+1']                                        # if minflow*0.9 < block AFTER
    
    bf = q_5d['t'].where((cond1 & cond2)==True,other=np.nan)                   # bf(t) = minflow(t) ONLY if both conditions are true
    
    bf = bf.interpolate(method='time',limit_direction='both')                  # interpolate NaNs
    
    if bf.sum() > 0:
        ml[20] =  bf.sum()*5 / np.sum(qdq)                                     # baseflow / tot flow (x 5 days per block)
    else:
        ml[20] = 0.0
    
    ml[21] = scalardiv(qy['min'].std(ddof=1) , qy['min'].mean()) * 100
    ml[22] = meanORmed(qy['min'],par['opt_median']) / par['catarea']
    
    mh[13] = qm['max'].std(ddof=1) * 100 / qm['max'].mean()
    mh[14] = (seriesdiv(qy['max'] , qy['med'])).median()

    mh[15] = scalardiv(pcflows2[0] , pcflows2[3])                              # 1pc / median
    mh[16] = scalardiv(pcflows2[1] , pcflows2[3])                              # 10pc / median
    mh[17] = scalardiv(pcflows2[2] , pcflows2[3])                              # 25pc / median
    
    log_qy_max = np.log10(qy['max'].clip(lower=zeroflows))
    mh[18] = (log_qy_max).std(ddof=1) * 100 / (log_qy_max).mean()
    
    log_qy_max = np.log10(qy['max'].clip(lower=zeroflows))                           # preparing some variables for mh19
    sumq1 = log_qy_max.sum()
    sumq2 = (log_qy_max**2).sum()
    sumq3 =(log_qy_max**3).sum()
    qstd = log_qy_max.std(ddof=1)
    ny = len(qy)
    
    if qstd==0 or ny==0:
        mh[19] = 0.0
    else:
        mh[19] = (((ny**2)*sumq3) - (3*ny*sumq1*sumq2) + (2*(sumq1**3))) / (ny*(ny-1)*(ny-2)*(qstd**3))

    mh[20] = meanORmed(qy['max'],par['opt_median']) / par['catarea']

    # high flow stats ---------------------------------------------------------------------
    hfthresh1 = pcflows[9]                                                     # threshold = median NON-ZERO daily
    hfthresh2 = pcflows[4]
    a1,qvol1,qmax1,b,c,d = calc.spellsabove(qdq,qdd,qdm,qdy,hfthresh1)         # stats for events above median x 1
    a3,qvol3,qmax3,b,c,d = calc.spellsabove(qdq,qdd,qdm,qdy,hfthresh1*3)       # stats for events above median x 3
    a7,qvol7,qmax7,b,c,d = calc.spellsabove(qdq,qdd,qdm,qdy,hfthresh1*7)       # stats for events above median x 7
    a75,qvol75,qmax75,b,c,d = calc.spellsabove(qdq,qdd,qdm,qdy,hfthresh2)      # stats for events above 75th (=25th exceedance)
    
    mh[21:28] = 0.0               # set equal to zero intially
    
                        #=============================
                        # Different formulation of MH21 / 22 / 23 in USGS R code
                        #
                        # MH21/22/23 are intended to be the average flood event
                        # volume as a proportion of the median.
                        # 
                        # The USGS EflowStats R code formulates them as the 
                        # average volume of events ==> IN EXCESS OF <== the
                        # threshold. In other words, the R code sums the flow
                        # for each day MINUS the median flow. This is not the
                        # entire flood volume, but the part of the flood 
                        # hydrograph above the median.
                        #
                        # The formulation here has been changed to be the
                        # entire event volume, so will produce slightly higher
                        # values for MH21/22/23.
                        #=============================

    if len(qvol1)>0: mh[21] = scalardiv(np.mean(qvol1) , pcflows[9])           # set mh21-27 only if there have actually been some events
    if len(qvol3)>0: mh[22] = scalardiv(np.mean(qvol3) , pcflows[9])
    if len(qvol7)>0: mh[23] = scalardiv(np.mean(qvol7) , pcflows[9])
    if len(qmax1)>0: mh[24] = scalardiv(np.mean(qmax1) , pcflows[9])
    if len(qmax3)>0: mh[25] = scalardiv(np.mean(qmax3) , pcflows[9])
    if len(qmax7)>0: mh[26] = scalardiv(np.mean(qmax7) , pcflows[9])
    if len(qmax75)>0: mh[27] = scalardiv(np.mean(qmax75) , pcflows[9])

    # seasonal stats ---------------------------------------------------------------------
    mavg,mmed,mmax,mmin,mcv = calc.monthly_stats(qdq,qdy,qdm)                # seasonal pattern stats
    
    for mon in range(0,12):
        iavg = mon + 12                     # iavg = 12 onwards
        icv = mon + 24                      # icv = 24 onwards
        imin = mon + 1                      # imin = 1 onwards
        imax = mon + 1                      # imax = 1 onwards
        
        ma[iavg] = meanORmed(mavg[:,mon],par['opt_median'])                    # ma12-23 = average daily flows in each month
        ma[icv]  = meanORmed( mcv[:,mon],par['opt_median'])                    # ma24-35 = cv of daily flows in each month
        ml[imin] = meanORmed(mmin[:,mon],par['opt_median'])                    # ml1-12 = min daily flow in each month
        mh[imax] = meanORmed(mmax[:,mon],par['opt_median'])                    # mh1-12 = max daily flow in each month
        
    return ma, ml, mh

# ==================================================================================================================
# Calculating indices for frequencies of low / high events
# Inputs qd, qm, qy, and par are generated in "CalcHITdaily" where this subroutine is called from.
def calcday_fl_fh(qd,qm,qy,par):

    fl = pd.Series(index=range(1,4),dtype='float64')
    fh = pd.Series(index=range(1,12),dtype='float64')
    
    # Low flows (LF) ----------------------------------------------------------------------
    zeroflows = par['opt_nonzero']                                             # zero flow cutoff in ml per day                                       
    qdq = qd['q'].to_numpy()
    qdy = qd['y'].to_numpy()
    qdm = qd['m'].to_numpy()
    qdd = qd['d'].to_numpy()

    #qd_nz,qdy_nz,qdm_nz,qdd_nz = calc.nonzeroflows(qdq,qdy,qdm,qdd,zeroflows)  # get a series of nonzero flows and matching dates
    qd_nz = qd[qd['q']>=zeroflows]['q'].to_numpy()
    
                        #=============================
                        # Notes about spell event identification
                        #
                        # Many indicators rely on spell event calculations.
                        # In the EflowStats R code from the USGS, spells are
                        # identified when the flow is LESS THAN the threshold, 
                        # and spells which span 2 calendar years are counted
                        # in both years (ie. double counted).
                        #
                        # Here the code identifies a spell as flow
                        # LESS THAN OR EQUAL TO the threshold, and only counts
                        # each spell based on its start date.
                        #
                        # For example, a spell starting on Dec 25th for a
                        # duration of 60 days is counted in the first year ONLY.
                        #=============================

    thresh = np.percentile(qd_nz,25)                                           # 25th pc (=75th exceedance)   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh) # spell stats
    if np.sum(annstarts)>0:
        fl[1] = meanORmed(annstarts,par['opt_median'])                         # if any spells, get average number per year
        fl[2] = np.std(annstarts,ddof=1) *100 / np.mean(annstarts)             # if any spells, std x 100 / mean
    else:
        fl[1] = fl[2] = 0.0
        
    thresh = np.mean(qdq) * 0.05                                               # 5pc of mean daily flow
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh) # spell stats
    if np.sum(annstarts)>0:
        fl[3] = meanORmed(annstarts,par['opt_median'])                         # if any spells, get average number per year
    else:
        fl[3] = 0.0
    
    # High flows based on means and percentiles -------------------------------------------
    thresh = np.percentile(qd_nz,75)                                           # 75th pc (=25th exceedance)   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[1] = meanORmed(annstarts,par['opt_median'])                         # if any spells, get average number per year
        fh[2] = np.std(annstarts,ddof=1) *100 / np.mean(annstarts)             # if any spells, std x 100 / mean
        fh[8] = fh[1]
    
    thresh = np.median(qd_nz) * 3.0                                            # median daily flow x 3         <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[3] = meanORmed(anndays,par['opt_median'])                           # avg days per year
        fh[6] = meanORmed(annstarts,par['opt_median'])                         # avg events per year
    
    thresh = np.median(qd_nz) * 7.0                                            # median daily flow x 7         <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[4] = meanORmed(anndays,par['opt_median'])                           # avg days per year
        fh[7] = meanORmed(annstarts,par['opt_median'])                         # avg events per year

    thresh = np.median(qd_nz)                                                  # median daily flow             <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[5] = meanORmed(annstarts,par['opt_median'])                         # avg events per year
    
    thresh = np.percentile(qd_nz,25)                                           # 25th pc (=75th exceedance)    <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[9] = meanORmed(annstarts,par['opt_median'])                         # avg events per year

    thresh = np.median(qy['min'])                                              # median of the annual minima   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[10] = meanORmed(annstarts,par['opt_median'])                        # avg events per year
    
    thresh = par['floodthresh']                                                # 1.67 year annual flood        <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        fh[11] = meanORmed(annstarts,par['opt_median'])                        # avg events per year
    
    return fl, fh

# ==================================================================================================================
# Calculating indices for durations of low / high events
# Inputs qd, qm, qy, and par are generated in "CalcHITdaily" where this subroutine is called from.
def calcday_dl_dh(qd,qm,qy,par):

    dl = pd.Series(index=range(1,21),dtype='float64')
    dh = pd.Series(index=range(1,25),dtype='float64')
    
    zeroflows = par['opt_nonzero']                                             # zero flow cutoff in ml per day                                       
    qdq = qd['q'].to_numpy()
    qdy = qd['y'].to_numpy()
    qdm = qd['m'].to_numpy()
    qdd = qd['d'].to_numpy()

    #qd_nz,dy_nz,dm_nz,dd_nz = calc.nonzeroflows(qdq,qdy,qdm,qdd,zeroflows)     # get a series of nonzero flows and matching dates
    qd_nz = qd[qd['q']>=zeroflows]['q'].to_numpy()
    
    qd_nzmed = np.median(qd_nz)
    
    # rolling stats -----------------------------------------------------------
    # min_1d,max_1d = calc.rolling_minmax(qdq,qdy,1)                             # rolling seasonal mins/maxs
    # min_3d,max_3d = calc.rolling_minmax(qdq,qdy,3)
    # min_7d,max_7d = calc.rolling_minmax(qdq,qdy,7)
    # min_1m,max_1m = calc.rolling_minmax(qdq,qdy,30)
    # min_3m,max_3m = calc.rolling_minmax(qdq,qdy,90)

    min_1d = qd['q'].resample('AS').min()
    max_1d = qd['q'].resample('AS').max()
    min_3d = qd['q'].rolling(3,center=True).mean().resample('AS').min()
    max_3d = qd['q'].rolling(3,center=True).mean().resample('AS').max()
    min_7d = qd['q'].rolling(7,center=True).mean().resample('AS').min()
    max_7d = qd['q'].rolling(7,center=True).mean().resample('AS').max()
    min_1m = qd['q'].rolling(30,center=True).mean().resample('AS').min()
    max_1m = qd['q'].rolling(30,center=True).mean().resample('AS').max()
    min_3m = qd['q'].rolling(90,center=True).mean().resample('AS').min()
    max_3m = qd['q'].rolling(90,center=True).mean().resample('AS').max()
    
    # low flow stats ----------------------------------------------------------
    dl[1] = meanORmed(min_1d,par['opt_median'])                                # average minima
    dl[2] = meanORmed(min_3d[1:-1],par['opt_median'])                          # [1:-1] strips out the first and last values
    dl[3] = meanORmed(min_7d[1:-1],par['opt_median'])
    dl[4] = meanORmed(min_1m[1:-1],par['opt_median'])
    dl[5] = meanORmed(min_3m[1:-1],par['opt_median'])
    
    dl[6] = scalardiv(np.std(min_1d,ddof=1), np.mean(min_1d)) * 100            # variability of minima
    dl[7] = scalardiv(np.std(min_3d[1:-1],ddof=1), np.mean(min_3d[1:-1])) * 100
    dl[8] = scalardiv(np.std(min_7d[1:-1],ddof=1), np.mean(min_7d[1:-1])) * 100
    dl[9] = scalardiv(np.std(min_1m[1:-1],ddof=1), np.mean(min_1m[1:-1])) * 100
    dl[10] = scalardiv(np.std(min_3m[1:-1],ddof=1), np.mean(min_3m[1:-1])) * 100
    
    dl[11] = scalardiv(np.mean(min_1d) , qd_nzmed)
    dl[12] = scalardiv(np.mean(min_7d[1:-1]) , qd_nzmed)
    dl[13] = scalardiv(np.mean(min_1m[1:-1]) , qd_nzmed)
    
    thresh = np.percentile(qd_nz,25)                                           # 25th pc (=75th exceedance)   <====
    dl[14] = scalardiv(thresh , qd_nzmed)
    
    thresh = np.percentile(qd_nz,10)                                           # 10th pc (=90th exceedance)   <====
    dl[15] = scalardiv(thresh , qd_nzmed)
    
    thresh = np.percentile(qd_nz,25)                                           # 25th pc (=75th exceedance)   <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh)    # spell stats
    if np.sum(annstarts)>0:
        starts_df = pd.DataFrame(starts,columns=['dur','int','y','m','d'])     # convert starts to a dataframe ready to aggregate
        starts_df.index = pd.to_datetime(starts_df.y*10000+starts_df.m*100+starts_df.d,format='%Y%m%d')
        ann = starts_df['dur'].resample('AS').mean().fillna(0)                 # get the avg event duration each year, fill nans
        dl[16] = meanORmed(ann,par['opt_median'])                              # if any spells, get median of yearly av durations
        dl[17] = ann.std(ddof=1) *100 / ann.mean()                             # if any spells, std x 100 / mean
    else:
        dl[16] = 0.0
        dl[17] = 0.0

    thresh = par['zeroflowday']                                                # spells of zero flow          <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh)    # spell stats
    if np.sum(annstarts)>0:
        dl[18] = meanORmed(anndays,par['opt_median'])                          # if any spells, get average number per year
        dl[19] = np.std(anndays,ddof=1) *100 / np.mean(anndays)                # if any spells, std x 100 / mean
    else:
        dl[18] = 0.0
        dl[19] = 0.0
    
    thresh = par['zeroflowmonth']
    dl[20] = len(qm[qm['sum']<=thresh]) / len(qm)                              # zero months / total months
    
    # high flow stats -------------------------------------------------------------------
    dh[1] = meanORmed(max_1d,par['opt_median'])                                # average maxima
    dh[2] = meanORmed(max_3d[1:-1],par['opt_median'])
    dh[3] = meanORmed(max_7d[1:-1],par['opt_median'])
    dh[4] = meanORmed(max_1m[1:-1],par['opt_median'])
    dh[5] = meanORmed(max_3m[1:-1],par['opt_median'])
    
    dh[6] = np.std(max_1d,ddof=1) * 100 / np.mean(max_1d)                      # variability of maxima
    dh[7] = np.std(max_3d[1:-1],ddof=1) * 100 / np.mean(max_3d[1:-1])
    dh[8] = np.std(max_7d[1:-1],ddof=1) * 100 / np.mean(max_7d[1:-1])
    dh[9] = np.std(max_1m[1:-1],ddof=1) * 100 / np.mean(max_1m[1:-1])
    dh[10] = np.std(max_3m[1:-1],ddof=1) * 100 / np.mean(max_3m[1:-1])
    
    dh[11] = scalardiv(np.mean(max_1d) , qd_nzmed)
    dh[12] = scalardiv(np.mean(max_7d[1:-1]) , qd_nzmed)
    dh[13] = scalardiv(np.mean(max_1m[1:-1]) , qd_nzmed)
    
    dh[14] = qm['avg'].quantile(0.95) / qm['avg'].mean()
    
    thresh = np.percentile(qd_nz,75)                                         # 75th pc (=25th exceedance)   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    if np.sum(annstarts)>0:
        starts_df = pd.DataFrame(starts,columns=['dur','int','y','m','d'])     # convert starts to a dataframe ready to aggregate
        starts_df.index = pd.to_datetime(starts_df.y*10000+starts_df.m*100+starts_df.d,format='%Y%m%d')
        ann = starts_df['dur'].resample('AS').mean().fillna(0)                 # get the avg event duration each year, fill nans
        dh[15] = ann.median()                                                  # if any spells, get median of yearly av durations
        dh[16] = ann.std(ddof=1) *100 / ann.mean()                             # if any spells, std x 100 / mean
    else:
        dh[15] = 0.0
        dh[16] = 0.0
    
    thresh = np.median(qd_nz)                                                  # median daily flow            <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[17] = meanORmed(starts[:,0],par['opt_median'])                          # mean event duration (col 0)

    thresh = np.median(qd_nz) * 3                                              # median daily flow x 3        <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[18] = meanORmed(starts[:,0],par['opt_median'])                          # mean event duration (col 0)

    thresh = np.median(qd_nz) * 7                                              # median daily flow x 7        <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[19] = meanORmed(starts[:,0],par['opt_median'])                          # mean event duration (col 0)

    thresh = np.percentile(qd_nz,75)                                         # 75th pc (=25th exceedance)   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[20] = meanORmed(starts[:,0],par['opt_median'])                          # mean event duration (col 0)

    thresh = np.percentile(qd_nz,25)                                         # 25th pc (=75th exceedance)   <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[21] = meanORmed(starts[:,0],par['opt_median'])                          # mean event duration (col 0)
    
                        # =============================
                        # Indices dh22/23/24 have been modified from the
                        # original EflowStats R code formulation. A note in
                        # the R code suggests that it might not be quite
                        # right anyway.
                        #
                        # Returning to the original Poff and Ward (1989) paper
                        # it looks like dh22/23 should be the raw mean interval
                        # and duration (days) and should NOT be calculated in
                        # days per year like EflowStats.
                        #
                        # the definition of dh24 is not clear in any reference
                        # so has been set to the mean number of non-flood days
                        # per year, which seems like the intent.
                        # =============================
    
    thresh = par['floodthresh']                                                # 1.67 year flood              <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    dh[22] = meanORmed(starts[:,1],par['opt_median'])                          # interval (mean days) (col 1)
    dh[23] = meanORmed(starts[:,0],par['opt_median'])                          # duration (mean days) (col 0)
    
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh) # spell stats BELOW the flood
    dh[24] = meanORmed(anndays,par['opt_median'])                              # flood free days (mean days per yr)

    return dl, dh

# ==================================================================================================================
# Calculating indices for timing and predictability of flows
# Inputs qd, qm, qy, and par are generated in "CalcHITdaily" where this subroutine is called from.
def calcday_ta_tl_th(qd,qm,qy,par):

    ta = pd.Series(index=range(1,4),dtype='float64')
    tl = pd.Series(index=range(1,5),dtype='float64')
    th = pd.Series(index=range(1,4),dtype='float64')
    
    # setting up colwell calculations ----------------------------------------
    qdq = qd['q'].to_numpy()
    qdy = qd['y'].to_numpy()
    qdm = qd['m'].to_numpy()
    qdd = qd['d'].to_numpy()
    qddoy = qd['doy'].to_numpy()
    
    logmeanflow = np.log10(np.mean(qdq))
    
                # ===========================
                # create bins as follows:
                #   bin1 = 0 x logmeanflow -> 0.1 x logmeanflow
                #   bin2 = 0.1 x logmeanflow -> 0.25 x logmeanflow
                #   ...
                #   bin12 = 2.25 x logmeanflow -> 100 x logmeanflow  (should be everything above 2.25)
                # ===========================
                
    bins = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 20])   # max bin based on gauge A5090503 with a whopping ratio of 15.73
    bins = bins * logmeanflow                                                  # multiply values by logmeanflow
    bins = 10**bins                                                            # reverse the log10
    bins[0] = 0.0                                                              # set first bin to zero
    
    p,c,m = calc.colwell(qdq,bins,qddoy,365)                                   # get colwell stats

    ta[1] = c                                                                  # constancy
    ta[2] = p                                                                  # predictability
    
                # ===========================
                # CHANGED FORMULATIONS
                #
                # ta3 and tl3 ---------------
                #
                # the formula as described in the HIT user manual and the 
                # EflowStats R code calculate these indices using a method
                # based on a rolling 2 month window throughout the entire
                # flow record. THIS IS INCORRECT, and a bit weird as indices go.
                #
                # IN CONTRAST, the original formulation described in Poff (1996) 
                # uses six 2-month windows as 'bins' to form a histogram, which
                # makes much more hydrological sense.
                #
                # the code presented here follows the intent of the original
                # Poff (1996) formulation, based on NUMBERS OF DAYS per
                # window, rather than numbers of events per window.
                #
                # tl4 and th3----------------
                #
                # the formulation as described in the HIT user manual and the 
                # EflowStats R code calculate these indices using a method
                # based on a 'length of spell event' basis.
                #
                # IN CONTRAST, Poff (1996) and Olden & Poff (2003) suggest an
                # approach based on a historgram representing days of the year
                # from 1 to 365.
                #
                # the code presented here follows the intent of the original
                # Poff (1996) formulation, based on HISTOGRAMS over the year
                # rather than event durations.
                # ===========================
                
    # 1.67 year flood statistics ---------------------------------------------
    thresh = par['floodthresh']                                                # 1.67 year flood              <====
    starts,qvol,qmax,spellon,annstarts,anndays = calc.spellsabove(qdq,qdd,qdm,qdy,thresh)   # spell stats
    
    spellon_sr = pd.Series(spellon,index=qd.index)                             # get series of spellon (binary spell on/off = 1/0)
    spellon_2m = spellon_sr.resample('2MS').sum()                              # resample to 2m intervals
    hist_2m = spellon_2m.groupby(spellon_2m.index.month).sum()                 # histogram of num spell days for each 2 month window
    hist_365d = spellon_sr.groupby(spellon_sr.index.dayofyear).sum()           # histogram of num spell days for each day of the year
    
    ta[3] = hist_2m.max() / hist_2m.sum()                                      # major spell window / all windows
    th[3] = hist_365d[hist_365d==0].count() / 365                              # number of non-flood days / 365
    
    # high/low flow day ------------------------------------------------------
    annvol,annmax,annmin,annmaxday,annminday,anncvs = calc.annual_stats(qdq,qdy,qddoy) # annual stats

    tl[1] = scistat.circmean(annminday,high=366,low=1)
    tl[2] = scistat.circstd(annminday,high=366,low=1)
    
    th[1] = scistat.circmean(annmaxday,high=366,low=1)
    th[2] = scistat.circstd(annmaxday,high=366,low=1)
    
    # 5 yr dry statistics ----------------------------------------------------
    thresh = par['drythresh']                                                  # 5 year low flow day              <====
    starts,spellon,annstarts,anndays = calc.spellsbelow(qdq,qdd,qdm,qdy,thresh)   # spell stats

    spellon_sr = pd.Series(spellon,index=qd.index)                             # get series of spellon (binary spell on/off = 1/0)
    spellon_2m = spellon_sr.resample('2MS').sum()                              # resample to 2m intervals
    hist_2m = spellon_2m.groupby(spellon_2m.index.month).sum()                 # histogram of num spell days for each 2 month window
    hist_365d = spellon_sr.groupby(spellon_sr.index.dayofyear).sum()           # histogram of num spell days for each day of the year
    
    tl[3] = hist_2m.max() / hist_2m.sum()                                      # major spell window / all windows
    tl[4] = hist_365d[hist_365d==0].count() / 365                              # number of non-lowflow days / 365
    
    return ta, tl, th

# ==================================================================================================================
# Calculating indices for rise and fall rates
# Inputs qd, qm, qy, and par are generated in "CalcHITdaily" where this subroutine is called from.
def calcday_ra(qd,qm,qy,par):

    ra = pd.Series(index=range(1,10),dtype='float64')

    # rise and fall ---------------------------------------------------------------------
    zeroflows = par['opt_nonzero']                                             # zero flow cutoff in ml per day
    
    qdq = qd['q'].to_numpy()
    qdy = qd['y'].to_numpy()

    chngrates,chngpc,numrevs,flipdays,annflips = calc.risefall(qdq,qdy)        # rise and fall stats
    
    rates = chngpc                                                             # 'rates' by default means pc change (today - prevday) / prevday
    if par['opt_risefall'] == 'absdiff':                                       # option to make 'rates' = absolute diff from prev day (today - prevday)
        rates = chngrates
    ratesr = rates[rates>0]
    ratesf = rates[rates<0]
    
    ra[1] = meanORmed(ratesr,par['opt_median'])                                # avg rise rate
    ra[2] = np.std(ratesr,ddof=1) * 100 / np.mean(ratesr)                      # cv of rise rates
    
    ra[3] = -1 * meanORmed(ratesf,par['opt_median'])                           # avg fall rate, modified to ensure result is always >=0
    ra[4] = -1 * np.std(ratesf,ddof=1) * 100 / np.mean(ratesf)                 # cv of fall rates
    
    ra[5] = np.count_nonzero(ratesr) / len(qd)                                          # num rises/yr
    
    ra[8] = meanORmed(annflips,par['opt_median'])                              # average number of flips/yr
    ra[9] = np.std(annflips,ddof=1) * 100 / np.mean(annflips)                  # cv of flips/yr
    
    qdq_log10 = np.where(qdq > 0, qdq, zeroflows)                                    # Log10 of flows, exchange the zeros for a positive value
    qdq_log10 = np.log10(qdq_log10)
    chngrates,chngpc,numrevs,flipdays,annflips = calc.risefall(qdq_log10,qdy)  # LOG10 rise and fall stats

    rates = chngpc                                                             # 'rates' by default means pc change (today - prevday) / prevday
    if par['opt_risefall'] == 'absdiff':                                       # option to make 'rates' = absolute diff from prev day (today - prevday)
        rates = chngrates
    ratesr = rates[rates>0]
    ratesf = rates[rates<0]

    ra[6] = np.median(ratesr)                                                  # median log10 rise
    ra[7] = -1 * np.median(ratesf)                                             # median log10 fall
    
    return ra

# ==================================================================================================================
# calculate statistics given DAILY flows for one catchment at a time -----
# Inputs are generated in "Stats_main.py" where this subroutine is called from.
def calcHITDaily(flow_df,opt_median,catskm2,units):

    catchments = flow_df.columns.to_list()
    numcats = len(catchments)
    
    results = pd.DataFrame(data=None,columns=catchments)                       # set up results dataframe (rows = stats, cols = sites)
    timings = pd.DataFrame(data=None,index=['setup',
                                            'pars',
                                            'm',
                                            'f',
                                            'd',
                                            't',
                                            'r',
                                            'out',
                                            'total'],columns=catchments)    
    
    loopreport = 50

    for icat,cat in enumerate(catchments):
        
        t1 = time.time()
        if icat % loopreport == 0:
            print ('      Catchment ' + str(icat+1) + ' of ' + str(numcats))

        single_flow = trimseries(flow_df[cat])
        
        if units=='mm':
            single_flow = single_flow / catskm2[cat]
            
        if (single_flow.max() - single_flow.min() > 1):                        # If its a valid series (not all 0's or 5's)
            
            # set up some basic stats and series ------------------------------
            qd = pd.DataFrame(single_flow)                                     # basic daily flow dataframe
            qd.columns = ['q']
            qd['y'] = qd.index.year
            qd['m'] = qd.index.month
            qd['d'] = qd.index.day
            qd['doy'] = qd.index.dayofyear
            
            qm = pd.DataFrame(single_flow.resample('MS').sum())                # aggregate to monthly for some stats
            qm.columns = ['sum']
            qm['avg'] = single_flow.resample('MS').mean()
            qm['min'] = single_flow.resample('MS').min()
            qm['max'] = single_flow.resample('MS').max()
            qm['y'] = qm.index.year
            qm['m'] = qm.index.month
            
            qy = pd.DataFrame(single_flow.resample('AS').sum())                # aggregate to annual for some stats
            qy.columns=['sum']
            qy['avg'] = single_flow.resample('AS').mean()
            qy['med'] = single_flow.resample('AS').median()
            qy['min'] = single_flow.resample('AS').min()
            qy['max'] = single_flow.resample('AS').max()
            qy['std'] = single_flow.resample('AS').std(ddof=1)
            qy['mmax'] = single_flow.resample('AS').max()
            qy['y'] = qy.index.year
            
            t2 = time.time()
            par = {}
            par['catarea'] = catskm2[cat]                                      # catchment area in km2

            ari = 1.666667
            par['floodthresh'] = calc.aepthresh(qd['q'].to_numpy(),
                                                qd['y'].to_numpy(),
                                                ari,'high')                      # 1.67yr flood thresh (60%ile) using ann series method
            ari = 5
            par['drythresh'] = calc.aepthresh(qd['q'].to_numpy(),
                                              qd['y'].to_numpy(),
                                              ari,'low')                       # 5yr flood thresh (20%ile) using ann series method
            par['zeroflowday'] = 0.1                                           # 0.1 ML/d
            par['zeroflowmonth'] = 10                                          # 10 ML/m
            
            par['opt_median'] = opt_median
            par['opt_nonzero'] = 0.1                                           # threshold for filtering out zero flow months/days
            
            par['opt_risefall'] = 'pcdiff'    #'absdiff'    #'pcdiff'          # option for rate of change calcs (absolute flow diff vs pc change)
            
            if units=='mm':
                par['catarea'] = 1.0
                par['zeroflowday'] = 0.1 / catskm2[cat]
                par['zeroflowmonth'] = 10 / catskm2[cat]
                par['opt_nonzero'] = 0.1 / catskm2[cat]
                
            # Get magnitude stats (avg, low, high) ----------------------------
            #print('    ma, ml, mh')
            t3 = time.time()
            ma, ml, mh = calcday_ma_ml_mh(qd,qm,qy,par)

            # Low flow and high flow stats ------------------------------------
            #print('    fl, fh')
            t4 = time.time()
            fl, fh = calcday_fl_fh(qd,qm,qy,par)
            
            # Duration stats --------------------------------------------------
            #print('    dl, dh')
            t5 = time.time()
            dl, dh = calcday_dl_dh(qd,qm,qy,par)
            
            # Timing stats ----------------------------------------------------
            #print('    ta, tl, th')
            t6 = time.time()
            ta, tl, th = calcday_ta_tl_th(qd,qm,qy,par)
            
            # Rise and fall stats ---------------------------------------------
            #print('    ra')
            t7 = time.time()
            ra = calcday_ra(qd,qm,qy,par)
            
            # combine into a single series with the right index ---------------
            #print('    combining results arrays')
            t8 = time.time()
            ma.index = 'MA' + ma.index.astype(str)
            ml.index = 'ML' + ml.index.astype(str)
            mh.index = 'MH' + mh.index.astype(str)
            fl.index = 'FL' + fl.index.astype(str)
            fh.index = 'FH' + fh.index.astype(str)
            dl.index = 'DL' + dl.index.astype(str)
            dh.index = 'DH' + dh.index.astype(str)
            ta.index = 'TA' + ta.index.astype(str)
            tl.index = 'TL' + tl.index.astype(str)
            th.index = 'TH' + th.index.astype(str)
            ra.index = 'RA' + ra.index.astype(str)
            
            # add to results dataframe ----------------------------------------
            allframes = [ma, ml, mh, fl, fh, dl, dh, ta, tl, th, ra]
            
            results.loc[:,cat] = pd.concat(allframes)
            t9 = time.time()
            
            timings.loc['setup',cat] = t2-t1
            timings.loc['pars',cat] = t3-t2
            timings.loc['m',cat] = t4-t3
            timings.loc['f',cat] = t5-t4
            timings.loc['d',cat] = t6-t5
            timings.loc['t',cat] = t7-t6
            timings.loc['r',cat] = t8-t7
            timings.loc['out',cat] = t9-t8
            timings.loc['total',cat] = t9-t1
            
    return results #,timings




