"""
Flow statistics routines

Trying to use Numba wherever possible

R Morden
24 Dec 2020 (yes I am working on Christmas Eve)
16 Feb 2021 (Added more)
5 Aug 2021 (Added Predictability calcs)

"""

import numpy as np
from numba import jit
import pandas as pd
import scipy.stats as scistat

# ==================================================================================================================
# Annual series
# calculate annual vols, mins, maxs, minmonth/day, maxmonth/day
# Works for MONTHLY or DAILY
#@jit(nopython=True)
def annual_stats(flow,dy,dstep):                             # dstep is either the month number (1-12) or the day number (1 to 366)

    numflows = len(flow)
    allyears = np.unique(dy)                                 # get a list of unique years
    numyears = len(allyears)

    annvol = np.zeros(numyears)                              # 1 column, 1 row per year
    anncvs = np.zeros(numyears)                              # 1 column, 1 row per year
    annstd = np.zeros(numyears)                              # 1 column, 1 row per year
    annavg = np.zeros(numyears)                              # 1 column, 1 row per year
    
    annmax = np.zeros(numyears)                              # 1 column, 1 row per year
    annmin = np.full(numyears,9999999.0)                     # 1 column, 1 row per year, need to initialise with a big number
    annmaxstep = np.zeros(numyears)                          # 1 column, 1 row per year
    annminstep = np.zeros(numyears)                          # 1 column, 1 row per year
    
    for i in range(numflows):
        thisyear = dy[i]
        thisstep = dstep[i]
        idxthisyear = np.where(allyears==thisyear)[0][0]     # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        
        if flow[i] < annmin[idxthisyear]:
            annmin[idxthisyear] = flow[i]
            annminstep[idxthisyear] = thisstep
        if flow[i] > annmax[idxthisyear]:
            annmax[idxthisyear] = flow[i]
            annmaxstep[idxthisyear] = thisstep
    
    for i in range(numyears):
        thisyear = allyears[i]
        cond = (dy==thisyear)
        annvol[i] = np.sum(flow[cond])
        annavg[i] = np.mean(flow[cond])
        annstd[i] = std_s(flow[cond],ddof=1)
        if annavg[i]>0:
            anncvs[i] = scalardiv(annstd[i] , annavg[i])
        else:
            anncvs[i] = 0.0
        
    return annvol,annmax,annmin,annmaxstep,annminstep,anncvs

# ==================================================================================================================
# Seasonal series
# calculate 12 monthly means, medians, mins, maxs
# Works for MONTHLY only
@jit(nopython=True)
def seasonal_stats(flow,dm):
    
    seasavg = np.zeros(12)
    seasmed = np.zeros(12)
    seasmax = np.zeros(12)
    seasmin = np.zeros(12)
    
    for i in range(12):
        seas = flow[dm==i+1]
        seasavg[i] = np.mean(seas)
        seasmed[i] = np.median(seas)
        seasmax[i] = np.max(seas)
        seasmin[i] = np.min(seas)
    
    return seasavg,seasmed,seasmax,seasmin

# ==================================================================================================================
# Annual series for each month
# calculate 12 monthly means, medians, mins, maxs in ML/d or ML/m
# Works for DAILY only 
@jit(nopython=True)
def monthly_stats(flow,dy,dm):
    
    allyears = np.unique(dy)                                   # get a list of unique years
    numyears = len(allyears)
    
    mavg = np.zeros((numyears,12))                             # set up results arrays
    mmed = np.zeros((numyears,12))
    mmin = np.zeros((numyears,12))
    mmax = np.zeros((numyears,12))
    mcv = np.zeros((numyears,12))
    
    for y in range(numyears):
        for m in range(12):
            thismonth = flow[(dy==allyears[y]) * (dm==m+1)]    # slice to get a single month of flows
            mavg[y,m] = np.mean(thismonth)                     # get stats for that month
            mmed[y,m] = np.median(thismonth)
            mmin[y,m] = np.min(thismonth)
            mmax[y,m] = np.max(thismonth)
            if mavg[y,m]==0:
                mcv[y,m] = 0.0
            else:
                mcv[y,m] = std_s(thismonth,ddof=1) / mavg[y,m] * 100
        
    return mavg,mmed,mmax,mmin,mcv                             # each variable is numyears x 12

# ==================================================================================================================
# Standard deviation for a sample (n-1)
# Using NUMBA, the usual "std" function cannot accept any arguments.
# This means that it defaults to the population std where ddof=0
# This routine gives the std where ddof can be specified.
# Inarray = input numpy array (1d)
# ddof = degrees of freedom, typically 1 for a hydrological sample
@jit(nopython=True)
def std_s(inarray,ddof):
    n = len(inarray)
    mean_s = np.mean(inarray)
    sum_sq_diff = np.sum((inarray - mean_s)**2)
    sd = np.sqrt(sum_sq_diff / (n-ddof))
    return sd

# ==================================================================================================================
# Spells below for a single 1d series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def spellsbelow(flows,dd,dm,dy,threshold):
    
    numflows = len(flows)                              # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)
    
    dur = []                                           # output variables
    intafter = []
    starty = []
    startm = []
    startd = []
    spellon = np.zeros(numflows)
    annstarts = np.zeros(numyears)
    anndays = np.zeros(numyears)
    
    count = 0                                          # looping variables
    prevflow = 999999999.0
    
    for i in range(numflows):
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]    # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        if flows[i] <= threshold:                      # spell event in progress
            if prevflow > threshold:                      # spell event just started
                count = count + 1                            # +1 spell
                dur.append(0)                                # start recording duration
                intafter.append(0)
                starty.append(dy[i])                         # record start date
                startm.append(dm[i])
                startd.append(dd[i])
                annstarts[iyr] = annstarts[iyr] + 1          # +1 start for this year
            spellon[i] = 1                                # mark this day as having active spell
            dur[count-1] = dur[count-1] + 1               # +1 duration
            anndays[iyr] = anndays[iyr] + 1               # +1 spell day for this year
        elif count >= 1:                               # if there has been at least 1 spell
            intafter[count-1] = intafter[count-1] + 1     # +1 to the interval after the event
            
        prevflow = flows[i]
        
    starts = np.asarray([dur,intafter,starty,startm,startd]).T
    
    return starts, spellon, annstarts, anndays

# ==================================================================================================================
# Spells above for a single 1d series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def spellsabove(flows,dd,dm,dy,threshold):
    
    numflows = len(flows)                              # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)
    
    dur = []                                           # output variables
    intafter = []
    qvol = []
    qmax = []
    starty = []
    startm = []
    startd = []
    spellon = np.zeros(numflows)
    annstarts = np.zeros(numyears)
    anndays = np.zeros(numyears)

    count = 0                                          # looping variables
    prevflow = 0.0
    
    for i in range(numflows):
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]    # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        if flows[i] > threshold:                       # spell event in progress
            if prevflow <= threshold:                     # spell event just started
                count = count + 1                            # +1 spell
                dur.append(0)                                # add a duration entry
                qvol.append(0.0)                             # add an event volume entry
                qmax.append(0.0)                             # add an event max flow entry
                intafter.append(0)                           # add an interval entry
                starty.append(dy[i])                         # record start date
                startm.append(dm[i])
                startd.append(dd[i])
                annstarts[iyr] = annstarts[iyr] + 1          # +1 start for this year
            
            spellon[i] = 1                                # mark this day as having active spell
            dur[count-1] = dur[count-1] + 1.0            # +1 duration
            qvol[count-1] = qvol[count-1] + flows[i]     # aggregate volume for this event
            qmax[count-1] = max(qmax[count-1],flows[i])  # get max vol for this event
            anndays[iyr] = anndays[iyr] + 1               # +1 spell day for this year
        
        elif count >= 1:                               # if there has been at least 1 spell
            intafter[count-1] = intafter[count-1] + 1     # +1 to the interval after the event

        prevflow = flows[i]                              # reset
    
    starts = np.asarray([dur,intafter,starty,startm,startd]).T
    
    return starts, qvol, qmax, spellon, annstarts, anndays

# ==================================================================================================================
# Flow percentiles
# based on an array of percentiles and a flow series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def flowpercentiles(flows,parray):
    
    if len(flows)==0:
        pflows = parray*0.0
    elif max(flows) < 0.1:
        pflows = parray*0.0
    else:
        parray_int = 100-parray
        pflows = np.percentile(flows,parray_int)
    
    return pflows

# ==================================================================================================================
# Calculate rates of rise and fall
# Assumed to be averaged across the entire record, dates not required
@jit(nopython=True)
def risefall(flows,dy):
    
    numflows = len(flows)                                                      # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)

    annflips = np.zeros(numyears)                                              # annflips is the number of reversals in each caendar year
    diff = np.zeros(numflows)                                                  # diff is the increase from the prev step in ML
    rate = np.zeros(numflows)                                                  # rate is the proportional change (increase from the prev timestep divided by the prev flow)
    change = np.zeros(numflows)                                                # change is a rise or fall flag (1 or -1 only)
    flip = np.zeros(numflows)                                                  # flip is set to 1 if there is a reversal
    
    for i in range(1,numflows):                                                # first step is ignored
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]    # 'where' returns a tuple of np arrays - need array entry 1 of tuple entry 1

        diff[i] = flows[i] - flows[i-1]                                        # calculate diff
        if flows[i-1] == 0:
            rate[i] = 0.0
        else:
            rate[i] = diff[i] / flows[i-1]

        if i==1:                                                               # on the first loop
            diff[0] = diff[1]                                                  # initialise the diff to avoid a flip
            rate[0] = rate[1]                                                  # initialse rate to match the second timestep
            if rate[0]<0:                                                      # initialise change to match the second timestep
                change[0] = -1
            elif rate[0]>0:
                change[0] = 1
            else:
                change[0] = -1                                                 # if the second timestep is stady, then assume falling
                
        if rate[i] < 0:                                                        # if falling
            change[i] = -1
        elif rate[i] > 0:                                                      # if rising
            change[i] = 1
        else:                                                                  # if steady then continue from the prev timestep
            change[i] = change[i-1]
        
        if change[i] != change[i-1]:                                           # if change not equal to the prev timestep, its a reversal
            flip[i] = 1
            annflips[iyr] = annflips[iyr] + 1
            
    reversals = np.sum(flip)                         # count flips
    
    return diff,rate,reversals,flip,annflips

# ==================================================================================================================
# Predictability, Constancy, Contingency
# Based on    Colwell RK. 1974. Predictability, Constancy, and Contingency of Periodic Phenomena. Ecology 55: 1148–53.
# I know numpy can do 2D histograms, but we'll see if numba can speed it up
# Inputs - flows=array of flow values, bins=numpy style upper bound of flow categories, seasons=array of months (1-12) or days(1-365)
@jit(nopython=True)
def colwell(flows,bins,seasons,nseasons):
    
    s = len(bins)-1                                                            # number of states, less one to account for the end bin value
    t = nseasons                                                               # number of seasons
    
    hist2d = np.zeros((s,t))                                                   # set up histogram array (r x c = states x seasons)
    #h_array = np.zeros((s,t))                                                  # set up prob array
    flowbins = np.digitize(flows,bins,right=False)                             # assign bin number to each flow
                                                                                  ## 'right=False' means that zero flows
                                                                                  ## will be assigned to the first bin if the
                                                                                  ## first value in the bin variable is zero.
    
    for i in range(len(flows)):                                                # 2d histogram of flows (bins/seasons)
        if seasons[i]<=nseasons:                                                 # if day=366, just skip it
            irow = flowbins[i]-1
            icol = seasons[i]-1
            hist2d[irow,icol] = hist2d[irow,icol] + 1  
    
    z = hist2d.sum()
    
    hx = 0.0
    for mnth in range(t):                                                      # sum counts per season, apply log eqn, sum = HX
        ratio = hist2d[:,mnth].sum() / z
        if ratio > 0:
            hx = hx - (ratio * np.log(ratio))
    
    hy = 0.0
    for state in range(s):                                                     # sum counts per state, apply log eqn, sum = HY
        ratio = hist2d[state,:].sum() / z
        if ratio > 0:
            hy = hy - (ratio * np.log(ratio))
    
    hxy = 0.0
    for irow in range(s):                                                         # sum individual counts, apply log eqn, sum = HXY
        for icol in range(t):
            val = hist2d[irow,icol]
            if val > 0:
                ratio = val / z
                hxy = hxy - (ratio * np.log(ratio))
        
    m = (hx + hy - hxy) / np.log(s)
    c = 1 - (hy / np.log(s))
    p = m + c
    
    return p,c,m    # p = predictibility, c = constancy, m = contingency
    
# ==================================================================================================================
# Calculations for flood or low flow with a specific AEP
# Method based on HIT manual
# q = daily flow (numpy array)
# y = year code for daily flows (numpy array)
# z = inverse of the normal distribution corresponding to the desired AEP (calculate beforehand using scipy.stats.norm.ppf(1-1/ARI))
# highlow = 'high' or 'low' depending on whether high or low flows are being calculated
# NOTE - as an annual series calculation, set the 1yr ARI flow to the min of the annual series 
#@jit(nopython=True)
def aepthresh(q,y,ari,highlow):
    
    # set up required variables -----------------
    allyears = np.unique(y)                                                    # get a list of unique years
    annseries = np.empty(len(allyears))                                        # annual min/max

    # Compute peak annual flows -----------------
    for iyr,yr in enumerate(allyears):                                         # loop through the years
        oneyrslice = q[y==yr]                                                  # get a single year of flow
       
        if highlow == 'high':                                                  # get the min or max of a single year
            annseries[iyr] = np.amax(oneyrslice)
        else:
            annseries[iyr] = np.amin(oneyrslice)

    if highlow == 'high':                                                  # get the z value
        z = scistat.norm.ppf(1-1/ari)
    else:
        z = scistat.norm.ppf(1/ari)
    
    # Take the logs of annmin/annmax, calculate the mean (M) and std dev (SD)
    log_annseries = np.log10(np.where(annseries==0,0.1,annseries))
    m = np.mean(log_annseries)
    sd = std_s(log_annseries, ddof=1)
    
    # Estimate the T=1.67 ARI flood (Qt) from this series from x = M + Z*SD
    # where the Z value corresponding to an AEP of 1/1.67 = norm.s.inv(1-1/1.67)
    #                                                     = -0.25
    # thus Qt = M-0.25*SD
    qt = 10**(m + (sd * z))
    
    if ari>1:
        if highlow == 'high':                                                  # but if the ari=1, just get the min of annual series
            qt = np.min(annseries)
        else:
            qt = np.max(annseries)
        
    return qt

# ==================================================================================================================
# Some index calculations have a user option for mean OR median
def meanORmed(arr,opt):                   
    
    if opt == 'median':
        return np.median(arr)
    elif opt =='mean':
        return np.mean(arr)

# ==================================================================================================================
# Trim a daily flow series to remove NaNs and partial years at the start and end
# Output should be a series which starts on 1st Jan and ends on 31st Dec with no NaNs
def trimseries(qin):
    
    qout = qin.dropna()                                                        # get the portion of the series without NaNs
    
    if qout.index[0].month > 1:                                                # trim to whole calendar years
        newstartdate = pd.to_datetime(str(qout.index[0].year + 1) + '-01-01')
        qout = qout[newstartdate:]
    if qout.index[-1].month < 12:
        newenddate = pd.to_datetime(str(qout.index[-1].year - 1) + '-12-31')
        qout = qout[:newenddate]
    
    return pd.Series(qout)                                                     # return a series trimmed to whole calendar years

# ==================================================================================================================
# Safely divide SCALARS without an error - if the divisor is zero return zero
def scalardiv(num,div):                   
    
    if div == 0:
        result = 0
    else:
        result = num / div
        
    return result

# ==================================================================================================================
# Safely divide ARRAYS without an error - if the divisor is zero return zero
def arraydiv(aa,bb):                   
    
    acopy = np.copy(aa)
    bcopy = np.copy(bb)
    acopy[acopy==0] = np.nan
    bcopy[bcopy==0] = np.nan
    result = aa / bb
    result[np.isnan(result)] = 0.0
    result[np.isinf(result)] = 0.0
    return result

# ==================================================================================================================
# Safely divide Panda series without an error - if the divisor is zero return zero
def seriesdiv(aa,bb):                   
    
    result = aa.div(bb.replace(0,np.nan)).fillna(0.0)
    return result

# ==================================================================================================================
# Print timer information to console
# t is a list of timer increments (float)
# 
# This is a code timer. Use the following code at the top of a function:
#
#      t = []
#      t.append(time.time()) #\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/
#
# ...and then repeat this at every point where you want an elapsed time:
#
#      t.append(time.time()) #\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/\_/
#
# ...and then put this at the end of the function:
#
#      t_print(t)

def t_print(t):
    
    inc = [0.0]                                                                # first inc = 0.0
    
    for i in range(1,len(t)):                                                  # subsequent incs = t(i)-t(i-1)
        inc.append(t[i] - t[i-1])
    
    print(' '.join(f"{x:.2f}" for x in inc))
    