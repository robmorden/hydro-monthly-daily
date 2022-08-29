# -*- coding: utf-8 -*-
"""
Ecohydrological classification of flow regimes at each site based on:
    Kennard MJ, Pusey BJ, Olden JD, et al. 2010. Classification of natural 
    flow regimes in Australia to support environmental flow management. 
    Freshw Biol 55: 171–93.

@author: rmorden
"""

import pandas as pd
import numpy as np
import csv
from numba import jit
import scipy.stats as scistat
import scipy.special as scispec
import math
from PathsFiles import paths,files    #,statslist_mon,statslist_day

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
# Calculations for flood or low flow with a specific AEP
# Method based on HIT manual
#   q = daily flow (pandas SERIES with full datetime index - NOT a dataframe! )
#   ari = flood annual recurrance interval
#   highlow = 'high' or 'low' depending on whether high or low flows are being calculated (only works for annual series)
#   method = 'annual' or 'partial'
#   ind = independance, used to select peaks over threshold for partial series (only needed for partial series)
# NOTE - for an annual series calculation, set the 1yr ARI flow to the min of the annual series
# NOTE - Annual series calcs can work with monthly data 
# DO NOT USE @JIT - JUST USE PANDAS INSTEAD, its easier to get the peaks and sort them out
def aepflow(q,ari,highlow,method,ind):
    
    if method == 'annual':
        
        if highlow == 'high':
            z = scistat.norm.ppf(1-(1/ari))                                    # get the z value
            annseries = q.resample('AS').agg(['max','idxmax'])                 # get the annual series
            annseries['log'] = annseries['max'].clip(lower=0.1)
            annseries['log'] = np.log10(annseries['log'])                      # get the log 10 of peaks (avoid log zero)
        
        else:
            z = scistat.norm.ppf(1/ari)                                        # get the z value
            annseries = q.resample('AS').agg(['min','idxmin'])                 # get the annual series
            annseries['log'] = annseries['min'].clip(lower=0.1)
            annseries['log'] = np.log10(annseries['log'])                      # get the log 10 of peaks (avoid log zero)

        m = annseries['log'].mean()
        sd = annseries['log'].std(ddof=1)
        
        # Estimate the T=1.67 ARI flood (Qt) from this series from x = M + Z*SD
        # where the Z value corresponding to an AEP of 1/1.67 = norm.s.inv(1-1/1.67)
        #                                                     = -0.25
        # thus Qt = M-0.25*SD

        qt = 10**(m + (sd * z))
        
        if ari==1:
            if highlow == 'high':                                              # but if the ari=1, just get the min of annual series
                qt = annseries['max'].min()
            else:
                qt = annseries['min'].max()
    
    elif method == 'partial':    

        n = len(np.unique(q.index.year))                                       # number of years
 
        window = (ind*2)+1                                                     # define window
        rollmax = q.rolling(window=window,center=True).max()                   # rolling window (2xind), fix to centre, calculate peak
        peaks = pd.DataFrame(True,index=q.index,columns=['peak'])              # create dataframe of 'True' values
        cond1 = (rollmax==qd['qs'])
        peaks = peaks.where(cond1,other=False)                                 # if the rolling max centre = q, then flag as peak
        peaks['eqdaychk'] = peaks['peak'].rolling(window=ind).sum()            # check in case there are multiple days with the same peak flow
        cond2 = (peaks['peak']) & (peaks['eqdaychk']==1)                       # flag days which are a peak, exclude subsequent equal flow days
        
        pot = q[cond2]                                                         # select only flagged days
        m = min(len(pot),n)                                                    # number of peaks to select = min (avail peaks, n)
        mpot = pot.nlargest(m).sort_values(ascending=True)                     # select m peaks over a threshold (='mpot')
            
        # Partial series flood estimation using method of L moments (ARR book 3, Chapter 2) ----------------
        # These calcs are based on Tony Ladsons blog and spreadsheet:
        # https://tonyladson.wordpress.com/2019/03/25/fitting-a-probability-model-to-pot-data/
        
        mpot = pd.DataFrame(mpot)
        flw = mpot.columns[0]
        mpot['i'] = range(1,m+1)                                               # number
        mpot['rank'] = m - mpot['i'] + 1                                       # rank in descending order
        mpot['i-1'] = mpot['i'] - 1
        mpot['n-i'] = n - mpot['i']
        mpot['diff'] = mpot['i-1'] - mpot['n-i']
        mpot['diff_x_q'] = mpot['diff'] * mpot[flw]
        mpot['ey'] = (mpot['rank']-0.4) / (n+0.2)                              # ey = expected prob in 1 year = plotting position
        mpot['ari'] = 1 / mpot['ey']                                           # ari for each flow
        
        mpot_subset = mpot.copy()  #loc[mpot['ari']<=10,:].copy()
        lambda1 = mpot_subset[flw].mean()                                             # L moment 1
        lambda2 = mpot_subset['diff_x_q'].sum() / scispec.comb(m,2) / 2               # L moment 2
        beta = lambda2 * 2                                                     # prediction coefficient
        qstar = lambda1 - beta                                                 # prediction coefficient
        
        mpot_subset['qfit'] = qstar - (beta * np.log(mpot_subset['ey']))                     # fitted values of q based on the model
        
        # get qt -----------------------------------
        ey = 1/ari
        qt = qstar - (beta * math.log(ey))                                     # predict using coefficients and natural log of ey
        
        if qt<0:
            qt=mpot_subset[flw].min()
            
    return qt

# ==================================================================================================================
# Spells above for a single 1d series
# Works for MONTHLY or DAILY
@jit(nopython=True)
def spellsabove(flows,dd,dm,dy,threshold,ind):
    
    numflows = len(flows)                                 # basic stats
    allyears = np.unique(dy)
    numyears = len(allyears)
    
    dur = []                                              # output variables
    intafter = []
    qvol = []
    qmax = []
    starty = []
    startm = []
    startd = []
    spellon = np.zeros(numflows)
    annstarts = np.zeros(numyears)
    anndays = np.zeros(numyears)

    ievent = -1                                           # looping variables
    prevflow = 0.0
    indTF = True
    
    for i in range(numflows):
        thisyear = dy[i]
        iyr = np.where(allyears==thisyear)[0][0]          # 'where' returns a tuple of numpy arrays - need array entry 1 of tuple entry 1
        if flows[i] > threshold:                          # spell event in progress
            if prevflow <= threshold:                         # spell event just started
                if indTF:                                         # if it IS independant
                    ievent = ievent + 1                               # +1 spell
                    dur.append(0)                                     # add a duration entry
                    qvol.append(0.0)                                  # add an event volume entry
                    qmax.append(0.0)                                  # add an event max flow entry
                    intafter.append(0)                                # add an interval entry
                    starty.append(dy[i])                              # record start date
                    startm.append(dm[i])
                    startd.append(dd[i])
                    annstarts[iyr] = annstarts[iyr] + 1               # +1 start for this year
                
                else:                                             # if its NOT independent
                    dur[ievent] = dur[ievent] + intafter[ievent]      # add the recent interval back into the spell duration
                    anndays[iyr] = anndays[iyr] + intafter[ievent]    # add the recent interval into the spell days count for this year
                    intafter[ievent] = 0                              # reset the interval to zero
                    
            spellon[i] = 1                                # mark this day as having active spell
            dur[ievent] = dur[ievent] + 1.0                   # +1 duration
            qvol[ievent] = qvol[ievent] + flows[i]            # aggregate volume for this event
            qmax[ievent] = max(qmax[ievent],flows[i])         # get peak vol for this event
            anndays[iyr] = anndays[iyr] + 1                   # +1 spell day for this year
        
        elif ievent >= 0:                             # if there has been at least 1 spell
            intafter[ievent] = intafter[ievent] + 1       # +1 to the interval after the event
            if intafter[ievent] >= ind:                   # spell is independent when interval is long enough
                indTF = True
            else:
                indTF = False
                
        prevflow = flows[i]                           # reset
    
    starts = np.asarray([dur,intafter,starty,startm,startd]).T
    
    return starts, qvol, qmax, spellon, annstarts, anndays

# ==================================================================================================================
# Main routine
classnames = ['Dummy',                                      #0
              'Stable baseflow',                            #1
              'Stable winter baseflow',                     #2
              'Stable summer baseflow',                     #3
              'Unpredictable baseflow',                     #4
              'Unpredictable winter rarely intermittent',   #5
              'Unpredictable winter intermittent',          #6
              'Unpredictable intermittent',                 #7
              'Predictable winter intermittent',            #8
              'Predictable winter highly intermittent',     #9
              'Predictable summer highly intermittent',     #10
              'Unpredictable summer highly intermittent',   #11
              'Variable summer extremely intermittent'      #12
              ]

# identify ML metrics and divide by long term mean daily flow -----------------

indicator_short_list = ['MA1',            # Mean daily flow                         **needed to standardise the other indices**
                        'ML4',            # Low flow discharge (75th %'ile)         **NEW**    **standardise**
                        'DL17',           # Number zero-flow days
                        'MA10',           # Mean June flows                                    **standardise**
                        'MA11',           # Mean July flows                                    **standardise**
                        'MA3',            # CV daily flow                           **NEW**
                        'MA12',           # Mean August flows                                  **standardise**
                        'MH9',            # Magnitude 1-year ARI                    **NEW**    **standardise**
                        'FH2',            # High flow spell count (>10th %'ile)     **NEW**
                        'MA17',           # CV January flows
                        'DH5',            # Annual max. 90-day means                           **standardise**
                        'MA22',           # CV June flows
                        'MH1',            # Median of annual maximum flows    (needed as 2nd best metric for split D8 instead of 1yr ARI flow, because 1yr flow couldn't be replicated)
                        'MA29']           # **needed to identify VERY low flow sites**

# Open daily flow data ----------------------------------------------------------
print('  Opening daily flows and catchment properties')

trim2000 = False                                                                # flag to compare results with Kennard (2000)

dayflow = pd.read_csv(                                                         # read in DAILY flow data
    paths['flow']+files['qday_ML'],
    na_values=-99.99,
    comment='#',
    index_col=0,
    parse_dates=True)

cats = pd.read_csv(                                                            # read catchment list with areas
     paths['flow']+files['catstats'],
     comment='#',
     index_col=0)
catstats = cats['Catchment Area (km2)']

# loop through each time series, trim and calculate -----------------------------
print('  Calculating 12 key indicators for each catchment')
catchments = dayflow.columns.to_list()
numcats = len(catchments)

daily_df = pd.DataFrame(data=None,index=catchments,columns=indicator_short_list)  # set up results dataframe (rows = sites, cols = stats)
loopreport = 50

for icat,cat in enumerate(catchments):
    #if cat!='210052': continue
    if icat % loopreport == 0:
        print ('      Catchment ' + str(icat+1) + ' of ' + str(numcats))

    single_flow = trimseries(dayflow[cat])
    
    if trim2000:                                                               # This trims the series to the
        newstartdate = pd.to_datetime('1965-01-01')                            # years 1965 to 2000, to compare  
        newenddate = pd.to_datetime('2000-12-31')                              # with the matching data from the 
        single_flow = single_flow[newstartdate:newenddate]                     # original Kennard (2010) paper
        
    if (single_flow.max() - single_flow.min() > 1):                            # If its a valid series (not all 0's or 5's)
        
        # set up initial dataseries -------------------------------------------
        catarea = catstats[cat]                                                # catchment area in km2
        
        qd = pd.DataFrame(single_flow)                                         # basic daily flow dataframe
        qd.columns = ['q']
        qd['qnz'] = qd['q'].where(cond=(qd['q']>=0.1),other=np.nan)            # non zero values
        qd['qs'] = qd['q'] / qd['q'].mean()                                    # standardised flow
        qd['y'] = qd.index.year
        qd['m'] = qd.index.month
        qd['d'] = qd.index.day
            
        qm = pd.DataFrame(single_flow.resample('MS').sum())                    # aggregate to monthly for some stats
        qm.columns = ['sum']
        qm['sumst'] = qm['sum'] / qd['q'].mean()
        qm['avg'] = qd['qs'].resample('MS').mean()
        qm['std'] = qd['qs'].resample('MS').std(ddof=1)
        qm['y'] = qm.index.year
        qm['m'] = qm.index.month
        
        qy = pd.DataFrame(single_flow.resample('AS').sum())                    # aggregate to annual for some stats
        qy.columns=['sum']
        qy['avg'] = qd['qs'].resample('AS').mean()
        qy['max'] = qd['qs'].resample('AS').max()

        qdq = qd['qs'].to_numpy()
        qdd = qd['d'].to_numpy()
        qdm = qd['m'].to_numpy()
        qdy = qd['y'].to_numpy()
        
        # calculate metrics ---------------------------------------------------
        MA1 = qd['qs'].mean()                                                  # mean of all flows
    
        ML4 = qd['qs'].quantile(0.25)                                          # 75th percentile exceedance from the flow duration curve
        
        countall = qd['q'].resample('AS').count()
        countzeros = countall - qd['qnz'].resample('AS').count()
        DL17 = countzeros.mean()                                               # Mean annual number of days having zero flow
        
        qm_groups = qm.groupby('m')
        avg = qm_groups['avg'].get_group(6)
        MA10 = avg.mean()                                                      # mean daily flow in June averaged across all Junes
        
        avg = qm_groups['avg'].get_group(7)
        MA11 = avg.mean()                                                      # mean daily flow in July averaged across all Julys
        
        MA3 = qd['qs'].std(ddof=1)/qd['qs'].mean()                             # CV in daily flows
        
        avg = qm_groups['avg'].get_group(8)
        MA12 = avg.mean()                                                      # mean daily flow in August averaged across all Augusts
        
        ind = 7
        MH9 = aepflow(qd['qs'],1,'high','partial',ind)                         # Magnitude of 1 year ARI flood events (partial series, expon distribution, l-moments fit).  Flood independence criteria = 7 days between peaks.
                                                                                 # Kennard (2010) seems to be using 
                                                                                 # an LP3 distribution for the partial series
                                                                                 # but the peak selection algorithm has major errors!
                                                                                 # It is ignoring higher peaks if the happened less
                                                                                 # than 7 days after a smaller peak
                                                                                 # Ignore this indicator, can't resolve. Use MH1 instead.
        
        ind = 1
        thresh = qd['qs'].quantile(0.9)                                        # 10th exceedance   <====
        starts,qvol,qmax,spellon,annstarts,anndays = spellsabove(qdq,qdd,qdm,qdy,thresh,ind) # spell stats
        if np.sum(annstarts)>0:
            FH2 = np.mean(annstarts)                                           # if any spells, get average number per year
        else:
            FH2 = 0.0
        
        avg = qm_groups['avg'].get_group(1)
        sd = qm_groups['std'].get_group(1)
        MA17 = (sd / avg).mean() #.fillna(0).mean()                            # CV of flows in Jan
                                                                                 # Kennard (2010) seems to be ignoring 
                                                                                 # months where mean=0 (division error)
        
        max_3m = qd['qs'].rolling(90,center=False).mean().resample('AS').max()
        DH5 = max_3m.mean() #[1:-1].mean()                                     # Annual max. 90-day means
                                                                                 # Kennard (2010) seems to be using
                                                                                 # rolling mean at the right of the window
                                                                                 # and using the first year even though
                                                                                 # it is incomplete!!
        avg = qm_groups['avg'].get_group(6)
        sd = qm_groups['std'].get_group(6)
        MA22 = (sd / avg).mean() #.fillna(0).mean()                            # CV of flows in June
                                                                                 # Kennard (2010) seems to be ignoring 
                                                                                 # months where mean=0 (division error)
        
        # MH1 = (qy['max'] / qy['avg']).fillna(0).median()                       # Median of annual maximum flows
        # MH1 = (qy['max'] / qy['avg']).median()                                 # Median of annual maximum flows
        MH1 = qy['max'].median()                                               # Median of annual maximum flows
                                                                                 # Kennard (2010) describes this as:
                                                                                 #   "Median of the highest annual daily flow 
                                                                                 #   divided by MADF averaged across all years"
                                                                                 # ...but this is ambiguous.
                                                                                 # A better description appears to be:
                                                                                 #   "Median of the (maximum daily flow each year)
                                                                                 #   divided by the (mean daily flow for the 
                                                                                 #   entire record)"
                                                                                 # OR...
                                                                                 #   "Median of the annual maxima of daily
                                                                                 #   standardised flow"
                                                                               
        MA29 = qy['sum'].mean() / catarea                                      # Mean annual flow divided by catchment area

        # Consolidate into a daily results dataframe
        daily_df.loc[cat,'MA1'] = MA1
        daily_df.loc[cat,'ML4'] = ML4
        daily_df.loc[cat,'DL17'] = DL17
        daily_df.loc[cat,'MA10'] = MA10
        daily_df.loc[cat,'MA11'] = MA11
        daily_df.loc[cat,'MA3'] = MA3
        daily_df.loc[cat,'MA12'] = MA12
        daily_df.loc[cat,'MH9'] = MH9
        daily_df.loc[cat,'FH2'] = FH2
        daily_df.loc[cat,'MA17'] = MA17
        daily_df.loc[cat,'DH5'] = DH5
        daily_df.loc[cat,'MA22'] = MA22
        daily_df.loc[cat,'MH1'] = MH1
        daily_df.loc[cat,'MA29'] = MA29
        
# set up decision tree splits -------------------------------------------------
#     - Each split is numbered based on Kennard et al (2010) Figure 8,
#       moving across the figure from the splits needed to decide the leftmost
#       category to the rightmost.
#     - Each category is numbered from 1 to 12 as per Figure 8. Note that some
#       categories appear twice. 
#     - Its still a fairly manual job to assign them, but its easier than
#       setting up a node link network just for a single tree.
print('  Applying CART decision tree to each site')

split = pd.DataFrame(index=daily_df.index,columns=["D%i" % i for i in range(1,14)])

split['D1'] =  (daily_df['ML4']  < 0.004)          # Low flow discharge (75th %'ile)<0.004
split['D2'] =  (daily_df['DL17'] > 257.4)          # Number zero-flow days>257.4
split['D3'] =  (daily_df['MA10'] < 0.09)           # Mean June flows<0.09
split['D4'] =  (daily_df['MA11'] < 1.33)           # Mean July flows<1.33
split['D5'] =  (daily_df['MA3']  > 2.88)           # CV daily flow>2.88
split['D6'] =  (daily_df['MA3']  > 4.76)           # CV daily flow>4.76
split['D7'] =  (daily_df['MA12'] > 1.60)           # Mean August flows>1.60
#split['D8'] =  (daily_df['MH9']  > 12.77)          # Magnitude 1-year ARI>12.77     # Problems calculating 1 year ARI peak - can't replicate peak selection method
split['D8'] =  (daily_df['MH1']  > 15.41)          # Median of ann max flows>15.41   # Replace MH9 with MH1 (2nd best indicator)
split['D9'] =  (daily_df['FH2']  > 5.33)           # High flow spell count (>10th %'ile)>5.33
split['D10'] = (daily_df['MA17'] < 1.64)           # CV January flows<1.64
split['D11'] = (daily_df['DH5' ] > 2.14)           # Annual max. 90-day means>2.14
split['D12'] = (daily_df['MA22'] > 0.49)           # CV June flows>0.49
split['D13'] = (daily_df['MA3']  > 1.08)           # CV daily flow>1.08

# combine decisions for each tree branch, label each site ---------------------
sitecategory = pd.DataFrame(index=daily_df.index,columns=['Category num','Category name','branchID','flowmmPerYear'])

cond = (split['D1'] & split['D2'])
sitecategory.loc[cond,'Category num'] = 12 
sitecategory.loc[cond,'Category name'] = classnames[12]
sitecategory.loc[cond,'branchID'] = 1

cond = (split['D1'] & (~ split['D2']) & split['D3'])
sitecategory.loc[cond,'Category num'] = 10 
sitecategory.loc[cond,'Category name'] = classnames[10]
sitecategory.loc[cond,'branchID'] = 2

cond = (split['D1'] & (~ split['D2']) & (~ split['D3']) & split['D4'])
sitecategory.loc[cond,'Category num'] = 11
sitecategory.loc[cond,'Category name'] = classnames[11]
sitecategory.loc[cond,'branchID'] = 3

cond = (split['D1'] & (~ split['D2']) & (~ split['D3']) & (~ split['D4']) & split['D5'])
sitecategory.loc[cond,'Category num'] = 9
sitecategory.loc[cond,'Category name'] = classnames[9]
sitecategory.loc[cond,'branchID'] = 4

cond = (split['D1'] & (~ split['D2']) & (~ split['D3']) & (~ split['D4']) & (~ split['D5']))
sitecategory.loc[cond,'Category num'] = 6 
sitecategory.loc[cond,'Category name'] = classnames[6]
sitecategory.loc[cond,'branchID'] = 5

cond = ((~ split['D1']) & split['D6'])
sitecategory.loc[cond,'Category num'] = 7 
sitecategory.loc[cond,'Category name'] = classnames[7]
sitecategory.loc[cond,'branchID'] = 6

cond = ((~ split['D1']) & (~ split['D6']) & split['D7'] & split['D8'] & split['D9'] & split['D10'])
sitecategory.loc[cond,'Category num'] = 5 
sitecategory.loc[cond,'Category name'] = classnames[5]
sitecategory.loc[cond,'branchID'] = 7

cond = ((~ split['D1']) & (~ split['D6']) & split['D7'] & split['D8'] & split['D9'] & (~ split['D10']))
sitecategory.loc[cond,'Category num'] = 8 
sitecategory.loc[cond,'Category name'] = classnames[8]
sitecategory.loc[cond,'branchID'] = 8

cond = ((~ split['D1']) & (~ split['D6']) & split['D7'] & split['D8'] & (~ split['D9']))
sitecategory.loc[cond,'Category num'] = 6 
sitecategory.loc[cond,'Category name'] = classnames[6]
sitecategory.loc[cond,'branchID'] = 9

cond = ((~ split['D1']) & (~ split['D6']) & split['D7'] & (~ split['D8']) & split['D11'])
sitecategory.loc[cond,'Category num'] = 2 
sitecategory.loc[cond,'Category name'] = classnames[2]
sitecategory.loc[cond,'branchID'] = 10

cond = ((~ split['D1']) & (~ split['D6']) & split['D7'] & (~ split['D8']) & (~ split['D11']))
sitecategory.loc[cond,'Category num'] = 1 
sitecategory.loc[cond,'Category name'] = classnames[1]
sitecategory.loc[cond,'branchID'] = 11

cond = ((~ split['D1']) & (~ split['D6']) & (~ split['D7']) & split['D12'])
sitecategory.loc[cond,'Category num'] = 4 
sitecategory.loc[cond,'Category name'] = classnames[4]
sitecategory.loc[cond,'branchID'] = 12

cond = ((~ split['D1']) & (~ split['D6']) & (~ split['D7']) & (~ split['D12']) & split['D13'])
sitecategory.loc[cond,'Category num'] = 3 
sitecategory.loc[cond,'Category name'] = classnames[3]
sitecategory.loc[cond,'branchID'] = 13

cond = ((~ split['D1']) & (~ split['D6']) & (~ split['D7']) & (~ split['D12']) & (~ split['D13']))
sitecategory.loc[cond,'Category num'] = 1 
sitecategory.loc[cond,'Category name'] = classnames[1]
sitecategory.loc[cond,'branchID'] = 14

sitecategory['flowmmPerYear'] = daily_df['MA29'] #* 365.25

sitecategory.to_csv(paths['flow']+'SiteCategories.csv',index=True,index_label='Site',quoting=csv.QUOTE_NONNUMERIC)


