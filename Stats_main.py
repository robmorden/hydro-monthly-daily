"""
Calculating monthly and daily flow statistics based on the
Hydrologic Index Tool (HIT), later reimplemented as the R package 'EflowStats'.

R Morden
17 Nov 2021

"""
import time
import pandas as pd
import numpy as np
from C2_CalcHITDaily import calcHITDaily
from C3_CalcHITMonthly import calcHITMonthly

from PathsFiles import paths,files

t = []                                                                         # initialise timings
t.append(time.time())
pd.set_option('compute.use_numba', True)                                       # always use NUMBA where possible

cats = pd.read_csv(                                                            # read catchment list with areas
     paths['flow']+files['catstats'],
     comment='#',
     index_col=0)
catstats = cats['Catchment Area (km2)']

print('  Opening daily flow file')

dayflow = pd.read_csv(                                                         # read in DAILY flow data
    paths['flow']+files['qday_ML'],
    na_values=-99.99,
    comment='#',
    index_col=0,
    parse_dates=True)

t.append(time.time())  
print('  Calculating daily flow stats')
daystats = calcHITDaily(dayflow,'mean',catskm2=catstats,units='ML')            # calculate DAILY stats
daystats.to_csv(paths['out']+'Qdaily_ML_171stats_py.csv')                      # save to file

t.append(time.time())  
print('    Time to complete daily stats (seconds) = ' + str(t[2] - t[1]))

print('  Aggregating to monthly') 

monflow = dayflow.resample('MS').agg(pd.Series.sum,skipna=False)               # resample daily to monthly

dayflowna = dayflow.isna()                                                     # these 3 lines are some special gymnastics required
monflowna = dayflowna.resample('MS').agg(pd.Series.sum,skipna=False)           #   to get pandas to add up nans as nans.
monflow = monflow.where(monflowna==0,np.nan)                                   #   Without it the sum of nans is 0. Dumb dumb dumb.

t.append(time.time())  
print('  Calculating monthly flow stats') 
monstats = calcHITMonthly(monflow,'mean',catskm2=catstats,units='ML')          # calculate MONTHLY stats
monstats.to_csv(paths['out']+'Qmonthly_ML_171stats_py.csv')                    # save to file

t.append(time.time())  
print('    Time to complete monthly stats (seconds) = ' + str(t[4] - t[3]))

print('  Finished!')                                                           # finalise
print('  Full execution time (seconds) = ' + str(t[4] - t[0]))
print('  Full execution time (minutes) = ' + str((t[4] - t[0])/60.0))
