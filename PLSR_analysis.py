# -*- coding: utf-8 -*-
"""
Partial least squares regression - analysis only

R Morden
March 2022

Modified June 2022:
    - Use only 10 fold cross validation throughout, no 20%/80% test/train split
    - This change means that the validation plots will include ALL sites, not just the 20% test sites.
    - In practice, run the cross validation to find the bet number of components, then..
    - ..use the n=20 kfold data to run the whole thing from scratch.

"""

import pandas as pd                                                            # external libraries
import numpy as np
import sys
from math import sqrt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import scipy.special as special

from PathsFiles import paths,files                                             # custom libraries

# =====================================================================================================
# Inverse yeo-johnson transformation
# Copied directly (with minor adjustments to fit here) from source code for scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/preprocessing/_data.py#L3127 (18 March 2022)
def inv_yeojohnson( x, lmbda):
    """Return inverse-transformed input x following Yeo-Johnson inverse
    transform with parameter lambda.
    """
    x_inv = np.zeros_like(x)
    pos = x >= 0

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.exp(x[pos]) - 1
    else:  # lmbda != 0
        x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
    else:  # lmbda == 2
        x_inv[~pos] = 1 - np.exp(-x[~pos])

    return x_inv

# =====================================================================================================
# Short routine to swap out circular indicators for sin/cos equivalents
def addsincos(df,swaplist,maxvals):
    
    df_new = df.copy()
    
    for i in range(len(swaplist)):
        
        col = swaplist[i]
        maxval = maxvals[i]
        colvalues = df_new.loc[:,col].copy()
        
        collist = df_new.columns                                               # make an list (index) of columns
        colnum_orig = collist.get_loc(col)                                     # get the index number of the swap column
        collist = collist.insert(colnum_orig,col+'_cos')                       # add a COS column in that location
        collist = collist.insert(colnum_orig,col+'_sin')                       # add a SIN column in that location
        collist = collist.drop(col)                                            # remove the swap column
        
        df_new[col+'_sin'] = np.sin(colvalues / maxval * 2 * np.pi)            # calculate sin
        df_new[col+'_cos'] = np.cos(colvalues / maxval * 2 * np.pi)            # calculate cos
        
        df_new = df_new[collist]                                               # reorder columns based on list created above
        
    return df_new

# =====================================================================================================
# Short routine to transform a dataframe
def tf(dfin):
    
    dfout = pd.DataFrame(index=dfin.index,columns=dfin.columns)
    dfout_lm = pd.DataFrame(index=dfin.columns,columns=['method','lambda','post_mean','post_sd'])
    for col in dfin.columns:
        
        if dfin[col].max() - dfin[col].min() < 0.1:                            # catch the case where all values are the same
            tf = dfin[col]
            lm = 1.0
            dfout_lm.loc[col,'method'] = 'bc'
        else:
            if dfin[col].min() < 0:                                            # if negative values, do a y-j transform
                tf,lm = stats.yeojohnson(dfin[col])
                dfout_lm.loc[col,'method'] = 'yj'
            else:
                tf,lm = stats.boxcox(dfin[col] + 0.01)                         # if positive values, add 0.01 and do a bc transform
                dfout_lm.loc[col,'method'] = 'bc'
        
        m = np.mean(tf)
        sd = np.std(tf,ddof=1)
        dfout_lm.loc[col,'lambda'] = lm
        dfout_lm.loc[col,'post_mean'] = m
        dfout_lm.loc[col,'post_sd'] = sd
        if sd==0.0:
            dfout.loc[:,col] = tf - m
        else:
            dfout.loc[:,col] = (tf - m)/sd
    
    return dfout, dfout_lm

# =====================================================================================================
# Short routine to >>REVERSE<< transform a dataframe
def tf_reverse(dfin,dfin_lm):  #pt_dict):   #,tf_mins):
    
    dfout = pd.DataFrame(index=dfin.index,columns=dfin.columns)
     
    for col in dfin.columns:
        lm = dfin_lm.loc[col,'lambda']
        meth = dfin_lm.loc[col,'method']
        m = dfin_lm.loc[col,'post_mean']
        sd = dfin_lm.loc[col,'post_sd']
        tf = (dfin[col] * sd) + m
        
        if meth == 'bc':
            tf = special.inv_boxcox(tf - 0.01,lm)
        else:
            tf = inv_yeojohnson(tf,lm)
        
        dfout.loc[:,col] = tf
        
    return dfout

# =====================================================================================================

print ('   Setting up')

savedata = True                                                                # options
valmodel = True

stattype = '_171stats'                                                         # file naming info

dayfile = 'Qdaily_ML_171stats_py.csv'                                          # basic input files
monfile = 'Qmonthly_ML_171stats_py.csv'

outliers = pd.read_csv(paths['flow'] + 'IncludedSites.csv',index_col=0)        # list of all sites to include/exclude
cats = pd.read_csv(paths['flow']+files['catstats'],comment='#',index_col=0)    # catchment list with areas and lat/long
circindicators = ['TL1','TH1']
circmaxvals_d = [366,366]
circmaxvals_m = [12,12]

# read data files -------------------------------------------------------------------------------------------
daily_raw = pd.read_csv(paths['out'] + dayfile,index_col=0).transpose()        # read the indicators
daily_df = daily_raw[outliers['RM_Include']=='include']                        # drop outlier sites
daily_df = addsincos(daily_df,circindicators,circmaxvals_d)                    # swap circular indicators for sin/cos versions

monthly_raw = pd.read_csv(paths['out'] + monfile,index_col=0).transpose()      # read the indicators
monthly_df = monthly_raw[outliers['RM_Include']=='include']                    # drop outlier sites
monthly_df = monthly_df.dropna(axis=1)                                         # remove indicators not calculated (roughly 1/3)
monthly_df = addsincos(monthly_df,circindicators,circmaxvals_m)                # swap circular indicators for sin/cos versions

# create model details ----------------------------------------------------------------------------
# classnames = ['Stable baseflow',                            #1
#               'Stable winter baseflow',                     #2
#               'Stable summer baseflow',                     #3
#               'Unpredictable baseflow',                     #4
#               'Unpredictable winter\nrarely intermittent',   #5
#               'Unpredictable winter\nintermittent',          #6
#               'Unpredictable\nintermittent',                 #7
#               'Predictable winter\nintermittent',            #8
#               'Predictable winter\nhighly intermittent',     #9
#               'Predictable summer\nhighly intermittent',     #10
#               'Unpredictable summer\nhighly intermittent',   #11
#               'Variable summer\nextremely intermittent'      #12
#               ]

flowclasses = pd.read_csv(paths['flow']+'SiteCategories.csv',index_col=0)
flowclasses = flowclasses[outliers['RM_Include']=='include']                   # drop outlier sites

subset_list = []
modelname_list = []
n_comp_list = []

subset_list.append(flowclasses['Category num'] >= 0)                           # 0 # all sites
modelname_list.append('_allsites')
n_comp_list.append(20)

subset_list.append(flowclasses['Category num'] >= 5)                           # 1 # intermittent sites
modelname_list.append('_intsites')
n_comp_list.append(20)

subset_list.append(flowclasses['Category num'] < 5)                            # 2 # perennial sites
modelname_list.append('_persites')
n_comp_list.append(12)

subset_list.append((flowclasses['Category num'] >= 5) & (flowclasses['Category num'] <= 8))     # 3 # slightly intermittent sites
modelname_list.append('_lintsites')
n_comp_list.append(12)

subset_list.append(flowclasses['Category num'] >= 9)                           # 4 # highly intermittent sites
modelname_list.append('_hintsites')
n_comp_list.append(13)

i = 0                                 # JUST CHANGE THIS BETWEEN MODEL RUNS   # Could make this a loop, but I prefer to run one at a time
subset = subset_list[i]
modelname = modelname_list[i]
n_comp = n_comp_list[i]

print ('   Running model >>' + modelname + '<<')

daily_df_subset = daily_df.loc[subset,:]
monthly_df_subset = monthly_df.loc[subset,:]

y_raw = daily_df_subset
x_raw = monthly_df_subset

# K-fold split and separate transformations -----------------------------------------------------------------
nfold = 10
kfsplit = KFold(nfold,shuffle=True,random_state=1)                             # set up the kfold split generator

print ('   Creating folds, then transforming train/test data in each fold')
ytrain =   [None] * nfold   # training data for each fold
xtrain =   [None] * nfold
yval =     [None] * nfold   # validation data for each fold
xval =     [None] * nfold
idxtrain = [None] * nfold   # index list for each fold
idxval =   [None] * nfold
pt_yt =    [None] * nfold   # transformations for each fold
pt_yv =    [None] * nfold
pt_xt =    [None] * nfold
pt_xv =    [None] * nfold

for i,(train,test) in enumerate(kfsplit.split(y_raw)):                         # set up folds in advance
    idxtrain[i]  = train
    idxval[i]    = test
    ytrain[i], pt_yt[i] = tf(y_raw.iloc[train,:])                              # transform, DO keep the transformation details
    yval[i],   pt_yv[i] = tf(y_raw.iloc[test, :])
    xtrain[i], pt_xt[i] = tf(x_raw.iloc[train,:])
    xval[i],   pt_xv[i] = tf(x_raw.iloc[test, :])

# run validation model to confirm the right number of components --------------------------------------------
if valmodel:
    print ('   Finding ideal number of orthogonal components')
    maxn = 50                                                                  # max number of components
    nrange = range(1,maxn+1)
    cv_RMSEout = pd.DataFrame(index=nrange)
    cv_R2tfout = pd.DataFrame(index=nrange)
    
    for n in nrange:                                                           # repeat kfold validation for 1 to n components
        sys.stdout.write('\r   Testing with n=' + str(n) + ' components')
        sys.stdout.flush()

        y_est_all = pd.DataFrame(index=y_raw.index,
                                 columns=y_raw.columns,
                                 dtype='float64')                              # set up blank results
        y_obs_all = y_est_all.copy()

        for i in range(nfold):
            plsr_cv = PLSRegression(n_components=n, scale=False)               # do the regression with >>training<< data
            plsr_cv.fit(xtrain[i],ytrain[i])
            y_est = plsr_cv.predict(xval[i])                                   # get estimates for >>validation<< data
            
            y_est_all.iloc[idxval[i],:] = y_est                                # slot into complete results series
            y_obs_all.iloc[idxval[i],:] = yval[i]
            
        cv_RMSEout.loc[n,'n components'] = n
        cv_RMSEout.loc[n,'RMSE'] = sqrt(mean_squared_error(y_obs_all, y_est_all))  # put RMSE results in dataframe
        
        cv_R2tfout.loc[n,'n components'] = n
        cv_R2tfout.loc[n,'R2'] = r2_score(y_obs_all, y_est_all)                # put R2 results in dataframe

        for ind in y_est_all.columns:
            obs = y_obs_all.loc[:,ind]
            est = y_est_all.loc[:,ind]
            cv_RMSEout.loc[n,ind] = sqrt(mean_squared_error(obs, est))         # put RMSE results in dataframe
            cv_R2tfout.loc[n,ind] = r2_score(obs, est)                         # put R2 results in dataframe
            
    cv_optimum = pd.Series(index=['n_comp','RMSE','R2'],dtype='float64')
    cv_optimum.loc['n_comp'] = n_comp                                          # record optimum n (predefined at top of code block)
    cv_optimum.loc['RMSE'] = cv_RMSEout.loc[n_comp,'RMSE']                     # record RMSE for optimum n
    cv_optimum.loc['R2'] = cv_R2tfout.loc[n_comp,'R2']                         # record R2 for optimum n
    
    if savedata:
        cv_RMSEout.to_csv(paths['plsr']+stattype+modelname+'_kfoldval_RMSEresults.csv')  # save crossfold validation results
        cv_R2tfout.to_csv(paths['plsr']+stattype+modelname+'_kfoldval_R2results.csv')
        cv_optimum.to_csv(paths['plsr']+stattype+modelname+'_kfoldval_optimum.csv')
    
    print('\n')

# run full validation model with optimised number of components ---------------------------------------------

# PLS regression (optimised ncomp) --------------------------------------------------------------------------
print ('   Running partial least squares regression (n='+str(n_comp)+' components)')

yv_obs = pd.DataFrame(index=y_raw.index,columns=y_raw.columns,dtype='float64') # set up blank results
yv_est = yv_obs.copy()
yv_est_raw = yv_obs.copy()
 
xv_obs = pd.DataFrame(index=x_raw.index,columns=x_raw.columns,dtype='float64')

for i in range(nfold):                                                         # for each fold....
    plsr = PLSRegression(n_components=n_comp, scale=False)                     # do the regression with >>training<< data
    plsr.fit(xtrain[i],ytrain[i])
    y_est = pd.DataFrame(plsr.predict(xval[i]),
                         index=yval[i].index,
                         columns=yval[i].columns)                              # get estimates for >>validation<< data
    y_est_lin = tf_reverse(y_est,pt_yt[i])                                     # reverse transformation back to linear space (use training data transformation)
    
    # slot data from each fold into complete dataframes
    yv_est.iloc[idxval[i],:] = y_est                                           # daily transf estimated
    yv_est_raw.iloc[idxval[i],:] = y_est_lin                                   # daily linear estimated
    
    yv_obs.iloc[idxval[i],:] = yval[i]                                         # daily transf observed
    xv_obs.iloc[idxval[i],:] = xval[i]                                         # monthly transf observed
    
# Create outputs per site -----------------------------------------------------------------------------------
site_results = pd.DataFrame(index=yv_obs.index,columns=['RMSE','R2','class num','class name'])
site_results['class num'] = flowclasses['Category num']                        # get class name and num from predefined file
site_results['class name'] = flowclasses['Category name']

for irow,row in enumerate(yv_obs.index):                                           # loop through each site
    site_rmse = sqrt(mean_squared_error(yv_obs.iloc[irow,:],yv_est.iloc[irow,:]))   # RMSE
    site_R2 = r2_score(yv_est.iloc[irow,:], yv_obs.iloc[irow,:])
    site_results.loc[row,'RMSE'] = site_rmse
    site_results.loc[row,'R2'] = site_R2

# Create outputs per regime class ---------------------------------------------------------------------------
class_rmse = pd.DataFrame(index=yv_obs.columns,columns=np.sort(flowclasses['Category num'].unique()))
for ind in class_rmse.index:                                                   # loop through each index and regime class
    for cl in class_rmse.columns:
        class_site_list = (flowclasses['Category num']==cl)                    # select all sites in a given regime class
        if len(yv_obs.loc[class_site_list,ind]) > 0:
            class_rmse.loc[ind,cl] = sqrt(mean_squared_error(yv_obs.loc[class_site_list,ind],yv_est.loc[class_site_list,ind]))
        
# Calculate correlations ------------------------------------------------------------------------------------
#xy_corr = pd.DataFrame(np.nan,index=yv_obs.columns,columns=['Rs'],dtype='float64')

yv_results = pd.DataFrame(np.nan,index=yv_obs.columns,columns=['R2 tf',        # set up validation results per indicator
                                                               'Rs x-obs tf',
                                                               'Rs y-est tf',
                                                               'R2 lin',
                                                               'Rs x-obs lin',
                                                               'Rs y-est lin',
                                                               'RMSE',
                                                               'group',
                                                               'groupnum',
                                                               'type'])
yv_results['group'] = yv_obs.columns.to_series().str[:2]
pltgroups = pd.Series(yv_results.loc[:,'group'].unique())
pltgroups_df = pd.DataFrame(np.arange(1,12),index=pltgroups,columns=['name'])
yv_results['groupnum'] = yv_results['group'].map(pltgroups_df['name'])

yv_results['type'] = 'dayonly'
# yv_results['RMSE'] = 0.0
# yv_results['R2 tf'] = 0.0
# yv_results['Rs y-est tf'] = 0.0
# yv_results['Rs x-obs tf'] = 0.0
# yv_results['R2 lin'] = 0.0
# yv_results['Rs y-est lin'] = 0.0
# yv_results['Rs x-obs lin'] = 0.0

for icol,col in enumerate(yv_obs.columns):
    yobs = yv_obs.iloc[:,icol]                                                 # goodness of fit (transformed)
    yest = yv_est.iloc[:,icol]
    yres = yobs - yest
    
    yv_results.loc[col,'RMSE'] = sqrt(mean_squared_error(yobs,yest))
        
    if yobs.abs().sum() != 0:                                                  # if the observed is NOT all zeros 
        yv_results.loc[col,'R2 tf'] = 1-( (yres**2).sum() / (yobs**2).sum() )  # R2 transformed (1 - sumsq residuals / sumsq obs)
        yv_results.loc[col,'Rs y-est tf'] = yobs.corr(yest,method='spearman')  # Spearman (est and observed, both transformed)
    
    if col in xv_obs.columns:
        yv_results.loc[col,'type'] = 'both'
        xobs = xv_obs.loc[:,col]
        if yobs.abs().sum() != 0:
            yv_results.loc[col,'Rs x-obs tf'] = yobs.corr(xobs,method='spearman') # Spearman (daily and monthly, both transformed)

    yobs = y_raw.iloc[:,icol]                                                  # goodness of fit (linear)
    yest = yv_est_raw.iloc[:,icol]
    yres = yobs - yest
    
    if yobs.abs().sum() != 0:                                                  # if the observed is NOT all zeros 
        yv_results.loc[col,'R2 lin'] = 1-( (yres**2).sum() / (yobs**2).sum() ) # R2 linear (1 - sumsq residuals / sumsq obs)
        yv_results.loc[col,'Rs y-est lin'] = yobs.corr(yest,method='spearman') # Spearman (est and observed, both linear)
    
    if col in xv_obs.columns:
        xobs = x_raw.loc[:,col]
        if yobs.abs().sum() != 0:
            yv_results.loc[col,'Rs x-obs lin'] = yobs.corr(xobs,method='spearman') # Spearman (daily and monthly, both linear)
            # if col in xv_obs.columns:
            #     xy_corr.loc[col,'Rs'] = yv_results.loc[col,'Rs x-obs lin']

# Run full model trained on all sites (just to get coeffs and loadings) -------------------------------------
ytrain_all, pt = tf(y_raw)                                                     # transform, DON'T keep the transformation details
xtrain_all, pt = tf(x_raw)

plsr = PLSRegression(n_components=n_comp, scale=False)                         # do the regression with >>training<< data
plsr.fit(xtrain_all,ytrain_all)

compnames = ['C%02d' % i for i in range(1, n_comp+1)]
xycoef = pd.DataFrame(plsr.coef_,index=x_raw.columns,columns=y_raw.columns)    # coefficients
xload = pd.DataFrame(plsr.x_rotations_,index=x_raw.columns,columns=compnames)  # loadings

# Create sorted list of most important variables ------------------------------------------------------------
xloadmax = pd.DataFrame(index=compnames,columns=['max1','max2','max3'])

for col in compnames:
    colsorted = xload.loc[:,col].abs().sort_values(ascending=False)
    xloadmax.loc[col,'max1'] = colsorted.index[0]
    xloadmax.loc[col,'load1'] = colsorted.iloc[0]
    xloadmax.loc[col,'max2'] = colsorted.index[1]
    xloadmax.loc[col,'load2'] = colsorted.iloc[1]
    xloadmax.loc[col,'max3'] = colsorted.index[2]
    xloadmax.loc[col,'load3'] = colsorted.iloc[2]
  
#xload['sumtop6'] = xload.iloc[:,0:6].abs().sum(axis=1)


# Export key data -------------------------------------------------------------------------------------------
if savedata:
    print('   Exporting results')
    
    filestub = stattype + modelname + '_data'
    
    y_raw.to_csv(paths['plsr'] + filestub +'_Y_obs_raw.csv')
    x_raw.to_csv(paths['plsr'] + filestub +'_X_obs_raw.csv')

    xv_obs.to_csv(paths['plsr'] + filestub +'_Xv_obs_tf.csv')
    yv_obs.to_csv(paths['plsr'] + filestub +'_Yv_obs_tf.csv')
    yv_est.to_csv(paths['plsr'] + filestub +'_Yv_est_tf.csv')
    yv_est_raw.to_csv(paths['plsr'] + filestub +'_Yv_est_raw.csv')

    xycoef.to_csv(paths['plsr'] + filestub +'_XYCoefficients.csv')
    xload.to_csv(paths['plsr'] + filestub +'_Xloadings.csv')
    xloadmax.to_csv(paths['plsr'] + filestub +'_XloadMaxPerCompnt.csv')

    yv_results.to_csv(paths['plsr'] + filestub +'_ModelFitByVariable.csv')

    class_rmse.to_csv(paths['plsr'] + filestub +'_ModelFitByClass.csv')
    site_results.to_csv(paths['plsr'] + filestub +'_ModelFitBySite.csv')

print('   Complete!')
