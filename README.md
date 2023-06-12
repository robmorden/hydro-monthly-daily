# hydro-monthly-daily
This repository includes code to calculate hydrological indicators for monthly and daily flow series, then compare them with PLSR.

For further information about the purpose of this code, please refer to the paper “Monthly flow indicators can be used to infer daily stream flow behaviour across Australia” (Morden et al., 2023). The paper is [free to read and download from here](https://doi.org/10.1016/j.jhydrol.2023.129078).

## Installation
I am a complete GitHub newbie as of August 2022. There is no prepared package to install, this is simply a repository of my code. If it looks useful, please download it and reference this repository or my paper Morden et al (2023).

## Overview
This repository has 3 basic parts:

1. **Stats** – This part calculates flow statistics based on daily streamflow data. In effect, this is a python version of the `EflowStats` tool produced by the USGS which [can be found here](https://github.com/USGS-R/EflowStats).
2. **RegimeClasses** – This part classifies the flow regime for each streamflow site into a regime ‘class’.
3. **PLSR** – This part uses partial least squares regression to compare flow statistics based on daily data with the same statistics based on monthly data, and use the monthly data to predict the daily data.

Flow data and catchment areas for 3 sample sites have been provided to help users understand the format requirements and how the code works.

Each of these parts is discussed below.

## Quick start
* If you simply want run the code to calculate flow indicators for a number of sites, set up your input data (refer to Stats/Inputs section below) then run the file `Stats_main.py`. It will organise everything for you.
* If you want to determine flow regime classes, set up your input data (refer to Regime classes/Inputs section below) then run the file `RegimeClasses.py`.
* If you want to do the PLSR calculations, you need to run `Stats_main.py`, then `RegimeClasses.py`, then `PLSR_analysis.py` in that order, one at a time.

## Stats
### Overview
Flow statistics, also known as flow indicators, are a common tool for describing specific facets of the observed flow regime such as magnitude, variability, or event duration, which provide triggers for various ecological processes and functions.

Many suites of flow indicators exist (Richter et al., 1996; Henriksen et al., 2006; Vogel et al., 2007; Kennard, Pusey, et al., 2010), and many approaches have been developed for using these indicators to assess flow regime alteration and ecological threats (Jumani et al., 2019). For this study we adopt the suite of 171 indicators originally collated by Olden and Poff (2003), and adapted by Henriksen et al. (2006). However, several modifications to the indicator calculation methods are required to address inconsistencies between previous studies. In addition, many of these indicators were developed in Northern America and Europe without consideration of highly ephemeral rivers. Accordingly, some indicators are adjusted to accommodate ephemeral rivers with a high proportion of zero-flow days.

Not all indicators can be meaningfully converted to a monthly timestep, an issue which has been noted in other studies where monthly indicators have been trialled (Laizé et al., 2014; Pumo et al., 2018). Some are effectively identical with a monthly or daily timestep, such as the ‘mean annual discharge’. Some indicators are not quite identical yet translate well to a monthly timestep, such as the ‘mean annual maximum daily discharge’ which corresponds to the ‘mean annual maximum monthly discharge’. Other indicators translate poorly to a monthly timestep, such as the ‘average daily rate of rise’ which corresponds to the ‘average monthly rate of rise’, the latter having little practical meaning. 

Of the 171 daily indicators, 115 can be meaningfully converted to a monthly calculation or were already in a monthly format. Two indicators based on circular statistics are replaced with their sine and cosine components, giving 173 daily and 117 monthly indicators overall. All indicators are consolidated into manageable categories based on different ‘facets’ of the flow regime as proposed by Olden and Poff (2003), namely magnitude (average / low / high), frequency (low / high), duration (low/high), timing, and rate of change.

The modifications to indicator calculations are outlined below:

* Where there was an option given in Henrikson et al. (2006) to calculate mean or median within an indicator, we chose the mean. Many sites in Australia are highly ephemeral where the median annual flow is zero or median flow for a given month or year is zero. Adopting medians would have resulted in a large number of zero indicator values.
* Indicators TL1 and TH1 are circular (mean date of annual minima/maxima), and are problematic for many correlation and regression techniques. In situations where this was an issue, they were removed and replaced with their sine and cosine equivalents, i.e. TL1 and TH1 were replaced with TL1_sin, TL1_cos, TH1_sin, and TH1_cos. In effect, the sine and cosine represent the spring/autumn and the summer/winter respectively of the annual maxima and minima.
* The method for calculating the rates of rise and fall were modified to reflect the proportional change in flow from the previous day, rather than the absolute volumetric change. This change applies to indicators RA1 through to RA7. Many sites have extremely high daily variability and can sometimes have large and rapid changes in flow rates. A simple difference approach does not work well in this environment, producing a noisy and highly skewed timeseries where calculation of the mean or median can be unstable. Using a proportional rate produces a more stable dataset, and consequently is more commonly used in Australia. From an ecological perspective, proportional changes in flow rates are also more likely to correlate with daily changes in depth and velocity in a stream channel.
* Some indicators involve calculations using the logarithm of flows, which can be problematic (Eng et al., 2017). Specifically, indicators MA4, MA9, MA10, MA11, MH18, MH19, TA1, and TA2 use LOG10 to scale flow data and then, crucially, calculate a ratio of flows. This creates some numerical instability, where the indicator results do not scale linearly with different flow units. In other words, the correlation between the daily and monthly versions of these indicators will be different depending on whether the units are ML/day, m3/day, or ft3/day. While the calculation methods provided for these indicators gave reasonable results in this study using units of ML/day, future treatments need to apply caution especially where the mean or median flow is less than approximately 10 units (i.e. the log10 of the flow is less than 1).
* For some indicators, the calculation methods given in the USGS code were substantially different to that proposed in Olden and Poff (2003). Where such differences were observed, the original calculation method was reviewed, and the more hydrologically informative approach was adopted. For example, DH22 and DH23 were changed to reflect the original methods (Olden & Poff, 2003; Poff & Ward, 1989) based on the mean duration and interval of an event greater than the 1.67 year ARI flood threshold.
 
### Inputs
The code requires the following input data:

*	**Daily streamflow data** – Dates should be in the first column in pandas-parseable format, with each site in subsequent columns. The first row should be column headings representing the site (gauge) name or number. Missing data is assumed to be `-99.9` and comments at the top of the file should have the prefix `#`.
*	**Catchment area information** – The code is customised to read the file `hrs_station_details_08_2020.csv` as obtained from the Australian Bureau of Meteorology as part of their Hydrologic Reference Station database. The catchment areas are in the field `Catchment Area (km2)`. The first column should be the site (gauge) name or number, *_exactly_* matching the column headings in the streamflow data.

Both of these inputs are easily adjusted to suit individual requirements, but it is essential that the flow data is in columns with each column heading matching the row index labels from the catchment area file.

### Outputs
The code produces a csv file called `Qdaily_ML_171stats_py.csv`, and another called `Qmonthly_ML_171stats_py.csv`. These files contain all indicator results arranged by site (column) and indicator (row).

### Options
* **Mean or median** - As per Henriksen et al. (2006), many indicators have an option to calculate a long term mean or median. This option is set in the `Stats-main.py` file when calling the `calcHITdaily` and `calcHITmonthly` functions. For both functions, the second parameter `opt_median` can be set to a string either `mean` or `median`.
* **Flows divided by catchment area** - In some situations, it may be useful to calculate indicators based on flow data standardised by area. If the `calcHITdaily` and `calcHITmonthly` functions are called with the last parameter set to `mm`, all flows will be divided by area prior to calculation. Specific indicators which rely on catchment areas are automatically modified accordingly.

### Code structure
If you want to run the code as is, just run 'stats_main.py'. It will do all of the necessary calculations.
* `Stats-main.py` loads the input data, calls the daily statistics routine, aggregates the inputs to monthly, runs the monthly statistics routine, then saves the output.
* `Stats_daily.py` takes daily flow inputs, then calculates all daily statistics and puts them in a DataFrame.
* `Stats_monthly.py` takes monthly flow inputs, then calculates all monthly statistics and puts them in a DataFrame.
* `Stats_functions.py` includes a range of functions for various hydrological calculations, such as rise and fall rates, or Colwell predictability calculations. Many of these functions rely on the [Numba](https://numba.pydata.org/numba-doc/latest/index.html) library and adopt the `@jit(nopython=True)` decorator. If you have never come across this, don’t panic. You can just comment out all of the `@jit` decorators and the code will work just fine. But if you leave them there the code will work waaaay faster for multiple sites. When using Numba, the entire routine should run in around 1 or 2 seconds per site. 

## Regime classes
### Overview
This code reads daily streamflow data and assigns it to a flow regime classification based on the CART model proposed in Kennard et al. (2010).

A type of decision tree, CART models assign outcome variables to classes using independent variables to determine how to proceed at each decision ‘fork’ in the tree. In this case, the CART model assigned sites to regime classes based on 11 daily flow indicators.

A minor modification to the CART model was required for one fork, replacing the ‘magnitude 1 year ARI’ indicator with the ‘median annual max flow’. The 1 year ARI peak flow indicator is calculated using a partial series approach, but the results from Kennard et al. (2010) could not be replicated exactly due to differences in the algorithms for identifying peaks over a threshold and fitting a probability model to the identified peaks. The next most important indicator for that particular fork was the median annual maximum flow (Kennard et al., 2010) which was adopted instead of the 1 year ARI peak flow as it could be successfully replicated.

### Inputs
The code requires the following input data:
* Daily streamflow data – Dates should be in the first column in pandas-parseable format, with each site in subsequent columns. The first row should be column headings representing the site (gauge) name or number. Missing data is assumed to be `-99.9` and comments at the top of the file should have the prefix `#`.
* Catchment area information – The code is customised to read the file `hrs_station_details_08_2020.csv` as obtained from the Australian Bureau of Meteorology as part of their Hydrologic Reference Station database. The catchment areas are in the field `Catchment Area (km2)`. The first column should be the site (gauge) name or number, *_exactly_* matching the column headings in the streamflow data.

Both of these inputs are easily adjusted to suit individual requirements, but it is essential that the flow data is in columns with each column heading matching the row index labels from the catchment area file.

### Outputs
The code produces a file called `SiteCategories.csv` which lists the class number, class name, the CART model branch, and the flow in mm/year for each site.

### Options
The variable `trim2000` ensures all input flow data begins in January 1965 and ends in December 2000 to allow closer comparison with the results from Kennard et al (2010).

## PLSR
### Overview
The indicators calculated in the stats section of this repository contain a high degree of collinearity and redundancy (Gao et al., 2009; Olden & Poff, 2003), making traditional regression techniques unsuitable. Partial least squares regression (PLSR, sometimes known as ‘Projection to Latent Structures Regression’) (Vinzi et al., 2010; Wold et al., 2001) is a technique which can identify links between two datasets, while also directly addressing the challenges of collinearity and redundancy.

Data is transformed prior to the PLSR analysis using either Yeo Johnson or Box Cox transformation. While [SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html) has a function for Yeo Johnson transformation, there was no practical reversing function available so this has been added as a custom function.

During the PLSR analysis, 10-fold cross validation is used to choose the best number of components to extract.

This code uses the [scikit-learn](https://scikit-learn.org/stable/) library to perform the regression and manage the k-fold cross validation.

### Inputs
The code requires the following input data:
*	Daily and monthly indicator results from the ‘stats’ code above.
*	Catchment area information – The code is customised to read the file `hrs_station_details_08_2020.csv` as obtained from the Australian Bureau of Meteorology as part of their Hydrologic Reference Station database. The catchment areas are in the field `Catchment Area (km2)`. The first column should be the site (gauge) name or number, *_exactly_* matching the column headings in the streamflow data.
*	A list of sites to include – the code reads a file structured similar to the catchment area information, but with a field called `RM_include` which is set to either `include` or `exclude` for each site. Apologies for the nominative field name, I know its poor practice but I guess old habits die hard.

### Outputs
This code produces a number of files including raw and transformed inputs, raw and transformed predictions, correlations, PLS component results, and PLS regression results.

### Options
* The `savedata` option allows the model to run without saving anything. The results can be inspected manually at run time.
* The `valmodel` option flags whether model validation should be run, saving a little time.
* There is a section of code which allows the PLSR to be run on a subset of sites based on flow regime classification. By default, it runs with all sites, but the relevant section can be manually commented/uncommented to process, say, only ephemeral sites.

## References
Eng, K., Grantham, T. E., Carlisle, D. M., & Wolock, D. M. (2017). Predictability and selection of hydrologic metrics in riverine ecohydrology. Freshwater Science, 36(4). https://doi.org/10.1086/694912

Gao, Y., Vogel, R. M., Kroll, C. N., Poff, N. L. R., & Olden, J. D. (2009). Development of representative indicators of hydrologic alteration. Journal of Hydrology, 374(1–2), 136–147. https://doi.org/10.1016/j.jhydrol.2009.06.009

Henriksen, J. A., Heasley, J., Kennen, J. G., & Nieswand, S. (2006). Users’ Manual for the Hydroecological Integrity Assessment Process Software (including the New Jersey Assessment Tools). http://www.usgs.gov/pubprod

Jumani, S., Deitch, M. J., Kaplan, D., Anderson, E. P., Krishnaswamy, J., Lecours, V., & Whiles, M. R. (2019). River fragmentation and flow alteration metrics: A review of methods and directions for future research. Environmental Research Letters, 15(12). https://doi.org/10.1088/1748-9326/ABCB37

Kennard, M. J., Pusey, B. J., Olden, J. D., Mackay, S. J., Stein, J. L., & Marsh, N. (2010). Classification of natural flow regimes in Australia to support environmental flow management. Freshwater Biology, 55(1), 171–193. https://doi.org/10.1111/j.1365-2427.2009.02307.x

Laizé, C. L. R., Acreman, M. C., Schneider, C., Dunbar, M. J., Houghton-Carr, H. A., Flörke, M., & Hannah, D. M. (2014). Projected flow alteration and ecological risk for pan-European rivers. River Research and Applications, 30(3), 299–314. https://doi.org/10.1002/rra.2645

Morden, R., Horne, A., Nathan, R., Bond, N. R., & Olden, J. D. (2023). Monthly flow indicators can be used to infer daily stream flow behaviour across Australia. Journal of Hydrology, 617, 129078. https://doi.org/10.1016/J.JHYDROL.2023.129078

Olden, J. D., & Poff, N. L. (2003). Redundancy and the choice of hydrologic indices for characterizing streamflow regimes. River Research and Applications, 19(2), 101–121. https://doi.org/10.1002/rra.700

Poff, N. L., & Ward, J. v. (1989). Implications of Streamflow Variability and Predictability for Lotic Community Structure: A Regional Analysis of Streamflow Patterns. Canadian Journal of Fisheries and Aquatic Sciences, 46(10), 1805–1818. https://doi.org/10.1139/F89-228

Pumo, D., Francipane, A., Cannarozzo, M., Antinoro, C., & Noto, L. V. (2018). Monthly Hydrological Indicators to Assess Possible Alterations on Rivers’ Flow Regime. Water Resources Management, 32(11), 3687–3706. https://doi.org/10.1007/s11269-018-2013-6

Richter, B. D., Baumgartner, J. v., Powell, J., & Braun, D. P. (1996). A Method for Assessing Hydrologic Alteration within Ecosystems. Conservation Biology, 10(4), 1163–1174. https://doi.org/10.1046/j.1523-1739.1996.10041163.x

Vinzi, V. E., Chin, W. W., Henseler, J., & Wang, H. (Eds.). (2010). Handbook of partial least squares: concepts, methods and applications. Springer. http://www.springer.com/series/7286

Vogel, R. M., Sieber, J., Archfield, S. A., Smith, M. P., Apse, C. D., & Huber-Lee, A. (2007). Relations among storage, yield, and instream flow. Water Resources Research, 43(5), 5403. https://doi.org/10.1029/2006WR005226

Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109–130. https://doi.org/10.1016/S0169-7439(01)00155-1
