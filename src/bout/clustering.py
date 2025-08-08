import pandas as pd
import numpy as np

import scipy
import scipy.stats as stats

def regress_around_peakIPI(intervals_ms, survival, values):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with within-bout intervals.

    These points are chosen around the most common inter-pulse interval.
    We know this interval and neighboring intervals are most likely to be within bout.
    """

    max_val = np.max(values)
    first_peak = np.where(values==max_val)[0][0]
    fast_inds = range(0, int(np.ceil(5*first_peak))+1)
    fast_coeff = stats.linregress(intervals_ms[fast_inds], survival[fast_inds])

    fast_process = dict()
    fast_process['metrics'] = fast_coeff
    fast_process['indices'] = fast_inds
    return fast_process


def regress_around_fast_intervals(intervals_ms, survival, values):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with within-bout intervals.

    These points are chosen to be all IPIs less than 2s.
    We know this interval and neighboring intervals are most likely to be within bout.
    """


    fast_inds = intervals_ms <= 2*1e3
    fast_coeff = stats.linregress(intervals_ms[fast_inds], survival[fast_inds])

    fast_process = dict()
    fast_process['metrics'] = fast_coeff
    fast_process['indices'] = fast_inds
    return fast_process


def regress_around_survival_threshold(intervals_ms, survival, values):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with within-bout intervals.

    These interval points to regress around are chosen using the top 10% of survival values.
    We can start to guess that these intervals could be within bout.
    """

    fast_inds = np.logical_and(survival >= (survival.max() * 0.90), survival <= (survival.max() * 1.0))
    fast_coeff = stats.linregress(intervals_ms[fast_inds], survival[fast_inds])

    fast_process = dict()
    fast_process['metrics'] = fast_coeff
    fast_process['indices'] = fast_inds
    return fast_process

def regress_around_slow_intervals(intervals_ms, survival):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with between-bout intervals.

    These interval points to regress around are chosen using values between 30-40% of the max survival.
    We have observed that these points have a strong linear relationship.
    They are also among intervals from 20 to 60min. This range is very likely between-bout.
    """

    # slow_inds = np.logical_and(survival >= (survival.max() * 0.25), survival <= (survival.max() * 0.65)) 
    slow_inds = np.logical_and(intervals_ms >= 60*1e3, survival >= (survival.max() * 0.1))
    slow_coeff = stats.linregress(intervals_ms[slow_inds], survival[slow_inds])

    slow_process = dict()
    slow_process['metrics'] = slow_coeff
    slow_process['indices'] = slow_inds
    return slow_process

def test_ipis_ms(ipis_ms, dates, location_sum_df):
    """
    The # of intervals calculated should be equal to the # of calls - (DATES)
    DATES is a constant here because for each date, the first call is not considered 
    as there is no previous call to calculate an interval for.
    """

    assert(len(ipis_ms) + len(dates) == len(location_sum_df))

def get_valid_ipis_ms(location_sum_df):
    """
    Gets the IPIs (Inter-Pulse Intervals) for a given location and frequency group
    using the 2022_bd2_summary files stored in data.

    Ignores IPIs generated for the first call of every date since we only want to consider intervals within nights.

    Returns a numpy array of IPIs in milliseconds.
    """

    intervals = pd.to_datetime(location_sum_df['call_start_time']) - pd.to_datetime(location_sum_df['call_end_time']).shift(1)
    location_sum_df.insert(0, 'time_from_prev_call_end_time', intervals)

    location_sum_df['ref_time'] = pd.DatetimeIndex(location_sum_df['call_start_time'])
    location_sum_df = location_sum_df.set_index('ref_time')

    first_calls_per_day = location_sum_df.resample('D').first()['call_start_time']
    first_valid_calls_per_day = pd.DatetimeIndex(first_calls_per_day.loc[~first_calls_per_day.isna()].values)
    location_sum_df.loc[first_valid_calls_per_day, 'time_from_prev_call_end_time'] = pd.NaT

    intervals = location_sum_df['time_from_prev_call_end_time'].values
    valid_intervals = intervals[~np.isnan(intervals)]
    ipis_ms = valid_intervals.astype('float32')/1e6

    test_ipis_ms(ipis_ms, first_valid_calls_per_day, location_sum_df)

    return ipis_ms

def get_histogram(location_sum_df, bin_step):
    """
    Uses the IPIs from a location and for a frequency group to compute and return a complete histogram.
    The interval width is set to be 10ms to provide good resolution for the most common IPIs.
    """

    ipis_ms = get_valid_ipis_ms(location_sum_df)
    hist_loc = np.histogram(ipis_ms, bins=np.arange(0, ipis_ms.max()+bin_step, bin_step))

    return ipis_ms, hist_loc

def get_log_survival(hist_loc):
    """
    Computes and returns a log-survivorship curve provided a histogram.
    """

    values, base = hist_loc[0], hist_loc[1]
    cumulative = (np.cumsum(values[::-1]))[::-1]
    survival = np.log(cumulative)
    intervals_ms = base[:-1]

    return intervals_ms, survival

def calculate_exponential_coefficients(process):
    """
    Computes and returns the exponential coefficients given the slope and intercept of a process.
    Using equations from Sibly et al. (1990) and Slater & Lester (1982)
    """
    
    process['lambda'] = -1*process['metrics'].slope
    process['num_intervals_slater'] = np.exp(process['metrics'].intercept) / process['lambda']

    return process


def get_bci_from_fagenyoung_method(fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Fagen & Young (1978) derived for minimizing total time misassigned.
    """

    bci = (1/(fast_process['lambda'] - slow_process['lambda'])) * np.log(fast_process['num_intervals_slater']/slow_process['num_intervals_slater'])

    misassigned_points = (fast_process['num_intervals_slater']*np.exp(-1*fast_process['lambda']*bci)) + (slow_process['num_intervals_slater']*(1 - np.exp(-1*slow_process['lambda']*bci)))

    return bci, misassigned_points

def get_bci_from_slater_method(intervals_ms, survival, fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Slater & Lester (1982) derived for minimizing # of events misassigned.
    """

    bci = (1/(fast_process['lambda'] - slow_process['lambda'])) * np.log((fast_process['num_intervals_slater']*fast_process['lambda'])/(slow_process['num_intervals_slater']*slow_process['lambda']))

    misassigned_points = (fast_process['num_intervals_slater']*np.exp(-1*fast_process['lambda']*bci)) + (slow_process['num_intervals_slater']*(1 - np.exp(-1*slow_process['lambda']*bci)))

    return bci, misassigned_points

def model(t, f_intervals, f_lambda, s_intervals, s_lambda):
    return (np.log((f_intervals*f_lambda*np.exp(-1*f_lambda*t))  + (s_intervals*s_lambda*np.exp(-1*s_lambda*t))))

def get_bci_from_sibly_method(intervals_ms, survival, fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Sibly et al. (1990) derived using NLIN curve-fitting techniques to model the regression lines with a single curve.
    """

    x0 = np.array([fast_process['num_intervals_slater'], fast_process['lambda'], slow_process['num_intervals_slater'], slow_process['lambda']], dtype='float64')
    nlin_inds = np.concatenate([fast_process['indices'], np.where(slow_process['indices']==True)[0]])
    cfit_sols = scipy.optimize.curve_fit(model, intervals_ms[nlin_inds].astype('float64'), survival[nlin_inds].astype('float64'), p0=x0)
    nlin_results = dict()
    nlin_results['solution'] = cfit_sols[0]
    nlin_results['fast_num_intervals'] = nlin_results['solution'][0]
    nlin_results['fast_lambda'] = nlin_results['solution'][1]
    nlin_results['slow_num_intervals'] = nlin_results['solution'][2]
    nlin_results['slow_lambda'] = nlin_results['solution'][3]

    bci_coeff = (1/(nlin_results['fast_lambda'] - nlin_results['slow_lambda'])) 
    nlin_results['bci'] = bci_coeff * np.log((nlin_results['fast_num_intervals']*nlin_results['fast_lambda'])/(nlin_results['slow_num_intervals']*nlin_results['slow_lambda']))

    fast_misassignments = (nlin_results['fast_num_intervals']*np.exp(-1*nlin_results['fast_lambda']*nlin_results['bci']))
    slow_missasignments = (nlin_results['slow_num_intervals']*(1 - np.exp(-1*nlin_results['slow_lambda']*nlin_results['bci'])))
    misassigned_points_optim = fast_misassignments + slow_missasignments

    return nlin_results, misassigned_points_optim