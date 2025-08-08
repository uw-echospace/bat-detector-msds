import pandas as pd
import numpy as np

import sys
sys.path.append('../src')

import bout.clustering as clstr


def construct_bout_metrics_from_classified_dets(fgroups_with_bouttags):
    """
    Reads in the dataframe of detected calls with bout tags.
    Uses these bout tags to create a new dataframe of bout metrics for the start and end times of each bout.
    Also includes the lowest frequency of a call within a bout as the lower bound for the bout
    and the highest frequency of a call within a bout as the upper bound frequency for the bout.
    Now, also included the number of detections captured within each bout.
    """

    location_df = fgroups_with_bouttags.copy()
    location_df.reset_index(drop=True, inplace=True)
    group_of_tagged_dets = location_df['freq_group'].unique().item()

    end_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout end', 'call_end_time'])
    start_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout start', 'call_start_time'])
    ref_end_times = location_df.loc[location_df['call_status']=='bout end', 'end_time_wrt_ref'].astype('float')
    ref_start_times = location_df.loc[location_df['call_status']=='bout start', 'start_time_wrt_ref'].astype('float')
    end_times = location_df.loc[location_df['call_status']=='bout end', 'end_time'].astype('float')
    start_times = location_df.loc[location_df['call_status']=='bout start', 'start_time'].astype('float')
    bout_starts = start_times_of_bouts.index
    bout_ends = end_times_of_bouts.index

    low_freqs = []
    high_freqs = []
    ref_time_cycle_start = []
    ref_time_cycle_end = []
    num_calls_per_bout = []
    for i, bout_start in enumerate(bout_starts):
        bat_bout = location_df.iloc[bout_start:bout_ends[i]+1]
        bat_bout = bat_bout.loc[bat_bout['class']!='MADE-UP FOR DC INVESTIGATION']
        pass_low_freq = np.min(bat_bout['low_freq'])
        pass_high_freq = np.max(bat_bout['high_freq'])
        start_cycle = bat_bout['cycle_ref_time'].values[0]
        end_cycle = bat_bout['cycle_ref_time'].values[-1]
        num_calls = len(bat_bout)
        low_freqs += [pass_low_freq]
        high_freqs += [pass_high_freq]
        ref_time_cycle_start += [start_cycle]
        ref_time_cycle_end += [end_cycle]
        num_calls_per_bout += [num_calls]

    bout_metrics = pd.DataFrame()
    bout_metrics['start_time_of_bout'] = start_times_of_bouts.values
    bout_metrics['end_time_of_bout'] = end_times_of_bouts.values
    bout_metrics['start_time_wrt_ref'] = ref_start_times.values
    bout_metrics['end_time_wrt_ref'] = ref_end_times.values
    bout_metrics['start_time'] = start_times.values
    bout_metrics['end_time'] = end_times.values
    bout_metrics['low_freq'] = low_freqs
    bout_metrics['high_freq'] = high_freqs
    bout_metrics['freq_group'] = group_of_tagged_dets
    bout_metrics['cycle_ref_time_start'] = ref_time_cycle_start
    bout_metrics['cycle_ref_time_end'] = ref_time_cycle_end
    bout_metrics['number_of_dets'] = num_calls_per_bout
    bout_metrics['bout_duration'] = end_times_of_bouts.values - start_times_of_bouts.values
    bout_metrics['bout_duration_in_secs'] = bout_metrics['bout_duration'].apply(lambda x : x.total_seconds())

    return bout_metrics

def construct_bout_metrics_from_location_df_for_freqgroups(location_df):
    """
    Given a location summary with tagged bout markers, construct and concatenate together bout metrics for each group
    """

    bout_metrics = pd.DataFrame()
    for group in location_df['freq_group'].unique():
        if group != '':
            tagged_freq_dets = location_df.loc[location_df['freq_group']==group].copy()
            if not(tagged_freq_dets.empty):
                freqgroup_bout_metrics = construct_bout_metrics_from_classified_dets(tagged_freq_dets)
                if len(freqgroup_bout_metrics)>0:
                    if len(bout_metrics) > 0:
                        bout_metrics = pd.concat([bout_metrics, freqgroup_bout_metrics])
                    else:
                        bout_metrics = freqgroup_bout_metrics.copy()

    return bout_metrics

def construct_bout_metrics_for_freqgroups_with_cycle_interval(location_df, data_params):
    """
    Given a location summary with tagged bout markers, construct and concatenate together bout metrics for each group
    """
    cycle_length = int(data_params['cur_dc_tag'].split('of')[1])
    time_on_in_mins = int(data_params['cur_dc_tag'].split('of')[0])
    time_on_in_secs = 60*time_on_in_mins

    bout_metrics = pd.DataFrame()
    for group in location_df['freq_group'].unique():
        if group != '':
            tagged_freq_dets = location_df.loc[location_df['freq_group']==group].copy()
            if not(tagged_freq_dets.empty):
                cycle_length_groups = tagged_freq_dets.groupby('cycle_ref_time', group_keys=False)
                fixed_dets = cycle_length_groups.apply(lambda x: add_placeholder_to_tag_dets_wrt_cycle(x, cycle_length))
                freqgroup_bout_metrics = construct_bout_metrics_from_classified_dets(fixed_dets)
                total_bout_dur_per_cycle = freqgroup_bout_metrics.groupby('cycle_ref_time_start', group_keys=False).apply(lambda x: check_bout_duration_per_cycle(x, time_on_in_secs))
                if len(freqgroup_bout_metrics)>0:
                    if len(bout_metrics) > 0:
                        bout_metrics = pd.concat([bout_metrics, freqgroup_bout_metrics])
                    else:
                        bout_metrics = freqgroup_bout_metrics.copy()

    return bout_metrics

def classify_bouts_in_detector_preds_for_freqgroups(batdetect2_predictions, bout_params):
    """
    Given a location summary and BCIs calculated for each group in location summary, tag each call in summary as the following:
    - Within Bout : Call existing inside a bout
    - Outside Bout: Call that is not a part of any bout
    - Bout Start: Within-bout call that starts a new bout
    - Bout End: Within-bout call that ends the bout
    """

    location_df = batdetect2_predictions.copy()
    location_df.insert(0, 'duration_from_last_call_ms', 0)
    location_df.insert(0, 'bout_tag', 0)
    location_df.insert(0, 'change_markers', 0)
    location_df.insert(0, 'call_status', '')
    result_df = pd.DataFrame()

    for group in location_df['freq_group'].unique():
        if group != '':
            freq_group_df = location_df.loc[location_df['freq_group']==group].copy()
            freq_group_df.reset_index(drop=True, inplace=True)
            if not(freq_group_df.empty):
                intervals = (pd.to_datetime(freq_group_df['call_start_time'].values[1:]) - pd.to_datetime(freq_group_df['call_end_time'].values[:-1]))
                ipis_f = intervals.to_numpy(dtype='float32')/1e6
                ipis_f = np.insert(ipis_f, 0, bout_params[f'{group}_bci'])

                freq_group_df['duration_from_last_call_ms'] =  ipis_f
                freq_group_df.loc[freq_group_df['duration_from_last_call_ms'] < bout_params[f'{group}_bci'], 'bout_tag'] = 1
                freq_group_df.loc[freq_group_df['duration_from_last_call_ms'] >= bout_params[f'{group}_bci'], 'bout_tag'] = 0
                wb_indices = freq_group_df.loc[freq_group_df['bout_tag']==1].index
                ob_indices = freq_group_df.loc[freq_group_df['bout_tag']==0].index
                freq_group_df.loc[wb_indices, 'call_status'] = 'within bout'
                freq_group_df.loc[ob_indices, 'call_status'] = 'outside bout'

                bout_tags = freq_group_df['bout_tag']
                change_markers = bout_tags.shift(-1) - bout_tags
                change_markers[len(change_markers)-1] = 0
                freq_group_df['change_markers'] = change_markers
                be_indices = freq_group_df.loc[freq_group_df['change_markers']==-1].index
                bs_indices = freq_group_df.loc[freq_group_df['change_markers']==1].index
                freq_group_df.loc[be_indices, 'call_status'] = 'bout end'
                freq_group_df.loc[bs_indices, 'call_status'] = 'bout start'

                num_bout_starts = len(freq_group_df.loc[freq_group_df['call_status']=='bout start'])
                num_bout_ends = len(freq_group_df.loc[freq_group_df['call_status']=='bout end'])
                if num_bout_starts != num_bout_ends:
                    freq_group_df.at[len(freq_group_df)-1, 'call_status'] = 'bout end'

                result_df = pd.concat([result_df, freq_group_df])

    return result_df

def add_placeholder_call_at_end_of_cycle(df, cycle_length):
    end_time_of_last_call = df.loc[len(df)-1, 'call_end_time']
    cycle_time_of_last_call = df.loc[len(df)-1, 'cycle_ref_time']
    next_cycle_time = cycle_time_of_last_call + pd.Timedelta(minutes=cycle_length)
    fake_call_start_time = next_cycle_time
    fake_call_end_time = next_cycle_time
    duration_from_last_ms = (next_cycle_time - end_time_of_last_call).total_seconds()*1000
    start_time_wrt_ref = (next_cycle_time - cycle_time_of_last_call).total_seconds()
    end_time_wrt_ref = start_time_wrt_ref

    mod_df = pd.concat([df, pd.DataFrame(df.loc[len(df)-1]).T], axis=0, ignore_index=True)

    mod_df.loc[len(mod_df)-1, 'call_status'] = 'bout end'
    mod_df.loc[len(mod_df)-1, 'change_markers'] = -1
    mod_df.loc[len(mod_df)-1, 'bout_tag'] = 0
    mod_df.loc[len(mod_df)-1, 'duration_from_last_call_ms'] = duration_from_last_ms
    mod_df.loc[len(mod_df)-1, 'start_time_wrt_ref'] = start_time_wrt_ref
    mod_df.loc[len(mod_df)-1, 'end_time_wrt_ref'] = end_time_wrt_ref
    mod_df.loc[len(mod_df)-1, 'ref_time'] = fake_call_start_time
    mod_df.loc[len(mod_df)-1, 'call_start_time'] = fake_call_start_time
    mod_df.loc[len(mod_df)-1, 'call_end_time'] = fake_call_end_time
    mod_df.loc[len(mod_df)-1, 'cycle_ref_time'] = cycle_time_of_last_call
    mod_df.loc[len(mod_df)-1, 'class'] = 'MADE-UP FOR DC INVESTIGATION'

    return mod_df

def add_placeholder_call_at_start_of_cycle(df):
    cycle_time_of_first_call = df.loc[0, 'cycle_ref_time']
    fake_call_start_time = cycle_time_of_first_call
    fake_call_end_time = cycle_time_of_first_call
    duration_from_last_ms = 0
    start_time_wrt_ref = 0
    end_time_wrt_ref = start_time_wrt_ref

    mod_df = pd.concat([pd.DataFrame(df.loc[0]).T, df], axis=0, ignore_index=True)

    mod_df.loc[0, 'call_status'] = 'bout start'
    mod_df.loc[0, 'change_markers'] = 1
    mod_df.loc[0, 'bout_tag'] = 0
    mod_df.loc[0, 'duration_from_last_call_ms'] = duration_from_last_ms
    mod_df.loc[0, 'start_time_wrt_ref'] = start_time_wrt_ref
    mod_df.loc[0, 'end_time_wrt_ref'] = end_time_wrt_ref
    mod_df.loc[0, 'ref_time'] = fake_call_start_time
    mod_df.loc[0, 'call_start_time'] = fake_call_start_time
    mod_df.loc[0, 'call_end_time'] = fake_call_end_time
    mod_df.loc[0, 'cycle_ref_time'] = cycle_time_of_first_call
    mod_df.loc[0, 'class'] = 'MADE-UP FOR DC INVESTIGATION'

    return mod_df

def add_placeholder_to_tag_dets_wrt_cycle(cycle_group, cycle_length):
    df = cycle_group.copy()
    df.reset_index(inplace=True, drop=True)

    first_call_within_another_bout = (df.loc[0, 'call_status']=='within bout')|(df.loc[0, 'call_status']=='bout end')
    if first_call_within_another_bout:
        mod_df = add_placeholder_call_at_start_of_cycle(df)
    else:
        mod_df = df.copy()

    last_call_within_another_bout = (mod_df.loc[len(mod_df)-1, 'call_status']=='within bout')|(mod_df.loc[len(mod_df)-1, 'call_status']=='bout start')
    if last_call_within_another_bout:
        mod_df = add_placeholder_call_at_end_of_cycle(mod_df, cycle_length)

    return mod_df

def check_bout_duration_per_cycle(cycle_group, time_on):
    df = cycle_group.copy()
    df.reset_index(inplace=True, drop=True)

    total_bout_duration_in_cycle = df['bout_duration_in_secs'].sum()
    assert total_bout_duration_in_cycle <= time_on

    return total_bout_duration_in_cycle

def generate_bout_metrics_for_location_and_freq(dc_applied_df, data_params, bout_params):
    """
    Given a location summary of calls dataframe, create an analogous location summary of bouts by:
    1) Calculating the BCI for each frequency group in the summary.
    2) Use the calculated BCI for each group to cluster bouts for that group.
    3) Put together all bout characteristics into the analogous dataframe.
    """
    time_on_in_mins = int(data_params['cur_dc_tag'].split('of')[0])

    tagged_dets = classify_bouts_in_detector_preds_for_freqgroups(dc_applied_df, bout_params)
    bout_metrics_fixed = construct_bout_metrics_for_freqgroups_with_cycle_interval(tagged_dets, data_params)
    test_bout_end_times_in_period(bout_metrics_fixed, time_on_in_mins)

    return bout_metrics_fixed

def get_bout_params_from_location(location_sum_df, data_params):
    """
    Given a location summary and the location it corresponds to, calculate the BCIs for each frequency group in the summary
    """

    bout_params = dict()
    bout_params['site_key'] = data_params['site_tag']

    for group in location_sum_df['freq_group'].unique():
        if group != '':
            freq_group_df = location_sum_df.loc[location_sum_df['freq_group']==group].copy()
            if not(freq_group_df.empty):
                ipis_loc, hist_loc = clstr.get_histogram(freq_group_df, 10)
                intervals_ms, survival = clstr.get_log_survival(hist_loc)
                fast_process = clstr.regress_around_fast_intervals(intervals_ms, survival, hist_loc[0])
                fast_process = clstr.calculate_exponential_coefficients(fast_process)
                slow_process = clstr.regress_around_slow_intervals(intervals_ms, survival)
                slow_process = clstr.calculate_exponential_coefficients(slow_process)
                bci, misassigned_points = clstr.get_bci_from_slater_method(intervals_ms, survival, fast_process, slow_process)
                bout_params[f'{group}_bci'] = bci


    return bout_params

def test_bout_end_times_in_period(bout_metrics, time_on):
    """
    A test function to see if the duty cycle effects from the location summary of all calls carried over to the location summary of all bouts.
    """
    assert(bout_metrics['end_time_wrt_ref'].max() <= 60*time_on)