import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


def calculate_total_time(data, fps=20, from_durations=False):
    """Calculate the total time spent in behaviors from frame intervals or durations."""
    total_time_data = {}
    for key, values in data.items():
        # Recursively process nested dictionaries
        if isinstance(values, dict):
            total_time_data[key] = calculate_total_time(values, fps=fps, from_durations=from_durations)
        else:
            # Filter out NaN values from the list
            if isinstance(values, list):
                values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            
            # Check if all values are NaN or empty
            if len(values) == 0:
                total_time_data[key] = [0]
            else:
                if from_durations:
                    # Sum durations directly if already in seconds
                    total_time_sec = sum(values)
                else:
                    # Calculate total frames from (start, end) tuples
                    total_frames = 0
                    for value in values:
                        start, end = value
                        total_frames += (end - start)
                    # Convert frames to seconds
                    total_time_sec = total_frames / fps
                total_time_data[key] = [total_time_sec]
    return total_time_data


def _calculate_pct_recursive(time_data, total_time_value):
    """Recursively calculate percentage of time spent in behaviors relative to total time."""
    pct_data = {}
    for key, values in time_data.items():
        if isinstance(values, dict):
            # Get session-specific total time if available
            if isinstance(total_time_value, dict):
                sess_total_time = total_time_value.get(key, np.nan)
            else:
                sess_total_time = total_time_value
            # Recursively process nested dictionaries
            pct_data[key] = _calculate_pct_recursive(values, sess_total_time)
        else:
            # Handle invalid or zero values
            if np.all(np.isnan(values)) or np.isnan(total_time_value) or total_time_value == 0:
                pct_data[key] = [0]
            else:
                # Calculate percentage
                pct = (values[0] / total_time_value) * 100
                pct_data[key] = [pct]
    return pct_data


def calculate_pct(data, total_time, fps=20, from_durations=False):
    """Calculate the percentage of total time spent in each behavior."""
    # First calculate total time for each behavior
    total_time_data = calculate_total_time(data, fps=fps, from_durations=from_durations)
    # Set initial total time value for recursion
    if isinstance(total_time, dict):
        initial_total_time = np.nan
    else:
        initial_total_time = total_time
    # Recursively calculate percentages
    pct_data = _calculate_pct_recursive(total_time_data, initial_total_time)
    return pct_data


def calculate_all_time(data, fps=20):
    """Calculate durations for all individual behavior bouts."""
    all_time_data = {}
    for key, values in data.items():
        # Recursively process nested dictionaries
        if isinstance(values, dict):
            all_time_data[key] = calculate_all_time(values, fps=fps)
        else:
            # Filter out NaN values from the list
            if isinstance(values, list):
                values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
            
            # Check if all values are NaN or empty
            if len(values) == 0:
                all_time_data[key] = [np.nan]
            else:
                # Calculate duration for each bout
                durations = []
                for value in values:
                    start, end = value
                    duration = (end - start) / fps
                    durations.append(duration)
                all_time_data[key] = durations
    return all_time_data


def calculate_average_duration(data, fps=20, from_durations=False):
    """Calculate the average duration of behavior bouts."""
    avg_duration_data = {}
    for key, values in data.items():
        # Recursively process nested dictionaries
        if isinstance(values, dict):
            avg_duration_data[key] = calculate_average_duration(values, fps=fps, from_durations=from_durations)
        else:
            if np.all(np.isnan(values)):
                avg_duration_data[key] = [np.nan]
            else:
                if from_durations:
                    # Use values directly if already durations
                    durations = values
                else:
                    # Convert (start, end) tuples to durations
                    durations = []
                    for value in values:
                        start, end = value
                        duration = (end - start) / fps
                        durations.append(duration)
                # Calculate mean duration
                avg_duration_data[key] = [np.mean(durations)]
    return avg_duration_data


def calculate_bouts(data, from_durations=False):
    """Count the number of behavior bouts."""
    count_data = {}
    for key, values in data.items():
        # Recursively process nested dictionaries
        if isinstance(values, dict):
            count_data[key] = calculate_bouts(values, from_durations=from_durations)
        else:
            # Return 0 if no valid data
            if np.all(np.isnan(values)):
                count_data[key] = [0]
            else:
                # Count number of bouts
                count_data[key] = [len(values)]
    return count_data

def _calculate_frequency_recursive(data, total_time_value):
    """Recursively calculate the frequency of behaviors in bouts per minute."""
    freq_data = {}
    for key, values in data.items():
        if isinstance(values, dict):
            # Get session-specific total time if available
            if isinstance(total_time_value, dict):
                sess_total_time = total_time_value.get(key, np.nan)
            else:
                sess_total_time = total_time_value
            # Recursively process nested dictionaries
            freq_data[key] = _calculate_frequency_recursive(values, sess_total_time)
        else:
            # Handle invalid or zero values
            if (
                np.all(np.isnan(values)) or
                np.isnan(total_time_value) or
                total_time_value == 0
            ):
                freq_data[key] = [0]
            else:
                # Calculate frequency as bouts per minute
                n_bouts = len(values)
                freq = n_bouts / (total_time_value / 60.0)  # per minute
                freq_data[key] = [freq]
    return freq_data


def calculate_frequency(data, total_time, fps=20, from_durations=False):
    """Calculate the frequency of behaviors in bouts per minute."""
    # Set initial total time value for recursion
    if isinstance(total_time, dict):
        initial_total_time = np.nan
    else:
        initial_total_time = total_time
    # Recursively calculate frequencies
    return _calculate_frequency_recursive(data, initial_total_time)


def calculate_1st_latency(data, fps=20):
    """Calculate the latency to the first behavior bout."""
    latency_data = {}
    for key, value in data.items():
        # Recursively process nested dictionaries
        if isinstance(value, dict):
            latency_data[key] = calculate_1st_latency(value, fps=fps)
        else:
            if np.all(np.isnan(value)):
                latency_data[key] = [np.nan]
            else:
                # Get start frame of first bout and convert to seconds
                start, end = value[0]
                latency = start / fps
                latency_data[key] = [latency]
    return latency_data


def calculate_all_latencies(data, fps=20):
    """Calculate latencies to all behavior bouts."""
    latency_data = {}
    for key, value in data.items():
        # Recursively process nested dictionaries
        if isinstance(value, dict):
            latency_data[key] = calculate_all_latencies(value, fps=fps)
        else:
            latencies = []
            if np.all(np.isnan(value)):
                latency_data[key] = [np.nan]
            else:
                # Calculate latency for each bout
                for val in value:
                    start, end = val
                    latency = start / fps
                    latencies.append(latency)
                latency_data[key] = latencies
    return latency_data


def calculate_latency(data, fps=20, first=False):
    """Calculate latency to behavior bouts, either first bout only or all bouts."""
    if first:
        # Calculate only first bout latency
        return calculate_1st_latency(data, fps=fps)
    else:
        # Calculate latencies for all bouts
        return calculate_all_latencies(data, fps=fps)


def calculate_std_duration(data, fps=20, from_durations=False):
    """Calculate the standard deviation of behavior bout durations."""
    std_duration_data = {}
    for key, values in data.items():
        # Recursively process nested dictionaries
        if isinstance(values, dict):
            std_duration_data[key] = calculate_std_duration(values, fps=fps, from_durations=from_durations)
        else:
            # Need at least 2 values to calculate std
            if np.all(np.isnan(values)) or len(values) < 2:
                std_duration_data[key] = [np.nan]
            else:
                if from_durations:
                    # Use values directly if already durations
                    durations = values
                else:
                    # Convert (start, end) tuples to durations
                    durations = []
                    for value in values:
                        start, end = value
                        duration = (end - start) / fps
                        durations.append(duration)
                # Calculate standard deviation
                std_duration_data[key] = [np.std(durations)]
    return std_duration_data


def _filter_recursive(data, start_frame, end_frame, behaviors, current_key=None, is_behavior_level=False):
    """Recursively filter behavior data by time range and behavior names."""
    if isinstance(data, dict):
        filtered_dict = {}
        for key, value in data.items():
            # Filter by behavior names if specified
            if behaviors is not None and isinstance(value, list):
                if key not in behaviors:
                    continue
            is_bhvr_level = isinstance(value, list)
            # Recursively filter nested data
            filtered_value = _filter_recursive(
                value, 
                start_frame=start_frame, 
                end_frame=end_frame, 
                behaviors=behaviors,
                current_key=key,
                is_behavior_level=is_bhvr_level
            )
            # Add to result if valid
            if is_bhvr_level:
                if isinstance(filtered_value, list):
                    filtered_dict[key] = filtered_value
            elif filtered_value:
                if isinstance(filtered_value, dict) and filtered_value:
                    filtered_dict[key] = filtered_value
                elif isinstance(filtered_value, list) and filtered_value:
                    # Exclude lists with single NaN value
                    if not (len(filtered_value) == 1 and isinstance(filtered_value[0], float) and np.isnan(filtered_value[0])):
                        filtered_dict[key] = filtered_value
        return filtered_dict if filtered_dict else {}
    elif isinstance(data, list):
        # No filtering if no time range specified
        if start_frame is None and end_frame is None:
            return data if data else []
        filtered_list = []
        for item in data:
            # Recursively filter list items
            filtered_item = _filter_recursive(
                item, 
                start_frame=start_frame, 
                end_frame=end_frame, 
                behaviors=behaviors,
                current_key=current_key,
                is_behavior_level=is_behavior_level
            )
            if filtered_item is not None:
                filtered_list.append(filtered_item)
        if is_behavior_level:
            return filtered_list
        return filtered_list if filtered_list else [np.nan]
    elif isinstance(data, tuple):
        # Filter (start, end) tuples by time range
        s, e = data
        if start_frame is None and end_frame is None:
            return (s, e)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = float('inf')
        # Exclude if outside range
        if e < start_frame or s > end_frame:
            return None
        # Clip to range boundaries
        if s < start_frame:
            s = start_frame
        if e > end_frame:
            e = end_frame
        return (s, e)
    else:
        return data


def _filter_series_in_time_range(bhvr_series_dict, toi, bhvr_list=None):
    """Filter pandas Series behavior data by time range and behavior names."""
    # Convert behavior list to set for faster lookup
    if bhvr_list is not None:
        if isinstance(bhvr_list, str):
            bhvr_list = [bhvr_list]
        bhvr_list = set(bhvr_list)
    # Parse time of interest parameter
    toi_dict = None
    default_start, default_end = None, None
    if toi is not None:
        if isinstance(toi, tuple):
            default_start, default_end = toi
        elif isinstance(toi, dict):
            toi_dict = toi
    filtered_dict = {}
    for sess_id, bhvr_data in bhvr_series_dict.items():
        # Get time range for this session
        if toi_dict is not None:
            if sess_id not in toi_dict:
                continue
            start_frame, end_frame = toi_dict[sess_id]
        elif default_start is not None:
            start_frame, end_frame = default_start, default_end
        else:
            start_frame, end_frame = None, None
        filtered_dict[sess_id] = {}
        for bhvr_name, series in bhvr_data.items():
            # Filter by behavior name if specified
            if bhvr_list is not None and bhvr_name not in bhvr_list:
                continue
            # Filter by time range
            if start_frame is None and end_frame is None:
                filtered_dict[sess_id][bhvr_name] = series.copy() if isinstance(series, pd.Series) else series
            elif isinstance(series, pd.Series):
                # Filter Series by index range and reindex from 0
                mask = (series.index >= start_frame) & (series.index < end_frame)
                filtered_series = series.loc[mask].copy()
                filtered_series.index = filtered_series.index - start_frame
                filtered_dict[sess_id][bhvr_name] = filtered_series
            else:
                filtered_dict[sess_id][bhvr_name] = series[start_frame:end_frame]
    return filtered_dict


def filter_behavior_in_time_range(data, toi=None, bhvr_list=None):
    """Filter behavior data by time range and optionally by behavior names."""
    # Detect if data contains pandas Series
    is_series_data = False
    for sess_id, sess_data in data.items():
        if isinstance(sess_data, dict):
            for bhvr_name, values in sess_data.items():
                if isinstance(values, pd.Series):
                    is_series_data = True
                break
        break
    # Use specialized function for Series data
    if is_series_data:
        return _filter_series_in_time_range(data, toi, bhvr_list)
    # Convert behavior list to set for faster lookup
    if bhvr_list is not None:
        if isinstance(bhvr_list, str):
            bhvr_list = [bhvr_list]
        bhvr_list = set(bhvr_list)
    # Parse time of interest parameter
    start_frame = None
    end_frame = None
    toi_dict = None
    if toi is not None:
        if isinstance(toi, tuple):
            start_frame, end_frame = toi
        elif isinstance(toi, dict):
            toi_dict = toi
            # Validate that all sessions in toi exist in data
            missing_keys = set(toi_dict.keys()) - set(data.keys())
            if missing_keys:
                raise KeyError(f"Can not find: {missing_keys}")
        else:
            raise TypeError("toi must be tuple or dict.")
        # Handle session-specific time ranges
        if toi_dict is not None:
            filtered_dict = {}
            for sess_id in data.keys():
                if sess_id in toi_dict:
                    start_frame, end_frame = toi_dict[sess_id]
                else:
                    start_frame, end_frame = None, None
                # Filter recursively
                filtered_value = _filter_recursive(
                    data[sess_id],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    behaviors=bhvr_list,
                    current_key=sess_id
                )
                if filtered_value:
                    filtered_dict[sess_id] = filtered_value
            
            return filtered_dict
    else:
        # No time filtering, just behavior filtering
        return _filter_recursive(data, start_frame, end_frame, bhvr_list)

def filter_coordinates_in_time_range(coord_dict, toi):
    """Filter coordinate arrays by session-specific time ranges."""
    filtered_coord_dict = {}
    
    for sess_id in coord_dict.keys():
        # Check if session has time range defined
        if sess_id not in toi:
            print(f"{sess_id} not found in toi")
            continue
        start_frame, end_frame = toi[sess_id]
        coords = coord_dict[sess_id]
        n_frames = coords.shape[0]
        # Validate and adjust frame boundaries
        if start_frame < 0:
            print(f"Warning: start_frame {start_frame} < 0 for {sess_id}, setting to 0")
            start_frame = 0
        if end_frame >= n_frames:
            print(f"Warning: end_frame {end_frame} >= n_frames {n_frames} for {sess_id}, setting to {n_frames-1}")
            end_frame = n_frames - 1
        # Slice coordinates array
        filtered_coords = coords[start_frame:end_frame]
        filtered_coord_dict[sess_id] = filtered_coords
    return filtered_coord_dict


def calculate_heatmap(
    coord_dict,
    sess_list=None,
    keypoint_idx=3,
    fps=20,
    xbins=12,
    ybins=10,
    min_speed=0,
    max_speed=300,
    time_range=None
):
    """Calculate spatial heatmaps with velocity information from coordinate data."""
    # Determine which sessions to process
    if sess_list is not None:
        sess_to_process = {sess_id: coord_dict[sess_id] for sess_id in sess_list if sess_id in coord_dict}
    else:
        sess_to_process = coord_dict
    results = {}
    for sess_id, coords in sess_to_process.items():
        n_frames = len(coords)
        # Get time range for this session
        if time_range is not None:
            if isinstance(time_range, dict):
                start_frame, end_frame = time_range.get(sess_id, (0, n_frames))
            else:
                start_frame, end_frame = time_range
        else:
            start_frame, end_frame = 0, n_frames
        # Validate frame boundaries
        start_frame = max(0, int(start_frame))
        end_frame = min(int(end_frame), n_frames)
        if start_frame >= end_frame:
            continue
        # Extract x and y coordinates for specified keypoint
        x = coords[start_frame:end_frame, keypoint_idx, 0].astype(float)
        y = coords[start_frame:end_frame, keypoint_idx, 1].astype(float)
        if x.size < 2 or y.size < 2:
            continue
        # Calculate velocity components
        vx = np.diff(x) * fps
        vy = np.diff(y) * fps
        speed = np.sqrt(vx ** 2 + vy ** 2)
        # Filter by speed range
        mask = (speed >= min_speed) & (speed <= max_speed)
        if not np.any(mask):
            continue
        x_filtered = x[:-1][mask]
        y_filtered = y[:-1][mask]
        vx_filtered = vx[mask]
        vy_filtered = vy[mask]
        speed_filtered = speed[mask]
        # Create 2D histogram for position counts
        count, xedges, yedges = np.histogram2d(x_filtered, y_filtered, bins=[xbins, ybins])
        # Calculate weighted histograms for velocity and speed
        grid_vx, _, _ = np.histogram2d(x_filtered, y_filtered, bins=[xedges, yedges], weights=vx_filtered)
        grid_vy, _, _ = np.histogram2d(x_filtered, y_filtered, bins=[xedges, yedges], weights=vy_filtered)
        speed_sum, _, _ = np.histogram2d(x_filtered, y_filtered, bins=[xedges, yedges], weights=speed_filtered)
        # Calculate average values per grid cell
        count_safe = np.where(count == 0, np.nan, count)
        grid_vx_avg = grid_vx / count_safe
        grid_vy_avg = grid_vy / count_safe
        speed_avg = speed_sum / count_safe
        # Mark cells with insufficient data as NaN
        valid_mask = count >= 1
        grid_vx_avg[~valid_mask] = np.nan
        grid_vy_avg[~valid_mask] = np.nan
        speed_avg[~valid_mask] = np.nan
        # Calculate frequency as percentage
        total_counts = np.sum(count)
        if total_counts > 0:
            hist_frequency = count / total_counts * 100
        else:
            hist_frequency = np.full_like(count, np.nan, dtype=float)
        hist_frequency[~valid_mask] = np.nan
        # Store results
        results[sess_id] = {
            'hist_frequency': hist_frequency,
            'speed_avg': speed_avg,
            'grid_vx_avg': grid_vx_avg,
            'grid_vy_avg': grid_vy_avg,
            'xedges': xedges,
            'yedges': yedges
        }
    return results


def merge_filtered_behaviors(*behavior_dicts, merged_name='merged_behavior'):
    """Merge multiple behavior dictionaries into one with unified time intervals."""
    merged_data = {}
    # Collect all session IDs across all behavior dictionaries
    all_sess_ids = set()
    for behavior_dict in behavior_dicts:
        all_sess_ids.update(behavior_dict.keys())
    for sess_id in all_sess_ids:
        # Collect all intervals from all behavior dictionaries for this session
        all_intervals = []
        for behavior_dict in behavior_dicts:
            if sess_id in behavior_dict:
                sess_behaviors = behavior_dict[sess_id]
                for behavior_name, intervals in sess_behaviors.items():
                    # Skip NaN placeholders
                    if not (len(intervals) == 1 and isinstance(intervals[0], float) and np.isnan(intervals[0])):
                        all_intervals.extend(intervals)
        # Handle case with no valid intervals
        if not all_intervals:
            merged_data[sess_id] = {merged_name: [np.nan]}
            continue
        # Sort intervals by start time
        all_intervals.sort(key=lambda x: x[0])
        # Merge overlapping or adjacent intervals
        merged_intervals = []
        current_start, current_end = all_intervals[0]
        for start, end in all_intervals[1:]:
            if start <= current_end + 1:  # Allow 1-frame gap
                # Extend current interval
                current_end = max(current_end, end)
            else:
                # Save current interval and start new one
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intervals.append((current_start, current_end))
        merged_data[sess_id] = {merged_name: merged_intervals}

    return merged_data


def flatten_bhvr_tuples(data, separator='_', only_top_key=True, fill_missing=True, fill_value=np.nan):
    """Flatten nested behavior dictionary structure into a single-level dictionary."""
    flattened = {}
    # Collect all leaf-level keys for filling missing values
    leaf_keys = set()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict):
                leaf_keys.update(v.keys())
    def _flatten_recursive(d, prefix='', top_key=None):
        """Helper function to recursively flatten nested dictionaries."""
        # Handle empty dictionaries
        if isinstance(d, dict) and not d:
            if fill_missing and prefix:
                target_key = prefix
                flattened[target_key] = [fill_value]
            return
        for key, value in d.items():
            if only_top_key:
                # Use only the top-level key, ignore nested keys
                new_key = prefix if prefix else key
                next_prefix = new_key
                next_top_key = top_key if top_key is not None else new_key
            else:
                # Create hierarchical key with separator
                new_key = f"{prefix}{separator}{key}" if prefix else key
                next_prefix = new_key
                next_top_key = top_key if top_key is not None else new_key
            if isinstance(value, dict):
                if value:
                    # Recursively process nested dictionary
                    _flatten_recursive(value, next_prefix, next_top_key)
                elif fill_missing:
                    # Fill missing nested values
                    if (not only_top_key) and leaf_keys:
                        for leaf in leaf_keys:
                            flattened[f"{new_key}{separator}{leaf}"] = [fill_value]
                    else:
                        target_key = new_key
                        flattened[target_key] = [fill_value]
            else:
                # Store leaf value
                flattened[new_key] = value

    _flatten_recursive(data)
    return flattened


def flatten_to_dataframe(flattened, value_column='value'):
    """Convert flattened behavior dictionary to a pandas DataFrame."""
    rows = []
    for group_name, values in flattened.items():
        if isinstance(values, list):
            if len(values) == 0:
                # Add NaN for empty lists
                rows.append({'group': group_name, value_column: np.nan})
            else:
                # Add one row per value
                for v in values:
                    rows.append({'group': group_name, value_column: v})
        else:
            # Add single value
            rows.append({'group': group_name, value_column: values})
    return pd.DataFrame(rows)


def sync_start_time(data, toi, fps=20):
    """Synchronize behavior data by adjusting frame indices to start from zero."""
    synced_data = {}
    for sess_id, sess_data in data.items():
        # Get start frame offset for this session
        if sess_id in toi:
            start_frame, _ = toi[sess_id]
        else:
            # No synchronization needed
            synced_data[sess_id] = sess_data
            continue
        if isinstance(sess_data, dict):
            synced_sess = {}
            for behavior, values in sess_data.items():
                if isinstance(values, pd.Series):
                    # Adjust Series index
                    synced_series = values.copy()
                    synced_series.index = synced_series.index - start_frame
                    synced_sess[behavior] = synced_series
                elif isinstance(values, list):
                    # Adjust tuples in list
                    synced_tuples = []
                    for item in values:
                        if isinstance(item, tuple) and len(item) == 2:
                            s, e = item
                            synced_tuples.append((s - start_frame, e - start_frame))
                        elif isinstance(item, float) and np.isnan(item):
                            synced_tuples.append(item)
                        else:
                            synced_tuples.append(item)
                    synced_sess[behavior] = synced_tuples
                else:
                    synced_sess[behavior] = values
            synced_data[sess_id] = synced_sess
        elif isinstance(sess_data, list):
            # Adjust list of tuples
            synced_tuples = []
            for item in sess_data:
                if isinstance(item, tuple) and len(item) == 2:
                    s, e = item
                    synced_tuples.append((s - start_frame, e - start_frame))
                elif isinstance(item, float) and np.isnan(item):
                    synced_tuples.append(item)
                else:
                    synced_tuples.append(item)
            synced_data[sess_id] = synced_tuples
        elif isinstance(sess_data, pd.Series):
            # Adjust Series index
            synced_series = sess_data.copy()
            synced_series.index = synced_series.index - start_frame
            synced_data[sess_id] = synced_series
        else:
            synced_data[sess_id] = sess_data
    return synced_data


def calculate_grouped_behavior_time(
    bhvr_tuples_dict,
    group_bhvr_dict,
    time_of_interest=None,
    total_frames=6000,
    fps=20
):
    """Calculate time spent in grouped behavior categories for each session."""
    results_by_session = {}
    # Initialize storage for category percentages
    category_percentages = {cat: [] for cat in group_bhvr_dict.keys()}
    
    for sess_id, behaviors_data in bhvr_tuples_dict.items():
        # Determine time range for this session
        if time_of_interest is not None:
            if isinstance(time_of_interest, dict):
                if sess_id not in time_of_interest:
                    continue
                start_frame, end_frame = time_of_interest[sess_id]
            else:
                start_frame, end_frame = time_of_interest
            session_total_frames = end_frame - start_frame
        else:
            start_frame, end_frame = 0, total_frames
            session_total_frames = total_frames
        
        sess_result = {}
        total_categorized_frames = 0
        
        # Process each behavior category
        for category, behavior_list in group_bhvr_dict.items():
            if not behavior_list:
                continue
            
            category_frames = 0
            # Sum frames for all behaviors in this category
            for behavior in behavior_list:
                if behavior in behaviors_data:
                    for bout in behaviors_data[behavior]:
                        bout_start, bout_end = bout
                        # Clip bout to time range
                        clipped_start = max(bout_start, start_frame)
                        clipped_end = min(bout_end, end_frame)
                        if clipped_start < clipped_end:
                            category_frames += (clipped_end - clipped_start)
            
            # Calculate time and percentage
            time_sec = category_frames / fps
            percentage = (category_frames / session_total_frames) * 100 if session_total_frames > 0 else 0
            
            sess_result[category] = {
                'frames': category_frames,
                'time_sec': time_sec,
                'percentage': percentage
            }
            total_categorized_frames += category_frames
            category_percentages[category].append(percentage)
        
        # Handle unlabeled/remainder category (empty behavior list)
        for category, behavior_list in group_bhvr_dict.items():
            if not behavior_list:
                remaining_frames = session_total_frames - total_categorized_frames
                remaining_frames = max(0, remaining_frames)
                time_sec = remaining_frames / fps
                percentage = (remaining_frames / session_total_frames) * 100 if session_total_frames > 0 else 0
                
                sess_result[category] = {
                    'frames': remaining_frames,
                    'time_sec': time_sec,
                    'percentage': percentage
                }
                category_percentages[category].append(percentage)
        
        results_by_session[sess_id] = sess_result
    
    # Calculate summary statistics across sessions
    summary = {}
    for category in group_bhvr_dict.keys():
        pcts = category_percentages[category]
        if pcts:
            summary[category] = {
                'avg_percentage': np.mean(pcts),
                'std_percentage': np.std(pcts),
                'all_percentages': pcts
            }
        else:
            summary[category] = {
                'avg_percentage': 0,
                'std_percentage': 0,
                'all_percentages': []
            }
    
    return {
        'by_session': results_by_session,
        'summary': summary
    }


def calculate_grouped_behavior_time_from_series(
    bhvr_series_dict,
    group_bhvr_dict,
    toi=None,
    fps=20
):
    """Calculate time spent in grouped behavior categories from pandas Series data."""
    frames_dict = {}
    time_dict = {}
    pct_dict = {}

    for sess_id, behaviors_data in bhvr_series_dict.items():
        # Convert to DataFrame if needed
        if isinstance(behaviors_data, pd.DataFrame):
            sess_df = behaviors_data
        else:
            sess_df = pd.DataFrame(behaviors_data)
        if sess_df.empty:
            pct_dict[sess_id] = {}
            continue
        # Get time range for this session
        if toi is not None:
            if isinstance(toi, dict):
                if sess_id not in toi:
                    continue
                start_frame, end_frame = toi[sess_id]
            else:
                start_frame, end_frame = toi
        else:
            start_frame = 0
            end_frame = len(sess_df)
        # Filter dataframe by time range
        if sess_df.index.dtype == 'int64' or sess_df.index.dtype == 'int32':
            mask = (sess_df.index >= start_frame) & (sess_df.index < end_frame)
            sess_df_filtered = sess_df.loc[mask]
        else:
            sess_df_filtered = sess_df.iloc[start_frame:end_frame]

        session_total_frames = len(sess_df_filtered)
        if session_total_frames == 0:
            continue

        frames_dict[sess_id] = {}
        time_dict[sess_id] = {}
        pct_dict[sess_id] = {}

        # Track all labeled frames
        all_labeled_mask = np.zeros(session_total_frames, dtype=bool)

        # Process labeled categories
        for category, behavior_list in group_bhvr_dict.items():
            if not behavior_list:
                continue
            # Create mask for this category
            category_mask = np.zeros(session_total_frames, dtype=bool)
            for behavior in behavior_list:
                if behavior in sess_df_filtered.columns:
                    behavior_values = sess_df_filtered[behavior].values
                    category_mask |= (behavior_values == 1)
            all_labeled_mask |= category_mask
            # Calculate statistics
            category_frames = int(np.sum(category_mask))
            time_sec = category_frames / fps
            percentage = (category_frames / session_total_frames) * 100 if session_total_frames > 0 else 0

            frames_dict[sess_id][category] = category_frames
            time_dict[sess_id][category] = time_sec
            pct_dict[sess_id][category] = percentage

        # Process unlabeled/remainder category (empty list in group_bhvr_dict)
        for category, behavior_list in group_bhvr_dict.items():
            if behavior_list:
                continue
            # Calculate unlabeled frames
            unlabeled_frames = session_total_frames - int(np.sum(all_labeled_mask))
            unlabeled_frames = max(0, unlabeled_frames)
            time_sec = unlabeled_frames / fps
            percentage = (unlabeled_frames / session_total_frames) * 100 if session_total_frames > 0 else 0

            frames_dict[sess_id][category] = unlabeled_frames
            time_dict[sess_id][category] = time_sec
            pct_dict[sess_id][category] = percentage

    return frames_dict, time_dict, pct_dict


def group_data_by_sess_id(data_dict, include_keywords=None, group_map=None):
    """Group session data by experimental conditions based on session ID keywords."""
    # Set default filtering keywords
    if include_keywords is None:
        include_keywords = ['WithRat']
    # Set default group mapping
    if group_map is None:
        group_map = {
            'sD': ('Single', 'Dom'), 
            'pD': ('Pair', 'Dom'), 
            'sS': ('Single', 'Sub'), 
            'pS': ('Pair', 'Sub')
        }
    # Initialize grouped data structure
    grouped_data = {group_name: {} for group_name in group_map.keys()}
    for sess_id, data in data_dict.items():
        # Filter by keywords
        if not all(keyword in sess_id for keyword in include_keywords):
            continue
        classified = False
        # Classify session into groups
        for group_name, keywords in group_map.items():
            if all(kw in sess_id for kw in keywords):
                # Aggregate behavior data for this group
                for behavior, values in data.items():
                    if behavior not in grouped_data[group_name]:
                        grouped_data[group_name][behavior] = None
                    # Handle different data types
                    if isinstance(values, list):
                        if grouped_data[group_name][behavior] is None:
                            grouped_data[group_name][behavior] = []
                        grouped_data[group_name][behavior].extend(values)
                    elif isinstance(values, (pd.Series, pd.DataFrame)):
                        if grouped_data[group_name][behavior] is None:
                            grouped_data[group_name][behavior] = values.copy()
                        else:
                            # Concatenate Series/DataFrames
                            grouped_data[group_name][behavior] = pd.concat(
                                [grouped_data[group_name][behavior], values], axis=0, ignore_index=True
                            )
                    else:
                        if grouped_data[group_name][behavior] is None:
                            grouped_data[group_name][behavior] = []
                        grouped_data[group_name][behavior].append(values)
                classified = True
                break
        if not classified:
            print(f"Ungrouped sess_id: {sess_id}")
    return grouped_data


def calculate_transition_matrices(bhvr_series_dict, boi, group_bhvr_dict=None, frame_gap_threshold=20, sess_list=None):
    """Calculate behavior transition probability matrices from time series data."""
    def get_main_behavior(row, behaviors_priority_list):
        """Get the first active behavior in priority order."""
        for behavior in behaviors_priority_list:
            if row.get(behavior, 0) == 1:
                return behavior
        return None
    
    def get_grouped_behavior(row, group_dict):
        """Get the grouped category for the first active behavior."""
        for group_name, behavior_list in group_dict.items():
            for behavior in behavior_list:
                if row.get(behavior, 0) == 1:
                    return group_name
        return None
    
    def compute_transition_matrix(main_behavior_df, states, frame_gap_threshold, skip_self_transitions=False, skip_state=None):
        """Compute normalized transition probability matrix from behavior sequence."""
        transition_matrix = pd.DataFrame(0, index=states, columns=states)
        
        # Sort by frame index and identify behavior blocks
        main_behavior_df = main_behavior_df.sort_values("frame_idx").reset_index(drop=True)
        behavior_change = (main_behavior_df["behavior"] != main_behavior_df["behavior"].shift(1))
        frame_discontinuity = (main_behavior_df["frame_idx"].diff() > 1)
        new_block = behavior_change | frame_discontinuity
        main_behavior_df["block_id"] = new_block.cumsum()
        
        # Aggregate blocks to get start/end frames
        blocks_info = main_behavior_df.groupby("block_id").agg({
            "frame_idx": ["min", "max"],
            "behavior": "first"
        })
        blocks_info.columns = ["start_frame", "end_frame", "behavior"]
        blocks_info.reset_index(drop=True, inplace=True)
        
        # Count transitions between consecutive blocks
        for i in range(len(blocks_info) - 1):
            b1 = blocks_info.loc[i, "behavior"]
            b2 = blocks_info.loc[i + 1, "behavior"]
            end_i = blocks_info.loc[i, "end_frame"]
            start_i1 = blocks_info.loc[i + 1, "start_frame"]
            gap = start_i1 - end_i
            
            # Skip self-transitions for certain states if requested
            if skip_self_transitions and skip_state and b1 == skip_state and b2 == skip_state:
                continue
            
            # Count transition if within gap threshold
            if gap <= frame_gap_threshold:
                if skip_self_transitions and skip_state:
                    if not (b1 == skip_state and b2 == skip_state):
                        transition_matrix.loc[b1, b2] += 1
                else:
                    transition_matrix.loc[b1, b2] += 1
            else:
                # Handle large gaps with gap_state or skip_state
                if 'gap_state' in states:
                    transition_matrix.loc[b1, "gap_state"] += 1
                    transition_matrix.loc["gap_state", b2] += 1
                elif skip_state and skip_state in states:
                    if b1 != skip_state:
                        transition_matrix.loc[b1, skip_state] += 1
                    if b2 != skip_state:
                        transition_matrix.loc[skip_state, b2] += 1
        
        # Normalize to get transition probabilities
        row_sums = transition_matrix.sum(axis=1)
        normalized_matrix = transition_matrix.copy()
        for state in normalized_matrix.index:
            if row_sums[state] != 0:
                normalized_matrix.loc[state] = normalized_matrix.loc[state] / row_sums[state]
        return normalized_matrix
    
    # Determine which sessions to process
    if sess_list is not None:
        sess_to_process = {sess_id: bhvr_series_dict[sess_id] 
                          for sess_id in sess_list if sess_id in bhvr_series_dict}
    else:
        sess_to_process = bhvr_series_dict
    results = {'fine_grained': {}}
    if group_bhvr_dict is not None:
        results['grouped'] = {}
    # Define all possible states including gap state
    all_states = list(boi) + ["gap_state"]
    for sess_id, behavior_data in sess_to_process.items():
        sess_df = pd.DataFrame(behavior_data)
        if sess_df.empty:
            continue
        # Extract main behavior for each frame
        main_behavior_list = []
        for frame_idx in sess_df.index:
            row = sess_df.loc[frame_idx]
            b = get_main_behavior(row, boi)
            main_behavior_list.append((frame_idx, b))
        main_behavior_df = pd.DataFrame(main_behavior_list, columns=["frame_idx", "behavior"])
        main_behavior_df.dropna(subset=["behavior"], inplace=True)
        # Compute fine-grained transition matrix
        if not main_behavior_df.empty:
            normalized_matrix = compute_transition_matrix(
                main_behavior_df, all_states, frame_gap_threshold
            )
            results['fine_grained'][sess_id] = normalized_matrix
        # Compute grouped transition matrix if requested
        if group_bhvr_dict is not None:
            grouped_behavior_list = []
            for frame_idx in sess_df.index:
                row = sess_df.loc[frame_idx]
                b = get_grouped_behavior(row, group_bhvr_dict)
                grouped_behavior_list.append((frame_idx, b))            
                grouped_behavior_df = pd.DataFrame(grouped_behavior_list, columns=["frame_idx", "behavior"])
            grouped_behavior_df.dropna(subset=["behavior"], inplace=True)
            if not grouped_behavior_df.empty:
                grouped_states = list(group_bhvr_dict.keys())
                # Identify unlabeled/gap category
                skip_state = None
                for cat, bhvrs in group_bhvr_dict.items():
                    if not bhvrs or 'gap_state' in bhvrs:
                        skip_state = cat
                        break
                # Compute grouped transition matrix with skip state
                normalized_grouped = compute_transition_matrix(
                    grouped_behavior_df, grouped_states, frame_gap_threshold,
                    skip_self_transitions=True, skip_state=skip_state
                )
                results['grouped'][sess_id] = normalized_grouped
    return results


def group_transition_matrices(trans_matrices_dict, include_keywords=None, group_map=None):
    """Group and average transition matrices by experimental conditions."""
    # Set default filtering keywords
    if include_keywords is None:
        include_keywords = ['WithRat']
    # Set default group mapping
    if group_map is None:
        group_map = {
            'sD': ('Single', 'Dom'),
            'pD': ('Pair', 'Dom'),
            'sS': ('Single', 'Sub'),
            'pS': ('Pair', 'Sub')
        }
    # Initialize grouped matrices storage
    grouped_matrices = {group_name: [] for group_name in group_map.keys()}
    for sess_id, matrix in trans_matrices_dict.items():
        # Filter by keywords
        if not all(keyword in sess_id for keyword in include_keywords):
            continue
        classified = False
        # Classify matrix into groups
        for group_name, keywords in group_map.items():
            if all(kw in sess_id for kw in keywords):
                grouped_matrices[group_name].append(matrix)
                classified = True
                break
        if not classified:
            print(f"Ungrouped sess_id: {sess_id}")
    # Average matrices within each group
    averaged_matrices = {}
    for group_name, matrices_list in grouped_matrices.items():
        if len(matrices_list) == 0:
            print(f"Warning: No matrices found for group '{group_name}'")
            averaged_matrices[group_name] = None
        elif len(matrices_list) == 1:
            averaged_matrices[group_name] = matrices_list[0].copy()
        else:
            # Collect all unique states
            all_indices = set()
            all_columns = set()
            for mat in matrices_list:
                all_indices.update(mat.index)
                all_columns.update(mat.columns)
            all_indices = sorted(list(all_indices))
            all_columns = sorted(list(all_columns))
            # Align all matrices to same dimensions
            aligned_matrices = []
            for mat in matrices_list:
                aligned = mat.reindex(index=all_indices, columns=all_columns, fill_value=0)
                aligned_matrices.append(aligned)
            # Calculate mean across aligned matrices
            stacked = np.stack([m.values for m in aligned_matrices], axis=0)
            avg_values = np.mean(stacked, axis=0)
            averaged_matrices[group_name] = pd.DataFrame(
                avg_values, 
                index=all_indices, 
                columns=all_columns
            )
    return averaged_matrices

def group_heatmap_results(heatmap_results, group_map=None, include_keywords=None):
    """Group and average heatmap results by experimental conditions."""
    # Set default group mapping
    if group_map is None:
        group_map = {
            'sD': ('Single', 'Dom'),
            'pD': ('Pair', 'Dom'),
            'sS': ('Single', 'Sub'),
            'pS': ('Pair', 'Sub')
        }
    # Initialize grouped storage
    grouped = {g: [] for g in group_map}
    for sess_id, data in heatmap_results.items():
        if data is None:
            continue
        # Filter by keywords if specified
        if include_keywords and not all(kw in sess_id for kw in include_keywords):
            continue
        # Classify into groups
        for group_name, kws in group_map.items():
            if all(kw in sess_id for kw in kws):
                grouped[group_name].append(data)
                break
    # Aggregate results within each group
    aggregated_results = {}
    for group_name, items in grouped.items():
        if not items:
            aggregated_results[group_name] = None
            continue
        # Use first item as template for grid edges
        base = items[0]
        xedges = base['xedges']
        yedges = base['yedges']
        # Average each field across sessions
        def _nanmean_stack(key):
            stack = [it[key] for it in items if key in it]
            return np.nanmean(np.stack(stack, axis=0), axis=0)
        aggregated_results[group_name] = {
            'hist_frequency': _nanmean_stack('hist_frequency'),
            'speed_avg': _nanmean_stack('speed_avg'),
            'grid_vx_avg': _nanmean_stack('grid_vx_avg'),
            'grid_vy_avg': _nanmean_stack('grid_vy_avg'),
            'xedges': xedges,
            'yedges': yedges
        }
    return aggregated_results


def calculate_behavior_pct_by_time_bins(bhvr_data, time_bins, fps=20):
    """Calculate behavior percentage across multiple time bins for each session."""
    # Generate labels for time bins
    key_labels = [f"{start//fps}-{end//fps}s" for start, end in time_bins]
    result = {}
    for sess_id in bhvr_data.keys():
        result[sess_id] = {}
        # Process each time bin
        for (start_frame, end_frame) in time_bins:
            bin_duration = (end_frame - start_frame) / fps
            # Filter data for this time bin
            toi = {sess_id: (start_frame, end_frame)}
            filtered_data = filter_behavior_in_time_range(bhvr_data, toi)
            pct_value = 0
            bhvr_name = None
            # Calculate percentage for this bin
            if sess_id in filtered_data:
                pct_data = calculate_pct(filtered_data, total_time=bin_duration, fps=fps)
                if sess_id in pct_data:
                    for bhvr_name, pct_list in pct_data[sess_id].items():
                        pct_value = pct_list[0] if pct_list else 0
                        break
            # Use generic name if no behavior found
            if bhvr_name is None:
                bhvr_name = 'behavior'
            # Store percentage for this bin
            if bhvr_name not in result[sess_id]:
                result[sess_id][bhvr_name] = []
            result[sess_id][bhvr_name].append(pct_value)
    return result, key_labels


def group_bhvr_pct_by_time_bins(pct_by_time_bins, group_map=None, include_keywords=None):
    """Group behavior percentage data by experimental conditions across time bins."""
    # Set default filtering keywords
    if include_keywords is None:
        include_keywords = ['WithRat']
    # Set default group mapping
    if group_map is None:
        group_map = {
            'sD': ('Single', 'Dom'),
            'pD': ('Pair', 'Dom'),
            'sS': ('Single', 'Sub'),
            'pS': ('Pair', 'Sub'),
        }
    # Collect behaviors and determine number of bins
    behaviors = set()
    n_bins = None
    for sess_dict in pct_by_time_bins.values():
        for bhvr, vals in sess_dict.items():
            behaviors.add(bhvr)
            n_bins = len(vals)
        if n_bins is not None:
            break
    behaviors = list(behaviors)
    # Initialize grouped storage with separate lists for each bin
    grouped = {g: {b: [] for b in behaviors} for g in group_map}
    for sess_id, bhvr_dict in pct_by_time_bins.items():
        # Filter by keywords
        if include_keywords and not all(kw in sess_id for kw in include_keywords):
            continue
        # Classify into groups
        target_group = None
        for g, kws in group_map.items():
            if all(kw in sess_id for kw in kws):
                target_group = g
                break
        if target_group is None:
            continue
        # Aggregate data for each behavior
        for bhvr, vals in bhvr_dict.items():
            if n_bins is None:
                n_bins = len(vals)
            # Initialize bin lists if needed
            if not grouped[target_group][bhvr]:
                grouped[target_group][bhvr] = [[] for _ in range(n_bins)]
            # Append values to corresponding bins
            for i, v in enumerate(vals):
                grouped[target_group][bhvr][i].append(v)
    # Return grouped data (keep lists for flexibility)
    grouped_mean = {g: {} for g in group_map}
    for g, bhvr_dict in grouped.items():
        for bhvr, bins in bhvr_dict.items():
            if not bins:
                continue
            # Store raw lists of values per bin
            grouped_mean[g][bhvr] = bins
    return grouped_mean


def merge_time_stage_dicts(pre_data, wr_data, post_data, time_suffixes=None):
    """Merge data from different time stages by adding suffixes to session IDs."""
    # Set default suffixes for time stages
    if time_suffixes is None:
        time_suffixes = ('_Pre', '_WR', '_Post')
    merged_dict = {}
    # Add suffixes and merge all stages
    for data_dict, suffix in zip([pre_data, wr_data, post_data], time_suffixes):
        for sess_id, sess_data in data_dict.items():
            new_sess_id = sess_id + suffix
            merged_dict[new_sess_id] = sess_data
    return merged_dict


def classify_and_pair(self_intervals, pair_intervals, partner_interval):
    """Classify and pair social behavior bouts based on temporal proximity and sequence."""
    pairs = []
    used_self = set()
    used_pair = set()
    # Sort intervals by start time
    self_intervals = sorted(self_intervals, key=lambda x: x[0])
    pair_intervals = sorted(pair_intervals, key=lambda x: x[0])
    # Try to pair each self interval with a partner interval
    for i, s_int in enumerate(self_intervals):
        if i in used_self:
            continue
        s_start, s_end = s_int
        best_match = None
        best_label = None
        min_time_diff = float('inf')
        # Find closest partner interval within threshold
        for j, p_int in enumerate(pair_intervals):
            if j in used_pair:
                continue
            p_start, p_end = p_int
            time_diff = abs(s_start - p_start)
            # Check if within partner interval threshold
            if time_diff <= partner_interval:
                # Classify interaction type based on timing
                if abs(s_start - p_start) < 1:
                    label = 'sp'  # Simultaneous
                elif s_start < p_start:
                    if p_start <= s_end:
                        label = 's2'  # Self starts, overlaps with partner
                    else:
                        label = 's1'  # Self starts before partner
                else:
                    if s_start <= p_end:
                        label = 'p2'  # Partner starts, overlaps with self
                    else:
                        label = 'p1'  # Partner starts before self
                # Keep closest match
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = j
                    best_label = label
        # Record pairing or mark as unpaired
        if best_match is not None:
            pairs.append((s_int, pair_intervals[best_match], best_label))
            used_self.add(i)
            used_pair.add(best_match)
        else:
            # Self-initiated without partner response
            pairs.append((s_int, None, 's3'))
            used_self.add(i)
    # Add unpaired partner intervals
    for j, p_int in enumerate(pair_intervals):
        if j not in used_pair:
            # Partner-initiated without self response
            pairs.append((None, p_int, 'p3'))
    return pairs


def find_paired_sessions(Soc_data_merged):
    """Find paired recording sessions by swapping animal identifiers in session IDs."""
    session_pairs = []
    processed = set()
    all_sessions = sorted(Soc_data_merged.keys())
    for session in all_sessions:
        if session in processed:
            continue
        # Define swap map for paired animals
        swap_map = {
            'Dom': 'Sub',
            'Sub': 'Dom',
            'WT': 'SF1',
            'SF1': 'WT',
        }
        # Generate partner session ID by swapping identifiers
        parts = session.split('_')
        parts = [swap_map.get(p, p) for p in parts]
        partner_session = '_'.join(parts)
        # Add pair if both sessions exist
        if partner_session and partner_session in Soc_data_merged:
            session_pairs.append((session, partner_session))
            processed.add(session)
            processed.add(partner_session)
    return session_pairs


def calculate_social_interaction(
    Soc_data_merged,
    partner_interval=30,
    fps=20
):
    """Calculate proactive and reactive social interactions between paired animals."""
    # Find all paired recording sessions
    session_pairs = find_paired_sessions(Soc_data_merged)
    print(f"Found {len(session_pairs)} paired sessions")
    # Initialize result dictionaries
    interaction_data = {}
    proactive_data = {}
    reactive_data = {}
    for animal1_session, animal2_session in session_pairs:
        # Initialize storage for both animals
        interaction_data[animal1_session] = {}
        proactive_data[animal1_session] = {}
        reactive_data[animal1_session] = {}
        interaction_data[animal2_session] = {}
        proactive_data[animal2_session] = {}
        reactive_data[animal2_session] = {}
        # Extract social behavior intervals
        animal1_intervals = Soc_data_merged.get(animal1_session, {}).get('Soc', [])
        animal2_intervals = Soc_data_merged.get(animal2_session, {}).get('Soc', [])
        # Clean and validate intervals
        animal1_intervals_clean = [
            (float(interval[0]), float(interval[1])) 
            for interval in animal1_intervals 
            if (isinstance(interval, (list, tuple, np.ndarray)) 
                and len(interval) == 2 
                and not np.isnan(interval).any())
        ]
        animal2_intervals_clean = [
            (float(interval[0]), float(interval[1])) 
            for interval in animal2_intervals 
            if (isinstance(interval, (list, tuple, np.ndarray)) 
                and len(interval) == 2 
                and not np.isnan(interval).any())
        ]
        # Skip if both animals have no social behaviors
        if not animal1_intervals_clean and not animal2_intervals_clean:
            continue
        # Classify and pair social interactions
        pairs = classify_and_pair(animal1_intervals_clean, animal2_intervals_clean, partner_interval)
        # Initialize result lists
        interaction_data[animal1_session]['interaction'] = []
        proactive_data[animal1_session]['proactive'] = []
        reactive_data[animal1_session]['reactive'] = []
        interaction_data[animal2_session]['interaction'] = []
        proactive_data[animal2_session]['proactive'] = []
        reactive_data[animal2_session]['reactive'] = []
        # Process each paired interaction
        for a1_int, a2_int, label in pairs:
            # Animal 1's perspective
            interaction_data[animal1_session]['interaction'].append(label)
            # Classify as proactive (s labels), reactive (p labels), or simultaneous (sp)
            proactive_data[animal1_session]['proactive'].append(100 if label in ['s1', 's2', 's3'] else (np.nan if label == 'sp' else 0))
            reactive_data[animal1_session]['reactive'].append(100 if label in ['p1', 'p2'] else (np.nan if label == 'sp' else 0))
            # Animal 2's perspective (swap s and p labels)
            label_a2 = label.replace('s', 'x').replace('p', 's').replace('x', 'p')
            interaction_data[animal2_session]['interaction'].append(label_a2)
            proactive_data[animal2_session]['proactive'].append(100 if label_a2 in ['s1', 's2', 's3'] else (np.nan if label_a2 == 'sp' else 0))
            reactive_data[animal2_session]['reactive'].append(100 if label_a2 in ['p1', 'p2'] else (np.nan if label_a2 == 'sp' else 0))
    return interaction_data, proactive_data, reactive_data, session_pairs
