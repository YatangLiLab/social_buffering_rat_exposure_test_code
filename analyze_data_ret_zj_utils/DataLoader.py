import numpy as np
import pandas as pd
import os
import yaml
import csv
import pickle


def read_hdf5_file(file_path, Cond_id):
    """Reads and processes HDF5 file for DeepLabCut data."""
    keypoint_names = [
        "Snout", "EarL", "EarR", "SpineF", "SpineM", "TailB",
        "TailM", "TailE", "LimbFL", "LimbFR", "LimbHL", "LimbHR"
    ]
    df = pd.read_hdf(file_path)
    scorer = df.columns.get_level_values('scorer')[0]
    df_data = df[scorer]
    coordinates_dict = {}
    confidences_dict = {}
    if Cond_id == 'Pair':
        # Get available individuals from the data
        available_individuals = df_data.columns.get_level_values(0).unique().tolist()
        if 'M1' in available_individuals and 'M2' in available_individuals:
            individual_pairs = [('M1', 'SF1'), ('M2', 'WT')]
        elif 'SF1' in available_individuals and 'WT' in available_individuals:
            individual_pairs = [('SF1', 'SF1'), ('WT', 'WT')]
        else:
            raise ValueError(f"Unknown individual format in file. Available: {available_individuals}")
        for file_individual_id, output_mouse_id in individual_pairs:
            n_frames = len(df_data)
            n_keypoints = len(keypoint_names)
            coords = np.zeros((n_frames, n_keypoints, 2))
            likelihoods = np.zeros((n_frames, n_keypoints))
            # Extract coordinates and likelihoods for each keypoint
            for kpt_idx, keypoint in enumerate(keypoint_names):
                coords[:, kpt_idx, 0] = df_data[file_individual_id][keypoint]['x'].values
                coords[:, kpt_idx, 1] = df_data[file_individual_id][keypoint]['y'].values
                likelihoods[:, kpt_idx] = df_data[file_individual_id][keypoint]['likelihood'].values
            # Identify missing data based on low likelihood
            missing_mask = likelihoods < 0.1
            coords_interp = coords.copy()
            # Interpolate missing coordinates
            for kpt_idx in range(n_keypoints):
                for dim in range(2):
                    col = coords_interp[:, kpt_idx, dim]
                    col_with_nan = col.copy()
                    col_with_nan[missing_mask[:, kpt_idx]] = np.nan
                    if not np.isnan(col_with_nan).all():
                        col_interp = pd.Series(col_with_nan).interpolate(
                            method='linear', limit_direction='both'
                        ).to_numpy()
                        coords_interp[:, kpt_idx, dim] = col_interp
            coordinates_dict[output_mouse_id] = coords_interp
            confidences_dict[output_mouse_id] = (~missing_mask).astype(float)
    elif Cond_id == 'Single':
        n_frames = len(df_data)
        n_keypoints = len(keypoint_names)
        coords = np.zeros((n_frames, n_keypoints, 2))
        likelihoods = np.zeros((n_frames, n_keypoints))
        # Extract coordinates and likelihoods for each keypoint
        for kpt_idx, keypoint in enumerate(keypoint_names):
            coords[:, kpt_idx, 0] = df_data[keypoint]['x'].values
            coords[:, kpt_idx, 1] = df_data[keypoint]['y'].values
            likelihoods[:, kpt_idx] = df_data[keypoint]['likelihood'].values
        # Identify missing data based on low likelihood
        missing_mask = likelihoods < 0.1
        coords_interp = coords.copy()
        # Interpolate missing coordinates
        for kpt_idx in range(n_keypoints):
            for dim in range(2):
                col = coords_interp[:, kpt_idx, dim]
                col_with_nan = col.copy()
                col_with_nan[missing_mask[:, kpt_idx]] = np.nan
                if not np.isnan(col_with_nan).all():
                    col_interp = pd.Series(col_with_nan).interpolate(
                        method='linear', limit_direction='both'
                    ).to_numpy()
                    coords_interp[:, kpt_idx, dim] = col_interp
        coordinates_dict['single'] = coords_interp
        confidences_dict['single'] = (~missing_mask).astype(float)
    return coordinates_dict, confidences_dict


def load_dlc_data(dlc_h5_folder, file_map_yaml_path):
    """Loads DeepLabCut data from HDF5 files using a file map."""
    with open(file_map_yaml_path, 'r') as f:
        file_map = yaml.safe_load(f)
    coordinates = {}
    confidences = {}
    for file_name in os.listdir(dlc_h5_folder):
        if file_name.endswith('.h5'):
            try:
                file_path = os.path.join(dlc_h5_folder, file_name)
                Cond_id = file_map[file_name]['Cond_id']
                coords_dict, confs_dict = read_hdf5_file(file_path, Cond_id)
                Sess_id_base = file_map[file_name]['Batch_id'] + '_' + \
                            file_map[file_name]['Cage_id'] + '_' + \
                            file_map[file_name]['Cond_id'] + '_'
                if Cond_id == 'Pair':
                    Mouse_ids = file_map[file_name]['Mouse_id']
                    Rank_ids = file_map[file_name]['Rank_id']
                    for Mouse_id, Rank_id in zip(Mouse_ids, Rank_ids):
                        Sess_id = Sess_id_base + Rank_id + '_' + Mouse_id + '_' + file_map[file_name]['Phase_id']
                        coordinates[Sess_id] = coords_dict[Mouse_id]
                        confidences[Sess_id] = confs_dict[Mouse_id]
                elif Cond_id == 'Single':
                    Mouse_id = file_map[file_name]['Mouse_id'][0]
                    Rank_id = file_map[file_name]['Rank_id'][0]
                    Sess_id = Sess_id_base + Rank_id + '_' + Mouse_id + '_' + file_map[file_name]['Phase_id']
                    coordinates[Sess_id] = coords_dict['single']
                    confidences[Sess_id] = confs_dict['single']
            except KeyError as e:
                print(f"KeyError {e} in file {file_name}: check file_map.yaml")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    print("All DeepLabCut data loaded.")
    return coordinates, confidences


def load_ev_data(ev_xlsx_folder):
    """Loads EthoVision raw data from Excel files."""
    kp_list = ['X center', 'Y center', 'X nose', 'Y nose', 'X tail', 'Y tail']
    coordinates = {}
    confidences = {}
    for file_name in os.listdir(ev_xlsx_folder):
        if file_name.startswith('Raw data') and file_name.endswith('.xlsx'):
            xlsx_file_path = os.path.join(ev_xlsx_folder, file_name)
            key = file_name[:-5]
            Batch_id = key.split('-')[1].split('_')[0]
            Cage_id = 'C'+key.split('_')[1]
            Cond = key.split('_')[2]
            Mouse_id = key.split('_')[3]
            Phase_id = key.split('_')[4]
            if Cond == "Pair":
                df1 = pd.read_excel(xlsx_file_path, sheet_name=0, skiprows=34)
                df2 = pd.read_excel(xlsx_file_path, sheet_name=1, skiprows=34)
                df1 = df1.drop(df1.index[0]).reset_index(drop=True)
                df2 = df2.drop(df2.index[0]).reset_index(drop=True)
                try:
                    data1 = df1[kp_list].replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
                    data2 = df2[kp_list].replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
                    arr1 = data1.to_numpy(dtype=float)
                    arr2 = data2.to_numpy(dtype=float)
                    arr1 = arr1.reshape(-1, 3, 2)
                    arr2 = arr2.reshape(-1, 3, 2)
                    missing_mask1 = np.isnan(arr1).any(axis=2)
                    missing_mask2 = np.isnan(arr2).any(axis=2)
                    arr1_interp = arr1.copy()
                    arr2_interp = arr2.copy()
                    # Interpolate missing data for first mouse
                    for kpt in range(arr1_interp.shape[1]):
                        for d in range(2):
                            col = arr1_interp[:, kpt, d]
                            if not np.isnan(col).all():
                                arr1_interp[:, kpt, d] = pd.Series(col).interpolate(
                                    method='linear', limit_direction='both'
                                ).to_numpy()
                    # Interpolate missing data for second mouse
                    for kpt in range(arr2_interp.shape[1]):
                        for d in range(2):
                            col = arr2_interp[:, kpt, d]
                            if not np.isnan(col).all():
                                arr2_interp[:, kpt, d] = pd.Series(col).interpolate(
                                    method='linear', limit_direction='both'
                                ).to_numpy()
                    confs1 = (~missing_mask1).astype(float)
                    confs2 = (~missing_mask2).astype(float)
                    coordinates[Batch_id+'_'+Cage_id+'_'+Cond+'_'+'Dom'+'_'+'SF1'+'_'+Phase_id] = arr1_interp
                    coordinates[Batch_id+'_'+Cage_id+'_'+Cond+'_'+'Sub'+'_'+'WT'+'_'+Phase_id] = arr2_interp
                    confidences[Batch_id+'_'+Cage_id+'_'+Cond+'_'+'Dom'+'_'+'SF1'+'_'+Phase_id] = confs1
                    confidences[Batch_id+'_'+Cage_id+'_'+Cond+'_'+'Sub'+'_'+'WT'+'_'+Phase_id] = confs2
                except KeyError as e:
                    print(f"Error processing {file_name}: {e}")
            elif Cond == "Single":
                df = pd.read_excel(xlsx_file_path, skiprows=34)
                df = df.drop(df.index[0]).reset_index(drop=True)
                try:
                    data = df[kp_list].replace('-', np.nan).apply(pd.to_numeric, errors='coerce')
                    arr = data.to_numpy(dtype=float)
                    arr = arr.reshape(-1, 3, 2)
                    missing_mask = np.isnan(arr).any(axis=2)
                    arr_interp = arr.copy()
                    n_frames, n_kpt, _ = arr_interp.shape
                    # Interpolate missing data
                    for k in range(n_kpt):
                        for d in range(2):
                            col = arr_interp[:, k, d]
                            if np.isnan(col).all():
                                continue
                            s = pd.Series(col)
                            arr_interp[:, k, d] = s.interpolate(method='linear', limit_direction='both').to_numpy()
                    confs = (~missing_mask).astype(float)
                    Rank_id = 'Dom' if Mouse_id == 'SF1' else 'Sub'
                    coordinates[Batch_id+'_'+Cage_id+'_'+Cond+'_'+Rank_id+'_'+Mouse_id+'_'+Phase_id] = arr_interp
                    confidences[Batch_id+'_'+Cage_id+'_'+Cond+'_'+Rank_id+'_'+Mouse_id+'_'+Phase_id] = confs
                except KeyError as e:
                    print(f"Error processing {file_name}: {e}")
    print("All EthoVision raw data loaded")
    return coordinates, confidences


def read_annotation_file(file_path, file_map):
    """Reads and parses annotation files for behavior data."""
    behavior_frames = {}
    behaviors = [
        'RatIn', 'RatOut', 'approach', 'approachEmRat', 'approachP',
        'followP', 'freezing', 'groomP', 'grooming', 'huddling',
        'investigateEmRat', 'investigation', 'jumping', 'rearing',
        'sniffP', 'stretch', 'tailrattling', 'watchP', 'withdrawal',
        'withdrawalEmRat'
    ]
    for behavior in behaviors:
        behavior_frames[behavior] = []
    if file_map[file_path.split('/')[-1]]['Cond_id'] == 'Single':
        with open(file_path, 'r') as file:
            lines = file.readlines()
        current_behavior = None
        for line in lines:
            if line.startswith('>'):
                current_behavior = line.strip()[1:]
                behavior_frames[current_behavior] = []
            elif line.strip() != '':
                try:
                    start_frame, stop_frame, _ = map(int, line.split())
                    behavior_frames[current_behavior].append((start_frame, stop_frame))
                except ValueError:
                    pass
    elif file_map[file_path.split('/')[-1]]['Cond_id'] == 'Pair':
        with open(file_path, 'r') as file:
            lines = file.readlines()
        ch1_dict = {}
        ch2_dict = {}
        current_dict = {}
        current_behavior = None
        for behavior in behaviors:
            ch1_dict[behavior] = []
            ch2_dict[behavior] = []
        for line in lines:
            if line.startswith('Ch1----------'):
                current_dict = ch1_dict
            elif line.startswith('Ch2----------'):
                current_dict = ch2_dict
            elif line.strip() != '':
                if line.startswith('>'):
                    current_behavior = line.strip()[1:]
                    current_dict[current_behavior] = []
                else:
                    try:
                        start_frame, stop_frame, _ = map(int, line.split())
                        current_dict[current_behavior].append((start_frame, stop_frame))
                    except ValueError:
                        pass
        behavior_frames = {'Ch1': ch1_dict, 'Ch2': ch2_dict}
    return behavior_frames


def load_annot_data(bento_annot_folder, file_map_yaml_path):
    """Loads annotation data from Bento annotation files using a file map."""
    with open(file_map_yaml_path, 'r') as f:
        file_map = yaml.safe_load(f)
    bhvr_tuples_dict = {}
    bhvr_tuples_dict_p = {}
    bhvr_tuples_dict_s = {}
    for file_name in os.listdir(bento_annot_folder):
        if file_name.endswith('.annot'):
            file_path = os.path.join(bento_annot_folder, file_name)
            behavior_frames = read_annotation_file(file_path, file_map)
            Batch_id = file_map[file_name]['Batch_id']
            Cage_id = file_map[file_name]['Cage_id']
            Cond_id = file_map[file_name]['Cond_id']
            Phase_id = file_map[file_name]['Phase_id']
            if Cond_id == 'Single':
                Mouse_id = file_map[file_name]['Mouse_id'][0]
                Rank_id = file_map[file_name]['Rank_id'][0]
                Sess_id_unified = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_id}_{Mouse_id}_{Phase_id}"
                bhvr_tuples_dict[Sess_id_unified] = behavior_frames
                Sess_id_original = f"{Batch_id}_{Cage_id}_{Cond_id}_{Phase_id}"
                bhvr_tuples_dict_s[Sess_id_original] = behavior_frames
            elif Cond_id == 'Pair':
                Mouse_ids = file_map[file_name]['Mouse_id']
                Rank_ids = file_map[file_name]['Rank_id']
                Ch1_frames = behavior_frames.get('Ch1', {})
                Ch2_frames = behavior_frames.get('Ch2', {})
                Sess_id_ch1 = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_ids[0]}_{Mouse_ids[0]}_{Phase_id}"
                Sess_id_ch2 = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_ids[1]}_{Mouse_ids[1]}_{Phase_id}"
                bhvr_tuples_dict[Sess_id_ch1] = Ch1_frames
                bhvr_tuples_dict[Sess_id_ch2] = Ch2_frames
                Sess_id_original = f"{Batch_id}_{Cage_id}_{Cond_id}_{Phase_id}"
                behavior_frames_flat = {'Ch1_' + behavior_type: frames for behavior_type, frames in Ch1_frames.items()}
                behavior_frames_flat.update({'Ch2_' + behavior_type: frames for behavior_type, frames in Ch2_frames.items()})
                bhvr_tuples_dict_p[Sess_id_original] = behavior_frames_flat
    return bhvr_tuples_dict, bhvr_tuples_dict_p, bhvr_tuples_dict_s


def convert_bhvr_tuples2series(bhvr_tuples_dict, total_frames=None, end_frame=None):
    """Converts behavior tuples to pandas Series."""
    bhvr_series_dict = {}
    for Sess_id, behavior_frames in bhvr_tuples_dict.items():
        bhvr_series_dict[Sess_id] = {}
        if end_frame is not None:
            if isinstance(end_frame, dict):
                Sess_end_frame = end_frame.get(Sess_id, None)
            else:
                Sess_end_frame = end_frame
        else:
            Sess_end_frame = None
        if total_frames is None:
            all_frames = []
            for behavior, tuples in behavior_frames.items():
                if tuples and not (len(tuples) == 1 and isinstance(tuples[0], float)):
                    for start, stop in tuples:
                        all_frames.extend([start, stop])
            max_frame = max(all_frames) if all_frames else 0
        else:
            max_frame = total_frames - 1
        if Sess_end_frame is not None:
            max_frame = Sess_end_frame
        for behavior_name, tuples in behavior_frames.items():
            series_data = np.zeros(max_frame + 1, dtype=int)
            if tuples and not (len(tuples) == 1 and isinstance(tuples[0], float)):
                for start, stop in tuples:
                    series_data[start:stop+1] = 1
            bhvr_series_dict[Sess_id][behavior_name] = pd.Series(series_data, name=behavior_name)
    return bhvr_series_dict


def convert_bhvr_series2tuples(bhvr_series_dict):
    """Converts behavior series back to tuples."""
    bhvr_tuples_dict = {}
    for Sess_id, behavior_series in bhvr_series_dict.items():
        bhvr_tuples_dict[Sess_id] = {}
        for behavior_name, series in behavior_series.items():
            results = []
            start_idx = None
            if isinstance(series, pd.Series):
                values = series.values
            else:
                values = np.array(series)
            for i in range(len(values)):
                if values[i] == 1 and start_idx is None:
                    start_idx = i
                elif values[i] == 0 and start_idx is not None:
                    results.append((start_idx, i - 1))
                    start_idx = None
            if start_idx is not None:
                results.append((start_idx, len(values) - 1))
            bhvr_tuples_dict[Sess_id][behavior_name] = results
    return bhvr_tuples_dict


def save_bhvr_dicts(bhvr_dict, save_folder, save_format='csv', file_prefix=''):
    """Saves behavior dictionaries to specified format files."""
    os.makedirs(save_folder, exist_ok=True)
    saved_files = []
    first_Sess = list(bhvr_dict.keys())[0]
    first_behavior = list(bhvr_dict[first_Sess].keys())[0]
    first_value = bhvr_dict[first_Sess][first_behavior]
    is_series_format = isinstance(first_value, pd.Series)
    if save_format == 'csv':
        if not is_series_format:
            data_to_save = convert_bhvr_tuples2series(bhvr_dict)
        else:
            data_to_save = bhvr_dict
        for Sess_id, behavior_series in data_to_save.items():
            df = pd.DataFrame(behavior_series)
            df.insert(0, 'background', 0)
            df.insert(0, '', df.index)
            filename = f'{file_prefix}{Sess_id}_labels.csv' if file_prefix else f'{Sess_id}_labels.csv'
            filepath = os.path.join(save_folder, filename)
            df.to_csv(filepath, index=False)
            saved_files.append(filepath)
    elif save_format == 'pkl':
        for Sess_id, behavior_data in bhvr_dict.items():
            filename = f'{file_prefix}{Sess_id}.pkl' if file_prefix else f'{Sess_id}.pkl'
            filepath = os.path.join(save_folder, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(behavior_data, f)
            saved_files.append(filepath)
    elif save_format == 'annot':
        if is_series_format:
            data_to_save = convert_bhvr_series2tuples(bhvr_dict)
        else:
            data_to_save = bhvr_dict
        for Sess_id, behavior_frames in data_to_save.items():
            annot_content = _create_annot_content(Sess_id, behavior_frames)
            filename = f'{file_prefix}{Sess_id}.annot' if file_prefix else f'{Sess_id}.annot'
            filepath = os.path.join(save_folder, filename)
            with open(filepath, 'w') as f:
                f.write(annot_content)
            saved_files.append(filepath)
    else:
        raise ValueError(f"Unsupported save format: {save_format}. Use 'csv', 'pkl', or 'annot'.")
    print(f"Saved {len(saved_files)} files to {save_folder}")
    return saved_files


def _create_annot_content(sess_id, behavior_frames):
    """Creates content for Bento annotation file."""
    all_frames = []
    for behavior, tuples in behavior_frames.items():
        if tuples:
            for start, stop in tuples:
                all_frames.extend([start, stop])
    max_frame = max(all_frames) if all_frames else 0
    fps = 20.0
    stop_time = max_frame / fps
    behavior_list = list(behavior_frames.keys())
    annotations_str = '\n'.join(behavior_list)
    annot_content = f"""Bento annotation file
Movie file(s):  {sess_id}.avi

Stimulus name: 
Annotation start time: 0.000000e+00
Annotation stop time: {stop_time:.6e}
Annotation framerate: {fps:.6f}

List of channels:
Channel_1

List of annotations:
{annotations_str}

Channel_1----------
"""
    for behavior, tuples in behavior_frames.items():
        annot_content += f">{behavior}\nStart\tStop\tDuration\n"
        if tuples:
            for start, stop in tuples:
                duration = stop - start
                annot_content += f"{start}\t{stop}\t{duration}\n"
        annot_content += "\n"
    return annot_content


def load_de_data(bhvr_csv_folder, file_map_yaml_path):
    """Loads behavior data from CSV files using a file map."""
    with open(file_map_yaml_path, 'r') as f:
        file_map = yaml.safe_load(f)
    bhvr_series_dict = {}
    for file_name in os.listdir(bhvr_csv_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(bhvr_csv_folder, file_name)
            data = pd.read_csv(file_path)
            data = data.drop(data.columns[[0, 1]], axis=1)
            Sess_id = file_name[:-11]
            parts = Sess_id.split('_')
            csv_file_name = Sess_id + '_labels.csv'
            if Sess_id.split('_')[-1] == '1channel' or 'Single' in Sess_id:
                Batch_id = parts[0]
                Cage_id = parts[1]
                Cond_id = parts[2]
                if len(parts) >= 5:
                    Mouse_id = parts[3]
                    Phase_id = '_'.join(parts[4:])
                else:
                    Mouse_id = 'SF1'
                    Phase_id = parts[3] if len(parts) > 3 else 'Unknown'
                if csv_file_name in file_map:
                    Rank_id = file_map[csv_file_name]['Rank_id'][0]
                    Mouse_id = file_map[csv_file_name]['Mouse_id'][0]
                Sess_id_unified = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_id}_{Mouse_id}_{Phase_id}"
                bhvr_series_dict[Sess_id_unified] = {}
                for behavior_name in data.columns:
                    bhvr_series_dict[Sess_id_unified][behavior_name] = data[behavior_name]
            elif Sess_id.split('_')[-1] == '2channel' or 'Pair' in Sess_id:
                Batch_id = parts[0]
                Cage_id = parts[1]
                Cond_id = parts[2]
                Phase_id = '_'.join(parts[3:])
                if csv_file_name in file_map:
                    Mouse_ids = file_map[csv_file_name]['Mouse_id']
                    Rank_ids = file_map[csv_file_name]['Rank_id']
                    Sess_id_ch1 = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_ids[0]}_{Mouse_ids[0]}_{Phase_id}"
                    Sess_id_ch2 = f"{Batch_id}_{Cage_id}_{Cond_id}_{Rank_ids[1]}_{Mouse_ids[1]}_{Phase_id}"
                bhvr_series_dict[Sess_id_ch1] = {}
                bhvr_series_dict[Sess_id_ch2] = {}
                for behavior_name in data.columns:
                    if behavior_name.startswith('Ch1_'):
                        clean_behavior_name = behavior_name[4:]
                        bhvr_series_dict[Sess_id_ch1][clean_behavior_name] = data[behavior_name]
                    elif behavior_name.startswith('Ch2_'):
                        clean_behavior_name = behavior_name[4:]
                        bhvr_series_dict[Sess_id_ch2][clean_behavior_name] = data[behavior_name]
    return bhvr_series_dict


def load_toi_dict(exp_time_dict):
    """Creates time-of-interest dictionaries from experiment time data."""
    pre_toi_dict = {}
    wr_toi_dict = {}
    post_toi_dict = {}
    for sess_id, times in exp_time_dict.items():
        rat_in_start, rat_in_end, rat_out_start, rat_out_end = times
        pre_toi_dict[sess_id] = (rat_in_start-2400, rat_in_start)
        wr_toi_dict[sess_id] = (rat_in_end, rat_in_end+6000)
        post_toi_dict[sess_id] = (rat_out_end, rat_out_end+6000)
    return pre_toi_dict, wr_toi_dict, post_toi_dict
