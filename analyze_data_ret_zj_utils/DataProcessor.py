import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def _filter_range_and_linear_interpolate(df, x_range=(0, 50), y_range=(0, 50), z_range=(0, 25)):
    """
    Filter values outside specified ranges and apply linear interpolation.
    """
    df_filtered = df.copy()
    for axis, (min_val, max_val) in zip(['x', 'y', 'z'], [x_range, y_range, z_range]):
        mask = (df_filtered[axis] < min_val) | (df_filtered[axis] > max_val)
        df_filtered.loc[mask, axis] = np.nan
    df_filtered[['x','y','z']] = df_filtered[['x','y','z']].interpolate(method='linear', limit_direction='both')
    return df_filtered


def _filter_axis_speed_and_repeat(df, axis_threshold=5):
    """
    Filter based on axis speed and repeat the previous value.
    """
    dx = np.abs(np.diff(df['x'], prepend=df['x'].iloc[0]))
    dy = np.abs(np.diff(df['y'], prepend=df['y'].iloc[0]))
    dz = np.abs(np.diff(df['z'], prepend=df['z'].iloc[0]))
    single_jump_mask = (dx > axis_threshold) | (dy > axis_threshold) | (dz > axis_threshold)
    df_fixed = df.copy()
    for i in np.where(single_jump_mask)[0]:
        if i > 0:
            df_fixed.iloc[i] = df_fixed.iloc[i - 1]
    return df_fixed, single_jump_mask


def _filter_drift_segment_and_linear_interpolate(df, axis_threshold=5, context=5, max_drift_length=20):
    """
    Filter drift segments and apply linear interpolation.
    """
    df = df.copy()
    coords = df[['x', 'y', 'z']].values
    deltas = np.diff(coords, axis=0)
    diffs = np.linalg.norm(deltas, axis=1)
    diffs = np.insert(diffs, 0, 0)
    dx = np.abs(np.diff(df['x'], prepend=df['x'].iloc[0]))
    dy = np.abs(np.diff(df['y'], prepend=df['y'].iloc[0]))
    dz = np.abs(np.diff(df['z'], prepend=df['z'].iloc[0]))

    axis_jump_mask = (dx > axis_threshold) | (dy > axis_threshold) | (dz > axis_threshold)
    jump_mask = (diffs > axis_threshold) | axis_jump_mask

    drift_mask = np.zeros(len(df), dtype=bool)
    i = 0
    while i < len(df) - context * 2:
        if jump_mask[i]:  # onset
            for j in range(i + 2, len(df) - 1):
                if j - i > max_drift_length:
                    break
                if jump_mask[j]:  # offset
                    pre_context = coords[max(0, i - context):i]
                    post_context = coords[j + 1:j + 1 + context]
                    if len(pre_context) + len(post_context) < 2:
                        continue
                    normal_center = np.concatenate([pre_context, post_context], axis=0).mean(axis=0)
                    drift_segment = coords[i + 1:j]
                    if len(drift_segment) == 0:
                        continue
                    drift_dist = np.linalg.norm(drift_segment - normal_center, axis=1)
                    if np.mean(drift_dist) > axis_threshold / 2:
                        drift_mask[i + 1:j] = True
                        i = j 
                        break
        i += 1
    df_fixed = df.copy()
    for axis in ['x', 'y', 'z']:
        s = df_fixed[axis]
        s[drift_mask] = np.nan
        df_fixed[axis] = s.interpolate(method='linear', limit_direction='both')

    return df_fixed, drift_mask


def _filter_axis_speed_and_linear_interpolate(df, axis_threshold=5):
    """
    Filter based on axis speed and apply linear interpolation.

    """
    dx = np.abs(np.diff(df['x'], prepend=df['x'].iloc[0]))
    dy = np.abs(np.diff(df['y'], prepend=df['y'].iloc[0]))
    dz = np.abs(np.diff(df['z'], prepend=df['z'].iloc[0]))
    single_jump_mask = (dx > axis_threshold) | (dy > axis_threshold) | (dz > axis_threshold)
    df_fixed = df.copy()
    jump_indices = np.where(single_jump_mask)[0]
    for i in jump_indices:
        if i == 0 or i == len(df) - 1:
            continue
        prev_idx = i - 1
        next_idx = i + 1
        while next_idx < len(df) and single_jump_mask[next_idx]:
            next_idx += 1
            if next_idx >= len(df):
                next_idx = i
                break
        if next_idx == i:
            df_fixed.iloc[i] = df_fixed.iloc[prev_idx]
        else:
            prev_val = df_fixed.iloc[prev_idx][['x', 'y', 'z']]
            next_val = df_fixed.iloc[next_idx][['x', 'y', 'z']]
            alpha = (i - prev_idx) / (next_idx - prev_idx)
            df_fixed.iloc[i, :3] = prev_val * (1 - alpha) + next_val * alpha
    return df_fixed, single_jump_mask


def filter_dlc_coordinates_range_interp(
    coord_dict: Dict[str, np.ndarray],
    x_range: Tuple[float, float] = (0, 50),
    y_range: Tuple[float, float] = (0, 50),
    keypoint_indices: Optional[list] = None
) -> Dict[str, np.ndarray]:
    """
    Filter coordinates outside specified ranges and apply linear interpolation.

    """
    filtered_dict = {}
    
    for sess_id, coords in coord_dict.items():
        n_frames, n_keypoints, _ = coords.shape
        filtered_coords = coords.copy()
        
        # Determine which keypoints to process
        kpt_indices = keypoint_indices if keypoint_indices is not None else range(n_keypoints)
        
        for kpt_idx in kpt_indices:
            # Create DataFrame for this keypoint
            df = pd.DataFrame({
                'x': coords[:, kpt_idx, 0],
                'y': coords[:, kpt_idx, 1],
                'z': np.zeros(n_frames)  # dummy z column
            })
            
            # Apply filtering
            df_filtered = _filter_range_and_linear_interpolate(
                df, x_range=x_range, y_range=y_range, z_range=(-np.inf, np.inf)
            )
            
            # Update coordinates
            filtered_coords[:, kpt_idx, 0] = df_filtered['x'].values
            filtered_coords[:, kpt_idx, 1] = df_filtered['y'].values
        
        filtered_dict[sess_id] = filtered_coords
    
    return filtered_dict


def filter_dlc_coordinates_speed_repeat(
    coord_dict: Dict[str, np.ndarray],
    axis_threshold: float = 5.0,
    keypoint_indices: Optional[list] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Filter based on axis speed and repeat the previous value.

    """
    filtered_dict = {}
    mask_dict = {}
    
    for sess_id, coords in coord_dict.items():
        n_frames, n_keypoints, _ = coords.shape
        filtered_coords = coords.copy()
        session_masks = np.zeros((n_frames, n_keypoints), dtype=bool)
        
        kpt_indices = keypoint_indices if keypoint_indices is not None else range(n_keypoints)
        
        for kpt_idx in kpt_indices:
            df = pd.DataFrame({
                'x': coords[:, kpt_idx, 0],
                'y': coords[:, kpt_idx, 1],
                'z': np.zeros(n_frames)
            })
            
            df_filtered, jump_mask = _filter_axis_speed_and_repeat(df, axis_threshold)
            
            filtered_coords[:, kpt_idx, 0] = df_filtered['x'].values
            filtered_coords[:, kpt_idx, 1] = df_filtered['y'].values
            session_masks[:, kpt_idx] = jump_mask
        
        filtered_dict[sess_id] = filtered_coords
        mask_dict[sess_id] = session_masks
    
    return filtered_dict, mask_dict


def filter_dlc_coordinates_drift_interp(
    coord_dict: Dict[str, np.ndarray],
    axis_threshold: float = 5.0,
    context: int = 5,
    max_drift_length: int = 20,
    keypoint_indices: Optional[list] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Filter drift segments and apply linear interpolation.

    """
    filtered_dict = {}
    mask_dict = {}
    
    for sess_id, coords in coord_dict.items():
        n_frames, n_keypoints, _ = coords.shape
        filtered_coords = coords.copy()
        session_masks = np.zeros((n_frames, n_keypoints), dtype=bool)
        
        kpt_indices = keypoint_indices if keypoint_indices is not None else range(n_keypoints)
        
        for kpt_idx in kpt_indices:
            df = pd.DataFrame({
                'x': coords[:, kpt_idx, 0],
                'y': coords[:, kpt_idx, 1],
                'z': np.zeros(n_frames)
            })
            
            df_filtered, drift_mask = _filter_drift_segment_and_linear_interpolate(
                df, axis_threshold, context, max_drift_length
            )
            
            filtered_coords[:, kpt_idx, 0] = df_filtered['x'].values
            filtered_coords[:, kpt_idx, 1] = df_filtered['y'].values
            session_masks[:, kpt_idx] = drift_mask
        
        filtered_dict[sess_id] = filtered_coords
        mask_dict[sess_id] = session_masks
    
    return filtered_dict, mask_dict


def filter_dlc_coordinates_speed_interp(
    coord_dict: Dict[str, np.ndarray],
    axis_threshold: float = 5.0,
    keypoint_indices: Optional[list] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Filter based on axis speed and apply linear interpolation.
    
    """
    filtered_dict = {}
    mask_dict = {}
    
    for sess_id, coords in coord_dict.items():
        n_frames, n_keypoints, _ = coords.shape
        filtered_coords = coords.copy()
        session_masks = np.zeros((n_frames, n_keypoints), dtype=bool)
        
        kpt_indices = keypoint_indices if keypoint_indices is not None else range(n_keypoints)
        
        for kpt_idx in kpt_indices:
            df = pd.DataFrame({
                'x': coords[:, kpt_idx, 0],
                'y': coords[:, kpt_idx, 1],
                'z': np.zeros(n_frames)
            })
            
            df_filtered, jump_mask = _filter_axis_speed_and_linear_interpolate(df, axis_threshold)
            
            filtered_coords[:, kpt_idx, 0] = df_filtered['x'].values
            filtered_coords[:, kpt_idx, 1] = df_filtered['y'].values
            session_masks[:, kpt_idx] = jump_mask
        
        filtered_dict[sess_id] = filtered_coords
        mask_dict[sess_id] = session_masks
    
    return filtered_dict, mask_dict


def apply_coordinate_filters(
    coord_dict: Dict[str, np.ndarray],
    filters: list = ['range', 'speed_interp'],
    x_range: Tuple[float, float] = (0, 50),
    y_range: Tuple[float, float] = (0, 50),
    axis_threshold: float = 5.0,
    context: int = 5,
    max_drift_length: int = 20,
    keypoint_indices: Optional[list] = None
) -> Dict[str, np.ndarray]:

    result = coord_dict.copy()
    
    for filter_name in filters:
        if filter_name == 'range':
            result = filter_dlc_coordinates_range_interp(
                result, x_range, y_range, keypoint_indices
            )
            print(f"Applied range filter: x{x_range}, y{y_range}")
            
        elif filter_name == 'speed_repeat':
            result, _ = filter_dlc_coordinates_speed_repeat(
                result, axis_threshold, keypoint_indices
            )
            print(f"Applied speed_repeat filter: threshold={axis_threshold}")
            
        elif filter_name == 'drift':
            result, _ = filter_dlc_coordinates_drift_interp(
                result, axis_threshold, context, max_drift_length, keypoint_indices
            )
            print(f"Applied drift filter: threshold={axis_threshold}, context={context}, max_length={max_drift_length}")
            
        elif filter_name == 'speed_interp':
            result, _ = filter_dlc_coordinates_speed_interp(
                result, axis_threshold, keypoint_indices
            )
            print(f"Applied speed_interp filter: threshold={axis_threshold}")
            
        else:
            print(f"Warning: Unknown filter '{filter_name}', skipping...")
    
    return result
