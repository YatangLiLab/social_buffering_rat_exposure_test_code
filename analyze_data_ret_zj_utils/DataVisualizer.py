import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.path import Path
import math

def draw_ethogram_split(
    bhvr_tuples_dict, 
    bhvr_params, 
    sess_list=None, 
    fps=20, 
    save_path=None
):
    """Draws split ethograms for each session."""
    if sess_list is not None:
        sess_to_plot = {sess_id: bhvr_tuples_dict[sess_id] for sess_id in sess_list if sess_id in bhvr_tuples_dict}
    else:
        sess_to_plot = bhvr_tuples_dict
    for sess_id, behavior_frames in sess_to_plot.items():
        parts = sess_id.split('_')
        Batch_id = parts[0]
        Cage_id = parts[1]
        Cond_id = parts[2]
        Rank_id = parts[3]
        Mouse_id = parts[4]
        Phase_id = '_'.join(parts[5:])
        all_frames = []
        for behavior_type, frame_ranges in behavior_frames.items():
            for start_frame, end_frame in frame_ranges:
                all_frames.extend([start_frame, end_frame])
        if not all_frames:
            print(f"{sess_id}: no behavior data")
            continue
        max_frame = max(all_frames)
        behavior_list = sorted([b for b in behavior_frames.keys() if behavior_frames[b]])
        plt.figure(figsize=(20, len(behavior_list) * 0.8), dpi=300)
        plt.gca().invert_yaxis()
        for idx, behavior_type in enumerate(behavior_list[::-1]):
            frame_ranges = behavior_frames[behavior_type]
            for frame_range in frame_ranges:
                start_frame, end_frame = frame_range
                color, zorder = bhvr_params.get(behavior_type, ('gray', 1))
                rect = plt.Rectangle(
                    (start_frame, idx - 0.4), 
                    end_frame - start_frame, 
                    0.8, 
                    facecolor=color, 
                    edgecolor='none',
                    linewidth=0.5,
                    zorder=zorder
                )
                plt.gca().add_patch(rect)
        total_time_sec = max_frame / fps
        total_time_min = total_time_sec / 60
        num_ticks = min(int(total_time_min) + 1, 20)
        time_ticks_frames = np.linspace(0, max_frame, num_ticks)
        time_ticks_labels = [f'{int(t/fps/60)}' for t in time_ticks_frames]
        plt.xticks(time_ticks_frames, time_ticks_labels)
        plt.xlabel('Time (min)', fontsize=12)
        plt.yticks(range(len(behavior_list)), behavior_list[::-1])
        plt.ylabel('Behavior', fontsize=12)
        plt.xlim(0, max_frame)
        plt.ylim(-0.5, len(behavior_list) - 0.5)
        title = f'Ethogram: {sess_id}\n{Cond_id} - {Mouse_id} ({Rank_id}) - {Phase_id}'
        plt.title(title, fontsize=14)
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=color[0], label=label) 
            for label, color in bhvr_params.items() 
            if label in behavior_list
        ]
        plt.legend(
            handles=legend_elements, 
            loc='center left', 
            bbox_to_anchor=(1, 0.5),
            fontsize=10
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/ethogram_{sess_id}.png', bbox_inches='tight')
        plt.show()


def draw_ethogram_merged(
    bhvr_tuples_dict, 
    bhvr_params, 
    sess_list=None, 
    fps=20, 
    save_path=None
):
    """Draws a merged ethogram for all sessions."""
    if sess_list is not None:
        sess_to_plot = {sess_id: bhvr_tuples_dict[sess_id] for sess_id in sess_list if sess_id in bhvr_tuples_dict}
    else:
        sess_to_plot = bhvr_tuples_dict
    if not sess_to_plot:
        print("No sessions to plot")
        return
    sess_ids = list(sess_to_plot.keys())
    n_sessions = len(sess_ids)
    fig, ax = plt.subplots(figsize=(25, n_sessions * 1.2), dpi=300)
    for sess_idx, sess_id in enumerate(sess_ids):
        behavior_frames = sess_to_plot[sess_id]
        all_frames = []
        for behavior_type, frame_ranges in behavior_frames.items():
            for start_frame, end_frame in frame_ranges:
                all_frames.extend([start_frame, end_frame])
        if not all_frames:
            continue
        max_frame = max(all_frames)
        y_pos = sess_idx
        for behavior_type, frame_ranges in behavior_frames.items():
            if not frame_ranges:
                continue
            color, zorder = bhvr_params.get(behavior_type, ('gray', 1))
            for start_frame, end_frame in frame_ranges:
                rect = plt.Rectangle(
                    (start_frame, y_pos - 0.4), 
                    end_frame - start_frame, 
                    0.8, 
                    facecolor=color, 
                    edgecolor='none',
                    linewidth=0.3,
                    zorder=zorder,
                    alpha=0.8
                )
                ax.add_patch(rect)
    max_frame_all = 0
    for sess_id in sess_ids:
        behavior_frames = sess_to_plot[sess_id]
        for behavior_type, frame_ranges in behavior_frames.items():
            for start_frame, end_frame in frame_ranges:
                max_frame_all = max(max_frame_all, end_frame)
    total_time_sec = max_frame_all / fps
    total_time_min = total_time_sec / 60
    num_ticks = min(int(total_time_min) + 1, 20)
    time_ticks_frames = np.linspace(0, max_frame_all, num_ticks)
    time_ticks_labels = [f'{int(t/fps/60)}' for t in time_ticks_frames]
    ax.set_xticks(time_ticks_frames)
    ax.set_xticklabels(time_ticks_labels)
    ax.set_xlabel('Time (min)', fontsize=14)
    ax.set_yticks(range(n_sessions))
    ax.set_yticklabels(sess_ids, fontsize=10)
    ax.set_ylabel('Session', fontsize=14)
    ax.set_xlim(0, max_frame_all)
    ax.set_ylim(-0.5, n_sessions - 0.5)
    ax.invert_yaxis()
    ax.set_title('Ethogram: All Sessions', fontsize=16)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color[0], label=label) 
        for label, color in bhvr_params.items()
    ]
    ax.legend(
        handles=legend_elements, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        ncol=1
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/ethogram_all_sessions.png', bbox_inches='tight')
    plt.show()


def draw_ethogram(
    bhvr_tuples_dict, 
    bhvr_params, 
    sess_list, 
    fps, 
    mode, 
    save_path
):
    """Draws ethograms in split or merged mode."""
    if mode == 'split':
        return draw_ethogram_split(bhvr_tuples_dict, bhvr_params, sess_list, fps, save_path)
    elif mode == 'merged':
        return draw_ethogram_merged(bhvr_tuples_dict, bhvr_params, sess_list, fps, save_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'split' or 'merged'.")


def draw_bar_plot(
    data, 
    ylabel='Value', 
    title=None, 
    figsize=(10, 6), 
    show_points=True, 
    save_path=None, 
    order=None, 
    color_dict=None, 
    plot_type='bar', 
    jitter=True
):
    """Draws bar plots with various types."""
    if isinstance(data, dict):
        rows = []
        all_groups = list(data.keys())
        for group_name, values in data.items():
            if isinstance(values, list):
                if len(values) == 0:
                    rows.append({'group': group_name, 'value': np.nan})
                else:
                    for v in values:
                        rows.append({'group': group_name, 'value': v})
            else:
                rows.append({'group': group_name, 'value': values})
        df = pd.DataFrame(rows)
    else:
        df = data.copy()
        all_groups = None
        if 'value' not in df.columns:
            value_col = [c for c in df.columns if c != 'group'][0]
            df = df.rename(columns={value_col: 'value'})
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if order is not None:
        if all_groups is not None:
            group_order = [g for g in order if g in all_groups]
            for g in all_groups:
                if g not in group_order:
                    group_order.append(g)
        else:
            existing_groups = df['group'].unique()
            group_order = [g for g in order if g in existing_groups]
            for g in existing_groups:
                if g not in group_order:
                    group_order.append(g)
    elif all_groups is not None:
        group_order = all_groups
    else:
        group_order = df['group'].unique().tolist()
    if color_dict is not None:
        colors = []
        for g in group_order:
            if g in color_dict:
                color_val = color_dict[g]
                if isinstance(color_val, tuple) and len(color_val) >= 1:
                    colors.append(color_val[0])
                else:
                    colors.append(color_val)
            else:
                colors.append('gray')
        use_palette = colors
    else:
        use_palette = 'Set2'
    if plot_type == 'box':
        sns.boxplot(data=df, x='group', y='value', palette=use_palette, ax=ax, 
                    order=group_order, width=0.5)
        if show_points:
            sns.stripplot(data=df, x='group', y='value', alpha=0.5, ax=ax, 
                          order=group_order, size=6, color='black', jitter=jitter)
    elif plot_type == 'bar':
        sns.barplot(data=df, x='group', y='value', palette=use_palette, ax=ax,
                    order=group_order, errorbar='sd', capsize=0.1)
        if show_points:
            sns.stripplot(data=df, x='group', y='value', alpha=0.5, ax=ax, 
                          order=group_order, size=6, color='black', jitter=jitter)
    elif plot_type == 'strip':
        sns.stripplot(data=df, x='group', y='value', palette=use_palette, ax=ax, 
                      order=group_order, size=8, jitter=jitter, alpha=0.5)
        for i, group in enumerate(group_order):
            group_data = df[df['group'] == group]['value'].dropna()
            if len(group_data) > 0:
                mean_val = group_data.mean()
                ax.hlines(mean_val, i - 0.2, i + 0.2, colors='black', linewidth=2)
    elif plot_type == 'violin':
        sns.violinplot(data=df, x='group', y='value', palette=use_palette, ax=ax,
                       order=group_order, inner='box')
        if show_points:
            sns.stripplot(data=df, x='group', y='value', alpha=0.5, ax=ax, 
                          order=group_order, size=4, color='black', jitter=jitter)
    else:
        raise ValueError(f"unsupported plot_type: {plot_type}. You can choose from 'box', 'bar', 'strip', 'violin'")
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path/f'{title}_{plot_type}.png', bbox_inches='tight')
    plt.show()


def draw_line_plot(
    data,
    xticks=None,
    ylabel='Value',
    title=None,
    figsize=(10, 6),
    color_dict=None,
    save_path=None,
    error_type='std',
):
    """Draws line plots with error bars."""
    
    def _extract_scalar_and_error(v, error_type='std'):
        if isinstance(v, (list, tuple, np.ndarray)):
            v_arr = np.array(v)
            mean_val = float(np.nanmean(v_arr))
            if error_type == 'std':
                err_val = float(np.nanstd(v_arr, ddof=1))  # sample std
            elif error_type == 'sem':
                err_val = float(scipy_stats.sem(v_arr, nan_policy='omit'))
            else:
                err_val = np.nan
            return mean_val, err_val
        else:
            return v, np.nan
    rows = []
    if all(isinstance(v, dict) for v in data.values()):
        for outer_key, inner in data.items():
            inner_keys = list(inner.keys())
            n_inner = len(inner_keys)
            # Prefer behaviors with list/array values
            for beh_key, beh_val in inner.items():
                series_label = outer_key if n_inner == 1 else f"{outer_key}:{beh_key}"
                if isinstance(beh_val, (list, tuple, np.ndarray)):
                    for i, v in enumerate(beh_val):
                        scalar_val, err_val = _extract_scalar_and_error(v, error_type)
                        rows.append({'x': xticks[i], 'series': series_label, 'value': scalar_val, 'yerr': err_val})
                else:
                    rows.append({'x': beh_key, 'series': series_label, 'value': beh_val, 'yerr': np.nan})
    else:
        raise ValueError("Unsupported dict value type for draw_line_plot")
    # Palette handling
    palette = color_dict if isinstance(color_dict, dict) else 'Set2'
    df_rows = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.lineplot(
        data=df_rows, x='x', y='value', hue='series', ax=ax,
        palette=palette, linewidth=2, marker='o'
    )
    # Overlay error bars
    series_list = df_rows['series'].unique()
    colors_map = {}
    # Get colors from seaborn lineplot
    for i, line in enumerate(ax.lines):
        if i < len(series_list):
            colors_map[series_list[i]] = line.get_color()
    for series_name in series_list:
        sub = df_rows[df_rows['series'] == series_name]
        if sub['yerr'].notna().any():
            x_vals = sub['x'].tolist()
            y_vals = sub['value'].values
            y_err = sub['yerr'].values
            # Only plot error bars where error is not NaN
            mask = ~np.isnan(y_err)
            if mask.any():
                x_plot = [x_vals[i] for i in np.where(mask)[0]]
                y_plot = y_vals[mask]
                err_plot = y_err[mask]
                color = colors_map.get(series_name, 'black')
                ax.errorbar(
                    range(len(x_plot)), 
                    y_plot,
                    yerr=err_plot,
                    fmt='none',
                    ecolor=color,
                    elinewidth=1.5,
                    alpha=0.7,
                    zorder=3
                )
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path/f'{title}.png', bbox_inches='tight')
    plt.show()


def draw_heatmap(
    heatmap_results,
    show_arrows=True,
    vmax_time=None,
    vmax_speed=None,
    save_path=None,
    figsize=(12, 5),
    cmap='viridis'
):
    """Draws position and speed heatmaps."""
    if not heatmap_results:
        print("No heatmap data")
        return {}
    fig_axes = {}
    for sess_id, data in heatmap_results.items():
        if data is None:
            continue
        hist_frequency = data['hist_frequency']
        speed_avg = data['speed_avg']
        grid_vx_avg = data['grid_vx_avg']
        grid_vy_avg = data['grid_vy_avg']
        xedges = data['xedges']
        yedges = data['yedges']
        if vmax_time is not None:
            time_max = vmax_time
        elif np.all(np.isnan(hist_frequency)):
            time_max = 0
        else:
            time_max = np.nanmax(hist_frequency)
        if vmax_speed is not None:
            speed_max = vmax_speed
        elif np.all(np.isnan(speed_avg)):
            speed_max = 0
        else:
            speed_max = np.nanmax(speed_avg)
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
        ax1 = axes[0]
        im1 = ax1.imshow(
            hist_frequency.T,
            cmap=cmap,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmin=0,
            vmax=time_max
        )
        fig.colorbar(im1, ax=ax1, label='Time (%)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_title(f'Position Heatmap\n{sess_id}')
        ax1.invert_yaxis()
        ax2 = axes[1]
        im2 = ax2.imshow(
            speed_avg.T,
            cmap=cmap,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmin=0,
            vmax=speed_max
        )
        fig.colorbar(im2, ax=ax2, label='Speed (pixels/s)')
        if show_arrows:
            X_centers = (xedges[:-1] + xedges[1:]) / 2
            Y_centers = (yedges[:-1] + yedges[1:]) / 2
            X_mesh, Y_mesh = np.meshgrid(X_centers, Y_centers)
            scale = 0.3
            head_scale = 0.01
            for i in range(X_mesh.shape[0]):
                for j in range(X_mesh.shape[1]):
                    vx_ = grid_vx_avg.T[i, j]
                    vy_ = grid_vy_avg.T[i, j]
                    if np.isnan(vx_) or np.isnan(vy_):
                        continue
                    mag = np.sqrt(vx_ ** 2 + vy_ ** 2)
                    if mag < 1e-5:
                        continue
                    dx = vx_ * scale
                    dy = vy_ * scale
                    x0 = X_mesh[i, j]
                    y0 = Y_mesh[i, j]
                    x1 = x0 + dx
                    y1 = y0 + dy
                    ax2.annotate(
                        '', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle=f'->, head_width={mag * head_scale:.2f}, head_length={mag * head_scale * 1.5:.2f}',
                            color='white',
                            linewidth=1,
                            mutation_scale=5,
                            shrinkA=0, shrinkB=0
                        )
                    )
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title(f'Speed Heatmap\n{sess_id}')
        ax2.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/heatmap_{sess_id}.png', bbox_inches='tight')
        plt.show()
        fig_axes[sess_id] = (fig, axes)


def draw_stacked_bar_plot(
    data,
    category_order=None,
    color_dict=None,
    title='Behavior Decision',
    ylabel='Behavior (%)',
    figsize=(8, 6),
    save_path=None
):
    """Draws stacked bar plots for behavior percentages."""
    # Normalize input to plot_data (list of {category: pct}) and group_labels
    if isinstance(data, dict) and 'by_session' in data:
        by_session = data['by_session']
        group_labels = list(by_session.keys())
        plot_data = [
            {cat: sess_data[cat]['percentage'] for cat in sess_data.keys()}
            for sess_id, sess_data in by_session.items()
        ]
        if category_order is None and group_labels:
            category_order = list(by_session[group_labels[0]].keys())
    elif isinstance(data, dict):
        # Direct pct_dict: {sess_id: {category: pct}}
        group_labels = list(data.keys())
        plot_data = [data[sess_id] for sess_id in group_labels]
        if category_order is None and group_labels:
            category_order = list(plot_data[0].keys())
    elif isinstance(data, list):
        plot_data = data
        group_labels = [f'Group {i+1}' for i in range(len(data))]
        if category_order is None and len(data) > 0:
            category_order = list(data[0].keys())
    else:
        raise ValueError("Unsupported format: expected dict (pct_dict or by_session) or list")
    n_groups = len(plot_data)
    n_categories = len(category_order)
    # Normalize colors into a dict palette (using only color_dict or defaults)
    if isinstance(color_dict, dict) and len(color_dict) > 0:
        palette_colors = color_dict
    else:
        default_colors = ['#ff9d43', '#679fce', '#ffffa0', '#62c05b', '#a0a0a4', '#FFFF00', '#FF00FF', '#00FFFF']
        palette_colors = {cat: default_colors[i % len(default_colors)] for i, cat in enumerate(category_order)}
    percentages = np.zeros((n_groups, n_categories))
    for i, pct_dict in enumerate(plot_data):
        for j, cat in enumerate(category_order):
            percentages[i, j] = pct_dict.get(cat, 0)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    x = np.arange(n_groups)
    bar_width = 0.6
    bottom = np.zeros(n_groups)
    bars_list = []
    for j, cat in enumerate(category_order):
        bars = ax.bar(
            x,
            percentages[:, j],
            bar_width,
            bottom=bottom,
            label=cat,
            color=palette_colors.get(cat, 'gray'),
            edgecolor='white',
            linewidth=0.5,
        )
        bars_list.append(bars)
        for i, (bar, pct) in enumerate(zip(bars, percentages[:, j])):
            if pct > 3:
                height = bottom[i] + pct / 2
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{pct:.1f}%',
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        bottom += percentages[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=12, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path/f'{title}.png', bbox_inches='tight')
    plt.show()


def draw_transition_heatmap(
    trans_matrices_dict,
    bhvr_abbrev_dict,
    matrix_order=None,
    threshold=0.0,
    figsize=(10, 8),
    cmap="Blues",
    vmin=0,
    vmax=100,
    title_prefix="Behavior Transition Matrix",
    save_path=None
):
    """Draws heatmaps for behavior transition matrices."""
    if matrix_order is None:
        matrix_order = list(bhvr_abbrev_dict.keys())
    new_labels = [bhvr_abbrev_dict.get(b, b) for b in matrix_order]
    results = {}
    for sess_id, trans_matrix in trans_matrices_dict.items():
        if trans_matrix is not None and not trans_matrix.empty:
            available_states = [s for s in matrix_order if s in trans_matrix.index]
            norm_trans_matrix = trans_matrix.reindex(
                index=available_states, 
                columns=available_states
            ).fillna(0)
            abbrev_labels = [bhvr_abbrev_dict.get(b, b) for b in available_states]
            df_mod = norm_trans_matrix.copy()
            df_mod.index = abbrev_labels
            df_mod.columns = abbrev_labels
            df_merged_rows = df_mod.groupby(level=0, sort=False).sum()
            df_merged_all = df_merged_rows.groupby(level=0, axis=1, sort=False).sum()
            df_merged_all = df_merged_all.where(df_merged_all > threshold, 0)
            row_sums = df_merged_all.sum(axis=1)
            safe_row_sums = row_sums.replace(0, np.nan)
            matrix_heatmap = df_merged_all.div(safe_row_sums, axis=0).fillna(0)
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            sns.heatmap(
                matrix_heatmap * 100,
                xticklabels=matrix_heatmap.columns,
                yticklabels=matrix_heatmap.index,
                cmap=cmap,
                annot=True,
                fmt=".1f",
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.invert_yaxis()
            plt.title(f"{title_prefix} (row→col)\n{sess_id}")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path/f'{title_prefix}_{sess_id}', bbox_inches='tight')
            plt.show()
            results[sess_id] = (fig, ax)
        else:
            print(f"{sess_id}: is empty, skipping.")


def draw_transition_network(
    trans_matrices_dict,
    bhvr_abbrev_dict,
    bhvr_color_dict=None,
    matrix_order=None,
    threshold=0.0,
    figsize=(8, 6),
    node_radius=0.05,
    label_offset=0.20,
    line_width_scale=10,
    title_prefix="Behavior Transition Network",
    show_legend=True,
    save_path=None
):
    """Draws network plots for behavior transitions."""
    if matrix_order is None:
        matrix_order = list(bhvr_abbrev_dict.keys())
    nodes_list = [bhvr_abbrev_dict.get(b, b) for b in matrix_order]
    if bhvr_color_dict is None:
        vals = np.linspace(0.1, 0.9, len(nodes_list))
        cmap = cm.hsv
        bhvr_color_dict = {abbr: mcolors.to_hex(cmap(v)) for abbr, v in zip(nodes_list, vals)}
    results = {}
    for sess_id, trans_matrix in trans_matrices_dict.items():
        if trans_matrix is not None and not trans_matrix.empty:
            available_states = [s for s in matrix_order if s in trans_matrix.index]
            norm_trans_matrix = trans_matrix.reindex(
                index=available_states, 
                columns=available_states
            ).fillna(0)
            abbrev_labels = [bhvr_abbrev_dict.get(b, b) for b in available_states]
            df_mod = norm_trans_matrix.copy()
            df_mod.index = abbrev_labels
            df_mod.columns = abbrev_labels
            df_merged_rows = df_mod.groupby(level=0, sort=False).sum()
            df_merged_all = df_merged_rows.groupby(level=0, axis=1, sort=False).sum()
            df_merged_all = df_merged_all.where(df_merged_all > threshold, 0)
            row_sums = df_merged_all.sum(axis=1)
            safe_row_sums = row_sums.replace(0, np.nan)
            matrix_network = df_merged_all.div(safe_row_sums, axis=0).fillna(0)
            current_nodes = list(matrix_network.index)
            n = len(current_nodes)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            R = 1
            pos = {node: (R * math.cos(angles[i]), R * math.sin(angles[i])) 
                for i, node in enumerate(current_nodes)}
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.set_aspect('equal', 'box')
            plt.axis('off')
            background_circle = plt.Circle(
                (0, 0), radius=R,
                facecolor='lightgray', edgecolor='none',
                alpha=0.1, zorder=1
            )
            ax.add_patch(background_circle)
            for b1 in current_nodes:
                for b2 in current_nodes:
                    val = matrix_network.loc[b1, b2]
                    if val > threshold:
                        x1, y1 = pos[b1]
                        x2, y2 = pos[b2]
                        color = bhvr_color_dict.get(b1, "gray")
                        line_width = val * line_width_scale
                        if b1 == b2:
                            angle = math.atan2(y1, x1)
                            A, B = 0.5, 0.5
                            p0 = (x1, y1)
                            p1 = (x1 + A * math.cos(angle) - B * math.sin(angle),
                                y1 + A * math.sin(angle) + B * math.cos(angle))
                            p2 = (x1 + A * math.cos(angle) + B * math.sin(angle),
                                y1 + A * math.sin(angle) - B * math.cos(angle))
                            p3 = (x1, y1)
                            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                            path = Path([p0, p1, p2, p3], codes)
                            patch = mpatches.PathPatch(
                                path, edgecolor=color, facecolor='none',
                                lw=line_width, alpha=0.7
                            )
                            ax.add_patch(patch)
                        else:
                            arrow = mpatches.FancyArrowPatch(
                                (x1, y1), (x2, y2),
                                connectionstyle="arc3, rad=0.2",
                                arrowstyle='-',
                                mutation_scale=12,
                                lw=line_width,
                                color=color,
                                alpha=0.7
                            )
                            ax.add_patch(arrow)
            for node in current_nodes:
                x, y = pos[node]
                color = bhvr_color_dict.get(node, "gray")
                circle = plt.Circle(
                    (x, y), radius=node_radius,
                    facecolor=color, edgecolor='black',
                    linewidth=2.0, zorder=3
                )
                ax.add_patch(circle)
                angle = math.atan2(y, x)
                label_x = x + label_offset * math.cos(angle)
                label_y = y + label_offset * math.sin(angle)
                ax.text(label_x, label_y, node,
                        ha='center', va='center',
                        fontsize=16, zorder=4)
            margin = 1.4
            ax.set_xlim(-margin, margin)
            ax.set_ylim(-margin, margin)
            if show_legend:
                handles = []
                for full_name, abbreviation in bhvr_abbrev_dict.items():
                    if abbreviation in current_nodes:
                        color = bhvr_color_dict.get(abbreviation, "gray")
                        label = f"{abbreviation}: {full_name}"
                        patch = mpatches.Patch(color=color, label=label)
                        handles.append(patch)
                fig.subplots_adjust(right=0.78)
                ax.legend(handles=handles,
                        loc='center left',
                        bbox_to_anchor=(1.00, 0.5),
                        frameon=False,
                        fontsize=10,
                        title="Behaviors",
                        title_fontsize=11)
            plt.tight_layout()
            plt.title(f"{title_prefix}\n{sess_id}", fontsize=14)
            if save_path:
                plt.savefig(save_path/f'{title_prefix}_{sess_id}', bbox_inches='tight')
            plt.show()
            results[sess_id] = (fig, ax)
        else:
            print(f"{sess_id}: is empty, skipping.")


def draw_histogram_merged(
    all_freezing_times, 
    bins='auto', 
    figsize=(8, 5),
    color='gray', 
    xlabel='Value',
    title='Histogram',
    save_path=None
):
    """Draws a merged histogram for freezing durations."""
    merged_durs = [d for ds in all_freezing_times.values() for d in ds]
    fig, ax = plt.subplots(figsize=figsize)
    if len(merged_durs) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_axis_off()
    ax.hist(merged_durs, bins=bins, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path/f'histogram_{title}.png', bbox_inches='tight')


def draw_histogram_split(
    all_freezing_times, 
    bins='auto', 
    figsize=(8, 5),
    color='gray', 
    xlabel='Value',
    title='Histogram',
    save_path=None
):    
    """Draws split histograms for freezing durations per session."""
    sess_ids = list(all_freezing_times.keys())
    n = len(sess_ids)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_axis_off()
    figures = []
    for sess_id in sess_ids:
        durs = all_freezing_times[sess_id]
        fig, ax = plt.subplots(figsize=figsize)
        if len(durs) == 0 or np.isnan(durs).all():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_axis_off()
        else:
            ax.hist(durs, bins=bins, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title(f'{title} - {sess_id}')
        plt.tight_layout()
        figures.append((fig, ax))
        if save_path:
            plt.savefig(save_path/f'histogram_{title}_{sess_id}.png', bbox_inches='tight')
    

def draw_histogram(
    all_freezing_times, 
    mode='merged', 
    bins='auto', 
    figsize=(8, 5),
    color='gray', 
    xlabel='Value',  
    title='Histogram', 
    save_path=None
):
    """Draws histograms for freezing durations in merged or split mode."""
    if mode == 'merged':
        return draw_histogram_merged(all_freezing_times, bins, figsize, color, xlabel, title, save_path)
    elif mode == 'split':
        return draw_histogram_split(all_freezing_times, bins, figsize, color, xlabel, title, save_path)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'split' or 'merged'.")
