import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.ticker as ticker

# Global settings to preserve view angles
global_view_settings = {'elev': 25, 'azim': -60}

def generate_review_dynamics_3d_chart(csv_path, output_path=None, elev=None, azim=None, show_plot=True):
    df = pd.read_csv(csv_path)
    dimensions = df.columns[1:-1]  # Exclude time_code and canonical_ids
    dim_count = len(dimensions)
    colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
    shift_amount = 0.12
    
    dim_data = {d: defaultdict(list) for d in dimensions}
    
    for _, row in df.iterrows():
        time_str = str(row['time_code'])
        date = datetime.strptime(time_str, "%m%d%Y")
        if date.year == 2025: continue
        canonical_ids = list(map(int, str(row['canonical_ids']).split(';')))
        
        for dim_idx, dim in enumerate(dimensions):
            scores = list(map(int, str(row[dim]).split(';')))
            for j, score in enumerate(scores):
                reviewer_id = canonical_ids[j]
                dim_data[dim][reviewer_id].append((date, score))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for dim_idx, dim in enumerate(dimensions):
        color = colors[dim_idx % len(colors)]
        for reviewer_id, values in dim_data[dim].items():
            values.sort(key=lambda x: x[0])
            dates, scores = zip(*values)
            xs = mdates.date2num(dates)
            ys = [reviewer_id + dim_idx * shift_amount] * len(xs)
            zs = scores
            ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.9)
            ax.scatter(xs, ys, zs, color=color, s=20, marker='o')

    ax.set_xlabel('Date')
    ax.set_ylabel('Reviewer Index')
    ax.set_zlabel('Score')
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # exactly every 7 days
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # force integer ticks
    ax.yaxis.grid(True)  # enable grid lines on y-axis

    legend_handles = [Line2D([0], [0], color=colors[i], lw=4, label=dimensions[i].capitalize()) 
                      for i in range(dim_count)]
    ax.legend(handles=legend_handles, title="Review Dimension", loc='upper left')

    # Use specified view or fall back to global
    ax.view_init(elev if elev is not None else global_view_settings['elev'],
                 azim if azim is not None else global_view_settings['azim'])
    
    # After plotting everything
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Scale factors: x, y, z (e.g., stretch y-axis)
    scales = [1, 1.5, 1]  # make y-axis 2x wider visually

    ax.get_proj = lambda: Axes3D.get_proj(ax) @ \
        [[scales[0], 0, 0, 0],
        [0, scales[1], 0, 0],
        [0, 0, scales[2], 0],
        [0, 0, 0, 1]]
    
    
    # plt.title(f'Review Dynamics Over Time: {os.path.basename(csv_path)}')
    plt.tight_layout()

    if output_path:
        # plt.savefig(output_path, dpi=300)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    if show_plot:
        plt.show()
    plt.close(fig)

def interactive_angle_setup(csv_sample_path):
    print("Displaying interactive figure to determine ideal viewing angles...")
    generate_review_dynamics_3d_chart(csv_sample_path)
    elev = float(input("Enter desired elevation angle (e.g., 25): "))
    azim = float(input("Enter desired azimuth angle (e.g., -60): "))
    global_view_settings['elev'] = elev
    global_view_settings['azim'] = azim
    print(f"Angle set: elev={elev}, azim={azim}")

def batch_generate_charts_from_folder(folder_path, output_folder=None):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            output_path = None
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, filename.replace('.csv', '.png'))
            generate_review_dynamics_3d_chart(
                csv_path,
                output_path=output_path,
                elev=global_view_settings['elev'],
                azim=global_view_settings['azim'],
                show_plot=False
            )

# Example:
# Step 1: Set preferred view
path = '/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2025/footprints/threshold_5/0aaaM31hLB_success.csv'
# interactive_angle_setup(path)

# Step 2: Generate all charts with fixed angle
root_in = "/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2025/footprints/threshold_5"
root_out = "/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2025/footprints_vis/threshold_5"
batch_generate_charts_from_folder("/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2025/footprints/threshold_5", "/home/jyang/projects/papercopilot/logs/openreview/venues/iclr/iclr2025/footprints_vis/threshold_5")

