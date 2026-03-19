import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import sys
import os
import glob
import math
import ast
from scipy.interpolate import UnivariateSpline

# Configuration
LOG_DIR = "/home/a3dobject/AUVSimData"
# Spline smoothing factors. Higher = smoother curve. Lower = tighter to the noise.
SMOOTH_POS = 5.0   
SMOOTH_VEL = 2.0   

def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def get_log_files(directory):
    if not os.path.exists(directory): return []
    files = glob.glob(os.path.join(directory, "*.parquet"))
    files.sort(key=os.path.getmtime)
    return files

def select_log_file():
    files = get_log_files(LOG_DIR)
    if not files:
        print(f"No log files found in: {LOG_DIR}")
        sys.exit(1)
    print(f"\nFound {len(files)} log files:")
    display_files = files[-10:] 
    start_index = len(files) - len(display_files) + 1
    for i, f in enumerate(display_files):
        idx = start_index + i
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"[{idx}] {os.path.basename(f):<40} | {size_mb:.2f} MB")
    user_input = input(f"Select file [1-{len(files)}] (default: latest): ").strip()
    return files[int(user_input)-1] if user_input.isdigit() else files[-1]

def parse_parquet_log(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    df = pd.read_parquet(filepath)
    
    data = {
        "odom_x": [], "odom_y": [], "odom_z": [], "t_odom": [],
        "imu_roll": [], "imu_pitch": [], "imu_yaw": [], "t_imu": [],
        "accel_x": [], "accel_y": [], "accel_z": [], "t_accel": [],
        "thrust_port": [], "thrust_stbd": [], "thrust_sway": [],
        "thrust_heave_b": [], "thrust_heave_s": [],
        "pressure": [], "t_pressure": []
    }

    if df.empty: return data
    start_time = df['timestamp'].min()

    for topic, group in df.groupby('topic'):
        times = (group['timestamp'] - start_time).tolist()
        if topic == "Odometry_Pose":
            for t, val in zip(times, group['data']):
                try:
                    p = ast.literal_eval(val) 
                    data["t_odom"].append(t)
                    data["odom_x"].append(p[0])
                    data["odom_y"].append(p[1])
                    data["odom_z"].append(p[2])
                except: pass
        elif topic == "IMU":
            for t, val in zip(times, group['data']):
                try:
                    q = ast.literal_eval(val)
                    r, p, y = quaternion_to_euler(q[0], q[1], q[2], q[3])
                    data["t_imu"].append(t)
                    data["imu_roll"].append(r)
                    data["imu_pitch"].append(p)
                    data["imu_yaw"].append(y)
                except: pass
        elif topic == "Accelerometer":
            for t, val in zip(times, group['data']):
                try:
                    a = ast.literal_eval(val)
                    data["t_accel"].append(t)
                    data["accel_x"].append(a[0])
                    data["accel_y"].append(a[1])
                    data["accel_z"].append(a[2])
                except: pass
        elif "Thruster" in topic:
            key = {
                "ThrusterSurgePort": "thrust_port",
                "ThrusterSurgeStarboard": "thrust_stbd",
                "ThrusterSway": "thrust_sway",
                "ThrusterHeaveBow": "thrust_heave_b",
                "ThrusterHeaveStern": "thrust_heave_s"
            }.get(topic)
            if key: data[key] = list(zip(times, group['data'].astype(float)))

    if data["odom_x"]:
        end_x, end_y, end_z = data["odom_x"][-1], data["odom_y"][-1], data["odom_z"][-1]
        data["dist_to_dock"] = [
            math.sqrt((x - end_x)**2 + (y - end_y)**2 + (z - end_z)**2)
            for x, y, z in zip(data["odom_x"], data["odom_y"], data["odom_z"])
        ]
        
    if len(data["t_odom"]) > 10:
        t = np.array(data["t_odom"])
        x, y, z = np.array(data["odom_x"]), np.array(data["odom_y"]), np.array(data["odom_z"])
        
        valid_idx = np.concatenate(([True], np.diff(t) > 0))
        t, x, y, z = t[valid_idx], x[valid_idx], y[valid_idx], z[valid_idx]
        data["t_odom_clean"] = t.tolist()
        data["dist_to_dock_clean"] = np.array(data["dist_to_dock"])[valid_idx].tolist()

        spline_x = UnivariateSpline(t, x, k=3, s=SMOOTH_POS)
        spline_y = UnivariateSpline(t, y, k=3, s=SMOOTH_POS)
        spline_z = UnivariateSpline(t, z, k=3, s=SMOOTH_POS)
        
        vx, vy, vz = spline_x.derivative(1)(t), spline_y.derivative(1)(t), spline_z.derivative(1)(t)
        
        smooth_speed = np.sqrt(vx**2 + vy**2 + vz**2)
        data["spline_speed"] = smooth_speed.tolist()
        
        spline_speed_func = UnivariateSpline(t, smooth_speed, k=3, s=SMOOTH_VEL)
        data["spline_accel"] = spline_speed_func.derivative(1)(t).tolist()
        
        data["smooth_x"] = spline_x(t).tolist()
        data["smooth_y"] = spline_y(t).tolist()
        data["smooth_z"] = spline_z(t).tolist()

    return data

def plot_trajectory(data):
    if not data.get('smooth_x'): return
    x, y, z = data['smooth_x'], data['smooth_y'], data['smooth_z']
    
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    
    # --- Hardcoded Dock Coordinates ---
    p_tail = np.array([0.0, 0.0])
    p_mouth = np.array([0.0, 4.53528])
    p_staging = np.array([0.0, 24.53528])  # 20 meters ahead of mouth
    p_obs = np.array([0.0, 2.26764])       # Midpoint between tail and mouth
    z_dock = 10.0                          # Fixed dock depth
    
    # Directional vectors for funnel geometry
    u_mouth = np.array([0.0, 1.0]) 
    u_perp = np.array([-1.0, 0.0])         # Perpendicular vector for frustum width
    
    # APF Constants
    APF_R_INF = 15.0
    NO_FLY_RADIUS = 5.0
    APF_K_ATT = 1.0
    APF_K_REP = 100.0
    APF_K_TAN = 300.0

    # --- Generate 5m Grid for APF Quiver Plot ---
    x_min = min(min(data['odom_x']), p_staging[0]) - 15
    x_max = max(max(data['odom_x']), p_staging[0]) + 15
    y_min = min(min(data['odom_y']), p_tail[1]) - 15
    y_max = max(max(data['odom_y']), p_staging[1]) + 15
    
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, 5.0),
        np.arange(y_min, y_max, 5.0)
    )
    
    U = np.zeros_like(grid_x)
    V = np.zeros_like(grid_y)
    
    # Calculate force vector at every grid intersection
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            p_grid = np.array([grid_x[i, j], grid_y[i, j]])
            
            # Attractive Force
            v_att = p_staging - p_grid
            dist_staging = np.linalg.norm(v_att)
            f_att = APF_K_ATT * (v_att / max(dist_staging, 0.001))
            
            # Repulsive & Tangential Forces
            f_rep = np.array([0.0, 0.0])
            f_tan = np.array([0.0, 0.0])
            
            v_obs = p_grid - p_obs
            d_obs = np.linalg.norm(v_obs)
            
            if d_obs < APF_R_INF:
                safe_d_boundary = max(d_obs - NO_FLY_RADIUS, 0.001)
                n_vec = v_obs / max(d_obs, 0.001)
                t_vec = np.array([n_vec[1], -n_vec[0]])
                
                mag = (1.0 / safe_d_boundary) - (1.0 / (APF_R_INF - NO_FLY_RADIUS))
                mag = max(mag, 0.0)
                
                f_rep = APF_K_REP * (mag**2) * n_vec
                f_tan = APF_K_TAN * (mag**2) * t_vec
                
            f_total = f_att + f_rep + f_tan
            
            # Normalize vector for clean quiver visualization
            norm_f = np.linalg.norm(f_total)
            if norm_f > 0:
                U[i, j] = f_total[0] / norm_f
                V[i, j] = f_total[1] / norm_f

    # --- Funnel Frustum Geometry ---
    mouth_half_width = 1.5 
    tail_half_width = 0.5
    
    c1 = p_mouth + u_perp * mouth_half_width
    c2 = p_mouth - u_perp * mouth_half_width
    c3 = p_tail - u_perp * tail_half_width
    c4 = p_tail + u_perp * tail_half_width
    
    funnel_x = [c1[0], c2[0], c3[0], c4[0], c1[0]]
    funnel_y = [c1[1], c2[1], c3[1], c4[1], c1[1]]

    # Plot Background Elements
    ax1.scatter(p_staging[0], p_staging[1], color='darkorange', marker='*', s=250, edgecolor='black', label='Staging Point', zorder=5)
    
    influence_circle = plt.Circle((p_obs[0], p_obs[1]), APF_R_INF, color='grey', fill=True, alpha=0.3, label='APF Influence Zone', zorder=1)
    ax1.add_patch(influence_circle)

    no_fly_circle = plt.Circle((p_obs[0], p_obs[1]), NO_FLY_RADIUS, color='red', fill=True, alpha=0.3, label='No Fly Zone', zorder=2)
    ax1.add_patch(no_fly_circle)

    # Plot the unfilled Funnel Frustum
    ax1.plot(funnel_x, funnel_y, color='black', linewidth=2.0, linestyle='-', label='Funnel Dock', zorder=6)
    
    # Plot the center of the APF
    ax1.scatter(p_obs[0], p_obs[1], color='black', marker='.', s=100, label='APF Center', zorder=7)

    # Draw Quiver (Vector Field)
    ax1.quiver(grid_x, grid_y, U, V, color='teal', alpha=0.4, 
               angles='xy', pivot='mid', label='APF Vector Field', zorder=3)

    # Plot Trajectory Data
    ax1.plot(data['odom_x'], data['odom_y'], color='#1f77b4', alpha=0.2, label='Noisy Path', zorder=4)
    ax1.plot(x, y, color='#1f77b4', linewidth=2.5, label='Smoothed Path', zorder=5)
    ax1.scatter(x[0], y[0], color='green', marker='o', s=100, edgecolor='black', label='Start', zorder=6)
    
    ax1.set_title("2D Top-Down Trajectory with APF Vector Field")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=9)

    # 3D Plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(data['odom_x'], data['odom_y'], data['odom_z'], color='#9467bd', alpha=0.2, label='Noisy Path')
    ax2.plot(x, y, z, color='#9467bd', linewidth=2, label='Smoothed Path')
    
    # Plot the 3D Funnel Frustum at the fixed depth
    funnel_z = [z_dock, z_dock, z_dock, z_dock, z_dock]
    # ax2.plot(funnel_x, funnel_y, funnel_z, color='black', linewidth=2.0)
    
    # Trajectory Start and End Markers
    ax2.scatter(x[0], y[0], z[0], color='green', marker='o', s=50, edgecolor='black', label='Start')
    ax2.scatter(x[-1], y[-1], z[-1], color='red', marker='X', s=50, edgecolor='black', label='End')
    
    ax2.set_title("3D AUV Trajectory")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Depth (m)")
    ax2.invert_zaxis() 
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_dynamics_and_plato(data):
    if not data.get('spline_speed'):
        print("Not enough trajectory data to plot speed and acceleration.")
        return

    t = data['t_odom_clean']
    speed = data['spline_speed']
    accel = data['spline_accel']
    dist = data['dist_to_dock_clean']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.plot(t, speed, color='blue', linewidth=2.5, label='Speed (Analytical)')
    ax1_accel = ax1.twinx()
    ax1_accel.plot(t, accel, color='red', linewidth=2.5, label='Acceleration (Analytical)')
    
    ax1.set_title("Mission Dynamics vs Time (Spline Fit)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (m/s)", color='blue')
    ax1_accel.set_ylabel("Acceleration (m/s²)", color='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_accel.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(dist, speed, color='darkgreen', linewidth=2.5, label='Forward Speed')
    ax2.set_xlim(max(dist), 0)
    ax2.set_title("PLATO Profile: Speed vs Distance to Dock")
    ax2.set_xlabel("Distance Remaining (m)")
    ax2.set_ylabel("Forward Speed (m/s)")
    ax2.axhline(y=0.1, color='r', linestyle='--', label='Min Docking Speed (0.1 m/s)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    path = select_log_file()
    data = parse_parquet_log(path)
    if data:
        plot_trajectory(data)
        plot_dynamics_and_plato(data)

if __name__ == "__main__":
    main()