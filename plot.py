import pandas as pd
import matplotlib.pyplot as plt
import argparse

def visualize_features(csv_path):
    # Load the exported CSV data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_path}'. Please check the path.")
        return

    # Check if frame_idx exists, otherwise use the index
    if 'frame_idx' in df.columns:
        x = df['frame_idx']
    else:
        x = df.index
        print("Warning: 'frame_idx' column not found. Using row index as the time axis.")

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Sign Language Segmentation Features over Time', fontsize=16)

    # 1. Plot Wrist Speeds
    axs[0].plot(x, df.get('left_wrist_speed', []), label='Left Wrist Speed', color='blue')
    axs[0].plot(x, df.get('right_wrist_speed', []), label='Right Wrist Speed', color='orange')
    axs[0].set_ylabel('Speed')
    axs[0].set_title('Wrist Kinematics (Speed)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Plot Wrist Accelerations
    axs[1].plot(x, df.get('left_wrist_accel', []), label='Left Wrist Accel', color='cyan')
    axs[1].plot(x, df.get('right_wrist_accel', []), label='Right Wrist Accel', color='red')
    axs[1].set_ylabel('Acceleration')
    axs[1].set_title('Wrist Kinematics (Acceleration)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Plot Distances
    axs[2].plot(x, df.get('inter_hand_distance', []), label='Inter-Hand Dist', color='purple')
    axs[2].plot(x, df.get('left_dist_to_rest', []), label='Left to Torso', color='green', linestyle='-.')
    axs[2].plot(x, df.get('right_dist_to_rest', []), label='Right to Torso', color='brown', linestyle='-.')
    axs[2].set_ylabel('Normalized Distance')
    axs[2].set_title('Spatial Relationships (Distances)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    # 4. Plot Activation & Handshape
    axs[3].plot(x, df.get('activation_ratio', []), label='Activation Ratio', color='black', linewidth=2)
    axs[3].plot(x, df.get('left_hand_spread', []), label='Left Spread', color='magenta', alpha=0.7)
    axs[3].plot(x, df.get('right_hand_spread', []), label='Right Spread', color='olive', alpha=0.7)
    axs[3].set_xlabel('Frame Index')
    axs[3].set_ylabel('Ratio / Spread')
    axs[3].set_title('Global Activation & Handshape')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # You can run this script from the terminal: python plot_features.py your_data.csv
    parser = argparse.ArgumentParser(description="Visualize Sign Language Segmentation Features")
    parser.add_argument("csv_file", type=str, help="Path to the exported CSV file", nargs='?', default="features_export.csv")
    args = parser.parse_args()
    
    visualize_features(args.csv_file)