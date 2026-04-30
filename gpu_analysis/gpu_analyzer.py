import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_gpu_data(log_file="gpu_usage_log.csv"):
    try:
        # Load the data
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Could not find {log_file}. Run the logger for a while first!")
        return

    # Convert timestamp string to actual datetime objects
    # nvidia-smi format: "YYYY/MM/DD HH:MM:SS.msec"
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')

    # Since multiple processes run at the same time, sum the memory for each exact timestamp
    total_mem_time = df.groupby('timestamp')['memory_mb'].sum().reset_index()

    # Extract Hour and Day of the Week
    total_mem_time['hour'] = total_mem_time['timestamp'].dt.hour
    total_mem_time['day_of_week'] = total_mem_time['timestamp'].dt.day_name()

    # Create a pivot table for the Heatmap (Average memory used for a specific hour on a specific day)
    heatmap_data = total_mem_time.pivot_table(
        values='memory_mb', 
        index='day_of_week', 
        columns='hour', 
        aggfunc='mean'
    )

    # Order the days logically
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Only include days we actually have data for to avoid errors
    days_present = [day for day in days_order if day in heatmap_data.index]
    heatmap_data = heatmap_data.reindex(days_present)

    # Plot the Heatmap
    plt.figure(figsize=(14, 6))
    
    # "YlOrRd" goes from Yellow (Empty/Good) to Orange to Red (Crowded/Bad)
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=False, linewidths=.5)
    
    plt.title("Server GPU Memory Usage: Heatmap of Crowded Times", fontsize=16)
    plt.xlabel("Hour of the Day (0 - 23)", fontsize=12)
    plt.ylabel("Day of the Week", fontsize=12)
    
    # Save it to a file so you can download and look at it
    plt.tight_layout()
    plt.savefig("gpu_crowd_heatmap.png", dpi=300)
    print("Graph saved as 'gpu_crowd_heatmap.png'. Look for the yellow/light spots to schedule your runs!")

if __name__ == "__main__":
    analyze_gpu_data()