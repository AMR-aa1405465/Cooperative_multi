from datetime import datetime
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import os

def smooth_data(data, method='moving_average', window=5, poly_order=2):
    """
    Smooth data using various methods.
    
    Parameters:
    -----------
    data : array-like
        Data to smooth
    method : str
        Smoothing method: 'moving_average', 'exponential', or 'savgol'
    window : int
        Window size for moving average or Savitzky-Golay filter
    poly_order : int
        Polynomial order for Savitzky-Golay filter
    """
    if method == 'moving_average':
        return pd.Series(data).rolling(window=window, center=True).mean()
    elif method == 'exponential': # Exponential weighted moving average
        return pd.Series(data).ewm(span=window, adjust=False).mean()
    elif method == 'savgol': # Savitzky-Golay filter
        if len(data) < window:
            return data
        return savgol_filter(data, window, poly_order)
    else:
        return data

def plot_columns(csv_files, columns_to_plot=None, figsize=(12, 6), style='seaborn', 
                smoothing=None, window=5, poly_order=2, show_original=False, 
                fig=None, ax=None, labels=None, use_runname_as_legend=False,
                x_axis_label='Not set', y_axis_label='Not set',timesteps_limit=None):
    """
    Plot single or multiple columns from multiple CSV files with optional smoothing.
    
    Parameters:
    -----------
    csv_files : str or list
        Single CSV file path or list of CSV file paths
    labels : list or None
        List of labels for each CSV file (defaults to filenames if None)
    columns_to_plot : str or list
        Single column name or list of column names to plot
    figsize : tuple
        Figure size (width, height)
    style : str
        Matplotlib style to use
    smoothing : str or None
        Smoothing method: 'moving_average', 'exponential', 'savgol', or None
    window : int
        Window size for smoothing
    poly_order : int
        Polynomial order for Savitzky-Golay filter
    show_original : bool
        If True, show both original and smoothed data
    fig : matplotlib.figure.Figure
        Existing figure to use
    ax : matplotlib.axes.Axes
        Existing axes to use
    use_runname_as_legend : bool
        If True, use runname as legend instead of labels
    x_axis_label : str
        Label for the x-axis
    y_axis_label : str
        Label for the y-axis
    """
    # Convert single file to list for consistent processing
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    
    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Handle labels
    if labels is None:
        labels = [os.path.basename(os.path.dirname(f)) for f in csv_files]
    
    # Show available columns from first file if none specified
    if columns_to_plot is None:
        df = pd.read_csv(csv_files[0])
        print("Available columns:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        return fig, ax
    
    # Convert single column to list
    if isinstance(columns_to_plot, str):
        columns_to_plot = [columns_to_plot]
    
    # Plot each column for each file
    for file_idx, csv_file in enumerate(csv_files):
        if timesteps_limit is not None:
            df = pd.read_csv(csv_file)[:timesteps_limit]
        else:
            df = pd.read_csv(csv_file)
        timesteps = range(len(df))
        
        for column in columns_to_plot:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in CSV file {csv_file}")
                continue
            
            # Modify labels to include both column and file name
            if use_runname_as_legend:
                column_label = f'{column} ({os.path.basename(os.path.dirname(csv_file))})'
            else:
                # column_label = f'{column} ({labels[file_idx]})'
                column_label = f'{labels[file_idx]}'
            
            if show_original or not smoothing:
                ax.plot(timesteps, df[column], 
                       label=f'{column_label} (original)', 
                       alpha=0 if smoothing else 1.0,
                       marker='o' if not smoothing else None,
                       markersize=4)
            
            if smoothing:
                smoothed_data = smooth_data(df[column], method=smoothing,
                                          window=window, poly_order=poly_order)
                ax.plot(timesteps, smoothed_data,
                    #    label=f'{column_label} ({smoothing})',
                       label=f'{column_label}',
                       linewidth=2)

    # Customize the plot
    # make the x-axis bold 
    ax.tick_params(axis='both', which='major', labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')


    
    ax.set_xlabel(x_axis_label, fontweight='bold',fontsize=12)
    # plt.ylabel("Average Head Immersion % (target=85%)", fontweight='bold', fontsize=12)
    ax.set_ylabel(y_axis_label, fontweight='bold',fontsize=12)
    title = 'Column Values Over Time'
    if smoothing:
        title += f' ({smoothing.replace("_", " ").title()} Smoothing)'
    # ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # put the legend inside the plot 
    ax.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

def plot_related_columns(csv_file, pattern, figsize=(12, 6), style='seaborn', 
                        smoothing=None, window=5, poly_order=2, show_original=False,
                        x_axis_label='Timestep', y_axis_label='Value'):
    """
    Plot all columns that match a certain pattern with optional smoothing.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    pattern : str
        Pattern to match in column names (e.g., 'MSP' or 'Head')
    figsize : tuple
        Figure size (width, height)
    style : str
        Matplotlib style to use
    smoothing : str or None
        Smoothing method: 'moving_average', 'exponential', 'savgol', or None
    window : int
        Window size for smoothing
    poly_order : int
        Polynomial order for Savitzky-Golay filter
    show_original : bool
        If True, show both original and smoothed data
    x_axis_label : str
        Label for the x-axis
    y_axis_label : str
        Label for the y-axis
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Find matching columns
    matching_columns = [col for col in df.columns if pattern in col]
    
    if not matching_columns:
        print(f"No columns found matching pattern '{pattern}'")
        return
    
    # Plot the matching columns
    return plot_columns(csv_file, matching_columns, figsize, style, 
                       smoothing, window, poly_order, show_original,
                       x_axis_label=x_axis_label, y_axis_label=y_axis_label)

# Example usage:
if __name__ == "__main__":
    # Add these imports at the top of the main block
    matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot

    def plot_in_thread(csv_files, columns, smoothing, window, show_original=False, labels=None, use_runname_as_legend=False,
                       x_axis_label='Timestep', y_axis_label='Value',timesteps_limit=None):
        try:
            with plt.style.context('seaborn-v0_8'):
                print(f"Using {labels} as labels")
                fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
                fig, ax = plot_columns(csv_files, columns, fig=fig, ax=ax, 
                                     smoothing=smoothing, window=window, 
                                     show_original=show_original, labels=labels, use_runname_as_legend=use_runname_as_legend,
                                     x_axis_label=x_axis_label, y_axis_label=y_axis_label,timesteps_limit=timesteps_limit)
                
                # Use the first runname for the directory structure
                runname = os.path.basename(os.path.dirname(csv_files[0] if isinstance(csv_files, list) else csv_files))
                plot_dir = f'results/comparison_plots'
                # create plot dir if not exists
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir, exist_ok=True)
                
                if isinstance(columns, list):
                    filename = '_'.join(columns)
                else:
                    filename = str(columns)
                current_time_hour_min_sec = datetime.now().strftime("%H_%M_%S")
                #create directory current_time_hour_min
                os.makedirs(f'{plot_dir}/{current_time_hour_min_sec}', exist_ok=True)
                filepath = f'{plot_dir}/{current_time_hour_min_sec}/plot_{filename}_{smoothing}_comparison.png'
                print(f"@{__name__}, Info: saving to", filepath)
                # plt.show()
                # input("Press Enter to continue...")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Error plotting {columns}: {str(e)}")

    # Example of plotting from multiple files
    num_requests = 10
    runnames = [f'msps_1_requests_{num_requests}_gamma_0.97_after_changing_and_immediate_avg',f'msps_2_requests_{num_requests}_gamma_0.97_after_changing_and_immediate_avg',f'msps_3_requests_{num_requests}_gamma_0.97_after_changing_and_immediate_avg',f'msps_4_requests_{num_requests}_gamma_0.97_after_changing_and_immediate_avg',f'msps_5_requests_{num_requests}_gamma_0.97_after_changing_and_immediate_avg']
    
    custom_labels = [f'msps=1,R={num_requests}',f'msps=2,R={num_requests}',f'msps=3,R={num_requests}',f'msps=4,R={num_requests}',f'msps=5,R={num_requests}']
    csv_files = [f'/Users/mac/Documents/Manual Library/Cooperative_game/results/{run}/summary_normalized.csv' 
                 for run in runnames]

    threads = []
    limit_timesteps = 10000
    # Example of plotting multiple files
    threads.append(threading.Thread(
        target=plot_in_thread,
        # args=(csv_files, 'total_reward', 'moving_average', 50, False, runnames, True)
        args=(csv_files, # csv_files
            #   'total_reward', # columns
              'normalized_total_reward', # columns
            #   'msp_q_0', # columns
            #   None,
              'moving_average', # smoothing
              1000, # window
              False, # show_original
              custom_labels, # labels
              False, # use_runname_as_legend
              'Training Episode', # x_axis_label
              'Total Reward (%)' # y_axis_label
              ,limit_timesteps # timesteps_limit
              )
    ))
    threads.append(threading.Thread(
        target=plot_in_thread,
        # args=(csv_files, 'total_reward', 'moving_average', 50, False, runnames, True)
        args=(csv_files, # csv_files
              'num_requests_fulfilled', # columns
            #   'msp_q_1', # columns
            #   None,
              'moving_average', # smoothing
              1000, # window
              False, # show_original
              custom_labels, # labels
              False, # use_runname_as_legend
              'Training Episode', # x_axis_label
              'Number of Requests Fulfilled' # y_axis_label,
              ,limit_timesteps # timesteps_limit
              )
    ))

    threads.append(threading.Thread(
        target=plot_in_thread,
        # args=(csv_files, 'total_reward', 'moving_average', 50, False, runnames, True)
        args=(csv_files, # csv_files
              'avg_head_imrvnss_alive', # columns
            #   'msp_q_1', # columns
            #   None,
              'moving_average', # smoothing
              1000, # window
              False, # show_original
              custom_labels, # labels
              False, # use_runname_as_legend
              'Training Episode', # x_axis_label
              'Immersion (x100%)' # y_axis_label,
              ,limit_timesteps # timesteps_limit
              )
    ))

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()