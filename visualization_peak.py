import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sum_levels(levels):
    l = np.array(levels)
    return 10*np.log10(np.sum(np.power(10,l/10)))

def leq(levels):
    l = np.array(levels)
    return 10*np.log10(np.mean(np.power(10,l/10)))

def plot_LA(df):
    figure = plt.figure(figsize=(25, 8))
    # plt.xlim(df['date'].iloc[0], df['date'].iloc[-1])
    # plt.plot(df['date'], df['LA'])
    plt.xlim(0, len(df['LA']))
    plt.plot(df['LA'])
    plt.title(f'LA Values over time')
    plt.xlabel('Time')
    plt.ylabel('LA')
    plt.show()

def plot_LA_night(df):
    # convert date column in datetime
    df['date'] = pd.to_datetime(df['date'])

    duration = df['date'].iloc[-1] - df['date'].iloc[0]
    duration = duration.total_seconds()
    print(f"Duration: {duration} seconds, {duration/60} minutes, {duration/3600} hours, {duration/3600/24} days")
    figure = plt.figure(figsize=(25, 8))

    # night period markings
    # start_date = df['date'].iloc[0].normalize()
    # end_date = df['date'].iloc[-1].normalize()
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    current_date = start_date
    while current_date <= end_date:
        night_start = current_date + pd.Timedelta(hours=20)
        night_end = current_date + pd.Timedelta(days=1, hours=7)
        plt.axvspan(night_start, night_end, color='grey', alpha=0.3, label='Night time' if current_date == start_date else "")
        
        plt.axvline(current_date, color='red', linestyle='--', linewidth=1)
        plt.axvline(current_date + pd.Timedelta(days=1), color='red', linestyle='--', linewidth=1)
        
        current_date += pd.Timedelta(days=1)
        

    plt.plot(df['date'], df['LA'])
    plt.xlim(df['date'].iloc[0], df['date'].iloc[-1])
    plt.title(f'LA Values over time')
    plt.xlabel('Time')
    plt.ylabel('LA')
    # set the legent outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_some_values(df):
    la = df['LA'].values

    # convert date to datetime and calculate how long the measurement lasted
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    max_la = np.max(la)
    min_la = np.min(la)
    median_la = np.median(la)
    std_la = np.std(la)
    mean_la = np.mean(la).round(2)  # Assuming you have a function leq() which computes mean differently
    # percentile90 = np.quantile(la, 0.90)

    # Print statistics
    print(f'Max: {max_la.round(2)}')
    print(f'Min: {min_la.round(2)}')
    print(f'Median: {median_la.round(2)}')
    print(f'Standard deviation: {std_la.round(2)}')
    # print(f'Percentile 90: {percentile90.round(2)}')

    # Rolling calculations
    window_size = 60  # window size in minutes if data is sampled every minute
    df['LA_mean'] = df['LA'].rolling(window=window_size, min_periods=1).mean()
    df['LA_median'] = df['LA'].rolling(window=window_size, min_periods=1).quantile(0.5)

    # Plotting
    plt.figure(figsize=(30, 7))
    plt.plot(df['date'], df['LA'], label='LA values')
    plt.plot(df['date'], df['LA_mean'], color='cyan', linestyle='-', linewidth=2, label=f'Leq Mean ({window_size} minutes)')
    plt.plot(df['date'], df['LA_median'], color='orange', linestyle='--', linewidth=2, label='Rolling Median (50th Percentile)')

    plt.axhline(max_la, color='r', linestyle='--', label='Max')
    plt.axhline(min_la, color='g', linestyle='--', label='Min')
    plt.axhline(mean_la, color='b', linestyle='--', label='Mean')
    # plt.axhline(percentile90, color='m', linestyle='--', label='90th Percentile')

    # Mark night time
    start_date = df['date'].iloc[0].normalize()
    end_date = df['date'].iloc[-1].normalize()
    current_date = start_date
    while current_date <= end_date:
        night_start = current_date + pd.Timedelta(hours=20)
        night_end = current_date + pd.Timedelta(days=1, hours=7)
        plt.axvspan(night_start, night_end, color='grey', alpha=0.3, label='Night time' if current_date == start_date else "")
        current_date += pd.Timedelta(days=1)

    plt.xlim(df['date'].iloc[0], df['date'].iloc[-1])
    plt.title('LA values over time with Statistics')
    plt.xlabel('Date')
    plt.ylabel('LA (dB)')
    # plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()