import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import pandas as pd

my_parser = argparse.ArgumentParser(description='List the content of a folder')

######################### Add the arguments #########################

# Train or test
my_parser.add_argument('--runname', type=str, required=True)
my_parser.add_argument('--window', type=int, default=1)
my_parser.add_argument('--limit', type=float, nargs='+', default=0)
my_parser.add_argument('--plot_many', type=bool, default=False)
my_parser.add_argument('--many_args', type=str, nargs='+', default=False)
my_parser.add_argument('--title', type=str, required=False)
my_parser.add_argument('--ycol', type=str, required=False)
my_parser.add_argument('--ycols', type=str, nargs='+', required=False)
my_parser.add_argument('--filename', type=str, required=False, default='summary.csv')
my_parser.add_argument('--save', type=bool, required=False, default=False)

args = my_parser.parse_args()


#Plot parameters
RUNNAME = args.runname
WINDOW = args.window
LIMIT = args.limit
PLOT_MANY = args.plot_many
many_args = args.many_args
Y_AXIS = args.ycol
Y_COLS = args.ycols
FILENAME = args.filename

command_text = ' '.join(sys.argv)
# print(command_text)
os.system(f"echo {command_text} >> ./results/{RUNNAME}/hist_command.txt")
# TITLE = args.title


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(df, x_axis, y_axis, window=1, title='Learning Curve', ylimit=None, xlimit=None, islog=False,
                 save=False, legend=""):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = df[x_axis], df[y_axis]
    y = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    font = {'family': 'normal',
            'weight': 'bold',
            'size': MEDIUM_SIZE}

    # plt.rc('font', )  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.xlabel('x-Axis', fontweight='bold')
    plt.ylabel('y-Axis', fontweight='bold')
    plt.plot(x, y)

    plt.grid(axis='y', which='both')
    if ylimit != None:
        plt.ylim(ylimit)
    if xlimit != None:
        plt.xlim(xlimit)
    if islog:
        plt.yscale(value='log')
    if x_axis == "timestep":
        x_axis = "Episode"
    plt.xlabel(x_axis)
    if y_axis == "battery%":
        y_axis = "Battery Left %"

    plt.ylabel(y_axis)
    #plt.title(TITLE + " Smoothed")
    #plt.title(title + " Smoothed")
    # plt.legend([legend])

    plt.ion()

    # if (save):
    #     random_number = np.random.randint(0,500)
    #     current_time = datetime.now().strftime("%H:%M:%S")
    #     os.makedirs(f'./results/{RUNNAME}/PLOTS/{current_time}/{y_axis}y/', exist_ok=True)
    #     plt.savefig(f'./results/{RUNNAME}/PLOTS/{current_time}/{y_axis}y/pic{random_number}.png')
    plt.show(block=False)
    return plt
    #input("Press [enter] to continue.")
    #plt.savefig('books_read.png')


def plot_many(x_axis, y_axis, ylimit, **kwargs):
    PLOT_KEYS = kwargs.keys()
    index = 0
    max_y = 0
    min_y = 0
    for runname in PLOT_KEYS:

        if (index == 0):
            max_y = max(kwargs.get(runname)[y_axis])
            min_y = min(kwargs.get(runname)[y_axis])
        if max(kwargs.get(runname)[y_axis]) > max_y:
            max_y = max(kwargs.get(runname)[y_axis])
        elif min(kwargs.get(runname)[y_axis]) < min_y:
            min_y = min(kwargs.get(runname)[y_axis])

        plot_results(kwargs.get(runname), x_axis=x_axis, y_axis=y_axis, ylimit=ylimit, window=WINDOW)
    plt.legend(PLOT_KEYS)
    plt.ylim((min_y - 1, max_y + 1))
    plt.grid()
    input("ok")


#%%

df = pd.read_csv(f"./results/{RUNNAME}/{FILENAME}")
df['timestep'] = [i for i in range(len(df.values))]
# df['battery%'] = df['battery%']*100

if PLOT_MANY:
    if LIMIT != 0:
        mydict = {}
        df = None
        for arg in many_args:
            df = pd.read_csv(f"./results/{arg}/{FILENAME}")
            df['timestep'] = [i for i in range(len(df.values))]
            # df['battery%'] = df['battery%'] * 100
            mydict[arg] = df

        if Y_AXIS == "all":
            for i in df.columns:
                plot_many(x_axis='timestep', y_axis=i, ylimit=LIMIT, **mydict)
        else:
            plot_many(x_axis='timestep', y_axis=Y_AXIS, ylimit=LIMIT, **mydict)
        # plot_many(x_axis='timestep', y_axis='averag_delay', ylimit=LIMIT, **mydict)
        # plot_many(x_axis='timestep', y_axis='cpu_utilization', ylimit=LIMIT, **mydict)

        # plot_results(df=df, x_axis='timestep', y_axis='reward', window=WINDOW,ylimit=LIMIT)
    else:

        mydict = {}
        df = None
        for arg in many_args:
            df = pd.read_csv(f"./results/{arg}/{FILENAME}")
            df['timestep'] = [i for i in range(len(df.values))]
            # df['battery%'] = df['battery%'] * 100
            mydict[arg] = df

        #print(mydict)
        if Y_AXIS == "all":
            for i in df.columns:
                pid = os.fork()  #forking children to plot
                if pid == 0:
                    plot_many(x_axis='timestep', y_axis=i, ylimit=LIMIT, **mydict)
                    exit(0)

            for i in df.columns:
                os.wait()  #waiting for all chilkdren to finish executing
        else:
            plot_many(x_axis='timestep', y_axis=Y_AXIS, ylimit=LIMIT, **mydict)
        # plot_many(x_axis='timestep', y_axis='cpu_utilization', ylimit=LIMIT, **mydict)
        # plot_many(x_axis='timestep', y_axis='averag_delay',ylimit=LIMIT, **mydict)
        # plot_many(x_axis='timestep', y_axis='cpu_utilization', ylimit=LIMIT, **mydict)

else:
    if LIMIT != 0:
        plot_results(df=df, x_axis='timestep', y_axis=Y_AXIS, window=WINDOW, ylimit=LIMIT)
        input("here")
        #plot_results(df=df, x_axis='timestep', y_axis='reward', window=WINDOW,ylimit=LIMIT)
    else:
        if Y_COLS is not None:
            conc = []
            for ycol in args.ycols:
                conc.append(ycol)
                # plot_results(df=df, x_axis='timestep', y_axis=Y_AXIS, window=WINDOW)
                plt.grid()
                y = plot_results(df=df, x_axis='timestep', y_axis=ycol, window=WINDOW,save=True,islog=False)
                plt.legend(conc)
                plt.tight_layout()
                random_number = np.random.randint(0, 500)
                current_time = datetime.now().strftime("%H:%M:%S")
                os.makedirs(f'./results/{RUNNAME}/PLOTS/{current_time}/{ycol}y/', exist_ok=True)
                plt.savefig(f'./results/{RUNNAME}/PLOTS/{current_time}/{ycol}y/pic{random_number}.png')

            #plt.legend([y for y in args.ycols])
        else:
            plot_results(df=df, x_axis='timestep', y_axis=Y_AXIS, window=WINDOW)

            plt.legend(["Total Util","Head Util","MSP Util"])
        input("Press any key to continue")

    #plot_results(df=df, x_axis='timestep', y_axis='reward', window=WINDOW,)
    #input("Press [enter] to continue.")

#Example
#python3 plotscript.py --runname myRUN95


#Plotting multiple plots on the same grid
#python3 plotscript.py --runname myRUN99 --plot_many True --many_args myRUN90 myRUN99 myRUN95 --limit 80 90 --window 5

#plotscript.py --runname myRUN99 --plot_many True --many_args myRUN90 myRUN99 myRUN97 myRUN95 --limit 80 90 --window 5


# python3 plotscript.py --runname ondemand_days_ahead --plot_many True --many_args seven_day_ahead six_days_ahead five_days_ahead four_days_ahead three_days_ahead two_days_ahead  one_days_ahead --window 100 --ycol Flow1_reward