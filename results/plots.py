import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


category='happy'
df = pd.read_csv('{}_results111.csv'.format(category))

def mse_plot(df):
    mse = df.loc[df['Choice'] == 'network']['MSE']

    plt.plot(mse)
    plt.suptitle('MSE score in RL simulation for song category = {}'.format(category), fontsize=15)
    plt.xlabel('iterations', fontsize=12)
    plt.ylabel('mean squared error', fontsize=12)
    plt.savefig('{}_mse.jpg'.format(category))

def epsilon_plot(df):
    mse = df['Epsilon']

    plt.plot(mse)
    plt.suptitle('Epsilon score in RL simulation for song category = {}'.format(category), fontsize=15)
    plt.xlabel('iterations', fontsize=12)
    plt.ylabel('epsilon', fontsize=12)
    plt.savefig('{}_epsilon.jpg'.format(category))


def q_value_plot(df):
    q = df.loc[df['Song'] == category].loc[df['Choice'] == 'network']['Q-Values'] #
    for entry in q:
        print(list(entry.strip()))
    # print(q)
    # plt.plot(q)
    # plt.show()

def choices_plot(df):
    plt.figure(figsize=(10,10))
    first = df.loc[: ,'Song'][:1500].value_counts()
    last = df.loc[: ,'Song'][1500:].value_counts()
    #print(first[0].shape)
    print(last.shape)
    first.plot(kind='bar', align='edge')
    last.plot(kind='bar', color='red', align='center')
   # lastlast = df.loc[: ,'Song'][2000:].value_counts().plot(kind='bar', align='edge', color='green')
    plt.ylim([0,1500])
    plt.legend(labels=['first 1500 iterations', 'last 1500 iterations'])
    plt.title('Distribution of picked song category,\n when category \'{}\' gives the highest reward'.format(category), fontsize=20)
    plt.savefig('{}_choices_distribution2.jpg'.format(category))

#epsilon_plot(df)
#choices_plot(df)
mse_plot(df)
