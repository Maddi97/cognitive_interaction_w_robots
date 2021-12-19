import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('happy_results.csv')
mse = df.loc[df['Choice'] == 'network']['MSE']

plt.plot(mse)
plt.suptitle('MSE score Simulation run song = happy', fontsize=15)
plt.xlabel('iterations', fontsize=12)
plt.ylabel('mean squared error', fontsize=12)
plt.savefig('happy_mse.pdf')


#first = df.loc[: ,'Song'][:1500].value_counts().plot(kind='bar')
#last = df.loc[: ,'Song'][1500:].value_counts().plot(kind='bar')



plt.show()
