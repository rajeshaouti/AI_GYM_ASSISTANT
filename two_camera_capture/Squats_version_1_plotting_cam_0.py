import csv
import pandas as pd
import matplotlib.animation as animation
import random
from itertools import count
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')

index = count()

X_D = []
Y_D = []

# side_view, axis = plt.subplots(1)


def animate(i):
    data = pd.read_csv('Cam_0_data.csv')

    P = data['Point_no']
    X0 = data['B_X0']
    Y0 = 480 - data['B_Y0']
    S_Y11 = data['y11']

    plt.cla()
    # axis[0].plot(P, S_Y11, color='g', label="Dumbbell_Y_axis_position")
    # # axis[1].ylabel('distance')
    plt.xlabel('frames')
    plt.plot(P, Y0, color='g', label="Dumbell Position")
    plt.legend(loc='upper left')
    plt.tight_layout()


ani = animation.FuncAnimation(plt.gcf(), animate, interval=10)
plt.show()
