import numpy as np
from matplotlib import pyplot as plt
import model

NY = 6
T = 5  # horizon length
DT = 0.1  # [s] time tick
state = None

x, y, cx, cy, ea, ed, oa, od = [], [], [], [], [], [], [], []
set = np.load('data/sample1.npy', allow_pickle=True)
input = set[:,0]
output = set[:,1]
cx = set[:,2]
cy = set[:,3]

for p in range(len(input)):
    plt.cla()
    xref = np.zeros((NY, T + 1))
    for i in range(NY):
        for j in range(T+1):
            xref[i, j] = input[p][j*NY+i]
    x0 = input[p][36:41]
    if state is None :
        state = model.State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
    x.append(state.x)
    y.append(state.y)
    plt.plot(cx[:p], cy[:p], 'k', x, y, 'b', state.x, state.y, 'xb', xref[:, 0], xref[:, 1], 'xr')
    # model.plot_car(state.x, state.y, state.yaw, steer=output[p][1])
    state = model.update_state(state, output[p][0], output[p][1], DT)
    plt.pause(0.01)

