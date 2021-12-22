"""
Generation of a dataset and some other usefull informations which can be post processed with "post_processing_average.py"
to create avaraged dataset.
Data is stored in "trajectory" directory

"""
"""
ACADO -- controls: acc, deltarate
Path tracking simulation with iterative linear model predictive control for speed and steer control
author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cvxpy
import math
import numpy as np
import sys
import acado
import post_processing_tools

sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
    import curve
except:
    raise

# generation parameters
nb_dataset = 5
nb_trajectory = 60

NX = 5  # [x, y, v, yaw, delta]
NY = 6
NYN = 4
NU = 2  # [accel, deltarate]
T = 5  # horizon length
# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q  = np.diag([1.0, 1.0, 0.5, 1.0, 0.01, 0.01])  # state cost matrix
Qf = np.diag([1.0, 1.0, 0.5, 1.0])  # state cost matrix
#Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 1000.0  # max simulation time
# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param
TARGET_SPEED = 5.0 / 3.6  # [m/s] target speed    
N_IND_SEARCH = 10  # Search index number
DT = 0.1  # [s] time tick
# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 1.32  # [m]
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(45.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
show_animation = False

class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, d=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.delta = d

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi
    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi
    return angle

def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB
    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)
    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)
    return A, B, C

def update_state(state, a, deltarate, dt, predict):
    # input check
    if deltarate > MAX_DSTEER:
        deltarate = MAX_DSTEER
    elif deltarate < -MAX_DSTEER:
        deltarate = -MAX_DSTEER
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / WB * math.tan(state.delta) * dt
    state.v = state.v + a * dt
    state.delta = state.delta + deltarate * dt
    if state.delta > MAX_STEER:
        state.delta = MAX_STEER
    elif state.delta < -MAX_STEER:
        state.delta = -MAX_STEER
    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED
    return state

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind, mind

def predict_motion(x0, oa, od, xref):
    xbar = np.zeros((NX, T + 1))
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]
    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2], d=x0[4])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):       
        state = update_state(state, ai, di, DT, True)        
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw
        xbar[4, i] = state.delta
    return xbar

def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC contorl with updating operational point iteratively
    """
    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T
    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")
    return oa, od, ox, oy, oyaw, ov

# MPC using ACADO
def linear_mpc_control(xref, xbar, x0, dref):
    # see acado.c for parameter details
    _x0=np.zeros((1,NX))  
    X=np.zeros((T+1,NX))
    U=np.zeros((T,NU))    
    Y=np.zeros((T,NY))    
    yN=np.zeros((1,NYN))    
    _x0[0,:]=np.transpose(x0)  # initial state    
    for t in range(T):
      Y[t,:] = np.transpose(xref[:,t])  # reference state
      X[t,:] = np.transpose(xbar[:,t])  # predicted state
    X[-1,:] = X[-2,:]    
    yN[0,:]=Y[-1,:NYN]         # reference terminal state
    X, U = acado.mpc(0, 1, _x0, X,U,Y,yN, np.transpose(np.tile(Q,T)), Qf, 0)    
    ox = get_nparray_from_matrix(X[:,0])
    oy = get_nparray_from_matrix(X[:,1])
    ov = get_nparray_from_matrix(X[:,2])
    oyaw = get_nparray_from_matrix(X[:,3])
    oa = get_nparray_from_matrix(U[:,0])
    odelta = get_nparray_from_matrix(U[:,1])
    return oa, odelta, ox, oy, oyaw, ov

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NY, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)
    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)
    if pind >= ind:
        ind = pind
    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    xref[4, 0] = 0.0
    xref[5, 0] = 0.0
    dref[0, 0] = 0.0  # steer operational point should be 0
    travel = 0.0
    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            xref[4, i] = 0.0
            xref[5, i] = 0.0
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            xref[4, i] = 0.0
            xref[5, i] = 0.0
            dref[0, i] = 0.0
    return xref, ind, dref

def check_goal(state, goal, tind, nind):
    if ourCurve and state.x >= horizon:
        return True
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.sqrt(dx ** 2 + dy ** 2)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(state.v) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False

def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    """
    goal = [cx[-1], cy[-1]]
    tmpArrayStates = []
    state = initial_state
    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0
    time = 0.0
    nextPlotTime = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    vref = [0.0]
    yawref = [0.0]
    delta = [0.0]
    ex = [0.0]
    ey = [0.0]
    ldu = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    odelta, oa = None, None
    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)
        x0 = [state.x, state.y, state.v, state.yaw, state.delta]  # current state
        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)
        if odelta is not None:
            di, ai = odelta[0], oa[0]
        # warm-up solver
        if True: #target_ind < 10:
          if abs(state.v) < 0.01:
            if sp[target_ind]<0:
             ai = -0.01
            else:
             ai =  0.01
        npState = []
        for i in xref:
            npState = np.concatenate((npState,i))
        npState = np.concatenate((npState,x0))
        npOutput = [ai, 4*di/math.pi] # We normalise the output : a in [-1;1] and delta from [-pi/4;pi/4] tot [-1;1]
        arrayState.append([npState, npOutput])
        tmpArrayStates.append([npState, npOutput])
        state = update_state(state, ai, di, DT, False)
        time = time + DT
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)
        yawref.append(pi_2_pi(cyaw[target_ind]))
        vref.append(sp[target_ind])
        ex.append(state.x-cx[target_ind])
        ey.append(state.y-cy[target_ind])
        isgoal = check_goal(state, goal, target_ind, len(cx))
        if isgoal:
            print("Goal")
            break
    arrayTrajectoriesStates.append(tmpArrayStates)
    print('time over')
    plt.show()
    return t, x, y, yaw, v, d, a

def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward
    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        move_direction = math.atan2(dy, dx)
        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
    speed_profile[-1] = 0.0
    return speed_profile

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw

ourCurve = False
horizon = 200
def compute():
    print(__file__ + " start!!")
    dl = 1.0  # course tick
    ourCurve = True
    (xa, ya) = curve.createCurveRec(horizon)
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(xa,ya)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)
    global data
    data.append([x, y, yaw, v, len(arrayState)])
    return cx, cy, t, x, y, yaw, v, d, a

data = []
arrayState = []
tmpArrayStates = []
arrayTrajectoriesStates = []
if __name__ == '__main__':
    x0_store_array = []
    for j in range(nb_dataset):
        previousIndex = 0
        for i in range(nb_trajectory):
            print('Dataset ' + str(j) + ' ; Trajectory ' + str(i))
            cx, cy, t, x, y, yaw, v, d, a = compute()
            x0 = [x[0], y[0], yaw[0], v[0], 0]
            x0_store_array.append(x0)
        np.save(f'./post_processing_data/trajectory/training{j}_data.npy', np.array(data, dtype=object))
        np.save(f'./post_processing_data/trajectory/x0_array{j}_data.npy', np.array(x0_store_array, dtype=object))
        np.save(f'./post_processing_data/trajectory/arrayTrajectoriesStates{j}.npy',np.array(arrayTrajectoriesStates, dtype=object))
        # post_processing_tools.displayData(data, arrayState)
        x0_store_array = []
        arrayState = []
        data = []
        arrayTrajectoriesStates = []