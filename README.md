# MPC Path Tracking

[Reference Code: PythonRobotics/PathTracking](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py)

[Reference Code: chhRobotics/Controllers/MPC](https://github.com/CHH3213/chhRobotics/blob/master/Controllers/MPC/mpc.ipynb)

## 1. How to run it

```sh
python3 model_predictive_speed_and_steer_control.py
```

## 2. How it works

* Step 1, input path data

```python
database = open(r'dict.eval_exp_path0.pickle','rb')
data = pickle.load(database)

path = np.array(data[b'364c19730f220279']['rst'][0])
ax = path[:,0]
ay = path[:,1]
```

* Step 2, using cubic spline to interpolate path trajectory

```python
# course tick
dl = 1.0

# input path waypoints from our data
# return interpolated x,y, yaw and curvature lists of interpolated path
# dl parameter means the distance of each point in the interpolated path
# we don't need to use the returned value 's', just forget about it.
cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
```

* Step 3, begin path tracking

```python
def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    """
    return t, x, y, yaw, v, d, a


database = open(r'dict.eval_exp_path0.pickle','rb')
data = pickle.load(database)
path = np.array(data[b'364c19730f220279']['rst'][0])
ax = path[:,0]
ay = path[:,1]

dl = 1.0
cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)

sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)
```

## 3. Now working on obstacle avoidance

* We need to import obstacle avoidance into cost function, the insert position lies in line 279 of [model_predictive_speed_and_steer_control.py](model_predictive_speed_and_steer_control.py)

  ```python
  cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
  ```

  I have tried insert obstacle penalty like this:

  ```python
  cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)
  for i in range(obstacles.shape[1]):
      obstacle = [obstacles[0,i], obstacles[1,i], obstacles[2,i], obstacles[3,i]]
      cost += cvxpy.quad_form(x[:, t] - obstacle, Obs)
  ```

  The result is, the car can detect the obstacle, but sometimes it directly stopped from start, sometimes it stopped when getting close to the obstacle, and sometimes it passes the obstacle instead of avoiding collisions with the obstacle.</br>

  This part still needs to be improved.

