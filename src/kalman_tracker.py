import numpy as np

# code credit: https://www.geeksforgeeks.org/python/kalman-filter-in-python/

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F # state transition matrix (system model)
        self.B = B # control matrix (effect of control input)
        self.H = H # observation matrix (how we measure the state)
        self.Q = Q # process noise covariance (uncertainty in the process)
        self.R = R # measurement noise covariance (uncertainty in the measurements)
        self.x = x0.copy() # initial state estimate
        self.P = P0.copy() # initial error covariance (initial uncertainty of state estimate)

    def predict(self, u=None): 
        ''' Predict next state '''
        if self.B is not None and u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x 
    
    def update(self, z): 
        ''' Update with a measurement z '''
        # innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # innovation (measurement minus prediction)
        y = z - (self.H @ self.x)

        # posterior mean
        self.x = self.x + K @ y

        # posterior covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x 
    

def make_bubble_kf(dt=1.0, process_pos=5.0, process_vel=1.0, process_r=0.5, meas_xy=3.0, meas_r=2.0, x0=(0,0,0,0,5)):
    '''
    Ex.: state x = [x, y, v_x, v_y, r]

    x, y: bubble center (pixels)
    v_x, v_y: bubble velocity (pixels/frame)
    r: bubble radius (pixels)
    '''
    # update x,y based on velocity
    F = np.array([[1,0,dt,0,0],
                  [0,1,0,dt,0],
                  [0,0,1, 0,0],
                  [0,0,0, 1,0],
                  [0,0,0, 0,1]], dtype=np.float32)
    
    # measure x, y and radius 
    H = np.array([[1,0,0,0,0],
                  [0,1,0,0,0],
                  [0,0,0,0,1]], dtype=np.float32)
    Q = np.diag([process_pos, process_pos, process_vel, process_vel, process_r]).astype(np.float32)
    R = np.diag([meas_xy, meas_xy, meas_r]).astype(np.float32)
    x0 = np.array(x0, dtype=np.float32).reshape(5,1)
    P0 = np.eye(5, dtype=np.float32)*10.0
    return KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, B=None)

def measurement_likelihood(kf, z):
    """
    Given a detection z (x,y,r), and a bubble's predicted state (mean + uncertainty),
    return how far away the detection is, statistically.
    """

    H = kf.H # measurement model
    R = kf.R # measurement noise
    x = kf.x # predicted state
    P = kf.P # predicted uncertainty

    # innovation = difference between predicted measurement and actual measurement
    y = z - (H @ x)

    # innovation covariance = how uncertain we expect that difference to be
    S = H @ P @ H.T + R

    # Mahalanobis distance:
    d2 = float(y.T @ np.linalg.inv(S) @ y)

    return d2







# maybe use bayesian statistic to see if it was a bubble then, is it still a bubble now

# waterline: run super low trheshold edge detector


# find way to evaluate performance, find wheer you have to manually labor 
# kalman filter? bayesian stats?