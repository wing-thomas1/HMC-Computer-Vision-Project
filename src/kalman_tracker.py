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

def make_kf(dt=1.0):
    # state: x, y, vx, vy
    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1,0],
                  [0,0,0,1]], dtype=np.float32)

    H = np.array([[1,0,0,0],
                  [0,1,0,0]], dtype=np.float32)

    Q = np.eye(4, dtype=np.float32) * 0.1
    R = np.eye(2, dtype=np.float32) * 5
    P0 = np.eye(4, dtype=np.float32) * 10

    x0 = np.zeros((4,1), dtype=np.float32)

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

class BubbleTrack:
    def __init__(self, kf, max_misses=10):
        self.kf = kf
        self.misses = 0
        self.max_misses = max_misses
        self.history = [] 

    def predict(self):
        x = self.kf.predict()
        self.history.append((float(x[0]), float(x[1])))
        return x
    
    def update(self, z):
        self.kf.update(z)
        self.misses = 0
    
    def miss(self):
        self.misses += 1
    
    def dead(self):
        return self.misses > self.max_misses


def pick_best_detection(kf, detections, threshold=25):
    """Return best detection based on statistical distance."""
    best = None
    best_d2 = float("inf")

    for (x, y, r) in detections:
        z = np.array([[x], [y], [r]], dtype=np.float32)
        d2 = measurement_likelihood(kf, z)

        if d2 < best_d2:
            best_d2 = d2
            best = (x, y, r)

    if best is not None and best_d2 < threshold:
        return best
    return None
