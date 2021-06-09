"""
Solution to the example given here:
https://en.wikipedia.org/wiki/Matrix_differential_equation
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
from scipy.integrate import solve_ivp


#===============================================================================
#                              Global Params
#===============================================================================
dur, dt = 2.0, 0.01
time = np.arange(0, dur, dt)
A = np.array([[3,-4],[4,-7]])
X0 = np.array([1,1])
F = np.array([0,0])

#===============================================================================
#                                 Functions
#===============================================================================
def get_XofT(A, F, X0, time):
    A_evals, A_evecs = np.linalg.eig(A)
    A_inv = np.linalg.inv(A)
    af = A_inv.dot(F)
    T = np.array(A_evecs)
    T_inv = np.linalg.inv(T)
    XofT = np.zeros((1,F.size, time.size))
    for ix, t in enumerate(time):
        XofT[0,:, ix] = -af + np.dot(T.dot(np.dot(np.diag(np.exp(A_evals*t)), T_inv)), X0 + af)
    return XofT


def get_KofT(XofT, F, A):
    K = np.zeros((1, XofT.shape[1], XofT.shape[-1]))
    for ix in range(XofT.shape[-1]):
        x, y = XofT[0, 0, ix], XofT[0, 1, ix]
        D = np.array([[x, -y], [-y, x-y]])
        D_pinv = np.linalg.pinv(D)
        K[0, :, ix] = np.dot(D_pinv.dot(A), XofT[0, :, ix])
    return K

def get_KofT_v2(XofT, F, A, dt):
    K = np.zeros((1, XofT.shape[1], XofT.shape[-1]))
    dXdt = np.gradient(XofT[0, :, :], dt, axis=1)
    for ix in range(XofT.shape[-1]):
        x, y = XofT[0, 0, ix], XofT[0, 1, ix]
        D = np.array([[x, -y], [-y, x-y]])
        D_pinv = np.linalg.pinv(D)
        K[0, :, ix] = np.dot(D_pinv, dXdt[:, ix]-F)
    return K


#===============================================================================
#                                 Run and Plot
#===============================================================================
XofT = get_XofT(A, F, X0, time)
KofT = get_KofT(XofT, F, A)
KofT_v2 = get_KofT_v2(XofT, F, A, dt)

fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
ax.plot(time, XofT[0, 0, :], label='x(t)', lw=2.5)
ax.plot(time, XofT[0, 1, :], label='y(t)', lw=2.5)
ax.plot(time, (2./3)*np.exp(time)+ (1./3)*np.exp(-5*time), label='x_true', ls='--')
ax.plot(time, (1./3)*np.exp(time)+ (2./3)*np.exp(-5*time), label='y_true', ls='--')
ax.set_xlabel('Time [s]', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('plots/wiki_XofT.png')
plt.close()

fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
ax.plot(time, KofT[0, 0, :], label='v1 x', lw=2.5)
ax.plot(time, KofT[0, 1, :], label='v1 y', lw=2.5)
ax.plot(time, KofT_v2[0, 0, :], label='v2 x', lw=2.5, ls='--')
ax.plot(time, KofT_v2[0, 1, :], label='v2 y', lw=2.5, ls='--')
ax.set_xlabel('Time [s]', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('plots/wiki_KofT.png')
plt.close()
