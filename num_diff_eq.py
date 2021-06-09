import numpy as np
np.random.seed(3301)
import scipy.linalg as sl
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')


#===============================================================================
#                             Linear Solution
#===============================================================================
k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 = np.random.rand(10)

# Rate constant matrix
A = np.array([[-(k0+k1), k2, 0, 0],
              [k1, -(k2+k3+k6+k7), k4, k8],
              [0, k3, -(k4+k5), 0],
              [0, k7, 0, -(k8+k9)]])

A_evals, A_evecs = np.linalg.eig(A)
A_inv = np.linalg.inv(A)
T = np.array(A_evecs)
T_inv = np.linalg.inv(T)
F = np.array([1, 0, 0, 0])

def get_XofT(A, F, X0, time):
    A_evals, A_evecs = np.linalg.eig(A)
    A_inv = np.linalg.inv(A)
    af = A_inv.dot(F)

    T = np.array(A_evecs)
    T_inv = np.linalg.inv(T)

    XofT = np.zeros((1,F.size, time.size))
    for ix, t in enumerate(time):
        # af = A_inv.dot(F[:,ix])
        XofT[0,:, ix] = -af + np.dot(T.dot(np.dot(np.diag(np.exp(A_evals*t)), T_inv)), X0 + af)

    return XofT

dur, res = 10, 0.1
time = np.arange(0, dur, res)
X0 = np.zeros(4)

XofT = get_XofT(A, F, X0, time)

fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
ax.plot(time, XofT[0, 0, :], label='[M]', lw=2.5)
ax.plot(time, XofT[0, 1, :], label='[B]', lw=2.5)
ax.plot(time, XofT[0, 2, :], label='[P]', lw=2.5)
ax.plot(time, XofT[0, 3, :], label='[C]', lw=2.5)
ax.set_xlabel('Time [s]', fontsize=20)
ax.set_ylabel('Concentration', fontsize=20)
plt.legend(fontsize=18)
plt.title('Concentration Evolution for Linear Case with Random $k_i$ Initialization', fontsize=22)
plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('XofT.png')
plt.close()
import sys; sys.exit()


#===============================================================================
#                              Paper Example 1
#===============================================================================
k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 = 1, 5, 0, 2, 0, 1, 1, 2, 0, 0.5

# Rate constant matrix
A = np.array([[-(k0+k1), k2, 0, 0],
              [k1, -(k2+k3+k6+k7), k4, k8],
              [0, k3, -(k4+k5), 0],
              [0, k7, 0, -(k8+k9)]])

X0 = np.ones(4)
dur, dt = 10, 0.1
time = np.arange(0, dur, dt)
F = np.zeros((4, time.size))
F[0, :] = 1


def get_XofT(A, F, X0, time):
    A_evals, A_evecs = np.linalg.eig(A)
    A_inv = np.linalg.inv(A)

    T = np.array(A_evecs)
    T_inv = np.linalg.inv(T)

    XofT = np.zeros((1,X0.size, time.size))
    for ix, t in enumerate(time):
        af = A_inv.dot(F[:, ix])
        XofT[0,:, ix] = -af + np.dot(T.dot(np.dot(np.diag(np.exp(A_evals*t)), T_inv)), X0 + af)

    return XofT


XofT = get_XofT(A, F, X0, time)
meas = np.array([1.01, 1.48, 1.77, 1.94, 2.05, 2.11])

fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
ax.plot(time, XofT[0, 0, :], label='[M]', lw=2.5)
ax.plot(time, XofT[0, 1, :], label='[B]', lw=2.5)
ax.plot(time, XofT[0, 2, :], label='[P]', lw=2.5)
ax.plot(time, XofT[0, 3, :], label='[C]', lw=2.5)
ax.plot(np.array([0, 1,2,3,4,5]), meas, 'k--', label='Paper [C]')
ax.set_xlabel('Time [s]', fontsize=20)
ax.set_ylabel('Concentration', fontsize=20)
plt.legend(fontsize=18)
plt.title('Linear Case, $\mathbf{F}(t) = (1,0,0,0)$', fontsize=25)
plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('XofT.png')
plt.close()


# Rate Matrix K(t)
def get_KofT(XofT, F, A):
    K = np.zeros((1, 10, XofT.shape[-1]))
    for ix in range(XofT.shape[-1]):
        M, B, P, C = XofT[0, 0, ix], XofT[0, 1, ix], XofT[0, 2, ix], XofT[0, 3, ix]
        D = np.array([[-M, -M, B, 0, 0, 0, 0, 0, 0, 0],
                      [0, M, -B, -B, P, 0, -B, -B, C, 0],
                      [0, 0, 0, B, -P, -P, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, B, -C, -C]])

        D_pinv = np.linalg.pinv(D)
        K[0, :, ix] = np.dot(D_pinv.dot(A), XofT[0, :, ix])

    return K


def get_KofT_v2(XofT, F, A, dt):
    K = np.zeros((1, 10, XofT.shape[-1]))
    dXdt = np.gradient(XofT[0, :, :], dt, axis=1)
    for ix in range(XofT.shape[-1]):
        M, B, P, C = XofT[0, 0, ix], XofT[0, 1, ix], XofT[0, 2, ix], XofT[0, 3, ix]
        D = np.array([[-M, -M, B, 0, 0, 0, 0, 0, 0, 0],
                      [0, M, -B, -B, P, 0, -B, -B, C, 0],
                      [0, 0, 0, B, -P, -P, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, B, -C, -C]])

        D_pinv = np.linalg.pinv(D)
        K[0, :, ix] = np.dot(D_pinv, (dXdt-F)[:, ix])

    return K


K = get_KofT(XofT, F, A)
K2 = get_KofT_v2(XofT, F, A, dt)
fig, ax = plt.subplots(3, 3, sharex=True, sharey=False, figsize=(14,9))
count = 0
for ii in range(3):
    for jj in range(3):
        ax[ii][jj].plot(time, K[0, count, :], 'C2-', label='$k_{}$'.format(count), lw=3, alpha=0.7)
        ax[ii][jj].plot(time, K2[0, count, :], label='$k\'_{}$'.format(count), ls='--')
        ax[ii][jj].set_title('$k_{}$'.format(count))
        ax[ii][jj].grid(True, which='both', ls='--', zorder=0)
        ax[ii][jj].legend()
        count+=1
plt.savefig('KofT.png')
plt.close()

##===============================================================================
##                              Wikipedia Example
##===============================================================================
#dur, dt = 2.0, 0.01
#time = np.arange(0, dur, dt)
#A = np.array([[3,-4],[4,-7]])
#X0 = np.array([1,1])
#F = np.array([0,0])

#def get_XofT(A, F, X0, time):
#    A_evals, A_evecs = np.linalg.eig(A)
#    A_inv = np.linalg.inv(A)
#    af = A_inv.dot(F)
#    T = np.array(A_evecs)
#    T_inv = np.linalg.inv(T)
#    XofT = np.zeros((1,F.size, time.size))
#    for ix, t in enumerate(time):
#        XofT[0,:, ix] = -af + np.dot(T.dot(np.dot(np.diag(np.exp(A_evals*t)), T_inv)), X0 + af)
#    return XofT

#XofT = get_XofT(A, F, X0, time)
#grad_x = np.gradient(XofT[0, :, :], dt, axis=1)
#plt.plot(time, XofT[0, 0, :], label='x(t)')
#plt.plot(time, grad_x[0, :], label='dx(t)/dt')
#plt.plot(time, (2./3)*np.exp(time)+ (-5./3)*np.exp(-5*time), label='true dx(t)/dt', ls='--')
#plt.legend()
#plt.savefig('compare_grad.png')
#plt.close()

## Rate Matrix K(t)
#def get_KofT(XofT, F, A):
#    K = np.zeros((1, XofT.shape[1], XofT.shape[-1]))
#    for ix in range(XofT.shape[-1]):
#        x, y = XofT[0, 0, ix], XofT[0, 1, ix]
#        D = np.array([[x, -y], [-y, x-y]])
#        D_pinv = np.linalg.pinv(D)
#        K[0, :, ix] = np.dot(D_pinv.dot(A), XofT[0, :, ix])
#    return K

#def get_KofT_v2(XofT, F, A, dt):
#    K = np.zeros((1, XofT.shape[1], XofT.shape[-1]))
#    dXdt = np.gradient(XofT[0, :, :], dt, axis=1)
#    for ix in range(XofT.shape[-1]):
#        x, y = XofT[0, 0, ix], XofT[0, 1, ix]
#        D = np.array([[x, -y], [-y, x-y]])
#        D_pinv = np.linalg.pinv(D)
#        K[0, :, ix] = np.dot(D_pinv, dXdt[:, ix]-F)
#    return K

#KofT = get_KofT(XofT, F, A)
#KofT_v2 = get_KofT_v2(XofT, F, A, dt)

#fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
#ax.plot(time, KofT[0, 0, :], label='v1 x', lw=2.5)
#ax.plot(time, KofT[0, 1, :], label='v1 y', lw=2.5)
#ax.plot(time, KofT_v2[0, 0, :], label='v2 x', lw=2.5, ls='--')
#ax.plot(time, KofT_v2[0, 1, :], label='v2 y', lw=2.5, ls='--')
## ax.plot(time, (2./3)*np.exp(time)+ (1./3)*np.exp(-5*time), label='x_true', ls='--')
## ax.plot(time, (1./3)*np.exp(time)+ (2./3)*np.exp(-5*time), label='y_true', ls='--')
#ax.set_xlabel('Time [s]', fontsize=20)
## ax.set_ylabel('Concentration', fontsize=20)
#plt.legend(fontsize=18)
#plt.grid(True, which='both', ls='--', zorder=0)
#plt.savefig('KofT.png')
#plt.close()


#fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
#ax.plot(time, XofT[0, 0, :], label='x(t)', lw=2.5)
#ax.plot(time, XofT[0, 1, :], label='y(t)', lw=2.5)
#ax.plot(time, (2./3)*np.exp(time)+ (1./3)*np.exp(-5*time), label='x_true', ls='--')
#ax.plot(time, (1./3)*np.exp(time)+ (2./3)*np.exp(-5*time), label='y_true', ls='--')
#ax.set_xlabel('Time [s]', fontsize=20)
#ax.set_ylabel('Concentration', fontsize=20)
#plt.legend(fontsize=18)
#plt.grid(True, which='both', ls='--', zorder=0)
#plt.savefig('XofT.png')
#plt.close()
