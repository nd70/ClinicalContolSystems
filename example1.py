import numpy as np
np.random.seed(7)
import scipy.linalg as sl
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')


#===============================================================================
#                                 Functions
#===============================================================================
def get_XofT(A, F, X0, time):
    """
    Solve:
        dX(t)/dt = A X(t) - F(t)
    for X(t) given the initial conditions X(0)

    Parameters
    ----------
    A : `numpy:ndarray`
        Matrix of rate constants. Shape (N x N)
    F : `numpy ndarray`
        Forcing function. Shape (I x T) where I is the
        number of inputs and T is the number of time steps
    X0 : `numpy.ndarray`
        Initial conditions. Shape (I,) where I is the
        number of inputs
    time : `numpy.ndarray`
        Times over which to evaluate the solution

    Returns
    -------
    XofT : `numpy.ndarray`
        X(t) for the differential equation
    """
    A_evals, A_evecs = np.linalg.eig(A)
    A_inv = np.linalg.inv(A)

    T = np.array(A_evecs)
    T_inv = np.linalg.inv(T)

    XofT = np.zeros((X0.size, time.size))
    for ix, t in enumerate(time):
        af = A_inv.dot(F[:, ix])
        XofT[:, ix] = -af + np.dot(T.dot(np.dot(np.diag(np.exp(A_evals*t)),
                                                T_inv)), X0 + af)

    return XofT


def get_XofT_v2(A, F, X0, time):
    """
    Solve:
        dX(t)/dt = A X(t) - F(t)
    for X(t) given the initial conditions X(0).

    This is similar to get_XofT() above but using matrix
    exponentiation. This is a bit cleaner, but ~25x slower.

    Parameters
    ----------
    A : `numpy:ndarray`
        Matrix of rate constants. Shape (N x N)
    F : `numpy ndarray`
        Forcing function. Shape (I x T) where I is the
        number of inputs and T is the number of time steps
    X0 : `numpy.ndarray`
        Initial conditions. Shape (I,) where I is the
        number of inputs
    time : `numpy.ndarray`
        Times over which to evaluate the solution

    Returns
    -------
    XofT : `numpy.ndarray`
        X(t) for the differential equation
    """
    A_inv = np.linalg.inv(A)
    XofT = np.zeros((X0.size, time.size))
    for ix, t in enumerate(time):
        af = A_inv.dot(F[:, ix])
        XofT[:, ix] = -af + np.dot(sl.expm(A*t), X0 + af)
    return XofT


def build_matrix(vals=None, alpha=None, mat='D'):
    """
    Construct the relevant matrices for the differential equations

    Parameters
    ----------
    vals : `numpy.ndarray`
        Either X(t_i) or k_i for 'D' or 'A' respectively
    alpha: `numpy.ndarray`
        values of alpha relating k_i to concentrations
    mat : `string`
        Matrix to return

    Returns
    -------
    mat : `numpy.ndarray`
        Filled in matrix
    """
    if mat=='D':
        M, B, P, C = vals
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = alpha
        mat = np.array([[((a1-a0)*M+a2*B)/a0, 0, -(a1*M + a2*B)/a6, 0],
                    [-(a1*M + a2*B)/a0, -(a3*B + a4*P)/a5,
                     (a1*M + (a2+a3-a6+a7)*B + a4*P +a8*C)/a6, -(a7*B+a8*C)/a9],
                    [0, (a3*B + (a4-a5)*P)/a5, -(a3*B + a4*P)/a6, 0],
                    [0, 0, -(a7*B + a8*C)/a6, (a7*B + (a8-a9)*C)/a9]])

    elif mat=='A':
        k0,k1,k2,k3,k4,k5,k6,k7,k8,k9 = vals
        mat = np.array([[-(k0+k1), k2, 0, 0],
                        [k1, -(k2+k3+k6+k7), k4, k8],
                        [0, k3, -(k4+k5), 0],
                        [0, k7, 0, -(k8+k9)]])
    return mat

def get_KofT(XofT, F, alphas, dt):
    """
    Solve K(t) = (D^T D)^{-1} D^T [dX/dt - F]

    Parameters
    ----------
    XofT : `numpy.ndarray`
        X(t) solution to the differential equation
    F : `numpy.ndarray`
        Forcing function
    alphas : `numpy.ndarray`
        values of alpha relating k_i to concentrations
    dt : `float`
        Time resolution (t[1]-t[0]). Used to compute the gradient
        of X(t)

    Returns
    -------
    K : `numpy.ndarray`
        Solution for K(t) as determined above
    """
    K = np.zeros((4, XofT.shape[-1]))
    dXdt = np.gradient(XofT, dt, axis=1)
    for ix in range(XofT.shape[-1]):
        D = build_matrix(vals=XofT[:, ix], alpha=alphas, mat='D')
        D_pinv = np.linalg.pinv(D)
        K[:, ix] = np.dot(D_pinv, (dXdt-F)[:, ix])

    return K

def determine_K(K, alpha):
    """
    Given the 4 independent k-values and alpha0,...alpha9,
    return the rest of the k-values

    Parameters
    ----------
    K : `numpy.ndarray`
        K(t) as found by get_KofT
    alphas : `numpy.ndarray`
        values of alpha relating k_i to concentrations

    Returns
    -------
    K : `numpy.ndarray`
        K(t) for k0,...k9
    """
    k0, k5, k6, k9 = K
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = alpha
    k1 = a1*(k6/a6 - k0/a0)
    k2 = a2*(k0/a0 - k6/a6)
    k3 = a3*(k5/a5 - k6/a6)
    k4 = a4*(k6/a6 - k5/a5)
    k7 = a7*(k9/a9 - k6/a6)
    k8 = a8*(k6/a6 - k9/a9)
    K = np.vstack((k0,k1,k2,k3,k4,k5,k6,k7,k8,k9))
    return K


def determine_alpha_from_k(rate_const, X0, epsilon=1e-6):
    """
    Given k0,...k9, and the initial conditions, determine
    alpha0,...alpha9

    Parameters
    ----------
    rate_const : `numpy.ndarray`
        all k_i constants
    X0 : `numpy.ndarray`
        Initial conditions
    epsilon : `float`
        Small non-zero number to avoid DivideByZero errors

    Returns
    -------
    alphas : `numpy.ndarray`
        values of alpha relating k_i to concentrations
    """
    k0,k1,k2,k3,k4,k5,k6,k7,k8,k9 = rate_const
    M, B, P, C = X0
    a0 = k0 / M
    a1 = k1 / (B - M + np.random.rand()*epsilon)
    a2 = k2 / (M - B + np.random.rand()*epsilon)
    a3 = k3 / (P - B + np.random.rand()*epsilon)
    a4 = k4 / (B - P + np.random.rand()*epsilon)
    a5 = k5 / P
    a6 = k6 / B
    a7 = k7 / (C - B + np.random.rand()*epsilon)
    a8 = k8 / (B - C + np.random.rand()*epsilon)
    a9 = k9 / C
    alphas = np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9])
    return alphas


def impulse_function(dim=4, dt=0.1, dur=10, method='sec'):
    """
    Construct the forcing function F(t) for all times

    Parameters
    ----------
    dim : `int`
        Number of differential equations to solve
    dt : `float`
        Time resolution
    dur : `float`
        Duration in seconds
    method : `str`
        Impulse pattern.
            'sec',  - the impulses will be applied once every second.
            'comb'  - the impulse is applied every other time-step.
            'const' - the impulse is applied at every time step.
            'delta' - the impulse is applied once at the beginning

    Returns
    -------
    F : `numpy.ndarray`
        F(t) forcing function
    """
    F = np.zeros((dim, int(dur/dt)))
    if method=='sec':
        inject = np.arange(0, int(dur/dt), int(1/dt))
        F[0, inject] = 1
    elif method=='comb':
        inject = np.arange(0, int(dur/dt), 2)
        F[0, inject] = 1
    elif method=='const':
        F[0, :] = 1
    elif method=='delta':
        F[0, 0] = 1
    return F

#===============================================================================
#                              Paper Example 1
#===============================================================================
# Set k values quoted in Example 1 in the paper
rate_const = np.array([1, 5, 0, 2, 0, 1, 1, 2, 0, 0.5])

# initial conditions
X0 = np.ones(4)

# construct 'A'
A = build_matrix(vals=rate_const, mat='A')

# given the rate constants, determine the alpha's
alphas = determine_alpha_from_k(rate_const, X0)

# set the duration and resolution
dur, dt = 10, 0.1
time = np.arange(0, dur, dt)

# get the forcing function
F = impulse_function(dim=X0.size, dt=dt, dur=dur, method='const')

# solve for X(t)
XofT = get_XofT(A, F, X0, time)

# solve for the four independent rate constants (k0, k5, k6, k9)
K2 = get_KofT(XofT, F, alphas, dt)

# use the four independent k and alphas to construct all rate constants
all_K = determine_K(K2, alphas)

# plot X(t)
fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(14,9))
ax.plot(time, XofT[0, :], label='[M]', lw=2.5)
ax.plot(time, XofT[1, :], label='[B]', lw=2.5)
ax.plot(time, XofT[2, :], label='[P]', lw=2.5)
ax.plot(time, XofT[3, :], label='[C]', lw=2.5)
ax.set_xlabel('Time [s]', fontsize=20)
ax.set_ylabel('Concentration', fontsize=20)
plt.legend(fontsize=18)
plt.title('Linear Case, $\mathbf{F}(t) = (1,0,0,0)$', fontsize=25)
plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('plots/test_XofT_paper_example1.png')
plt.close()

# plot first 9 of K(t) = (k_0(t),... k_9(t))^T.
# This is our recovery as a function of time, all k's are constant
fig, ax = plt.subplots(3, 3, sharex=True, sharey=False, figsize=(14,9))
count = 0
for ii in range(3):
    for jj in range(3):
        ax[ii][jj].axhline(rate_const[count], label='True $k_{}$'.format(count))
        ax[ii][jj].plot(time, all_K[count, :], color='orange',
                        label='$k_{}$'.format(count), ls='--')
        ax[ii][jj].set_title('$k_{}$'.format(count))
        ax[ii][jj].grid(True, which='both', ls='--', zorder=0)
        ax[ii][jj].legend()
        count+=1
plt.savefig('plots/test_KofT_paper_example1.png')
plt.close()
