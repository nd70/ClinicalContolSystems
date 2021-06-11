import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import library as lib


#===============================================================================
#                              Paper Example 1
#===============================================================================
# Set k values quoted in Example 1 in the paper
rate_const = np.array([1, 5, 0, 2, 0, 1, 1, 2, 0, 0.5])

# initial conditions
convergence_factor = np.random.rand(4)*1e-2
X0 = np.ones(4) + convergence_factor

# construct 'A'
A = lib.build_matrix(vals=rate_const, mat='A')

# given the rate constants, determine the alpha's
alphas = lib.determine_alpha_from_k(rate_const, X0)

# set the duration and resolution
dur, dt = 10, 0.1
time = np.arange(0, dur, dt)

# get the forcing function
F = lib.impulse_function(dim=X0.size, dt=dt, dur=dur, method='const')

# solve for X(t)
XofT = lib.get_XofT(A, F, X0, time)

# solve for the four independent rate constants (k0, k5, k6, k9)
independent_K = lib.get_KofT(XofT, F, alphas, dt)

# use the four independent k and alphas to construct all rate constants
all_K = lib.determine_K(independent_K, alphas)

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
plt.savefig('plots/XofT_paper_example1.png')
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
plt.savefig('plots/KofT_paper_example1.png')
plt.close()
