import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import library as lib


#===============================================================================
#                Paper Example 1 with Different Forcing Functions
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
dur, dt = 14, 0.01
time = np.arange(0, dur, dt)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(14,14))

F, tt = lib.impulse_function(dt=dt, dur=dur, forcing=[1]*dur)
XofT = lib.get_XofT(A, F, X0, tt)
ax[0].plot(time, XofT[0, :], lw=2.5, label='[M]')
ax[0].plot(time, XofT[1, :], lw=2.5, label='[B]')
ax[0].plot(time, XofT[2, :], lw=2.5, label='[P]')
ax[0].plot(time, XofT[3, :], lw=2.5, label='[C]')
ax[0].axhline(2./6, color='k', ls='--', alpha=0.75, zorder=0)
ax[0].axhline(1./6, color='k', ls='--', alpha=0.75, zorder=0)
ax[0].axhline(4./6, color='k', ls='--', alpha=0.75, zorder=0)
ax[0].set_ylabel('Concentration', fontsize=20)
ax[0].set_ylim([0, 1.5])
ax[0].grid(True, which='both', ls='--', zorder=0)
ax[0].set_title('constant forcing function', fontsize=25)
ax[0].legend(fontsize=18)

F, tt = lib.impulse_function(dt=dt, dur=dur, forcing=[1]*7+[0]*7)
XofT = lib.get_XofT(A, F, X0, tt)
ax[1].plot(time, XofT[0, :], lw=2.5, label='[M]')
ax[1].plot(time, XofT[1, :], lw=2.5, label='[B]')
ax[1].plot(time, XofT[2, :], lw=2.5, label='[P]')
ax[1].plot(time, XofT[3, :], lw=2.5, label='[C]')
ax[1].set_ylabel('Concentration', fontsize=20)
ax[1].set_ylim([0, 1.5])
ax[1].grid(True, which='both', ls='--', zorder=0)
ax[1].set_title('on-off forcing function', fontsize=25)
ax[1].legend(fontsize=18)

F, tt = lib.impulse_function(dt=dt, dur=dur, forcing=[1,0]*7)
XofT = lib.get_XofT(A, F, X0, tt)
ax[2].plot(time, XofT[0, :], lw=2.5, label='[M]')
ax[2].plot(time, XofT[1, :], lw=2.5, label='[B]')
ax[2].plot(time, XofT[2, :], lw=2.5, label='[P]')
ax[2].plot(time, XofT[3, :], lw=2.5, label='[C]')
ax[2].set_xlabel('Time [s]', fontsize=20)
ax[2].set_ylabel('Concentration', fontsize=20)
ax[2].set_ylim([0, 1.5])
ax[2].grid(True, which='both', ls='--', zorder=0)
ax[2].set_title('intermittent forcing function', fontsize=25)
ax[2].legend(fontsize=18)

plt.grid(True, which='both', ls='--', zorder=0)
plt.savefig('plots/forcing_function.png')
plt.close()
