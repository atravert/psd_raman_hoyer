"""
Analysis of Raman MES data from Hoyer et al. (2024)
"""
#%%
import numpy as np
import spectrochempy as scp
import utilities as u
from importlib import reload

# read raw raman data
D_1 = u.read_raman_raw('data/Fig_2-a-c_raw-data.txt')
D_1.y -= D_1.y[0] 
D_1.plot(title='raw spectra')
scp.show()

# remove spikes between spectra
# uncomment to plot the 1st spectrum (to check that there is no spike)
# D_1[0].plot()
# scp.show()

D_2 = D_1.copy()
diff = D_1[1:] - D_1[:-1]
spectrum, spike = np.where(diff.data > 100)
for spec_idx, spike_idx in zip(spectrum, spike):
    D_2.data[spec_idx+1, spike_idx] = 0.5 * (D_1.data[spec_idx+2, spike_idx] + D_1.data[spec_idx, spike_idx])

D_2.plot(title='raw spectra, despiked')
scp.show()

# %%
#  statistic analysis in the time domain: PCA

pca = scp.PCA(n_components=5)
pca.fit(D_2)

_ = pca.plot_scree()
print(pca)
# interpretation of the screeplot: the 1st component describes 11.5% of the variance. The next ones are of the same order of magnitude and are mostly noise  

# Assuming a single reaction the the loading is the unique reaction spectrum rt and the score is proportional to the extent of reaction x. The sigtns in PCA
# is arbitrary: I multiply by -1.   

rt = - pca.loadings[0]
x = - pca.scores[:,0]

rt.plot()
x.plot()

# the scores show a clear trend of a decreasing (overall) signal with superimposed oscillations.
#  The oscillations reflect periodic perturbations, while the global downward trend suggests a slower evolution over time.

# compute the variance of x within each cycle 
data = np.array(x.data).reshape(N, n)   # (cycles, points_per_cycle)

var_per_cycle = data.var(axis=1)     

# %%
# statistic analysis in the time domain: variance within each cycle.

# first, reshape to get a 3 way tensor (periods x spectra in each period x wavenumbers)
N = 41 # cycles
n = 60 # spectra per period
D_3way = scp.NDDataset(D_2.data.reshape(N, n, D_2.shape[1]))
times = D_2.y.data.reshape(N, n) 
ave_rel_times = np.mean(times - np.expand_dims(times[:,0], axis=1), axis=0)
D_3way.units = D_2.units
D_3way.title = D_2.title
D_3way.set_coordset(z=scp.Coord(np.arange(1, N+1, 1), title='period', units=None),
                        y=scp.Coord(ave_rel_times, title='time', units='s'),
                        x=D_1.x)

# Next subtract from each cycle the cycle-averaged spectrum. 
D_3way_centered_percycle = D_3way.copy()

cycle_mean = D_3way.data.mean(axis=1)          # shape: (N cycles, wavenumber)
D_3way_centered_percycle.data = D_3way.data - cycle_mean[:, np.newaxis, :]

# compute the variance in each cycle
var_per_cycle = D_3way_centered_percycle[:,:,400.:800.].data.var(axis=1)  # shape (N cycles, wavenumber) 
var_scalar_per_cycle = var_per_cycle.mean(axis=1) # shape (N cycles, wavenumber)


# %%
cycles = np.arange(1, N + 1)
plt.plot(cycles, var_scalar_per_cycle, "o-")
plt.xlabel("cycle")
plt.ylabel("mean variance")
plt.ylim(250, 350)
plt.show()

# %%
# fit: var = a * cycle + b
slope, intercept = np.polyfit(cycles, var_scalar_per_cycle, 1)
print("slope =", slope)

# %%
# plot the first and last cycles
D_3way_centered_percycle[1,:,400.:800.].plot()
D_3way_centered_percycle[-1,:,400.:800.].plot()
# %%
# Analyse 
# %% 
# carry out the PSD 

psd =  scp.PSD()  # inituialize the object with defaults (k = 1, matrix product) 
psd.fit(D_3way)   # fit
 

_ = psd.prs.plot()    # plot the phase resolved sopectra (prs attribute)


# %%


# %%
# noise analysis 
reload(u)
D_m_ = D_m[:, 800.:900.]
for D_ in [D_m_, D_m_[::2], D_m_[::3], D_m_[::4], D_m_[::5], D_m_[::6]]:
    print(f'n={D_.shape[0]}, eta_PSD={u.noise_reduction(D_)}')

D_m_ = D_m[:, 800.:900.]
for D_ in [D_m_, D_m_[::2], D_m_[::3], D_m_[::4], D_m_[::5], D_m_[::6]]:
    print(f'n={D_.shape[0]}, eta_PSD={u.noise_reduction(D_, k=3)}')
# %%
# PARAFAC decomposition on difference spectra (ref = first spectrum of the series)

D_3way_d_1 = D_3way[1:] - D_3way[1,0,:]
D_3way_d_1.name = 'diff, ref=1st spectrum of whole series'
reload(u)
pl = u.PerspectiveAxesSpectraPlot(D_3way_d_1)
pl.plot(mode="lines", cmap='viridis' ) # cmap='coolwarm'  
# %%

for rank in range(1, 3):
    cp = u.CP(n_components=int(rank))
    cp.fit(D_3way_d_1)

    if rank == 1:
        SSE_1 = cp.SSE
        print(
            f'rank={rank} -- LOSS={SSE_1 / SSE_1:.2f} --  EV={cp.explained_variance:.2f} -- CC={cp.core_consistency:.2f}')
    else:
        print(
            f'rank={rank} -- LOSS={cp.SSE / SSE_1:.2f} --  EV={cp.explained_variance:.2f}  -- CC={cp.core_consistency:.2f}')


cp.n_components = 1
cp.fit(D_3way_d_1)

for load in (cp.A, cp.B, cp.C):
        load.T.plot(title=load.name)
        scp.show()
        
    # %%
# CANDECOMP/PARAFAC (CP) decomposition on diff with average initial spectra (ref = t0)
D_3way_d_2 = D_3way[1:] - D_3way[1:,0,:]
D_3way_d_2.name = 'diff, ref= 1st spectra of each period'

for rank in range(1, 4):
    cp = u.CP(n_components=int(rank))
    cp.fit(D_3way_d_2)

    if rank == 1:
        SSE_1 = cp.SSE
        print(
            f'rank={rank} -- LOSS={SSE_1 / SSE_1:.2f} --  EV={cp.explained_variance:.2f} -- CC={cp.core_consistency:.2f}')
    else:
        print(
            f'rank={rank} -- LOSS={cp.SSE / SSE_1:.2f} --  EV={cp.explained_variance:.2f}  -- CC={cp.core_consistency:.2f}')


cp.n_components = 2
cp.fit(D_3way_d_2)

for load in (cp.A, cp.B, cp.C):
    load.T.plot(title=load.name)
    scp.show()


# %%
# CANDECOMP/PARAFAC (CP) decomposition on diff with average initial spectra (ref = t0)
D_3way_d_3 = D_3way[1:] - np.mean(D_3way[1:,0,:].data, axis=0)
D_3way_d_3.name = 'diff, ref=averaged 1st spectrum of each period'

for rank in range(1, 3):
    cp = u.CP(n_components=int(rank))
    cp.fit(D_3way_d_3)

    if rank == 1:
        SSE_1 = cp.SSE
        print(
            f'rank={rank} -- LOSS={SSE_1 / SSE_1:.2f} --  EV={cp.explained_variance:.2f} -- CC={cp.core_consistency:.2f}')
    else:
        print(
            f'rank={rank} -- LOSS={cp.SSE / SSE_1:.2f} --  EV={cp.explained_variance:.2f}  -- CC={cp.core_consistency:.2f}')

cp.n_components = 1
cp.fit(D_3way_d_3)

for load in (-cp.A, cp.B, -cp.C):
    load.T.plot(title=load.name)
    scp.show()

# ==> suggests a single component

# %%
############################################################################################################
# read raw raman data, pure CeO2
D_3 = u.read_hess_raman_raw('data/hess/Fig_2-b-d_raw-data.txt')
D_3.y -= D_3.y[0] 
D_3.plot()
scp.show()


# remove spikes between spectra
# uncomment to plot the 1st spectrum (to check that there is no spike)
D_3[0].plot()
scp.show()

D_4 = D_3.copy()
diff = D_3[1:] - D_3[:-1]
spectrum, spike = np.where(diff.data > 120)
for spec_idx, spike_idx in zip(spectrum, spike):
    D_4.data[spec_idx+1, spike_idx] = 0.5 * (D_3.data[spec_idx+2, spike_idx] + D_3.data[spec_idx, spike_idx])

D_4.plot()
scp.show()

# %%
# reshape to get a 3 way tensor (periods x spectra in each period x wavenumbers)
N = 41 # periods
n = 60 # spectra per period
D_3way_Ce = scp.NDDataset(D_4.data.reshape(N, n, D_4.shape[1]))
times = D_4.y.data.reshape(N, n) 
ave_rel_times = np.mean(times - np.expand_dims(times[:,0], axis=1), axis=0)
D_3way_Ce.units = D_4.units
D_3way_Ce.title = D_4.title
D_3way_Ce.set_coordset(z=scp.Coord(np.arange(1, N+1, 1), title='period', units=None),
                        y=scp.Coord(ave_rel_times, title='time', units='s'),
                        x=D_4.x)

# average over periods and plot
D_m_Ce = np.mean(D_3way_Ce, axis=0)

# %%
# CANDECOMP/PARAFAC (CP) decomposition on diff with average initial spectra (ref = t0)
D_3way_Ce_d = D_3way_Ce[1:] - np.mean(D_3way_Ce[1:,0,:].data, axis=0)
#D_3way_Ce_d = D_3way_Ce[1:] - D_3way_Ce[1:,0,:]
D_3way_Ce_d.name = 'diff, ref=averaged 1st spectrum of each period'

for rank in range(1, 3):
    cp = u.CP(n_components=int(rank))
    cp.fit(D_3way_Ce_d)

    if rank == 1:
        SSE_1 = cp.SSE
        print(
            f'rank={rank} -- LOSS={SSE_1 / SSE_1:.2f} --  EV={cp.explained_variance:.2f} -- CC={cp.core_consistency:.2f}')
    else:
        print(
            f'rank={rank} -- LOSS={cp.SSE / SSE_1:.2f} --  EV={cp.explained_variance:.2f}  -- CC={cp.core_consistency:.2f}')

cp.n_components = 1
cp.fit(D_3way_Ce_d)

for load in (cp.A, cp.B, cp.C):
    load.T.plot(title=load.name)
    scp.show()

# ==> suggests a single component


# %%
# reconstructed data

S = np.tensordot(factors[1], factors[2].T, axes=1)
Dhat = np.tensordot(factors[0], S, axes=0).squeeze()

# plot the difference between the original data and the reconstructed data
res = Dhat - D_3way_diff
plt.plot(D_3way_diff.reshape(40*60, 984).T)
plt.plot(res.reshape(40*60, 984).T - 500, 'r')
plt.show()

plt.plot(D_3way_diff[:,1,:].T)
plt.show()


A

# %%
weights, factors = tl.decomposition.parafac(D_3way_diff[:], rank=1)

# np.dot(factor[0], factor[1])
# %%
# genberate an animated gif of the difference spectra
from matplotlib.animation import FuncAnimation

speca = D_diff_me[0].data.squeeze()
stedev = D_diff_std[0].data.squeeze()

fig, ax = plt.subplots()
# plot spectrum and line to show the std
fill_1 = ax.fill_between(np.arange(len(speca)), speca - 3 * stedev, speca + 3 * stedev, alpha=0.05, linewidth=0.0)
fill_2 = ax.fill_between(np.arange(len(speca)), speca - 2 * stedev, speca + 2 * stedev, alpha=0.1, linewidth=0.0)
fill_3 = ax.fill_between(np.arange(len(speca)), speca - 1 * stedev, speca + 1 * stedev, alpha=0.15, linewidth=0.0)
line = ax.plot(speca)[0]
ax.set(ylim = [-150, 150])

def update(frame):
    line.set_ydata(D_diff_me[frame].data.squeeze())

A = FuncAnimation(fig=fig, func=update, frames=59, interval=100)
A.save(filename="a.gif", writer="pillow")

# %%
    # plot spectrum and line to show the std





speca = D_diff_me[n_spec].data.squeeze()
stedev = D_diff_std[n_spec].data.squeeze()

# plot spectrum and line to show the std
fig, ax = plt.subplots()
ax.plot(speca)
ax.fill_between(np.arange(len(speca)), speca - 2 * stedev, speca + 2 * stedev, alpha=0.2, linewidth=0.0)
plt.show()

# averaged modulation excitation spectra and standard deviation (exclude the first period)
D_me = np.mean(D_3way[1:], axis=0)
D_std = np.std(D_3way[1:], axis=0)

D_me = scp.NDDataset(D_me)
D_me.set_coordset(y=scp.Coord(np.arange(0, n, 1), title='time', units='s'), x=D_2.x)
D_me.plot()
scp.show()

D_std = scp.NDDataset(D_std)
D_std.set_coordset(y=scp.Coord(np.arange(0, n, 1), title='time', units='s'), x=D_2.x)
D_std.plot()
scp.show()

# %%

speca = D_m[0].data.squeeze()

stedev = D_std[0].data.squeeze()

# plot spectrum and line to show the std
fig, ax = plt.subplots()
ax.plot(D_m[0].data.squeeze())
ax.fill_between(np.arange(len(spec)), spec - 2 * stedev, spec + 2 * stedev, alpha=0.2, linewidth=0.0)
ax.plot(D_m[29].data.squeeze())
ax.fill_between(np.arange(len(spec)), spec - 2 * stedev, spec + 2 * stedev, alpha=0.2, linewidth=0.0)
plt.show()




# difference spectra
D_d = (D_m - D_m[0])[1:]
D_d.plot()
scp.show()

# denoise with PCA
pca = scp.PCA(n_components=5)
pca.fit(D_d)
D_d_denoised = pca.inverse_transform()
D_d_denoised.plot()
scp.show()


# %%
# PSD

D_psd = scp.dot(u.K_psd(D_d.y, quadrature='trapezoid'), D_d)
D_psd.plot()
scp.show()



# %%
D_1_ave = u.read_hess_raman_averaged('data/hess/Fig_2-a.txt')
D_1_ave.plot()
scp.show()

# %%
# Difference spectra
D_1_d = D_1_ave - D_1_ave[0]
D_1_d.plot(title='diff spectra, ref=1st spectrum')
scp.show()

# %%
# Difference spectra
D_1_d[:30].plot(title='diff spectra, ref=1st spectrum, 1st half period')
scp.show()

#  Difference specyra
D_1_d[30:].plot(title='diff spectra, ref=1st spectrum, 2nd half period')
scp.show()

# %%
# Denoise with ACP
pca = scp.PCA(n_components=2)
pca.fit(D_1_d)
D_1_d_denoised = pca.inverse_transform()
D_1_d_denoised.plot()
scp.show()


# %%
D_2 = u.read_hess_raman_raw('data/hess/Fig_2-b-d_raw-data.txt')
D_2.plot()
scp.show()

# %%
D_3 = u.read_hess_uv_raw('data/hess/Fig_3-a-c_raw-data.txt')
D_3.plot()
scp.show()
