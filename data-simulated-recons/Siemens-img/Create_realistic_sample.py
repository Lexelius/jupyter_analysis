# Computes the phase shifts and amplitudes of 1 micron gold deposited on 0.5 micron Si3N4, compared to vacuum.
import numpy as np
from PIL import Image

#%% Calculate the values

# data from https://henke.lbl.gov/optical_constants/getdb2.html
Energy = 8000.  # eV
delta = 4.77303183E-05
beta = 4.96011535E-06

# Silicon nitrade wafer
delta_Si3N4 = 1.12793196E-05
beta_Si3N4 = 1.69130473E-07

# Complex index of refraction of the material
n_ = 1 - delta - beta*1j  # (real and imaginary parts)
n_Si3N4 = 1 - delta_Si3N4 - beta_Si3N4*1j

heV = 4.135667696*10**(-15)  # [eVs] Planck's constant
c = 299792458  # [m/s] speed of light

lmbda = heV*c/Energy  # [m]
d = 1e-6  # material thickness
d_Si3N4 = 0.5e-6

# Phase shift
phase_shift_Au = 2 * np.pi * d * n_.real / lmbda
phase_shift_vacAu = 2 * np.pi * d * 1 / lmbda
phase_shift1 = phase_shift_vacAu - phase_shift_Au  # [rad] difference in phase shift between light passing through vacuum and through Au.

phase_shift_Si3N4 = 2 * np.pi * d_Si3N4 * n_Si3N4.real / lmbda
phase_shift_vacSi3N4 = 2 * np.pi * d_Si3N4 * 1 / lmbda
phase_shift2 = phase_shift_vacSi3N4 - phase_shift_Si3N4  ### Phase shift difference of light transmitted through wafer compared to vacuum.
phase_shift_sample = phase_shift_Au + phase_shift_Si3N4
phase_shift_vac = phase_shift_vacAu + phase_shift_vacSi3N4
phase_shift = phase_shift_vac - phase_shift_sample  ### Phase shift difference of light transmitted through Au + wafer compared to vacuum.

# Amplitude
I0 = 1
I_T_sample = I0 * np.exp(-4*np.pi * beta * d / lmbda) * np.exp(-4*np.pi * beta_Si3N4 * d_Si3N4 / lmbda)
I_T_wafer = I0 * np.exp(-4*np.pi * beta_Si3N4 * d_Si3N4 / lmbda)

# R = np.abs((n_ - 1) / (n_ + 1))**2
# T = 1 - R # ? but the absorption part though?
print("Phase shift of transmitted light:", phase_shift)
print("Transmitted light:", I_T_sample, I_T_wafer)


print(f'\n phase_shift_Au       {phase_shift_Au}\n phase_shift_vacAu       {phase_shift_vacAu}\n phase_shift1       {phase_shift1}\n phase_shift_Si3N4       {phase_shift_Si3N4}\n phase_shift_vacSi3N4       {phase_shift_vacSi3N4}\n phase_shift2 *     {phase_shift2}\n phase_shift_sample       {phase_shift_sample}\n phase_shift_vac       {phase_shift_vac}\n phase_shift *     {phase_shift}')



#%% Insert values into Siemens array and save

fname = '/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/41566_2016_Article_BFnphoton2015279_Fig1_HTML.png'
img = Image.open(fname)
imgarr = np.asarray(img)


imgarr_complexobj = np.asarray(imgarr[:, :, 1], dtype='complex128')
# The complex object is given by: Amplitude * exp(i*phaseshift)
imgarr_complexobj[np.where(imgarr[:, :, 1] < 140)] = I_T_wafer * np.exp(phase_shift2*1j)   # wafer
imgarr_complexobj[np.where(imgarr[:, :, 1] >= 140)] = I_T_sample * np.exp(phase_shift*1j)  # wafer + Au siemens star

# square crop
sh = imgarr_complexobj.shape
margin = (max(sh) - min(sh)) // 2
imgarr_complexobj_crop = imgarr_complexobj[:, margin:-margin]

np.save('/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/siemens_array.npy', imgarr_complexobj_crop)
