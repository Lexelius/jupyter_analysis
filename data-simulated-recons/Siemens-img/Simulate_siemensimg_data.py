"""
Copied (and modified) from:
https://ptycho.github.io/tutorials/notebooks/18_21_Advanced_Topics/20_simulating_data.html


Datasets that could be of interest for making data-simulations:
  Siemens stars:
    Sample MA71 (asterix looking star):
      /data/visitors/nanomax/20211244/2023120608/process/0002_MA71/scan_000059/ptycho_ptypy_GPU_crop-1024_bin-2_Pmodes-4_dist-7.140/rec/rec_scan_000059_ML_pycuda_4000.ptyr     # has around 8x7 stars
      /data/visitors/nanomax/20211244/2023120608/process/0002_MA71/scan_000058/ptycho_ptypy_GPU_crop-512_bin-2_Pmodes-2_dist-7.140/rec/rec_scan_000058_ML_pycuda_4000.ptyr      # has around 5x6 stars
      /data/visitors/nanomax/20211244/2023120608/process/0002_MA71/scan_000057/ptycho_ptypy_GPU_crop-1024_bin-1_Pmodes-2_dist-7.140/rec/rec_scan_000057_ML_pycuda_4000.ptyr     # has around 4-5x5-6 stars placed mainly in the lower left corner, has a lot of background!
    "Old" siemens sample:
      /data/visitors/nanomax/20211244/2023120608/process/0001_setup/scan_000010/ptycho_ptypy_GPU_crop-1024_bin-1_Pmodes-1_dist-7.140/rec/rec_scan_000010_ML_pycuda_4000.ptyr    # has around 7x7 stars
      /data/visitors/nanomax/20211244/2023120608/process/0001_setup/scan_000007/ptycho_ptypy_GPU_crop-1024_bin-1_Pmodes-1_dist-7.140/rec/rec_scan_000007_ML_pycuda_4000.ptyr    # has around 5-6x5-6 stars, with noisy edges



"""
import ptypy
import ptypy.utils as u
from ptypy import io
import os
import numpy as np
from mpi4py import MPI
import time
from PIL import Image
from ptypy.core import xy
ptypy.load_gpu_engines(arch="cuda")
ptypy.load_ptyscan_module("livescan") ## Only used for getting the ../ptypy/experiment/ - path.


# fname_obj = '/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/41566_2016_Article_BFnphoton2015279_Fig1_HTML.png'
# obj_arr = u.rgb2complex(np.array(Image.open(fname_obj)))
obj_arr = np.load('/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/siemens_array.npy')

fname_probe = '/home/reblex/ptycho_ptypy/ptypy/ptypy/resources/moon.png'
probe_arr = u.rgb2complex(np.array(Image.open(fname_probe)))
sim_probe = "/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/realistic_probe256.npy"#"/data/staff/nanomax/reblex/data-simulated-recons/NTT_scan_001190/original_recon/NTT_1190_startframe2912_crop256_dist367_defocus980_a0.8_00/dumps/dump_scan_000000_DM_pycuda_1000.ptyr"
probe = np.load(sim_probe)
########## Using a gaussian probe:
# Copied from https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
size = 256
fwhm = 40
x = np.arange(0, size, 1, float)
y = x[:,np.newaxis]
x0 = y0 = size // 2  # center
probe_gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
##################################
n_px = 10  # stepsize of scan positions i pixels

defocus_um = 980  #
scannr = 0000
intensity = 1e12  #1e5
sample = f'simg_256px_Au-Si3N4_step{n_px}px_{intensity:.0e}'
out_dir0 = f'/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/simulated_data/{sample}'
nr = 0
out_dir = f'{out_dir0}_{nr:02d}/'
while os.path.isdir(f'{out_dir0}_{nr:02d}/'):
    nr += 1
out_dir = f'{out_dir0}_{nr:02d}/'

out_dir_data = out_dir + 'data/'
out_dir_dumps = out_dir + 'dumps/'
out_dir_scripts = out_dir + 'scripts/'
out_dir_rec = out_dir + 'rec/'

path_data = out_dir_data + 'data_scan_' + str(scannr).zfill(6) + '.ptyd'  # the file with the prepared data
path_dumps = out_dir_dumps + 'dump_scan_' + str(scannr).zfill(6) + '_%(engine)s_%(iterations)04d.ptyr'  # intermediate results
path_rec = out_dir_rec + 'rec_scan_' + str(scannr).zfill(6) + '_%(engine)s_%(iterations)04d.ptyr'  # final reconstructions (of each engine)

# stuff to only do once
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:

    # create output directories if it does not already exists
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_data, exist_ok=True)
    os.makedirs(out_dir_dumps, exist_ok=True)
    os.makedirs(out_dir_scripts, exist_ok=True)
    os.makedirs(out_dir_rec, exist_ok=True)

    # copy this file into this directory with a tag made from the time and date this script was run
    try:
        os.system('cp ' + str(__file__) + ' ' + out_dir_scripts + time.strftime("%Y-%m-%d_%H%M") + '_' + str(__file__).split('/')[-1])  ##
    except:
        print("Couldn't copy Recon-file")
    try:
        os.system('cp ' + '/home/reblex/RelayServer/relayserver.py' + ' ' + out_dir_scripts + time.strftime("%Y-%m-%d_%H%M") + '_' + '/home/reblex/RelayServer/relayserver.py'.split('/')[-1])  ##
    except:
        print("Couldn't copy Relay-file")
    try:
        os.system('cp ' + ptypy.experiment.__path__[0] + '/livescan.py' + ' ' + out_dir_scripts + time.strftime("%Y-%m-%d_%H%M") + '_' + 'livescan.py')
    except:
        print("Couldn't copy LiveScan-file")
#%%
# Create parameter tree
p = u.Param()

# Set verbose level, can be "interactive", "info" or "debug"
p.verbose_level = "info"

p.frames_per_block = 500

# Basic I/O settings (no files saved in this case)
p.io = u.Param()
p.io.home = out_dir_rec  # where to save the final reconstructions
p.io.rfile = path_rec  ## None  # how to name those files for the final reconstructions
p.io.autosave = u.Param(active=True) # False
p.io.autosave.rfile = path_dumps  # where to save the intermediate reconstructions and how to name them
p.io.autosave.interval = 100
p.io.interaction = u.Param(active=True)

# Live-plotting during the reconstruction
p.io.autoplot = u.Param()
p.io.autoplot.active = False #  True
p.io.autoplot.threaded = False
p.io.autoplot.layout = 'default'  # "jupyter"
p.io.autoplot.interval = 10

# Scan model
p.scans = u.Param()
p.scans.scan_00 = u.Param()
p.scans.scan_00.name = 'BlockFull'
# p.scans.scan_00.name = 'Full'
p.scans.scan_00.coherence = u.Param()
p.scans.scan_00.coherence.num_probe_modes = 1

# Scan data (simulation) parameters
# using typical values for I08-1 instrument
p.scans.scan_00.data = u.Param()
p.scans.scan_00.data.name = 'SimScan'
p.scans.scan_00.data.center = (size//2, size//2)#(1350, 1250)  ###
p.scans.scan_00.data.energy = 8.0  # 0.7
p.scans.scan_00.data.distance = 3.670  # 0.072
p.scans.scan_00.data.psize = 75e-6  # 22e-6
p.scans.scan_00.data.shape = size#1024#(2162, 2068) ##512
p.scans.scan_00.data.save = 'append' ##
p.scans.scan_00.data.dfile = path_data  ## once all data is collected, save it as .ptyd file
p.scans.scan_00.data.add_poisson_noise = False
# p.scans.scan_00.data.detector = 'GenericCCD32bit' ##
# p.scans.scan_00.data.orientation = (False, True, False)

# Scanning parameters
p.scans.scan_00.data.xy = u.Param()
steps = 20
stepsize = n_px*1.1849530945572919e-07#2.9623827363932297e-08#50e-9
# uncomment the 3 lines below to add noise to positions
# pos__ = xy.spiral_scan(dr=stepsize, r=steps*stepsize/2, maxpts=None)
# pos__ += np.random.rand(pos__.shape[0],pos__.shape[1])*50e-9 / 3
# p.scans.scan_00.data.xy.override = pos__
## p.scans.scan_00.data.xy.override = 'spiral'#np.array((-y, x)).T * 1e-6  # "spiral"  # Options: None, ‘round’, ‘raster’, ‘spiral’ or array-like
p.scans.scan_00.data.xy.model = 'raster'#'spiral'#'raster'
p.scans.scan_00.data.xy.spacing = stepsize#50e-9
p.scans.scan_00.data.xy.steps = steps#50 #2912  # 10
#### spiral,  spacing = 50e-9,
# steps:  50    |  40    | 100
#         1964  |  1257  | 7854
# You have to explicitly disable extent!
p.scans.scan_00.data.xy.extent = None


# Object to be used for simulation
p.scans.scan_00.data.sample = u.Param()
# p.scans.scan_00.data.sample.model = "recon" ##
p.scans.scan_00.data.sample.model = obj_arr ##u.rgb2complex(np.array(u.imload(sim_image)))
p.scans.scan_00.data.sample.process = u.Param()
# p.scans.scan_00.data.sample.process.offset = (0,200)
# p.scans.scan_00.data.sample.process.zoom = 0.25#0.5
p.scans.scan_00.data.sample.process.formula = None
p.scans.scan_00.data.sample.process.density = None
p.scans.scan_00.data.sample.process.thickness = None
p.scans.scan_00.data.sample.process.ref_index = None
p.scans.scan_00.data.sample.process.smoothing = None
p.scans.scan_00.data.sample.fill = 1.0+0.j
# p.scans.scan_00.data.sample.recon = u.Param()  ##
# p.scans.scan_00.data.sample.recon.rfile = sim_image  ##

# Detector parameters
# p.scans.scan_00.data.detector = u.Param()
# p.scans.scan_00.data.detector.dtype = np.uint64
# p.scans.scan_00.data.detector.full_well = 2**64-1#2**32-1
# p.scans.scan_00.data.detector.psf = None
p.scans.scan_00.data.plot = False #True  ## False
## No detector. If you use a detector you always get Poisson noise added
p.scans.scan_00.data.detector = None

##################################################################################################### Used before testing:
"""
# Illumination to be used for simulation
p.scans.scan_00.data.illumination = u.Param()
probe = io.h5read(sim_probe, '/content/probe')['/content/probe']['Sscan00G00']['data'][0] ##
p.scans.scan_00.data.illumination.model = probe  ## "recon"
p.scans.scan_00.data.illumination.recon = u.Param()
# p.scans.scan_00.data.illumination.recon.rfile = sim_probe
# p.scans.scan_00.data.illumination.photons = 5.9e11  #None  # 1e11 ## could also set around 5.92e+11
p.scans.scan_00.data.illumination.aperture = None
# p.scans.scan_00.data.illumination.propagation = u.Param()
# p.scans.scan_00.data.illumination.propagation.parallel = 1. * defocus_um * 1e-6  # 50e-6

# Initial illumination for reconstruction
p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = probe  # "recon"
p.scans.scan_00.illumination.recon = u.Param()
# p.scans.scan_00.illumination.recon.rfile = sim_probe
# p.scans.scan_00.illumination.photons = 5.9e11 ##None
p.scans.scan_00.illumination.aperture = None
# p.scans.scan_00.illumination.propagation = u.Param()
# p.scans.scan_00.illumination.propagation.parallel = 1. * defocus_um * 1e-6  # 50e-6
"""

#####################################################################################################
# Used for testing:
#####################################################################################################
# Illumination to be used for simulation
p.scans.scan_00.data.illumination = u.Param()
p.scans.scan_00.data.illumination.model = probe#_gaussian#"recon"#probe_arr  ## "recon"
# p.scans.scan_00.data.illumination.recon = u.Param()
# p.scans.scan_00.data.illumination.recon.rfile = sim_probe
p.scans.scan_00.data.illumination.photons = intensity#10e20#9.40e+10#5.9e11  #None  # 1e11 ## could also set around 5.92e+11
p.scans.scan_00.data.illumination.aperture = None
# p.scans.scan_00.data.illumination.aperture = u.Param()
# p.scans.scan_00.data.illumination.aperture.form = 'circ'
# p.scans.scan_00.data.illumination.aperture.size = 7.5836928e-05#3*7.6e-05#7.5836928e-05#7.5836928e-06##None
# p.scans.scan_00.data.illumination.propagation = u.Param()
# p.scans.scan_00.data.illumination.propagation.parallel = 1. * defocus_um * 1e-6  # 50e-6

# Initial illumination for reconstruction
p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = probe#_gaussian#"recon"#probe_arr  # "recon"
# p.scans.scan_00.illumination.recon = u.Param()
# p.scans.scan_00.illumination.recon.rfile = sim_probe
# p.scans.scan_00.illumination.photons = None#9.40e+10#5.9e11 ##None
p.scans.scan_00.illumination.aperture = None
# p.scans.scan_00.illumination.aperture = u.Param()
# p.scans.scan_00.illumination.aperture.form = 'circ'
# p.scans.scan_00.illumination.aperture.size = 7.5836928e-05#3*7.6e-05##7.5836928e-05#7.5836928e-06##None
# p.scans.scan_00.illumination.propagation = u.Param()
# p.scans.scan_00.illumination.propagation.parallel = 1. * defocus_um * 1e-6  # 50e-6


#####################################################################################################
#####################################################################################################






# Reconstruction parameters
p.engines = u.Param()
p.engines.engine = u.Param()
p.engines.engine.name = 'DM_pycuda' #'DM' #'DM_pycuda'
p.engines.engine.numiter = 3000
# p.engines.engine.numiter_contiguous = 1
p.engines.engine.alpha = 0.8  # 0.9
# p.engines.engine.clip_object = (0, 1)          # Default = None, Clip object amplitude into this interval
p.engines.engine.probe_support = None
p.engines.engine.probe_update_start = 6500
p.engines.engine.fourier_relax_factor = 0.0

t0 = time.time()
P = ptypy.core.Ptycho(p, level=5)#, data_type='double')
dt = time.time() - t0
print(f'Simulation took {dt // 60} min {dt % 60} sec.')
# print(f'Simulation took {time.time() - t0} seconds.')
