import ptypy
import ptypy.utils as u
import os
import numpy as np
from mpi4py import MPI
import time

ptypy.load_gpu_engines(arch="cuda")

# Root directory of tutorial data
tutorial_data_home = "/home/reblex/Documents/Diamond-Workshop-2023/"

# Probe/object to be used for simulation
path_to_probe = "small_data/dls_i08_nanogold_recon.ptyr"
path_to_image = "small_data/painting.jpg"
sim_probe = os.path.join(tutorial_data_home, path_to_probe)
sim_image = os.path.join(tutorial_data_home, path_to_image)

#%% Output names

scannr = 0000
sample = 'Diam_no-zoom-or-offset_ML'
out_dir0 = f'/data/staff/nanomax/reblex/data-simulated-recons/Diamond/simulated_data/{sample}'
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
        cmdstr = 'cp ' + str(__file__) + ' ' + out_dir_scripts + time.strftime("%Y-%m-%d_%H%M") + '_' + str(__file__).split('/')[-1]
        print(f"Couldn't copy Recon-file when running this line:\n{cmdstr}\n")


#%%


# Create parameter tree
p = u.Param()

# Set verbose level, can be "interactive", "info" or "debug"
p.verbose_level = "info"#"interactive"

# Basic I/O settings (no files saved in this case)
p.io = u.Param()
p.io.rfile = path_rec#None
p.io.autosave = u.Param(active=True)
p.io.autosave.rfile = path_dumps
p.io.autosave.interval = 100
p.io.interaction = u.Param(active=True)

# Live-plotting during the reconstruction
p.io.autoplot = u.Param()
p.io.autoplot.active = False
p.io.autoplot.threaded = False
p.io.autoplot.layout = 'default'#"jupyter"
p.io.autoplot.interval = 10

# Scan model
p.scans = u.Param()
p.scans.scan_00 = u.Param()
p.scans.scan_00.name = 'BlockFull'
p.scans.scan_00.coherence = u.Param()
p.scans.scan_00.coherence.num_probe_modes=1

# Scan data (simulation) parameters
# using typical values for I08-1 instrument
p.scans.scan_00.data = u.Param()
p.scans.scan_00.data.name = 'SimScan'
p.scans.scan_00.data.energy = 0.7
p.scans.scan_00.data.distance = 0.072
p.scans.scan_00.data.psize = 22e-6
p.scans.scan_00.data.shape = 512
p.scans.scan_00.data.save = 'append'
p.scans.scan_00.data.dfile = path_data

# Scanning parameters
p.scans.scan_00.data.xy = u.Param()
p.scans.scan_00.data.xy.model = "raster"
p.scans.scan_00.data.xy.spacing = 50e-9
p.scans.scan_00.data.xy.steps = 10

# Illumination to be used for simulation
p.scans.scan_00.data.illumination = u.Param()
p.scans.scan_00.data.illumination.model = "recon"
p.scans.scan_00.data.illumination.recon = u.Param()
p.scans.scan_00.data.illumination.recon.rfile = sim_probe
p.scans.scan_00.data.illumination.photons = 1e11
p.scans.scan_00.data.illumination.aperture = None
p.scans.scan_00.data.illumination.propagation = u.Param()
p.scans.scan_00.data.illumination.propagation.parallel = 50e-6

# Object to be used for simulation
p.scans.scan_00.data.sample = u.Param()
p.scans.scan_00.data.sample.model = u.rgb2complex(np.array(u.imload(sim_image)))
p.scans.scan_00.data.sample.process = u.Param()
#p.scans.scan_00.data.sample.process.offset = (0,200)
#p.scans.scan_00.data.sample.process.zoom = 0.5
p.scans.scan_00.data.sample.process.formula = None
p.scans.scan_00.data.sample.process.density = None
p.scans.scan_00.data.sample.process.thickness = None
p.scans.scan_00.data.sample.process.ref_index = None
p.scans.scan_00.data.sample.process.smoothing = None
p.scans.scan_00.data.sample.fill = 1.0+0.j

# Detector parameters
p.scans.scan_00.data.detector = u.Param()
p.scans.scan_00.data.detector.dtype = np.uint32
p.scans.scan_00.data.detector.full_well = 2**32-1
p.scans.scan_00.data.detector.psf = None
p.scans.scan_00.data.plot = False

# Initial illumination for reconstruction
p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = "recon"
p.scans.scan_00.illumination.recon = u.Param()
p.scans.scan_00.illumination.recon.rfile = sim_probe
p.scans.scan_00.illumination.photons = None
p.scans.scan_00.illumination.aperture = None
p.scans.scan_00.illumination.propagation = u.Param()
p.scans.scan_00.illumination.propagation.parallel = 50e-6

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda'
p.engines.engine00.numiter = 1000#40
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.alpha = 0.9
p.engines.engine00.probe_support = None
p.engines.engine00.probe_update_start = 9000#0

p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML_pycuda'
p.engines.engine01.numiter = 2000#40
p.engines.engine01.numiter_contiguous = 1
p.engines.engine01.probe_support = None
p.engines.engine01.probe_update_start = 9000#0



P = ptypy.core.Ptycho(p, level=5)
