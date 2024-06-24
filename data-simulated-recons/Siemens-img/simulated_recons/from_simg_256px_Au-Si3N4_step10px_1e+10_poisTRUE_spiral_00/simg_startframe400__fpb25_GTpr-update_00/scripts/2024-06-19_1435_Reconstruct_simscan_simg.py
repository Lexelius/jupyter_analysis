import os
import sys
import time
import socket
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from distutils.version import LooseVersion
from mpi4py import MPI
from ptypy import io

setting = 'gpu'
if float(ptypy.version[:3]) >= float('0.5'):
    ptypy.load_ptyscan_module("livescan")
    if setting != 'cpu':
        ptypy.load_gpu_engines(arch="cuda")
    # ptypy.load_gpu_engines(arch="cuda"):::'DM_pycuda', 'DM_pycuda_nostream',  ptypy.load_gpu_engines(arch="serial"):::'DM_serial',  ptypy.load_gpu_engines(arch="ocl"):::'DM_ocl'

print(ptypy.__file__)
############################################################################
# hard coded user input
############################################################################


beamtime_basedir        = '/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/simulated_recons/from_simg_256px_Au-Si3N4_step10px_1e+10_poisTRUE_spiral_00'  # '/home/reblex/Documents/Reconstructions/NM_scannr_1190'
GT_fname                = '/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/simulated_data/simg_256px_Au-Si3N4_step10px_1e+12_00/data/data_scan_000000.ptyd'
bgfile                  = None
scannr                  = 0
overwrite               = False ### True only for debugging, if False increments scannr used in writing path to a unique path.

detector                = 'eiger4m'
detector_binning_rec    = None       # binning on top of that done only in ptypy
cropping                = 256##800 #256
distance_m              = 3.670#3.740   # distance between the sample and the detector in meters
defocus_um              = 980#800  # distance between the focus and the sample plane in micro meters -> used for inital probe
probe_modes             = 1
fpb                     = 25
start_frame             = 400  #1257
numitcont               = 1
divnoise = (0.5, 1.0)
alpha                   = 0.8#0.75
sample                  = f'simg_startframe{str(start_frame).ljust(4,"_")}_fpb{str(fpb).ljust(2,"_")}_GTpr-update'


############################################################################
# some preparations before the actual reconstruction
# ############################################################################

#out_dir0 = f'{beamtime_basedir}/{sample}_fpb{str(fpb).ljust(2,"_")}_startframe{str(start_frame).ljust(4,"_")}_itcont{numitcont}_crop{cropping}{str(f"_posref-{str(do_pos_ref[0])[0]}{str(do_pos_ref[1])[0]}_maxshift{int(maxshift*1e9)}nm_refstart{posref_start}" if any(do_pos_ref) else "")}_dist{int(distance_m*100)}_bin{detector_binning_rec}_defocus{defocus_um}_a{alpha}'
#out_dir0 = f'{beamtime_basedir}/{sample}_startframe{str(start_frame).ljust(4,"_")}_crop{cropping}_fpb{str(fpb).ljust(2,"_")}_dist{int(distance_m*100)}_defocus{defocus_um}_a{alpha}{str(f"_posref-{str(do_pos_ref[0])[0]}{str(do_pos_ref[1])[0]}_maxshift{int(maxshift*1e9)}nm_refstart{posref_start}" if any(do_pos_ref) else "")}'
out_dir0 = f'{beamtime_basedir}/{sample}'
out_dir = f'{out_dir0}_{scannr:02d}/'
if not overwrite:
    while os.path.isdir(f'{out_dir0}_{scannr:02d}/'):
        scannr += 1
    out_dir = f'{out_dir0}_{scannr:02d}/'

out_dir_data = out_dir + 'data/'
out_dir_dumps = out_dir + 'dumps/'
out_dir_scripts = out_dir + 'scripts/'
out_dir_rec = out_dir + 'rec/'

# and what the files are supposed to be called
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



############################################################################
# creating the parameter tree
############################################################################

# General parameters
p = u.Param()
p.verbose_level = 3  # 'interactive'
p.run = 'scan%d' % scannr
p.min_frames_for_recon = start_frame  #500#180  #10#cases[case]['start_frame']  # 55		## Minimum number of frames loaded before starting iterations
p.frames_per_block = fpb  # cases[case]['frames_per_block'] #100 #400 ####default: 100000 #### should overrule min_frames at


# where to put the reconstructions
p.io = u.Param()
p.io.home = out_dir_rec  # where to save the final reconstructions
p.io.rfile = path_rec  # how to name those files for the final reconstructions
p.io.interaction = u.Param()
p.io.interaction.active = True
p.io.interaction.server = u.Param()


IPaddr = socket.gethostbyname(socket.gethostname())  ## Used o enable interactions outside the node
p.io.interaction.server.address = f'tcp://{IPaddr}'  ##'tcp://172.18.1.26'  # cn25 ## default = 'tcp://127.0.0.1'
p.io.interaction.server.port = 5560  ### IF YOU FORGET TO INCLUDE "tcp://" IN ADDRESS: ptypy think this port is used but it's not, and then tries to many times and just ends up not connecting to a port, even though it says that is does..!!!
p.io.interaction.client = u.Param()
p.io.interaction.client.address = f'tcp://{IPaddr}'  ## 'tcp://172.18.1.26'
p.io.interaction.client.port = 5560
p.io.autosave = u.Param()
p.io.autosave.active = True
p.io.autosave.rfile = path_dumps  # where to save the intermediate reconstructions and how to name them
p.io.autoplot = u.Param()
p.io.autoplot.active = False  #  True
# p.io.autoplot.interval = 1		# default = 1 (>-1)
p.io.autoplot.threaded = False  # default = True
# p.io.autoplot.dump = True
# p.io.autoplot = u.Param(active=True, threaded=False, dump=True, layout='minimal') ##u.Param(active=False)


# Scan parameters
p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'BlockFull' ##'Full'
p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = probe_modes  # number of probe modes
p.scans.scan00.coherence.num_object_modes = 1  # number of object modes

p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'LiveScan'
p.scans.scan00.data.detector = detector
p.scans.scan00.data.xMotor = 'pseudo/x'
p.scans.scan00.data.yMotor = 'pseudo/y'
p.scans.scan00.data.positions_multiplier = 1.  # Multiplicative factor that converts motor positions to metres.
p.scans.scan00.data.relay_host = 'tcp://127.0.0.1'
p.scans.scan00.data.relay_port = 45678
p.scans.scan00.data.shape = cropping  # size of the window of the diffraction patterns to be used in pixel
p.scans.scan00.data.crop_at_RS = None#cropping
p.scans.scan00.data.rebin = detector_binning_rec  #None
p.scans.scan00.data.rebin_at_RS = None  #detector_binning_rec
p.scans.scan00.data.save = 'append'
p.scans.scan00.data.dfile = path_data  # once all data is collected, save it as .ptyd file
p.scans.scan00.data.center = (128,128)#None  #(1284, 802)   # center of the diffraction pattern (y,x) in pixel or None -> auto
p.scans.scan00.data.auto_center = None
p.scans.scan00.data.xMotorFlipped = False  # should be opposite to what is used with NanomaxContrast
p.scans.scan00.data.yMotorFlipped = False  #True  # should be opposite to what is used with NanomaxContrast
p.scans.scan00.data.orientation = (False, False, False) #(False, True, False) # is correct for sim_files_05#(False, True, True) is correct for sim_files_06
                                   #  {'merlin': (False, False, True),
                                   # 'pilatus': None,
                                   # 'eiger': None,
                                   # 'eiger4m': (False, True, False),
                                   # 'zylafop': (True, True, True)}[detector]
p.scans.scan00.data.distance = distance_m  # distance between sample and detector in [m]
p.scans.scan00.data.psize = {'pilatus': 172e-6,
                             'merlin': 55e-6,
                             'eiger': 75e-6,
                             'eiger4m': 75e-6,
                             'zylafop': 6.5e-6}[detector]
# p.scans.scan00.data.energy = energy_keV    # incident photon energy in [keV], now read from file

# Params after hackaton
p.scans.scan00.data.min_frames = fpb  #has to be the same as fpb atm (something that has to be fixed in the core)  ##cases[case]['min_frames']  # 1		## Minimum number of frames loaded by each node/process
p.scans.scan00.data.block_wait_count = 1  ##cases[case]['bwc']  # 0#1
p.scans.scan00.data.frames_per_iter = None ## Load a fixed number of frames in between each iteration, default = None
p.scans.scan00.data.load_parallel = None

# Initial illumination for reconstruction
GT_probe = io.h5read(GT_fname, '/info/illumination')['/info/illumination']
p.scans.scan00.illumination = u.Param()
p.scans.scan00.illumination.model = GT_probe.model
p.scans.scan00.illumination.aperture = GT_probe.aperture


#############################################################

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda' #'DM' #'DM_pycuda'
p.engines.engine00.numiter = 3000
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.alpha = alpha
# p.engines.engine00.clip_object = (0, 1)          # Default = None, Clip object amplitude into this interval
p.engines.engine00.probe_support = None
p.engines.engine00.probe_update_start = 2  #6500  # default = 2
p.engines.engine00.fourier_relax_factor = 0.0

t0 = time.time()
P = ptypy.core.Ptycho(p, level=5)#, data_type='double')
dt = time.time() - t0
print(f'Simulation took {dt // 60} min {dt % 60} sec.')