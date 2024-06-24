"""
Modifies copies of original raw data files and updates it with the data obtained from
simulations instead.
"""
import numpy as np
from ptypy import io
import h5py
sufx = 'simg_256px_Au-Si3N4_step10px_1e+10_poisTRUE_spiral_00'  # Name of the reconstruction-folder created when running "Simulate_siemensimg_data.py"

# Read raw data and simulated data
basedir = '/data/staff/nanomax/reblex/data-simulated-recons/Siemens-img/simulated_data/'
h5_raw_fname = basedir + '001190.h5'  ## Contains only link to the diffraction patterns
hdf5_raw_fname = basedir + 'scan_001190_eiger4m.hdf5'
simdata_fname = basedir + sufx + '/data/data_scan_000000.ptyd'

h5_raw = io.h5read(h5_raw_fname)
hdf5_raw = io.h5read(hdf5_raw_fname)
simdata = io.h5read(simdata_fname)
simdata_diff = simdata['chunks']['0']['data']
simdata_pos = simdata['chunks']['0']['positions']
simdata_energy = simdata['info']['energy'] * 1e3
print('Done reading data.')

# Some modifications to make the files work with the RS-simulators
h5_mod = h5_raw  # changes to h5_mod will also be applied to h5_raw
h5_mod['entry']['measurement']['eiger4m']['frames'] = h5_mod['entry']['measurement']['eiger4m']['thumbs:'][:simdata_pos.shape[0]]

h5_mod['entry']['measurement']['pseudo']['x'] = simdata_pos[:, 1]
h5_mod['entry']['measurement']['pseudo']['y'] = simdata_pos[:, 0]

h5_mod['entry']['snapshots']['pre_scan']['energy'] = np.array([simdata_energy])

# hdf5_mod = hdf5_raw
# hdf5_mod['entry']['measurement']['Eiger']['data'] = simdata_diff


io.h5write(basedir + 'sim_files_'+sufx+'/000000.h5', h5_mod)
print('Done writing h5-data.')


# io.h5write(basedir + 'sim_files/scan_001190_eiger4m.hdf5', hdf5_mod)
hdf5_raw_f = h5py.File(hdf5_raw_fname, 'r')
f = h5py.File(basedir + 'sim_files_'+sufx+'/scan_000000_eiger4m.hdf5', 'w')
hdf5_raw_f.copy(source='entry', dest=f)
f['entry']['measurement']['Eiger'].pop('data')
f['entry']['measurement']['Eiger'].create_dataset(name='data', data=simdata_diff)

hdf5_raw_f.close()
f.close()
print('Done writing hdf5-data.')
