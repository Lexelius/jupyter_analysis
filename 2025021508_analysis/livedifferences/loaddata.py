"""
Loads the data from from files that are neccessary for the other notebooks.
"""
from ptypy import io
import numpy as np
from ptypy.utils import rmphaseramp
from skimage.restoration import unwrap_phase
from skimage.registration import phase_cross_correlation
import h5py


class LoadData:
    def __init__(self, offline_fname='', realtime_fname='', samplename=''):
        """ Constructor for this class. """

        #offline
        self.fname1 = offline_fname
        #real-time
        self.fname2 = realtime_fname
        self.samplename = samplename
    
        
    def rampweight(self, ob, scale=3):
        """Get the weight for where to remove phase ramp"""
        weights = np.zeros(ob.shape)#np.zeros_like(ob)
        M, N = weights.shape
        weights[int(M // scale):int(M - M // scale), int(N // scale):int(N - N // scale)] = 1  # centered box
        mask = np.ones_like(weights, dtype=bool)
        edge = 30
        mask[ M//edge:M - M//edge , N//edge:N - N//edge  ] = False
        weights[mask] = 0
        return weights
    
    def rampweight_(self, ob, scale=60):
        """Get the weight for where to remove phase ramp, scale given in percentage of original shape."""
        weights = np.zeros(ob.shape)#np.zeros_like(ob)
        M, N = weights.shape
        fraction = scale / 100
        start_M = int((1 - fraction) / 2 * M)
        end_M = int((1 + fraction) / 2 * M)
        start_N = int((1 - fraction) / 2 * N)
        end_N = int((1 + fraction) / 2 * N)
        weights[start_M:end_M, start_N:end_N] = 1  
        #weights[int(M // scale):int(M - M // scale), int(N // scale):int(N - N // scale)] = 1  # centered box
        #mask = np.ones_like(weights, dtype=bool)
        #edge = 30
        #mask[ M//edge:M - M//edge , N//edge:N - N//edge  ] = False
        #weights[mask] = 0
        return weights

    
    def get_margin(self, arr, scale=100):
        """
        Get cordinates for cropping the array. 
        :param scale: [float] percentage of how much to keep of the array.
        """
        M, N = arr.shape
        fraction = scale / 100
        start_M = int((1 - fraction) / 2 * M)
        end_M = int((1 + fraction) / 2 * M)
        start_N = int((1 - fraction) / 2 * N)
        end_N = int((1 + fraction) / 2 * N)
        return [start_M, end_M, start_N, end_N]


    def get_vminvmax(self, arr, std_factor=3):
        """
        Get vmin and vmax for plotting the array 'arr' with imshow. 
        :param std_factor: [float] factor deciding how big of a range vmin and vmax will have.
        """
        if np.iscomplexobj(arr):
            print(f'WARNING: Input array is complex, will only calculating vmin and vmax on np.abs(arr).')
            arr = np.abs(arr)
        vmin = np.mean(arr) - std_factor*np.std(arr)
        vmax = np.mean(arr) + std_factor*np.std(arr)
        return vmin, vmax
        

    
    def load(self):
        """"Loads the reconstructions from file"""
        if len(self.fname1) > 0:
            self.obj1 = io.h5read(self.fname1, 'content/obj/Sscan00G00/data')['content/obj/Sscan00G00/data'][0]
            self.pr1 = io.h5read(self.fname1, 'content/probe/Sscan00G00/data')['content/probe/Sscan00G00/data'][0]
            self.obj1_psize = io.h5read(self.fname1, 'content/obj/Sscan00G00/_psize')['content/obj/Sscan00G00/_psize'][0]
            self.pr1_psize = io.h5read(self.fname1, 'content/probe/Sscan00G00/_psize')['content/probe/Sscan00G00/_psize'][0]
            
        if len(self.fname2) > 0:
            self.obj2 = io.h5read(self.fname2, 'content/obj/Sscan00G00/data')['content/obj/Sscan00G00/data'][0]
            self.pr2 = io.h5read(self.fname2, 'content/probe/Sscan00G00/data')['content/probe/Sscan00G00/data'][0]
            self.obj2_psize = io.h5read(self.fname2, 'content/obj/Sscan00G00/_psize')['content/obj/Sscan00G00/_psize'][0]
            self.pr2_psize = io.h5read(self.fname2, 'content/probe/Sscan00G00/_psize')['content/probe/Sscan00G00/_psize'][0]
        
        
        #self.obj1 = io.h5read(self.fname1, 'content/obj/Sscan00G00/data')['content/obj/Sscan00G00/data'][0]
        #self.obj2 = io.h5read(self.fname2, 'content/obj/Sscan00G00/data')['content/obj/Sscan00G00/data'][0]
        #self.pr1 = io.h5read(self.fname1, 'content/probe/Sscan00G00/data')['content/probe/Sscan00G00/data'][0]
        #self.pr2 = io.h5read(self.fname2, 'content/probe/Sscan00G00/data')['content/probe/Sscan00G00/data'][0]

        #self.obj1_psize = io.h5read(self.fname1, 'content/obj/Sscan00G00/_psize')['content/obj/Sscan00G00/_psize'][0]
        #self.obj2_psize = io.h5read(self.fname2, 'content/obj/Sscan00G00/_psize')['content/obj/Sscan00G00/_psize'][0]
        #self.pr1_psize = io.h5read(self.fname1, 'content/probe/Sscan00G00/_psize')['content/probe/Sscan00G00/_psize'][0]

       

    
    def get_pos(fname):
        """Returns the positions from chunked ptyd files"""
        with h5py.File(fname, 'r') as f:
            nrchunks = len(f['chunks'].keys())
            pos = np.array([[],[]]).T
            for k in range(nrchunks):
                pos = np.concatenate((pos, f['chunks'][str(k)]['positions'][:]))
        return pos

    def process_data(self):
        """Get the amplitudes and phases of the objects, and some naive differences"""
        self.obj1_abs = np.abs(self.obj1)
        self.obj2_abs = np.abs(self.obj2)
        self.obj1_abslog = np.log(self.obj1_abs)
        self.obj2_abslog = np.log(self.obj2_abs)

        self.obj1_phase = np.angle(self.obj1)
        self.obj2_phase = np.angle(self.obj2)
        self.w1 = self.rampweight(self.obj1, scale=10)#100) #5) #np.abs(rampweight(obj1, scale=3)-1)
        self.w2 = self.rampweight(self.obj2, scale=10)#100) #5) #np.abs(rampweight(obj2, scale=3)-1)

        self.obj1_ramp, self.ramp1 = rmphaseramp(self.obj1, self.w1, return_phaseramp=True)#rampweight(obj1, scale=3)
        self.obj2_ramp, self.ramp2 = rmphaseramp(self.obj2, self.w2, return_phaseramp=True)
        self.obj1_phaseramp = np.angle(self.obj1_ramp)
        self.obj2_phaseramp = np.angle(self.obj2_ramp)
        self.obj1_phaseramp_unwrapped = unwrap_phase(self.obj1_phaseramp) # unwrap_phase must have real input & output
        self.obj2_phaseramp_unwrapped = unwrap_phase(self.obj2_phaseramp)

        self.obj1_rmramp_abs = np.abs(self.obj1_ramp)
        self.obj2_rmramp_abs = np.abs(self.obj2_ramp)
        self.obj1_rmramp_abslog = np.log(self.obj1_rmramp_abs)
        self.obj2_rmramp_abslog = np.log(self.obj2_rmramp_abs)
        """try:
            self.diff_rmramp_abslog = self.obj1_rmramp_abslog - self.obj2_rmramp_abslog
            self.diff_abslog_od = self.obj1_abslog - self.obj2_abslog # optical density
            self.diff_phase = self.obj1_phase - self.obj2_phase
            self.diff_phaseramp = self.obj1_phaseramp - self.obj2_phaseramp
            self.diff_phaseramp_unwrapped = self.obj1_phaseramp_unwrapped - self.obj2_phaseramp_unwrapped
            self.diff_phaseramp_unwrapped_ramp = np.angle(rmphaseramp(  np.exp(1j*self.diff_phaseramp_unwrapped), self.rampweight(np.exp(1j*self.diff_phaseramp_unwrapped), scale=3)  ))
            
            #marg = 100#500#250
            #self.shift1, self.error, self.phasediff = phase_cross_correlation(self.obj1_phaseramp_unwrapped[marg:-marg,marg:-marg], self.obj2_phaseramp_unwrapped[marg:-marg,marg:-marg], upsample_factor=1000)
        except ValueError:
            print('Will not calculate diffs, object shapes are probably different shapes.')
"""
        

    def add_padding(self):
        # Compare GT data with recons from simulations with DIFFERENT POSITIONS.
        self.sh = self.obj_abs.shape
        self.sh1 = self.obj1_abs.shape
        self.sh2 = self.obj2_abs.shape

        if self.sh != self.sh1:
            self.padrow1 = (self.sh[0]-self.sh1[0])//2
            self.padcol1 = (self.sh[1]-self.sh1[1])//2

            self.obj1_pad = np.pad(self.obj1, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_abs_pad = np.pad(self.obj1_abs, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_abslog_pad = np.pad(self.obj1_abslog, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_phase_pad = np.pad(self.obj1_phase, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_ramp_pad = np.pad(self.obj1_ramp, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_rmramp_abs_pad = np.pad(self.obj1_rmramp_abs, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_rmramp_abslog_pad = np.pad(self.obj1_rmramp_abslog, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_phaseramp_pad = np.pad(self.obj1_phaseramp, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))
            self.obj1_phaseramp_unwrapped_pad = np.pad(self.obj1_phaseramp_unwrapped, ((self.padrow1,self.sh[0]-self.padrow1-self.sh1[0]),(self.padcol1,self.sh[1]-self.padcol1-self.sh1[1])))

        if self.sh != self.sh2:
            self.padrow2 = (self.sh[0]-self.sh2[0])//2
            self.padcol2 = (self.sh[1]-self.sh2[1])//2

            self.obj2_pad = np.pad(self.obj2, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_abs_pad = np.pad(self.obj2_abs, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_abslog_pad = np.pad(self.obj2_abslog, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_phase_pad = np.pad(self.obj2_phase, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_ramp_pad = np.pad(self.obj2_ramp, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_rmramp_abs_pad = np.pad(self.obj2_rmramp_abs, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_rmramp_abslog_pad = np.pad(self.obj2_rmramp_abslog, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_phaseramp_pad = np.pad(self.obj2_phaseramp, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))
            self.obj2_phaseramp_unwrapped_pad = np.pad(self.obj2_phaseramp_unwrapped, ((self.padrow2,self.sh[0]-self.padrow2-self.sh2[0]),(self.padcol2,self.sh[1]-self.padcol2-self.sh2[1])))



##### In progress: (make LD functions available without specifying rec-fnames)
# class group(LoadData):
#     def __init__(self):
#         # Initialize the parent class (Shape) with color
#         super().__init__(self) 
    
    
# class item(group):
#     def __init__(self):
#         super().__init__(self) 
#         def __init__(self, offline_fname, realtime_fname, samplename=''):
#         """ Constructor for this class. """

#         #offline
#         self.fname1 = offline_fname
#         #real-time
#         self.fname2 = realtime_fname
#         self.samplename = samplename

