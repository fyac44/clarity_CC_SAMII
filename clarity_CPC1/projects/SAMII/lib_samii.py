from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

class samii:
    def __init__(self, ear, cfs, tstamps, pre_silence):

        # Information data
        self.mi = np.array(ear['mi'])    # Mutual information
        self.ti = np.array(ear['Si'])    # Transmitted information
        self.pi = np.array(ear['Ri'])    # Received information

        # Center frequencies and analysis windows
        self.cfs = np.array(cfs)
        self.tstamps = np.array(tstamps)
        self.ncfs = len(self.cfs) # Number of center frequencies
        self.total_awindows = len(self.tstamps) # Total analysis windows
        self.pre_silence = pre_silence # Time in seconds previous to the target 
                                       # stimulus. It is used to estimate the 
                                       # information threshold due to sporadic
                                       # firing rate

        # Obtain values
        self.info_thrs, self.mi_thrs = self._obtain_infothrs() # Information thresholds per CF
        self.wmatrix = self._obtain_weight_matrix() # Weightening matrix
        mi_samples = np.array([self.mi[cf,:]>self.mi_thrs[cf] 
                                    for cf in range(self.ncfs)])
        self.ti_samples = np.array([self.ti[cf,:]>self.info_thrs[cf]
                                    for cf in range(self.ncfs)])
        self.out_samples = self.ti_samples * mi_samples
        [self.wmi, self.samii] = self._calculate_samii()
        return
    
    def _obtain_weight_matrix(self):
        # To take into account the most important samples in the calculation of the
        # samii, a weightening matrix is used over the mutual information signal. 
        # The weights depend on the the sum of the received and transmitted 
        # information.
        mi_diff = np.array([self.mi[cf,:] - self.mi_thrs[cf] for cf in range(self.ncfs)])
        mi_diff[mi_diff < 0] = 0
        weight_matrix = self.ncfs*np.array(
            [   np.ones(self.ncfs) * 
                (1/self.pi[:,s]) / 
                ((1/self.pi[:,s]).sum())
            for s in range(self.total_awindows)
            ]).transpose()
        # weight_matrix = np.ones((self.total_awindows, self.ncfs))
        weight_matrix[weight_matrix != weight_matrix] = 0
        weight_matrix[self.pi == 0] = 0
        weight_matrix[self.mi == 0] = 0
        return weight_matrix

    def _obtain_infothrs(self):
        pre_samples = self.tstamps < self.pre_silence
        thrs = [    self.ti[cf,:][pre_samples].mean() + 
                    3*self.ti[cf,:][pre_samples].std()
                for cf in range(self.ncfs)]
        
        mi_thrs = [ self.mi[cf,:][pre_samples].mean() + 
                    3*self.mi[cf,:][pre_samples].std()
                for cf in range(self.ncfs)]
        return thrs, mi_thrs

    def _calculate_samii(self):
        '''
        samii is the weightened average of "mutual information" (mi) within the 
        samples where that is over the spontaneous spike activity.

        Input:
            - ear: Is a dictionary that contains matrices for the mutual information
                   ['mi'], perceived information ['pi'], and transmitted information
                   ['ti']. Every row corresponds to a center frequency in "self.cfs"
                   and every column is the instantaneous measure of information at 
                   every time stamp in "self.tstamps".
        
        Notes:
            "Information" is defined as the entropy of the spike activity in a group
            of fibers.
        '''

        weighted_mi = self.mi*self.out_samples#*self.wmatrix
        #weighted_mi[weighted_mi != weighted_mi] = 0
        #weighted_mi[self.ti == 0] = 0
        total_info_samples = self.ti_samples.sum()
        
        # mutual information speech intelligibility index
        samii = weighted_mi.sum() / total_info_samples

        return weighted_mi, samii

class experiment:
    def __init__(self, data, name, pre_silence):

        self.name = name

        # Ears
        self.left = data['left']
        self.right = data['right']
        self.binaural = data['binaural']

        # Information about center frequencies and sample time
        self.cfs = np.array(data['CFs'])
        self.tstamps = np.array(data['timeStamps'])

        # Mutual information speech intelligibility index
        self.samii_l = samii(self.left,self.cfs,self.tstamps, pre_silence)
        self.samii_r = samii(self.right,self.cfs,self.tstamps, pre_silence)
        self.samii_b = samii(self.binaural,self.cfs,self.tstamps, pre_silence)
        return
    
    def get_samii(self):
        return max(self.samii_l.samii, self.samii_r.samii, self.samii_b.samii)

    def _plot_ear(self, samii, ofile):

        # plot code
        fig1, ax_ti = plt.subplots()
        fig2, ax_pi = plt.subplots()
        fig3, ax_mi = plt.subplots()
        fig, (ax_mir, ax_aux1, ax_aux2) = plt.subplots(1, 3, sharex=True, sharey=True)

        # Plot Mutual Information
        im_mi = ax_mi.pcolormesh(self.tstamps, self.cfs, samii.mi, cmap='gray')
        ax_mi.set_title('Mutual Information "I(S|R)"')
        ax_mi.set_yscale('log')
        ax_mi.set_yticks([300, 1000, 7000])

        # Plot Sacaling Factor
        im_mir = ax_mir.pcolormesh(self.tstamps, self.cfs, samii.wmatrix, cmap='gray')
        ax_mir.set_title('Weighting Matrix "ω"')
        ax_mir.set_yscale('log')

        # Plot Transmitted Information
        im_ti = ax_ti.pcolormesh(self.tstamps, self.cfs, samii.ti, cmap='gray')
        ax_ti.set_title('Transmitted Information "H(S)"')
        ax_ti.set_yscale('log')
        ax_ti.set_yticks([300, 1000, 7000])

        # Plot Recived Information
        im_pi = ax_pi.pcolormesh(self.tstamps, self.cfs, samii.pi, cmap='gray')
        ax_pi.set_title('Perceived Information "H(R)"')
        ax_pi.set_yscale('log')
        ax_pi.set_yticks([300, 1000, 7000])

        # Plot samples
        im_aux1 = ax_aux1.pcolormesh(self.tstamps, self.cfs, 1*samii.ti_samples + 1*samii.out_samples, cmap='gray')
        ax_aux1.set_title('Used samples "Z" and "Z_R"')
        ax_aux1.set_yscale('log')

        # Plot Output
        im_aux2 = ax_aux2.pcolormesh(self.tstamps, self.cfs, samii.wmi, cmap='gray')
        ax_aux2.set_title('Output')
        ax_aux2.set_yscale('log')

        # Axes
        ax_aux1.set_xlabel('Time [s]')
        ax_aux2.set_xlabel('Time [s]')
        ax_mi.set_xlabel('Time [s]')
        ax_pi.set_xlabel('Time [s]')
        ax_ti.set_xlabel('Time [s]')
        ax_mi.set_ylabel('Frequency [Hz]')
        ax_ti.set_ylabel('Frequency [Hz]')
        ax_pi.set_ylabel('Frequency [Hz]')
        ax_aux1.set_ylabel('Frequency [Hz]')

        # Colorbars
        fig3.colorbar(im_mi, ax=ax_mi, label='Information [bits]')
        fig.colorbar(im_mir, ax=ax_mir, label='Information weight')
        fig1.colorbar(im_ti, ax=ax_ti, label='Information [bits]')
        fig2.colorbar(im_pi, ax=ax_pi, label='Information [bits]')
        fig.colorbar(im_aux1, ax=ax_aux1, label='Unused-Transmitted-Used')
        fig.colorbar(im_aux2, ax=ax_aux2, label='Weighted information [bits]')
        fig.tight_layout()
        fig.set_size_inches(9, 5)
        fig1.savefig(ofile+'ti.png')
        fig2.savefig(ofile+'pi.png')
        fig3.savefig(ofile+'mi.png')
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        plt.close(fig)
        return
    
    def generate_plots(self, odir='./'):
        ofile = odir + self.name
        self._plot_ear(self.samii_l, ofile + '_left.png')
        self._plot_ear(self.samii_r, ofile + '_right.png')
        self._plot_ear(self.samii_b, ofile + '_binaural.png')
        return