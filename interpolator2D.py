import numpy as np
from numpy import polynomial
from scipy.interpolate import CubicSpline,RBFInterpolator
from scipy.interpolate import CloughTocher2DInterpolator as CT2D

import deepcr_helper as crhp
from deepcr_helper import get_pulsetime, get_phase_constant, get_fluence
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.utilities import units,fft

# class interpolator2D:
#     def __init__(self,efields,positions):
#         self._positions = positions
#         amplitudes = []
#         phases = []
#         for efield in efields:
#             _t = efield.get_times()
#             sampling_rate = 1./(_t[1]-_t[0])
#             amps = efield.get_frequency_spectrum(sa)

class interpolator2D:
    def __init__(self,efields,position2d,
                 sampling_rate=2*units.GHz,trace_length=2048):
        self._positions = position2d
        self._amplitudes = [] ## amplitudes on common grid
        self._residual_phasor = [] ## phase residual in common grid
        self._pulse_time = [] ## 3 for 3pol
        self._phase_constant = [] ## 1 for each pol
        self._trace_start_time = []
        self._log_amplitudes = []

        
        self._sampling_rate = sampling_rate
        self._frequencies = np.fft.rfftfreq( trace_length,1./self._sampling_rate)
        
        for efield in efields:
            t = efield.get_times()
            pulse_time = np.array(get_pulsetime(efield))
            
            trace_sampling_rate = 1./(t[1]-t[0])
            trace_start_time = t[0]
            
            #waveform = efield.get_trace()
            #spectrum = fft.time2freq( waveform,trace_sampling_rate )
            spectrum = efield.get_frequency_spectrum()
#             print("spectrum",spectrum)
            
            ff = efield.get_frequencies()
            dff = ff[1]-ff[0]
            shifted_spectrum = np.array([sp * np.exp(1.j*2*np.pi*ff*(pulse_time[ip] - trace_start_time)) for ip,sp in enumerate(spectrum)])
            phase_constant = np.exp( 1.j*np.angle(np.sum(shifted_spectrum,axis=-1)))
            
            residual_phase =np.array([ np.angle(ssp/phase_constant[ip])  for ip,ssp in enumerate(shifted_spectrum)])
            residual_phasor = np.exp(1.j*residual_phase)

            amp = []
            logamp = []
            res_phasor = []
            for pol in range(3):
                sp = spectrum[pol]
                Aspline = CubicSpline(ff,np.abs(sp),bc_type="clamped",extrapolate=True)
                df = self._frequencies[1]-self._frequencies[0]
                _amp = Aspline(self._frequencies) #* ( dff/df)**0.5
                #_amp = np.interp(self._frequencies,ff,np.abs(sp))
#                print("spline amp",_amp)

                logAspline = CubicSpline(ff,np.log10(np.abs(sp)),bc_type="natural",extrapolate=False ) 
                _logamp = logAspline(self._frequencies) #+ np.log10(dff/df)/2
                amp.append(_amp)
                logamp.append(_logamp)
                
                res_phase = -np.unwrap(np.angle(residual_phasor[pol]))
                Pspline = CubicSpline(ff,res_phase,extrapolate=False)
                _resphase = Pspline(self._frequencies)
                res_phasor.append(np.exp(-1.j*_resphase))
                
            
            
            self._amplitudes.append(np.array(amp)) 
            self._log_amplitudes.append(np.array(logamp))
            
            self._residual_phasor.append(np.array(res_phasor))
            
            self._pulse_time.append(pulse_time)
            self._phase_constant.append(phase_constant)
            self._trace_start_time.append(trace_start_time)
        
        self._amplitude_interp = []
        self._log_amplitude_interp = []
#         self._residual_phasor_interp = []
#         print(len(self._pulse_time),len(self._positions))

#         self._pulse_time_interp = CT2D(self._positions,self._pulse_time)
        self._pulse_time_interp = RBFInterpolator(self._positions,self._pulse_time,
                                                  kernel='thin_plate_spline',smoothing=0.,degree=5)
        self._phase_constant_interp = CT2D(self._positions,self._phase_constant,fill_value=1.+0.j)
#         self._phase_constant_interp = RBFInterpolator(self._positions,self._phase_constant,
#                                                   kernel='thin_plate_spline',smoothing=0.,degree=5)
        for pol in range(3):
            amp_intp = CT2D(self._positions,np.array(self._amplitudes)[:,pol,:],fill_value=0.)
            logamp_intp = CT2D(self._positions,np.array(self._log_amplitudes)[:,pol,:],fill_value=0.)
            
            self._amplitude_interp.append(amp_intp)
            self._log_amplitude_interp.append(logamp_intp)

            
    def __call__(self,target2d,prepulse=100*units.ns):
        return_efields = []
        for target in target2d:
            distance = np.linalg.norm( np.array(self._positions) - np.array(target),axis=-1 )
            argmin = np.argmin(distance)
            
            freq = self._frequencies
            
#            recon_amp =np.array([ self._amplitude_interp[pol](target[0],target[1]) for pol in range(3)])
#             print('recon_amp',recon_amp)
            recon_amp =np.array([ 10**(self._log_amplitude_interp[pol](target[0],target[1])) for pol in range(3)])


            recon_phase_const = np.array(self._phase_constant_interp(target[0],target[1]))
            recon_phase_const = np.exp(1.j*np.angle(recon_phase_const))
            
            recon_pulse_time = np.array(self._pulse_time_interp([target])[0])
            recon_res_phasor = np.array(self._residual_phasor)[argmin]
            
            trace_start_time = np.min(recon_pulse_time) - prepulse#self._trace_start_time[argmin]
#             print(recon_pulse_time)
            
            recon_phasor = np.array([ np.exp(-1.j*(2.*np.pi*freq*(recon_pulse_time[pol] - trace_start_time))) for pol in range(3)])
            recon_phasor = np.array([ recon_phase_const[pol]*recon_phasor[pol] for pol in range(3)])
            recon_phasor = np.array([ recon_res_phasor[pol]*recon_phasor[pol] for pol in range(3)])
            recon_Efft = np.array([ recon_amp[pol]*recon_phasor[pol] for pol in range(3)])
#             recon_trace = np.array([ fft.freq2time(e,self._sampling_rate)for e in recon_Efft])
            
#             print('recon_trace',recon_Efft)
#             print(trace_start_time,recon_pulse_time)
#             plt.plot(freq,np.abs(recon_Efft.T),ls='--')
            for i in range(3): 
                pass
                #plt.plot(fft.freq2time(recon_Efft[i],self._sampling_rate))
                #plt.plot(freq,-np.unwrap(np.angle(recon_phasor[i])))
                #plt.plot(freq,-np.unwrap(np.angle(recon_res_phasor[i])))
                #plt.plot(np.abs(recon_Efft[i]) )
            
            recon_E = ElectricField(0)
            recon_E.set_position([target[0],target[1],0])
#            recon_E.set_trace(recon_trace,self._sampling_rate)
            recon_E.set_frequency_spectrum(recon_Efft,self._sampling_rate)
            recon_E.set_trace_start_time(trace_start_time)
            
            return_efields.append(recon_E)
        return return_efields
            
             

        