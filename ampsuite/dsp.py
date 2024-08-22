import pandas as pd
import numpy as np
import scipy as scipy
import warnings
from scipy import signal
import matplotlib.pyplot as plt

PI = 4*np.arctan(1)
EFACT = 1/np.log(10)

def bandpass(data, sampling_rate, low_freq=None, high_freq=None, center_freq=None, 
             octave_ratio=None, order=2, zerophase=True, filter_type='butter', figure=False):
    """
    IIR-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``order`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design) and :func:`scipy.signal.sosfilt` (for applying the filter).

    :param data: Data to filter.
    :param sampling_rate: Sampling rate in Hz.
    :param low_freq: Pass band low corner frequency.
    :param high_freq: Pass band high corner frequency.
    :param center_freq: Center frequency for bandpass filter.
    :param octave_ratio: Octave ratio for bandpass filter.
    :param order: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards. This results in twice the filter order but zero phase shift in the resulting filtered trace.
    :param filter_type: Type of filter to use. Default is 'butter'.
    :param figure: If True, plot the frequency response of the filter.
    :return: Filtered data.
    """
    
    nyquist = 0.5 * sampling_rate
    
    if center_freq != None:
        low_freq = np.round(fc / 2**(1/(2*octave_ratio)),decimals=3)
        high_freq = np.round(fc * 2**(1/(2*octave_ratio)),decimals=3)
    
    # checks
    if high_freq - nyquist > -1e-6:
        msg = f"Selected high frequency ({high_freq}) of bandpass is at or above nyquist ({nyquist}). The data will be high-passed instead."
        warnings.warn(msg)
        return highpass(data, sampling_rate, low_freq, order=order,
                        zerophase=zerophase,filter_type=filter_type, figure=figure)
    
    if low_freq > nyquist:
        msg = f"Selected low frequency ({high_freq}) is above Nyquist ({nyquist})."
        raise ValueError(msg)
        
        
    #z, p, k = signal.iirfilter(order, Wn, btype='bandpass', fs=sampling_rate, output='zpk')

    sos = signal.iirfilter(order, [low_freq,high_freq], btype='bandpass', ftype=filter_type,fs=sampling_rate, output='sos')
    
    if figure:
        w, h = signal.sosfreqz(sos, fs=sampling_rate)
        if zerophase:
            # Compute frequency response for filter passed twice
            h = h**2
        # Plot frequency response of the filter
        fig, ax = plt.subplots()
        # ax.plot(w, 20 * np.log10(abs(h)))
        # ax.plot(w, abs(h))
        ax.plot(w, abs(h))
        ax.set(title='Butterworth bandpass filter frequency response',
               xlabel='Frequency [Hz]', ylabel='Amplitude')
        ax2 = ax.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g-', label='Phase Response')
        ax2.set_ylabel('Phase [rad]', color='g')
        ax.axvline(low_freq, color='r', ls='--', alpha=0.5)
        ax.axvline(high_freq, color='r', ls='--', alpha=0.5)

        ax.grid()
        #plt.savefig('butter_4-8.png')
        plt.show()
    
    if zerophase:
        return signal.sosfiltfilt(sos, data)
    else:
        return signal.sosfilt(sos, data)
    
def highpass(data, sampling_rate, freq, order=2, zerophase=True, filter_type='butter', figure=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design) and :func:`scipy.signal.sosfilt` (for applying the filter).

    :param data: Data to filter.
    :param sampling_rate: Sampling rate in Hz.
    :param freq: Filter corner frequency.
    :param order: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards. This results in twice the number of corners but zero phase shift in the resulting filtered trace.
    :param filter_type: Type of filter to use. Default is 'butter'.
    :param figure: If True, plot the frequency response of the filter.
    :return: Filtered data.
    """
    nyquist = 0.5 * sampling_rate
    # raise for some bad scenarios
    if freq > nyquist:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    
    sos = signal.iirfilter(order, freq, btype='highpass', ftype=filter_type, fs=sampling_rate, output='sos')
    
    if figure:
        w, h = signal.sosfreqz(sos, fs=sampling_rate)
        if zerophase:
            # Compute frequency response for filter passed twice
            h = h**2
        # Plot frequency response of the filter
        fig, ax = plt.subplots()
        # ax.plot(w, 20 * np.log10(abs(h)))
        # ax.plot(w, abs(h))
        ax.plot(w, abs(h))
        ax.set(title='Butterworth bandpass filter frequency response',
               xlabel='Frequency [Hz]', ylabel='Amplitude')
        ax2 = ax.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g-', label='Phase Response')
        ax2.set_ylabel('Phase [rad]', color='g')
        ax.axvline(freq, color='r', ls='--', alpha=0.5)
        ax.grid()
        #plt.savefig('butter_4-8.png')
        plt.show()
    
    if zerophase:
        return signal.sosfiltfilt(sos, data)
    else:
        return signal.sosfilt(sos, data)

def lowpass(data, sampling_rate, freq, order=2, zerophase=True, filter_type='butter', figure=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design) and :func:`scipy.signal.sosfilt` (for applying the filter).

    :param data: Data to filter.
    :param sampling_rate: Sampling rate in Hz.
    :param freq: Filter corner frequency.
    :param order: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards. This results in twice the number of corners but zero phase shift in the resulting filtered trace.
    :param filter_type: Type of filter to use. Default is 'butter'.
    :param figure: If True, plot the frequency response of the filter.
    :return: Filtered data.
    """
    nyquist = 0.5 * sampling_rate
    
    # raise for some bad scenarios
    if freq > nyquist:
        freq = nyquist
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high cut."
        warnings.warn(msg)
    sos = iirfilter(order, freq, btype='lowpass', ftype='butter',
                        output='sos')
    if figure:
        w, h = signal.sosfreqz(sos, fs=sampling_rate)
        if zerophase:
            # Compute frequency response for filter passed twice
            h = h**2
        # Plot frequency response of the filter
        fig, ax = plt.subplots()
        # ax.plot(w, 20 * np.log10(abs(h)))
        # ax.plot(w, abs(h))
        ax.plot(w, abs(h))
        ax.set(title='Butterworth bandpass filter frequency response',
               xlabel='Frequency [Hz]', ylabel='Amplitude')
        ax2 = ax.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w, angles, 'g-', label='Phase Response')
        ax2.set_ylabel('Phase [rad]', color='g')
        ax.axvline(freq, color='r', ls='--', alpha=0.5)
        ax.grid()
        #plt.savefig('butter_4-8.png')
        plt.show()
    
    if zerophase:
        return signal.sosfiltfilt(sos, data)
    else:
        return signal.sosfilt(sos, data)

def find_nearest_index(array, value):
    """
    Find the index of the nearest value in an array.

    :param array: Array to search.
    :param value: Value to find the nearest to.
    :return: Tuple with the nearest value and its index.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def taper(data, max_percentage, npts, sampling_rate, type='hann', max_length=None,side='both', **kwargs):
        """
        Taper the trace.
        Optional (and sometimes necessary) options to the tapering function can
        be provided as kwargs. See respective function definitions in
        `Supported Methods`_ section below.
        :type type: str
        :param type: Type of taper to use for detrending. Defaults to
            ``'hann'``.  See the `Supported Methods`_ section below for
            further details.
        :type max_percentage: None, float
        :param max_percentage: Decimal percentage of taper at one end (ranging
            from 0. to 0.5).
        :type max_length: None, float
        :param max_length: Length of taper at one end in seconds.
        :type side: str
        :param side: Specify if both sides should be tapered (default, "both")
            or if only the left half ("left") or right half ("right") should be
            tapered.
        .. note::
            To get the same results as the default taper in SAC, use
            `max_percentage=0.05` and leave `type` as `hann`.
        .. note::
            If both `max_percentage` and `max_length` are set to a float, the
            shorter tape length is used. If both `max_percentage` and
            `max_length` are set to `None`, the whole trace will be tapered.
        ``'cosine'``
            Cosine taper, for additional options like taper percentage see:
            :func:`obspy.signal.invsim.cosine_taper`.
        ``'barthann'``
            Modified Bartlett-Hann window. (uses:
            :func:`scipy.signal.barthann`)
        ``'bartlett'``
            Bartlett window. (uses: :func:`scipy.signal.bartlett`)
        ``'blackman'``
            Blackman window. (uses: :func:`scipy.signal.blackman`)
        ``'blackmanharris'``
            Minimum 4-term Blackman-Harris window. (uses:
            :func:`scipy.signal.blackmanharris`)
        ``'bohman'``
            Bohman window. (uses: :func:`scipy.signal.bohman`)
        ``'boxcar'``
            Boxcar window. (uses: :func:`scipy.signal.boxcar`)
        ``'chebwin'``
            Dolph-Chebyshev window. (uses: :func:`scipy.signal.chebwin`)
        ``'flattop'``
            Flat top window. (uses: :func:`scipy.signal.flattop`)
        ``'gaussian'``
            Gaussian window with standard-deviation std. (uses:
            :func:`scipy.signal.gaussian`)
        ``'general_gaussian'``
            Generalized Gaussian window. (uses:
            :func:`scipy.signal.general_gaussian`)
        ``'hamming'``
            Hamming window. (uses: :func:`scipy.signal.hamming`)
        ``'hann'``
            Hann window. (uses: :func:`scipy.signal.hann`)
        ``'kaiser'``
            Kaiser window with shape parameter beta. (uses:
            :func:`scipy.signal.kaiser`)
        ``'nuttall'``
            Minimum 4-term Blackman-Harris window according to Nuttall.
            (uses: :func:`scipy.signal.nuttall`)
        ``'parzen'``
            Parzen window. (uses: :func:`scipy.signal.parzen`)
        ``'slepian'``
            Slepian window. (uses: :func:`scipy.signal.slepian`)
        ``'triang'``
            Triangular window. (uses: :func:`scipy.signal.triang`)
        """
        type = type.lower()
        side = side.lower()
        side_valid = ['both', 'left', 'right']
        npts = npts
        if side not in side_valid:
            raise ValueError("'side' has to be one of: %s" % side_valid)
        # retrieve function call from entry points
        #func = 
        if type == 'dpss':
            from scipy.signal import windows
            func = getattr(windows, 'dpss')
        elif type == 'cosine':
            from obspy.signal import invsim
            func = getattr(invsim, 'cosine_taper')
        else:
            from scipy import signal
            func = getattr(signal, type)
        
        #func = _get_function_from_entry_point('taper', type)
        # store all constraints for maximum taper length
        max_half_lengths = []
        if max_percentage is not None:
            max_half_lengths.append(int(max_percentage * npts))
        if max_length is not None:
            max_half_lengths.append(int(max_length * sampling_rate))
        if np.all([2 * mhl > npts for mhl in max_half_lengths]):
            msg = "The requested taper is longer than the trace. " \
                  "The taper will be shortened to trace length."
            warnings.warn(msg)
        # add full trace length to constraints
        max_half_lengths.append(int(npts / 2))
        # select shortest acceptable window half-length
        wlen = min(max_half_lengths)
        # obspy.signal.cosine_taper has a default value for taper percentage,
        # we need to override is as we control percentage completely via npts
        # of taper function and insert ones in the middle afterwards
        #if type == "cosine":
        #    kwargs['p'] = 1.0
        # tapering. tapering functions are expected to accept the number of
        # samples as first argument and return an array of values between 0 and
        # 1 with the same length as the data
        if 2 * wlen == npts:
            if type == 'dpss':
                #Common choices for the time half bandwidth product are: 2.5, 3, 3.5, and 4.
                #You can specify the bandwidth of the Slepian sequences in Hz by defining 
                #the time half bandwidth product as NW/Fs, where Fs is the sample rate.
                nw = 2.5
                m = (2*wlen)
                if nw == m/2:
                    nw = 2

                taper_sides = func(M=m,NW=nw, Kmax=None, **kwargs)
            elif type == 'kaiser':
                #Attenuation parameter. Beta requires a real number. 
                #Larger absolute values of Beta result in greater stopband attenuation, 
                #or equivalently greater attenuation between the main lobe and first side lobe.
                #beta_window_shape: 0 Rectangular, 5 Similar to a Hamming, 6 Similar to a Hann, 
                #8.6 Similar to a Blackman
                taper_sides = func(M=(2*wlen),beta=8.6)
            else:
                taper_sides = func(2 * wlen, **kwargs)
        else:
            if type == 'dpss':
                nw = 2.5
                m = (2*wlen+1)
                if nw == m/2:
                    nw = 2
                #Common choices for the time half bandwidth product are: 2.5, 3, 3.5, and 4.
                taper_sides = func(M=m,NW=nw, Kmax=None, **kwargs)
            elif type == 'kaiser':
                #beta_window_shape: 0 Rectangular, 5 Similar to a Hamming, 6 Similar to a Hann, 8.6 Similar to a Blackman
                taper_sides = func(M=(2*wlen+1),beta=8.6)
            elif type == 'cosine':
                taper_sides = func(2 * wlen + 1,p = max_percentage, sactaper=True, **kwargs)
            else:
                taper_sides = func(2 * wlen + 1, **kwargs)
        if side == 'left':
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - wlen)))
        elif side == 'right':
            taper = np.hstack((np.ones(npts - wlen),
                               taper_sides[len(taper_sides) - wlen:]))
        else:
            taper = np.hstack((taper_sides[:wlen], np.ones(npts - 2 * wlen),
                               taper_sides[len(taper_sides) - wlen:]))

        # Convert data if it's not a floating point type.
        if not np.issubdtype(data.dtype, np.floating):
            data = np.require(data, dtype=np.float64)

        data *= taper
        return data


def taper_adjust(taper_percent, window_length):
        """
        Calculate the time adjustment needed for tapering data.

        Parameters:
        taper_percent (float): The decimal percentage of the taper.
        window_length (float): The length of the data window in seconds.

        Returns:
        time_adjust (float): The time in seconds to adjust the arrival time to account for tapering.
        """
        taper_adjust = 1.0-(taper_percent*2)
        #number of seconds to adjust arrival for tapering data
        time_adjust = (((window_length)/taper_adjust)-(window_length))/2
        return time_adjust

def rms(x,n_samples,base='log10',unit='meters'):
    """
    Compute the RMS of an input signal over a given window length.
    
    Parameters: 
    x (array-like): Input signal, must be indexed/cut to the measurement window.
    n_samples (int): Length of the measurement window.
    base (str): Base of the logarithm, either 'log10' (default) or something else. 
    unit (str): Unit of the input data, either 'meters' (default) or 'nanometers'.
        
    Returns:
    rmsamp (float): RMS amplitude of the input signal in the specified unit and base.
    
    Note: If using 'nanometers' as the unit, the result will be -9 if rmsvalue is 0 
          and base is 'log10'. numpy.nan is returned instead of -999.
    """
    x_square = np.square(np.ravel(x))
    x_sum = np.sum(x_square)
    rmsvalue = np.sqrt(x_sum/n_samples)
    #rmsvalue = x_sum/n_samples # used for a quick comparison
    
    #python decon will scale waveform to be in meters, if not scaled
    if unit == 'nanometers':
        if base == 'log10':
            rmsamp = np.log10(rmsvalue) - 9 if rmsvalue > 0 else np.nan #-999
        else:
            rmsamp = rmsvalue
    #python decon will scale waveform to be in meters
    elif unit == 'meters':
        if base == 'log10':
            rmsamp = np.log10(rmsvalue) if rmsvalue > 0 else np.nan
        else:
            rmsamp = rmsvalue
    return rmsamp



def validate_length_zeropad(data_a,data_b):
    """
    Check if two arrays have the same length and zero pad the shorter one if not.
    
    Parameters:
    data_a (array-like): First input array.
    data_b (array-like): Second input array.
    
    Returns:
    data_a (array): First array, zero padded if it was shorter.
    data_b (array): Second array, zero padded if it was shorter.
    """
    if(len(data_a) == len(data_b)):
        print('npts equal')
    else:
        print('npts not equal')
        if len(data_a) > len(data_b):
            npts_diff = abs(len(data_a) - len(data_b))
            data_b = pad_zeros(data_b, npts_diff)
        elif len(data_a) < len(data_b):
            npts_diff = abs(len(data_a) - len(data_b))
            data_a = pad_zeros(data_a, npts_diff)
            
    return data_a, data_b

def spec_log10(amp_spectra,unit):
    """
    Compute the log10 of an amplitude spectrum, considering the input unit.
    
    Parameters:
    amp_spectra (array-like): Input amplitude spectrum.
    unit (str): Unit of the input data, either 'meters' or 'nanometers'.
    
    Returns:
    log10_spec (array): Log10 of the input amplitude spectrum.
    """
    if unit == 'meters':
        return np.log(amp_spectra)*EFACT - 9
    elif unit == 'nanometers':
        return np.log(amp_spectra)*EFACT


def bandpower(x, fs, nfft, fmin, fmax, window = 'boxcar'):
    """
    Compute the power in a frequency band of a signal using Welch's method.
    
    Parameters:
    x (array-like): Input signal.
    fs (float): Sampling frequency of the signal.
    nfft (int): Length of the FFT used, must be even.
    fmin (float): Lower boundary of the frequency band of interest.
    fmax (float): Upper boundary of the frequency band of interest.
    
    Returns:
    bandpower (float): Power in the specified frequency band.
    """
    f, Pxx = scipy.signal.periodogram(x,fs=fs,window=window, nfft=nfft)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    #return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])



def bandpower_rms(bandpower_value,base,unit):
    """
    Compute the RMS from the power in a frequency band.
    
    Parameters:
    bandpower_value (float): Power in the frequency band.
    base (str): Base of the logarithm, either 'log10' or something else.
    unit (str): Unit of the input data, either 'meters' or 'nanometers'.
    
    Returns:
    rmspower (float): RMS computed from the power in the specified unit and base.
    """
    rmsvalue = np.sqrt(bandpower_value)
    if unit == 'meters':
        if base == 'log10':
            rmspower = np.log(rmsvalue)*EFACT - 9 if rmsvalue > 0 else np.nan
        else:
            rmspower = rmsvalue
    elif unit == 'nanometers':
        if base == 'log10':
            rmspower = np.log(rmsvalue)*EFACT if rmsvalue > 0 else np.nan
        else:
            rmspower = rmsvalue
    return rmspower


def column(matrix, i):
    """
    Return a specific column from a matrix as a list.

    Parameters:
    matrix (list of lists): The input matrix.
    i (int): The index of the column to return.

    Returns:
    column_list (list): The i-th column of the matrix.
    """
    return [row[i] for row in matrix]



def sacfft(data,sampling_rate):
    """
    Compute the FFT of a signal in the same way as SAC (Seismic Analysis Code).

    Parameters:
    data (array-like): Input signal.
    sampling_rate (float): Sampling rate of the signal.

    Returns:
    mag_fft (array-like): Magnitude of the FFT.
    phase_fft (array-like): Phase of the FFT.
    freqs (array-like): Frequencies corresponding to the FFT values.
    """
    Fs=sampling_rate           # sampling frequency
    nyquist=Fs/2;                  # Nyquist frequency

    # Next highest power of 2 greater than or equal to
    # length(x):

    nfft=int(2**(np.ceil(np.log(len(data))/np.log(2))))
    # Take fft, padding with zeros, length(FFTX)==NFFT

    fftx=np.fft.fft(data,n = nfft);
    #avearge each fft result
    fftx = np.mean(fftx,axis=1)# matlab mean(fftx,2)???
    #sum each fft result
    #FFTX = sum(FFTX,2);

    NumUniquePts = np.ceil((nfft+1)/2)
    # fft is symmetric, throw away second half
    idx = linspace(0,NumUniquePts-1,NumUniquePts)
    fftx=fftx[idx];
    mag_fft=np.abs(fftx);            # Take magnitude of X
    phase_fft=np.angle(fftx);            # Take phase of X

    # Multiply by 2 to take into account the fact that we
    # threw out second half of FFTX above
    #---MATLAB APPROACH
    mag_fft=mag_fft*2;

    mag_fft[0]= mag_fft[0]/2
    if not np.remainder(NFFT,2):
        mag_fft[len(mag_fft)]=mag_fft[len(mag_fft)]/2
    # comment out above to follow SAC and Glenn's approach below

    # Scale the FFT so that it is not a function of the 
    # length of x.


    #Division by N: amplitude = abs(fft (signal)/N), where "N" is the signal length;


    #---SAC APPROACH
    #MX=MX*delta; # Glenn's approach
    
    vec = linspace(0,NumUniquePts-1,NumUniquePts)
    freqs=ff_vec.T*2*nyquist/nfft;
    
    return mag_fft,phase_fft,freqs

#padding routines below
def pad_zeros(x,npts):
    """
    Pad an array with zeros at the end.
    
    Parameters: 
    x (array-like): Input array.
    npts (int): Number of zeros to pad.
    
    Returns:
    padded_array (array): Input array padded with zeros at the end.
    """
    return np.pad(x, (0, npts), 'constant')

def mtspec_pad(x,nfft):
    """
    Pad an array with zeros to reach the desired length for an FFT.
    
    Parameters:
    x (array-like): Input array.
    nfft (int): Desired length of the FFT.
    
    Returns:
    x_out (array): Input array padded with zeros to the desired FFT length.
    """
    if(nfft > len(x)):
        npts_diff = abs(len(x) - nfft)
        x_out =  pad_zeros(x, npts_diff)
    else:
        x_out = x
    return x_out
    

def pad_with(vector, pad_width, iaxis, kwargs):
    """
    Pad an array with a specified value.
    
    Parameters:
    vector (array-like): Input array to pad.
    pad_width (tuple): Number of values padded to the edges of each axis.
    iaxis (int): Axis along which to pad.
    kwargs (dict): Keyword arguments, expecting 'padder' which is the value to pad with.
    
    Returns:
    vector (array): Padded array.
    """
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:                      # <-- the only change (0 indicates no padding)
        vector[-pad_width[1]:] = pad_value

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    """
    Pad an array with zeros along a specified axis to a target length.
    
    Parameters:
    array (np.ndarray): Input array to pad.
    target_length (int): The length to pad the array to.
    axis (int): The axis along which to pad (default=0).
    
    Returns:
    padded_array (np.ndarray): The padded array.
    """
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def pad_zeros_even():
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

