# -*- coding: utf-8 -*-
"""
Python wrapper providing access to C-language API provided by HTDDSDll.h.

Example usage:
ht = HantekDDS()
ht.drive_periodic(amplitude=2.0, frequency=12000., function = 'sine')
"""

import ctypes
import os
import numpy as np
import threading
import time

this_dir = os.path.dirname(os.path.realpath(__file__))
HTDLL = ctypes.windll.LoadLibrary(this_dir + '/HTDDSDll')

FMAX = 200e6  # Maximum DAC sample rate


class HantekDDS():

    """ Interface to Hantek 1025G function generator."""

    def __init__(self, dev=0):
        """ Default constructor."""
        self.idVendor = '0x0483'
        self.idProduct = '0x5726'
        self.revNumber = '0200'
        self.dev_id = None
        self.connect(dev)
        self.param_dict = {'square_duty': 0.5,
                           'triangle_duty': 0.5,
                           'trap_rise': 0.2,
                           'trap_high': 0.3,
                           'trap_fall': 0.2,
                           'exp_mode': 'decay',
                           'exp_time': 0.001,
                           }
        self.halt = False

    def connect(self, dev=0):
        """ Verify existence of device and get device id."""
        for attempt in range(10):
            n_devices = search()
            if n_devices:
                if bool(check(dev)):
                    self.dev_id = dev
                    print('Connected as device {:d}'.format(dev))
                    return
        print('ERROR: Failed to establish connection with HantekDDS.')

    def set_divisor(self, div):
        """ Set the DAC sample rate divisor."""
        setDivNum(self.dev_id, div)

    def set_waveform(self, data):
        """ Set the waveform buffer."""
        data = np.array(data)  # Ensure data is in ndarray
        dlen = len(data)
        if dlen > 4096:
            print('WARNING: Hantek 1025G buffer limited to 4096 samples -- ' +
                  'Truncating data to 4096 elements')
            data = data[:4096]
        if download(self.dev_id, data.tobytes(), dlen):
            pass
        else:
            print('HantekDDS method set_waveform() returned failure code.')

    def drive_periodic(self, amplitude=1.0, frequency=1000.0,
                       offset=0, phase=0, function='sine', **kwargs):
        """ Direct the device to drive a periodic function.

        Args:
          amplitude: Amplitude of signal in Volts (1/2 V_pp)
          frequency: Frequency of signal in Hertz
          offset: Offset of signal in Volts.
          phase: Offset of phase in periods (phase = 0 is equivalent to
            phase=1, phase = 0.5 is 180 degrees out of phase from phase = 0.
          function: Type of periodic function to drive.

            Valid Types
            -----------
              'sine': Sine wave
              'square': Square wave, param 'square_duty' for high duty cycle
                        specified as float in range (0,1)
              'triangle': Triangle wave, param 'duty' for rise duty cycle
                          specified as float in range (0,1)
              'ramp': Synonym for 'triangle'
              'sawtooth': Synonym for 'triangle'
              'trap': Trapezoidal wave, params 'trap_rise', 'trap_high',
                      and 'trap_fall' for duty cycle for rise, high, and fall
                      segments specified as floats > 0 with sum <= 1.
              'exp': Exponential wave, params 'exp_mode' (may be 'rise' or
                     'saturate') and 'exp_time' (time constant in seconds).
        """
        for key,val in kwargs.items():
            if key in self.param_dict:
                self.param_dict[key] = val

        frequency = validate_frequency(frequency)
        phase = phase % 1
        if (amplitude + abs(offset)) > 3.5:
            print('WARNING: amplitude and offset specified will cause ' +
                  'the signal to be clipped.  Consider confining the range ' +
                  'to ±3.5 volts.')

        if function not in periodic_functions.keys():
            print('WARNING: function type for periodic function not found. ' +
                  'Valid values are ' +
                  'Defaulting to sine wave.')
            function = 'sine'

        div, length = get_freq_settings(frequency)
        f = periodic_functions[function]
        signal = f(amplitude, frequency, offset, phase,
                   length, **(self.param_dict))
        digital = np.short(voltage_adc(signal))
        self.set_waveform(digital)
        self.set_divisor(div)

    def frequency_scan(self, f_start, f_end, nsteps=10, delta=0, dwell=5,
                       ltype='linear', n_repeats=1, amplitude=1.0, offset=0.0):
        """ Scan through a range of frequencies.

        Args:
          f_start: Scan starting frequency in Hertz
          f_end: Scan ending frequency in Hertz
          nsteps: The number of steps to take between f_start and f_end.
          delta: The arithmetic or geometric difference between steps.
                 If non-zero, this setting will override nsteps.
                 Additionally, the frequency sweep may go past f_end.
          n_repeats: The number of times to loop through the entire frequency
                     scan.  Set to -1 to make continuous.
        """
        f_start = validate_frequency(f_start)
        f_end = validate_frequency(f_end)
        if ltype in ['log', 'logarithmic']:
            if delta > 0:
                nsteps = np.ceil(np.log(f_end / f_start) / np.log(delta))
                fspace = np.product(np.append(f_start, delta*np.ones(nsteps)))
            else:
                fspace = np.logspace(np.log10(f_start),
                                     np.log10(f_end), nsteps)
        if ltype in ['lin', 'linear']:
            if delta != 0:
                fspace = np.arange(f_start, f_end + delta, delta)
            else:
                fspace = np.linspace(f_start, f_end, nsteps)
            fspace = np.linspace(f_start, f_end, nsteps)
        self.scanner = threading.Thread(target=self.freq_scan_threaded,
                                        args=(fspace, amplitude,
                                              offset, dwell, n_repeats))
        self.halt = False
        self.scanner.start()
#        self.freq_scan_threaded(fspace, amplitude, offset, dwell, n_repeats)

    def freq_scan_threaded(self, fspace, amp, offset, dwell, n_repeats):
        """ Frequency scan to be started in non-blocking threaded process."""
        queue = int(n_repeats)
        while queue != 0:
            queue -= 1
            for f in fspace:
                if self.halt:
                    return None
                self.drive_periodic(amplitude=amp, frequency=f, offset=offset)
                time.sleep(dwell)

    def stop(self):
        """ Halt the threaded scan process."""
        self.halt = True


def search():
    """ Search for connected Hantek DDS devices.

    Will not return identity of devices, only the number connected.
    """
    fh = HTDLL[7]  # function handle for DDSSearch()
    return fh()


def setFrequency(dev, f, npoints, nperiods):
    """ Set parameters appropriate for a particular frequency.

    Args:
    f: frequency to be set
    """
    fh = HTDLL[11]
    return fh(dev, f, npoints, nperiods)


def getMeasure():
    """ Retrieve the value from the frequency/counter function."""
    raise NotImplementedError


def setSingleWave(dev, state):
    """ Set device to output continuously, or to output only on trigger."""
    fh = HTDLL[13]
    fh(dev, state)


def resetCounter():
    """ Reset the count of the counter function."""
    fh = HTDLL[6]
    return fh(dev)


def setTrigger(dev, internal, falling):
    """ Set trigger parameters."""
    fh = HTDLL[14]
    return fh(dev, internal, falling)


def setDigitalIn():
    """ Read input values (low or high) from digital input ports.

    Lowest four bits of the read array are read from the 4 input pins.
    """
    raise NotImplementedError


def setDIOMode():
    """ Switch between programmable output and generator output."""
    raise NotImplementedError


def setDigitalOut():
    """ Set the output values (low or high) of the digital output ports.

    Lowest twelve bits of the output array are set on the 12 output pins.
    """
    raise NotImplementedError


def download(dev, buf, num):
    """ Transfer a waveform specification to the device waveform buffer.

    Args:
      dev: Device index
      buf: Pointer to short type array (only 12 bits used)
      num: Number of elements in buffer

    Returns:
      bool: 1 on success, 0 on failure
    """
    fh = HTDLL[2]
    return fh(dev, buf, num)


def check(dev):
    """ Check the status of the device.

    Determine whether or not the device can be seen by the operating system

    Args:
      dev: Index of Hantek device.

    Returns:
      bool: status of connection (True = success, False = failure)

    Argument ``dev`` seems to be an internal index, presumably out of the
    number of Hantek devices connected.  If only one Hantek device is
    connected, ``dev`` should probably always be 0.
    """
    fh = HTDLL[1]
    return fh(dev)


def setPowerOnOutput(dev, state):
    """ Set whether waveform should output upon device power-on

    If true, output stored function.  Otherwise, wait for a trigger (?) or
    explicit command to output.
    """
    fh = HTDLL[12]
    return fh(dev, state)


def getDivNum(dev):
    """ Get the DAC frequency divisor

    Args:
      dev: device index

    Returns:
      int: Divisor index number
    """
    fh = HTDLL[4]
    return fh(dev)


def setDivNum(dev, div):
    """ Set the DAC sampling rate divisor.

    Args:
      div: divisor to apply to DAC sampling frequency.

    Values of div from 1:n give sampling rates of fmax / (2 * div).
    When div == 0, sampling rate is ``FMAX`` = 200MS/sec.
    """
    fh = HTDLL[10]
    return fh(dev, div)


def validate_frequency(frequency):
    if frequency > 1e8:
        print('WARNING: frequency {:g} is outside the '.format(frequency) +
              'possible range of frequencies for the Hantek 1025G.  ' +
              'Setting frequency to 10MHz for your "convenience".')
        frequency = 1e7
    if frequency < 1:
        print('WARNING: frequency {:g} is outside the '.format(frequency) +
              'possible range of frequencies for the Hantek 1025G.  ' +
              'Setting frequency to 10Hz for your "convenience".')
        frequency = 10
    return frequency


def get_freq_settings(frequency):
    """ Compute the DAC divisor and sample length for a a target frequency.

    Args:
      frequency: Target frequency in Hertz.

    Returns:
      int: divisor to be passed to setDivNum
      int: length to be passed to signal synthesis routines

    This function returns the same information that setFrequency does, but
    without modifying the DAC divisor.
    """
    frequency = validate_frequency(frequency)
    divisor = int(FMAX/4096/frequency)
    length = int(FMAX / max(1, 2 * divisor) / frequency)
    return divisor, length


def voltage_adc(voltage):
    """ Convert voltage in the range from -3.5V to +3.5V to a 12-bit level."""
    return np.minimum(4095, np.maximum(0, (3.5 - voltage) / 1.70898375e-3))

# Periodic Functions
# ==================


def sine_wave(amplitude, frequency, offset, phase, length, **kwargs):
    """ Construct one period of a sine wave."""
    arg = np.linspace(0, 2*np.pi, length, endpoint=False)
    signal = amplitude * np.sin(arg) + offset
    return np.roll(signal, int(phase * length))


def square_wave(amplitude, frequency, offset, phase, length, **kwargs):
    """ Construct one period of a square wave."""
    duty = kwargs['square_duty']
    cutoff = int(duty * length)
    signal = np.empty(length)
    signal[:cutoff] = amplitude + offset
    signal[cutoff:] = -amplitude + offset
    return np.roll(signal, int(phase * length))


def triangle_wave(amplitude, frequency, offset, phase, length, **kwargs):
    """ Construct one period of a triangle (sawtooth, ramp) wave."""
    duty = kwargs['triangle_duty']
    signal = np.empty(length)
    cutoff = int(duty * length)
    signal[:cutoff] = np.linspace(-amplitude, amplitude,
                                  cutoff, endpoint=False)
    signal[cutoff:] = np.linspace(amplitude, -amplitude,
                                  length - cutoff, endpoint=False)
    signal += offset
    return np.roll(signal, int(phase * length))


def exponential(amplitude, frequency, offset, phase, length, **kwargs):
    """ Construct an exponentially decaying/saturating wave."""
    tau = kwargs['exp_time']
    exp_mode = kwargs['exp_mode']
    time = np.linspace(0, 1/frequency, length, endpoint=False)
    if exp_mode == 'saturate':
        signal = amplitude * (1 - np.exp(-time / tau) + offset)
    if exp_mode == 'decay':
        signal = amplitude * (np.exp(-time / tau) + offset)

    return np.roll(signal, int(phase * length))


def trapezoid(amplitude, frequency, offset, phase, length, **kwargs):
    """ Construct a trapezoidal wave."""
    d_rise = kwargs['trap_rise']
    d_high = kwargs['trap_high']
    d_fall = kwargs['trap_fall']
    if d_high + d_rise + d_fall > 1:
        print('Duty cycle greater than 1 specified for trapezoidal wave. ' +
              'Reseting to reasonable parameters.')
        d_rise = 0.2
        d_high = 0.3
        d_fall = 0.2
    l_r = c_r = int(d_rise * length)
    l_h = int(d_high * length)
    c_h = c_r + l_h
    l_f = int(d_fall * length)
    c_f = c_h + l_f
    signal = np.empty(length)
    signal[:c_r] = np.linspace(-amplitude, amplitude, l_r, endpoint=False)
    signal[c_r:c_h] = amplitude
    signal[c_h:c_f] = np.linspace(amplitude, -amplitude, l_f, endpoint=False)
    signal[c_f:] = -amplitude
    return np.roll(signal, int(phase * length))

# Dictionary of all periodic functions
periodic_functions = {'sine': sine_wave,
                      'square': square_wave,
                      'triangle': triangle_wave,
                      'sawtooth': triangle_wave,
                      'ramp': triangle_wave,
                      'trap': trapezoid,
                      'exp': exponential}

class HantekDDS_test():

    """ Class for providing interfaces for HantekDDS without requiring connected hardware."""

    def __init__(self, dev=0):
        """ Default constructor."""
        self._waveform = None
        self.param_dict = {'square_duty': 0.5,
                           'triangle_duty': 0.5,
                           'trap_rise': 0.2,
                           'trap_high': 0.3,
                           'trap_fall': 0.2,
                           'exp_mode': 'decay',
                           'exp_time': 0.001,
                           }
        self.halt = False

    def connect(self, dev=0):
        """ Verify existence of device and get device id."""
        pass

    def set_divisor(self, div):
        """ Set the DAC sample rate divisor."""
        pass

    def set_waveform(self, data):
        """ Set the waveform buffer."""
        data = np.array(data)  # Ensure data is in ndarray
        dlen = len(data)
        if dlen > 4096:
            print('WARNING: Hantek 1025G buffer limited to 4096 samples -- ' +
                  'Truncating data to 4096 elements')
            data = data[:4096]
        self._waveform = data

    def drive_periodic(self, amplitude=1.0, frequency=1000.0,
                       offset=0, phase=0, function='sine', **kwargs):
        """ Direct the device to drive a periodic function.

        Args:
          amplitude: Amplitude of signal in Volts (1/2 V_pp)
          frequency: Frequency of signal in Hertz
          offset: Offset of signal in Volts.
          phase: Offset of phase in periods (phase = 0 is equivalent to
            phase=1, phase = 0.5 is 180 degrees out of phase from phase = 0.
          function: Type of periodic function to drive.

            Valid Types
            -----------
              'sine': Sine wave
              'square': Square wave, param 'square_duty' for high duty cycle
                        specified as float in range (0,1)
              'triangle': Triangle wave, param 'duty' for rise duty cycle
                          specified as float in range (0,1)
              'ramp': Synonym for 'triangle'
              'sawtooth': Synonym for 'triangle'
              'trap': Trapezoidal wave, params 'trap_rise', 'trap_high',
                      and 'trap_fall' for duty cycle for rise, high, and fall
                      segments specified as floats > 0 with sum <= 1.
              'exp': Exponential wave, params 'exp_mode' (may be 'rise' or
                     'saturate') and 'exp_time' (time constant in seconds).
        """
        for key,val in kwargs.items():
            if key in self.param_dict:
                self.param_dict[key] = val

        frequency = validate_frequency(frequency)
        phase = phase % 1
        if (amplitude + abs(offset)) > 3.5:
            print('WARNING: amplitude and offset specified will cause ' +
                  'the signal to be clipped.  Consider confining the range ' +
                  'to ±3.5 volts.')

        if function not in periodic_functions.keys():
            print('WARNING: function type for periodic function not found. ' +
                  'Valid values are ' +
                  'Defaulting to sine wave.')
            function = 'sine'

        div, length = get_freq_settings(frequency)
        f = periodic_functions[function]
        signal = f(amplitude, frequency, offset, phase,
                   length, **(self.param_dict))
        digital = np.short(voltage_adc(signal))
        self.set_waveform(digital)
        self.set_divisor(div)

    def frequency_scan(self, f_start, f_end, nsteps=10, delta=0, dwell=5,
                       ltype='linear', n_repeats=1, amplitude=1.0, offset=0.0):
        """ Scan through a range of frequencies.

        Args:
          f_start: Scan starting frequency in Hertz
          f_end: Scan ending frequency in Hertz
          nsteps: The number of steps to take between f_start and f_end.
          delta: The arithmetic or geometric difference between steps.
                 If non-zero, this setting will override nsteps.
                 Additionally, the frequency sweep may go past f_end.
          n_repeats: The number of times to loop through the entire frequency
                     scan.  Set to -1 to make continuous.
        """
        f_start = validate_frequency(f_start)
        f_end = validate_frequency(f_end)
        if ltype in ['log', 'logarithmic']:
            if delta > 0:
                nsteps = np.ceil(np.log(f_end / f_start) / np.log(delta))
                fspace = np.product(np.append(f_start, delta * np.ones(nsteps)))
            else:
                fspace = np.logspace(np.log10(f_start),
                                     np.log10(f_end), nsteps)
        if ltype in ['lin', 'linear']:
            if delta != 0:
                fspace = np.arange(f_start, f_end + delta, delta)
            else:
                fspace = np.linspace(f_start, f_end, nsteps)
            fspace = np.linspace(f_start, f_end, nsteps)
        self.scanner = threading.Thread(target=self.freq_scan_threaded,
                                        args=(fspace, amplitude,
                                              offset, dwell, n_repeats))
        self.halt = False
        self.scanner.start()
#        self.freq_scan_threaded(fspace, amplitude, offset, dwell, n_repeats)

    def freq_scan_threaded(self, fspace, amp, offset, dwell, n_repeats):
        """ Frequency scan to be started in non-blocking threaded process."""
        queue = int(n_repeats)
        while queue != 0:
            queue -= 1
            for f in fspace:
                if self.halt:
                    return None
                self.drive_periodic(amplitude=amp, frequency=f, offset=offset)
                time.sleep(dwell)

    def stop(self):
        """ Halt the threaded scan process."""
        self.halt = True
