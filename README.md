# Hantek_1025G
Python wrapper for Hantek 1025G C-language API provided by HTDDSDll.h

## Overview
This module uses ctypes to provide an interface to API functions
provided by HTDDSDll.h, provided in Hantek DDS device development
files (e.g., `HT1025G_Software/SDK/DLL/HTDDSDll.{h, dll}`).

Due to the absence of Linux drivers, this module works only on Windows
platforms.  While 64-bit versions of the DLL and header files exist, no
effort has been made to verify functionality with these drivers, and
32 bit versions of python should probably be used if making use of this
module.

## Notes
It seems that the `set_frequency` method should not generally be used,
as it can leave the device in an inconsistent state.  Instead,
`set_divisor` should be used.  The only functional difference between
these two methods is that set_frequency takes a frequency argument
together with the waveform length in order to compute and set the
divisor, while set_divisor takes the divisor directly.  The
`drive_periodic` method handles the computation of the appropriate
divisor, but without the risk of leaving the device in an inconsistent
state (which simply necessitates a power-cycling of the device).

## License
The MIT License (MIT)

Copyright (c) 2015, Atlanta Instrumentation and Measurement, LLC

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Author
Kevin D. Nielson (2015.04.19)

[AIM - Atlanta Instrumentation and Measurement, LLC](www.aimatlanta.com)