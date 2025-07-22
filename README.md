xcorr_py is a Python implementation of the cross-correlation score as implemented by Comet.

As of 2025/07/22, xcorr_py generates the same scores as Comet 2025.02.0.

This is just a example program.  Modified residues and higher charged fragments have not
yet been implemented are easy to do and I'm happy to add that support in if helpful to
anyone.

Lines 14, 15, and 16 is where you set the fragment paramters such as fragment_bin_width,
fragment_bin_offset, and use_flanking_peaks (aka theoretical_fragment_ions in Comet).

The "test" directory contains the same spectra data and peptides for a Comet search as
are encoded in xcorr_py.
