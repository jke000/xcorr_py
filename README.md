xcorr_py is a Python implementation of the [Comet](https://github.com/UWPR/Comet) cross-correlation score.

- xcorr_py generates the exact same xcorr scores as [Comet 2025.02.0](https://github.com/UWPR/Comet/releases/tag/v2025.02.0)
and presumably earlier/later versions.

- Another independent Python xcorr implementation by M. Maccoss [is available here with pyXcorrDIA](https://github.com/maccoss/pyXcorrDIA).

- This is just a example program.  Modified residues and higher charged fragments have not
yet been implemented but are easy to do by extending `calculate_fragment_ion_bins()` to generate
the proper set of `fragment_ion_bins_uniq`. I'm happy to add that support in if helpful to
anyone.

- Lines 14, 15, and 16 is where you set the fragment parameters such as
[fragment_bin_width](https://uwpr.github.io/Comet/parameters/parameters_202502/fragment_bin_tol.html),
[fragment_bin_offset](https://uwpr.github.io/Comet/parameters/parameters_202502/fragment_bin_offset.html),
and use_flanking_peaks (aka [theoretical_fragment_ions](https://uwpr.github.io/Comet/parameters/parameters_202502/theoretical_fragment_ions.html)
in Comet). Low res and high res fragment ion settings, with and without flanking peaks, have been
tested and confirmed to match Comet.

- The "test" directory contains the same spectra data and peptides for a Comet search as
are encoded in xcorr_py.
