import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class FastXcorr:
    """
    Fast cross-correlation score implementation based on the SEQUEST algorithm.
    
    This implementation follows the optimized approach described in Eng et al. (2008)
    that avoids FFTs and uses direct dot products for efficient scoring.
    """
    
    def __init__(self, bin_width: float = 0.02,
                       bin_offset: float = 0.0,
                       use_flanking_peaks: bool = True):
        """
        Initialize the FastXcorr scorer.
        
        Args:
            bin_width: Width of mass bins in daltons
            bin_offset: Offset for mass binning
        """
        self.bin_width = bin_width
        self.bin_offset = bin_offset
        self.use_flanking_peaks = use_flanking_peaks 
        self.amino_acid_masses = {
            'A':  71.037113805, 'R': 156.101111050, 'N': 114.042927470, 'D': 115.026943065,
            'C': 103.009184505, 'E': 129.042593135, 'Q': 128.058577540, 'G':  57.021463735,
            'H': 137.058911875, 'I': 113.084064015, 'L': 113.084064015, 'K': 128.094963050,
            'M': 131.040484645, 'F': 147.068413945, 'P':  97.052763875, 'S':  87.032028435,
            'T': 101.047678505, 'W': 186.079312980, 'Y': 163.063328575, 'V':  99.068413945 
        }
        self.water_mass = 18.0105647
        self.proton_mass = 1.00727646688
        
    def preprocess_spectrum(self, spectrum: List[Tuple[float, float]], 
                          charge: int = 2, print_debug: bool = False) -> np.ndarray:
        """
        Preprocess experimental spectrum following SEQUEST protocol.
        
        Args:
            spectrum: List of (mass, intensity) tuples
            charge: Precursor charge state
            flank_peaks: 0=no, 1=use flanking peaks
            
        Returns:
            Preprocessed spectrum array ready for xcorr calculation
        """

        # Convert to numpy array and filter by mass range
        spec_array = np.array(spectrum)
        max_mass = max(spec_array[:, 0])
        if len(spec_array) == 0:
            return np.zeros(int((max_mass / self.bin_width) + 1 - self.bin_offset))
            
        valid_peaks = spec_array[spec_array[:, 0] <= max_mass]
        
        # Create binned spectrum ; set max_bin to +75 beyond last mass for fastxcorr offsets
        max_bin = int((max_mass / self.bin_width) + 1 - self.bin_offset) + 2 + 75
        binned_spectrum = np.zeros(max_bin)
        
        for mass, intensity in valid_peaks:
            bin_idx = int((mass / self.bin_width) + 1 - self.bin_offset)
            sqrt_intensity = math.sqrt(max(0, intensity))
            if 0 <= bin_idx < max_bin:
                binned_spectrum[bin_idx] = max(binned_spectrum[bin_idx], sqrt_intensity)

        if print_debug:
            print("experimental spectrum (binned_spectrum) before normalization...")
            for index, value in enumerate(binned_spectrum):
                if value != 0:
                    print(f"{index}: {value:.3f}")

        # Normalize intensities in windows (simplified version)
        # Set max intensity to 50 in each of the 10 windows
        # To match Comet exactly, apply a mininum cutoff of 0.05 of the maximum intensity
        num_windows = 10
        highest_ion =  int((max_mass / self.bin_width) + 1 - self.bin_offset)
        window_size = (int)(highest_ion / num_windows) + 1;

        """
        print(f"OK window_size {window_size}")
        print(f"OK highest_ion {highest_ion}, max_mass {max_mass}")
        print(f"OK len(binned_spectrum) {len(binned_spectrum)}")
        """

        if window_size > 0:
            for i in range(0, highest_ion, window_size):
                end_idx = min(i + window_size, len(binned_spectrum))
                window = binned_spectrum[i:end_idx]
                if np.max(window) > 0:
                    # Calculate cutoff as 5% of this window's maximum
                    min_intensity_cutoff = 0.05 * np.max(window)
                    # Create a mask for values above the cutoff
                    mask = window > min_intensity_cutoff
                    # Only normalize values above the cutoff
                    normalized_window = window.copy()
                    if np.any(mask):
                        normalized_window[mask] = window[mask] / np.max(window) * 50.0
                    binned_spectrum[i:end_idx] = normalized_window

        # check if flanking peaks need to be incorporated
        final_binned_spectrum = np.zeros(max_bin)
        if self.use_flanking_peaks == False:
           final_binned_spectrum = binned_spectrum
        else:
            for i in range(len(binned_spectrum)):
                final_binned_spectrum[i] = binned_spectrum[i]
    
                # Add 0.5 * value from i-1 (if it exists)
                if i > 0:
                    final_binned_spectrum[i] += 0.5 * binned_spectrum[i-1]
    
                # Add 0.5 * value from i+1 (if it exists)
                if i < len(binned_spectrum) - 1:
                    final_binned_spectrum[i] += 0.5 * binned_spectrum[i+1]

        if print_debug:
            print("experimental spectrum (binned_spectrum) after normalization...")
            for index, value in enumerate(final_binned_spectrum):
                if value != 0:
                    print(f"{index}: {value:.3f}")

        return final_binned_spectrum


    def apply_xcorr_preprocessing(self, spectrum: np.ndarray, print_debug: bool = False) -> np.ndarray:
        """
        Apply the fast xcorr preprocessing as described in equation 6 of the paper.
        Args:
            spectrum: Preprocessed spectrum array
        Returns:
            Spectrum ready for xcorr calculation with correction factor applied
        """
        corrected_spectrum = np.zeros_like(spectrum)
        spectrum_length = len(spectrum)
        
        # Initialize sum for i=0
        sum_offsets = 0.0
        for tau in range(-75, 76):
            if tau == 0:
                continue
            neighbor_idx = tau
            if 0 <= neighbor_idx < spectrum_length:
                sum_offsets += spectrum[neighbor_idx]
        
        mean_offset = sum_offsets / 150
        corrected_spectrum[0] = spectrum[0] - mean_offset
        
        # For each subsequent i, update the sliding window
        for i in range(1, spectrum_length):
            # When moving from i-1 to i, the window shifts:
            # Old window: [(i-1)-75, (i-1)+75] excluding (i-1)
            # New window: [i-75, i+75] excluding i
            
            # Remove the leftmost element of the old window: (i-1)-75 = i-76
            remove_idx = i - 76
            if 0 <= remove_idx < spectrum_length:
                sum_offsets -= spectrum[remove_idx]
                
            # Add the rightmost element of the new window: i+75
            add_idx = i + 75
            if 0 <= add_idx < spectrum_length:
                sum_offsets += spectrum[add_idx]
                
            # Remove the old center (i-1) and add it back since we excluded it
            if 0 <= i - 1 < spectrum_length:
                sum_offsets += spectrum[i - 1]
                
            # Remove the new center (i) since we exclude tau=0
            if 0 <= i < spectrum_length:
                sum_offsets -= spectrum[i]
            
            mean_offset = sum_offsets / 150
            corrected_spectrum[i] = spectrum[i] - mean_offset

        if print_debug:
            print("experimental spectrum after fast xcorr processing...")
            for index, value in enumerate(corrected_spectrum):
                if value != 0:
                    print(f"{index}: {value:.6f}")
        
        return corrected_spectrum


    def calculate_fragment_ion_bins(self, peptide_sequence: str,
                                     charge: int = 2,
                                     max_mass: float = 5000.0) -> np.ndarray:
        """
        Calculate theoretical spectrum for a peptide sequence.
        Args:
            peptide_sequence: Peptide sequence string
            charge: Maximum fragment charge state to consider
            max_mass: Maximum peptide mass to consider for max fragment bin
        Returns:
            fragment_ion_bins_uniq: the set of fragment bins
        """
        fragment_ion_bins = []
        max_bin = int(max_mass / self.bin_width) + 1
        theoretical_spectrum = np.zeros(max_bin)
        
        # Calculate b-ions (N-terminal fragments)
        # b-ions are formed by cleavage at the peptide bond, keeping the N-terminal portion
        cumulative_mass = 0.0
        for i in range(len(peptide_sequence) - 1):  # Exclude last amino acid
            aa = peptide_sequence[i]
            if aa in self.amino_acid_masses:
                cumulative_mass += self.amino_acid_masses[aa]
                # Calculate b-ions for all charge states from 1 to max charge
                for fragment_charge in range(1, charge + 1):
                    # b-ion mass = (cumulative amino acid mass + fragment_charge * proton) / fragment_charge
                    b_ion_mz = (cumulative_mass + fragment_charge * self.proton_mass) / fragment_charge
                    bin_idx = int((b_ion_mz / self.bin_width) + 1 - self.bin_offset)
                    if 0 <= bin_idx < max_bin:
                        fragment_ion_bins.append(bin_idx)
        
        # Calculate y-ions (C-terminal fragments)
        # y-ions are formed by cleavage at the peptide bond, keeping the C-terminal portion
        cumulative_mass = self.water_mass  # Start with H2O
        for i in range(len(peptide_sequence) - 1, 0, -1):  # Reverse direction, exclude first amino acid
            aa = peptide_sequence[i]
            if aa in self.amino_acid_masses:
                cumulative_mass += self.amino_acid_masses[aa]
                # Calculate y-ions for all charge states from 1 to max charge
                for fragment_charge in range(1, charge + 1):
                    # y-ion mass = (cumulative amino acid mass + H2O + fragment_charge * proton) / fragment_charge
                    y_ion_mz = (cumulative_mass + fragment_charge * self.proton_mass) / fragment_charge
                    bin_idx = int((y_ion_mz / self.bin_width) + 1 - self.bin_offset)
                    if 0 <= bin_idx < max_bin:
                        fragment_ion_bins.append(bin_idx)
    
        fragment_ion_bins_uniq = set(fragment_ion_bins)

        return fragment_ion_bins_uniq
    
    def calculate_xcorr(self, experimental_spectrum: np.ndarray, 
                       fragment_bins: set) -> float:
        """
        Calculate the fast xcorr score using dot product.
        
        Args:
            experimental_spectrum: Fast xcorr preprocessed experimental spectrum
            fragment_bins:  Set of fragment ion bin locations
            
        Returns:
            Cross-correlation score which is a simple sum across the
            fast xcorr processed spectrum.
        """
        xcorr = 0.0
        arraysize = experimental_spectrum.size

        for bin_idx in fragment_bins:
            if bin_idx < arraysize :
                xcorr += experimental_spectrum[bin_idx]

        xcorr = xcorr * 0.005  # this handles theoretical spectrum intensities of 50 and dividing raw xcorr by 1e4

        xcorr = round(xcorr, 3)   # match Comet and round xcorr to 3 decimal points
        
        return xcorr
    
    def score_peptides(self, spectrum: List[Tuple[float, float]], 
                      peptide_sequences: List[str], 
                      charge: int = 2, 
                      max_mass: float = 5000.0) -> List[Tuple[str, float]]:
        """
        Score multiple peptide sequences against an experimental spectrum.
        
        Args:
            spectrum: Experimental spectrum as list of (mass, intensity) tuples
            peptide_sequences: List of peptide sequences to score
            charge: Precursor charge state
            max_mass: Maximum mass to consider
            
        Returns:
            List of (peptide_sequence, xcorr_score) tuples sorted by score
        """
        # Preprocess experimental spectrum
        exp_spectrum = self.preprocess_spectrum(spectrum, charge, print_debug=False)
        exp_spectrum_corrected = self.apply_xcorr_preprocessing(exp_spectrum, print_debug=False)


          
        """
        # This will print bin index, intensity, and corrected value, skipping bins where both are zero.
        print("\nBin\tIntensity\tprocessed intensity")
        for idx, (intensity, corrected) in enumerate(zip(exp_spectrum, exp_spectrum_corrected)):
            if intensity != 0.0 or corrected != 0.0:
                print(f"{idx}\t{intensity:.6f}\t{corrected:.6f}")
        """

        
        # Score each peptide
        charge = 1
        scores = []
        for peptide in peptide_sequences:
            # Calculate theoretical spectrum
            fragment_bins = self.calculate_fragment_ion_bins(peptide, charge, max_mass)
            
            # Calculate xcorr score
            xcorr_score = self.calculate_xcorr(exp_spectrum_corrected, fragment_bins)
            
            scores.append((peptide, xcorr_score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    

# Example usage and testing
if __name__ == "__main__":
    # Create FastXcorr instance
    xcorr_scorer = FastXcorr()
    
    # Example experimental spectrum (mass, intensity pairs)
    experimental_spectrum = [
       (147.1128,  10.684),
       (164.0706,   8.172),
       (248.1605,  26.432),
       (292.1292,  23.903),
       (379.1612,  15.469),
       (385.2194,  95.589),
       (386.3205,  95.589),
       (472.2514, 416.629),
       (516.2201,  19.546),
       (600.3100, 293.581),
       (617.2678,  17.995)
    ]
    # Example peptide sequences to score
    peptide_sequences = [
        "YQSHTK",           # Target peptide
        "YSQHTK",           # Similar wrong peptide
        "YHQSTK" 
    ]
    
    # Score peptides
    print("Scoring peptides against experimental spectrum...")
    
    # First, let's see what the preprocessed spectrum looks like
    exp_spectrum = xcorr_scorer.preprocess_spectrum(experimental_spectrum, charge=2, print_debug=False)
    
    """
    print("\nPreprocessed experimental spectrum (exp_spectrum):")
    print(f"Length: {len(exp_spectrum)}")
    print(f"Non-zero values: {np.count_nonzero(exp_spectrum)}")
    """
    
    """
    # Print non-zero values with their indices (mass bins)
    print("\nNon-zero intensity values:")
    print("Bin Index\tIntensity")
    print("-" * 25)
    for i, intensity in enumerate(exp_spectrum):
        if intensity > 0:
            print(f"{i}\t\t{intensity:.4f}")
    """
    
    scores = xcorr_scorer.score_peptides(experimental_spectrum, peptide_sequences, charge=2)
    
    # Debug output for YQSHTK peptide to verify b-ions and y-ions
    print("\n" + "="*60)
    print("DEBUG: Detailed analysis for YQSHTK peptide")
    print("="*60)
    
    dSum = 0

    target_peptide = "YQSHTK"
    fragment_charge = 1
    if target_peptide in peptide_sequences:
        # Get the corrected experimental spectrum
        exp_spectrum = xcorr_scorer.preprocess_spectrum(experimental_spectrum, fragment_charge, print_debug=False)
        exp_spectrum_corrected = xcorr_scorer.apply_xcorr_preprocessing(exp_spectrum, print_debug=False)
        
        # Calculate theoretical spectrum for YQSHTK
        fragment_bins = xcorr_scorer.calculate_fragment_ion_bins(target_peptide, fragment_charge, max_mass=5000.0)
        
        print(f"\nAnalyzing peptide: {target_peptide}")
        print("Amino acid sequence: Y-Q-S-H-T-K")
        
        # Calculate and display b-ions
        print("\nB-ions (N-terminal fragments):")
        cumulative_mass = 0.0
        for i in range(len(target_peptide) - 1):  # Exclude last amino acid
            aa = target_peptide[i]
            if aa in xcorr_scorer.amino_acid_masses:
                cumulative_mass += xcorr_scorer.amino_acid_masses[aa]
                b_ion_mass = cumulative_mass + xcorr_scorer.proton_mass  # +H
                bin_idx = int((b_ion_mass / xcorr_scorer.bin_width) + 1 - xcorr_scorer.bin_offset)
                
                xcorr_value = 0
                if 0 <= bin_idx < len(exp_spectrum_corrected):
                    xcorr_value = exp_spectrum_corrected[bin_idx]
                dSum += xcorr_value
                print(f"mass {b_ion_mass:.6f}, binned ion index {bin_idx}, fast xcorr at {bin_idx}: {xcorr_value:.4f}, xcorr_sum {dSum:.4f}")
        
        # Calculate and display y-ions
        print("\nY-ions (C-terminal fragments):")
        cumulative_mass = xcorr_scorer.water_mass + xcorr_scorer.proton_mass  # Start with H2O + H+
        for i in range(len(target_peptide) - 1, 0, -1):  # Reverse direction, exclude first amino acid
            aa = target_peptide[i]
            if aa in xcorr_scorer.amino_acid_masses:
                cumulative_mass += xcorr_scorer.amino_acid_masses[aa]
                y_ion_mass = cumulative_mass  # Already includes H2O + H+
                bin_idx = int((y_ion_mass / xcorr_scorer.bin_width) + 1 - xcorr_scorer.bin_offset)
                
                xcorr_value = 0
                if 0 <= bin_idx < len(exp_spectrum_corrected):
                    xcorr_value = exp_spectrum_corrected[bin_idx]
                dSum += xcorr_value
                print(f"mass {y_ion_mass:.6f}, binned ion index {bin_idx}, fast xcorr at {bin_idx}: {xcorr_value:.4f}, xcorr_sum {dSum:.4f}")
        
    
    print("\n" + "="*60)
    
    print("\nResults (sorted by xcorr score):")
    print("-" * 40)
    for peptide, score in scores:
        print(f"{peptide:<20}\t{score:.4f}")
    
    if scores:
        top_peptide, top_score = scores[0]
        print(f"\nTop hit: {top_peptide}, xcorr {top_score:.4f}")
