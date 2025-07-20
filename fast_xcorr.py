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
    
    def __init__(self, bin_width: float = 1.0005079, bin_offset: float = 0.4):
        """
        Initialize the FastXcorr scorer.
        
        Args:
            bin_width: Width of mass bins in daltons
            bin_offset: Offset for mass binning
        """
        self.bin_width = bin_width
        self.bin_offset = bin_offset
        self.amino_acid_masses = {
            'A':  71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
            'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G':  57.02146,
            'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
            'M': 131.04049, 'F': 147.06841, 'P':  97.05276, 'S':  87.03203,
            'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V':  99.06841
        }
        self.water_mass = 18.01056
        self.proton_mass = 1.007276
        
    def preprocess_spectrum(self, spectrum: List[Tuple[float, float]], 
                          charge: int = 2, max_mass: float = 4000.0,
                          flank_peaks: int = 1) -> np.ndarray:
        """
        Preprocess experimental spectrum following SEQUEST protocol.
        
        Args:
            spectrum: List of (mass, intensity) tuples
            charge: Precursor charge state
            max_mass: Maximum mass to consider
            flank_peaks: 0=no, 1=use flanking peaks
            
        Returns:
            Preprocessed spectrum array ready for xcorr calculation
        """

        """
        max_mass is maximum mass from input spectrum
        max_mass = max(spec_array[:, 0], initial=0.0)
        """

        # Convert to numpy array and filter by mass range
        spec_array = np.array(spectrum)
        max_mass = max(spec_array[:, 0])
        if len(spec_array) == 0:
            return np.zeros(int((max_mass / self.bin_width) + 1 - self.bin_offset))
            
        valid_peaks = spec_array[spec_array[:, 0] <= max_mass]
        
        # Create binned spectrum
        max_bin = int((max_mass / self.bin_width) + 1 - self.bin_offset) + 2
        binned_spectrum = np.zeros(max_bin)
        
        for mass, intensity in valid_peaks:
            bin_idx = int((mass / self.bin_width) + 1 - self.bin_offset)
            sqrt_intensity = math.sqrt(max(0, intensity))
            if flank_peaks == 0:
                if 0 <= bin_idx < max_bin:
                    binned_spectrum[bin_idx] = max(binned_spectrum[bin_idx], sqrt_intensity)
            else:
                for spread in [-1, 0, 1]:
                    idx = bin_idx + spread
                    if 0 <= idx < max_bin:
                        if spread == 0:
                            binned_spectrum[idx] = max(binned_spectrum[idx], sqrt_intensity)
                        else:
                            binned_spectrum[idx] = max(binned_spectrum[idx], 0.5 * sqrt_intensity)

        # Normalize intensities in windows (simplified version)
        # SEQUEST normalizes max intensity to 50 in each of the 10 windows
        num_windows = 10
        window_size = (int)(max_bin / num_windows) + 1;

        if window_size > 0:
            for i in range(0, len(binned_spectrum), window_size):
                end_idx = min(i + window_size, len(binned_spectrum))
                window = binned_spectrum[i:end_idx]
                if np.max(window) > 0:
                    binned_spectrum[i:end_idx] = window / np.max(window) * 50.0
        
        #print("binned_spectrum...")
        #for index, value in enumerate(binned_spectrum):
        #    if value != 0:
        #        print(f"{index}: {value}")


        return binned_spectrum


    def apply_xcorr_preprocessing(self, spectrum: np.ndarray) -> np.ndarray:
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
        
        return corrected_spectrum


    def calculate_fragment_ion_bins(self, peptide_sequence: str, 
                                     charge: int = 2, 
                                     max_mass: float = 4000.0) -> np.ndarray:
        """
        Calculate theoretical spectrum for a peptide sequence.
        
        Args:
            peptide_sequence: Peptide sequence string
            charge: Precursor charge state
            max_mass: Maximum mass to consider
            
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
                
                # b-ion mass = cumulative amino acid mass + proton
                b_ion_mass = cumulative_mass + self.proton_mass
                bin_idx = int((b_ion_mass / self.bin_width)  + 1 - self.bin_offset)
                if 0 <= bin_idx < max_bin:
                    fragment_ion_bins.append(bin_idx)
        
        # Calculate y-ions (C-terminal fragments)
        # y-ions are formed by cleavage at the peptide bond, keeping the C-terminal portion
        cumulative_mass = self.water_mass + self.proton_mass  # Start with H2O + H+
        for i in range(len(peptide_sequence) - 1, 0, -1):  # Reverse direction, exclude first amino acid
            aa = peptide_sequence[i]
            if aa in self.amino_acid_masses:
                cumulative_mass += self.amino_acid_masses[aa]
                
                # y-ion mass = cumulative amino acid mass + H2O + H+
                y_ion_mass = cumulative_mass
                bin_idx = int((y_ion_mass / self.bin_width)  + 1 - self.bin_offset)
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
        
        return xcorr
    
    def score_peptides(self, spectrum: List[Tuple[float, float]], 
                      peptide_sequences: List[str], 
                      charge: int = 2, 
                      max_mass: float = 4000.0) -> List[Tuple[str, float]]:
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
        use_flank_peaks = 1
        exp_spectrum = self.preprocess_spectrum(spectrum, charge, max_mass, use_flank_peaks)
        exp_spectrum_corrected = self.apply_xcorr_preprocessing(exp_spectrum)


        """
        # This will print bin index, intensity, and corrected value, skipping bins where both are zero.
        print("\nBin\tIntensity\tprocessed intensity")
        for idx, (intensity, corrected) in enumerate(zip(exp_spectrum, exp_spectrum_corrected)):
            if intensity != 0.0 or corrected != 0.0:
                print(f"{idx}\t{intensity:.6f}\t{corrected:.6f}")
        """

        
        # Score each peptide
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
       (116.034220, 100),
       (147.112804, 100),
       (229.118284, 100),
       (248.160483, 100),
       (286.139747, 100),
       (373.171776, 100),
       (377.203076, 100),
       (464.235104, 100),
       (502.214369, 100),
       (521.256568, 100),
       (603.262047, 100),
       (634.340632, 100),
    ]
    # Example peptide sequences to score
    peptide_sequences = [
        "DIGSETK",           # Target peptide
        "DGISETK",           # Similar wrong peptide
        "STLAFNLK",
        "ASLQFTMK",
        "PEPTIDER"
    ]
    
    # Score peptides
    print("Scoring peptides against experimental spectrum...")
    
    # First, let's see what the preprocessed spectrum looks like
    exp_spectrum = xcorr_scorer.preprocess_spectrum(experimental_spectrum, charge=2, max_mass=4000.0)
    
    """
    print(f"\nPreprocessed experimental spectrum (exp_spectrum):")
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
    
    # Debug output for DIGSETK peptide to verify b-ions and y-ions
    print("\n" + "="*60)
    print("DEBUG: Detailed analysis for DIGSETK peptide")
    print("="*60)
    
    target_peptide = "DIGSETK"
    if target_peptide in peptide_sequences:
        # Get the corrected experimental spectrum
        exp_spectrum = xcorr_scorer.preprocess_spectrum(experimental_spectrum, charge=2, max_mass=4000.0)
        exp_spectrum_corrected = xcorr_scorer.apply_xcorr_preprocessing(exp_spectrum)
        
        # Calculate theoretical spectrum for DIGSETK
        fragment_bins = xcorr_scorer.calculate_fragment_ion_bins(target_peptide, charge=2, max_mass=4000.0)
        
        print(f"\nAnalyzing peptide: {target_peptide}")
        print(f"Amino acid sequence: D-I-G-S-E-T-K")
        
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
                print(f"mass {b_ion_mass:.6f}, binned ion {bin_idx}, fast xcorr at {bin_idx}: {xcorr_value:.6f}")
        
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
                print(f"mass {y_ion_mass:.6f}, binned ion {bin_idx}, fast xcorr at {bin_idx}: {xcorr_value:.6f}")
        
    
    print("\n" + "="*60)
    
    print("\nResults (sorted by xcorr score):")
    print("-" * 40)
    for peptide, score in scores:
        print(f"{peptide:<20}\t{score:.4f}")
    
    if scores:
        top_peptide, top_score = scores[0]
        print(f"\nTop hit: {top_peptide}, xcorr {top_score:.4f}")
