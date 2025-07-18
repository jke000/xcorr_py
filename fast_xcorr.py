import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class FastXCorr:
    """
    Fast cross-correlation score implementation based on the SEQUEST algorithm.
    
    This implementation follows the optimized approach described in Eng et al. (2008)
    that avoids FFTs and uses direct dot products for efficient scoring.
    """
    
    def __init__(self, bin_width: float = 1.0005079, bin_offset: float = 0.4):
        """
        Initialize the FastXCorr scorer.
        
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
                          charge: int = 2, max_mass: float = 4000.0) -> np.ndarray:
        """
        Preprocess experimental spectrum following SEQUEST protocol.
        
        Args:
            spectrum: List of (mass, intensity) tuples
            charge: Precursor charge state
            max_mass: Maximum mass to consider
            
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
            return np.zeros(int(max_mass / self.bin_width) + 1 - self.bin_offset)
            
        valid_peaks = spec_array[spec_array[:, 0] <= max_mass]
        
        # Create binned spectrum
        max_bin = int((max_mass / self.bin_width) + 1 - self.bin_offset)
        binned_spectrum = np.zeros(max_bin)
        
        for mass, intensity in valid_peaks:
            bin_idx = int((mass / self.bin_width) + 1 - self.bin_offset)
            if 0 <= bin_idx < max_bin:
                # Take square root of intensity (SEQUEST preprocessing)
                binned_spectrum[bin_idx] = max(binned_spectrum[bin_idx], 
                                             math.sqrt(max(0, intensity)))
        
        # Normalize intensities in windows (simplified version)
        # SEQUEST normalizes max intensity to 50 in each 10 m/z window
        window_size = int(10.0 / self.bin_width)
        if window_size > 0:
            for i in range(0, len(binned_spectrum), window_size):
                end_idx = min(i + window_size, len(binned_spectrum))
                window = binned_spectrum[i:end_idx]
                if np.max(window) > 0:
                    binned_spectrum[i:end_idx] = window / np.max(window) * 50.0
        
        print("binned_spectrum...")
        print(binned_spectrum[:500])

        return binned_spectrum


    def apply_xcorr_preprocessing(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply the fast xcorr preprocessing as described in equation 6 of the paper.
        
        Args:
            spectrum: Preprocessed spectrum array
            
        Returns:
            Spectrum ready for xcorr calculation with correction factor applied
        """
        # Calculate the correction factor using offsets from -75 to +75 (excluding 0)
        corrected_spectrum = np.zeros_like(spectrum)
        spectrum_length = len(spectrum)

        for i in range(spectrum_length):
            sum_offsets = 0.0

            for tau in range(-75, 76):
                if tau == 0:
                    continue

                neighbor_idx = i + tau
                if 0 <= neighbor_idx < spectrum_length:
                    sum_offsets += spectrum[neighbor_idx]

            mean_offset = sum_offsets / 150
            corrected_spectrum[i] = spectrum[i] - mean_offset

            if spectrum[i] == 50:
                print("spectrum:"  {i}, {spectrum[i}, {mean_offset})

        print("fast xcorr spectrum...")
        print(corrected_spectrum[:600])


        return corrected_spectrum


    def calculate_theoretical_spectrum(self, peptide_sequence: str, 
                                     charge: int = 2, 
                                     max_mass: float = 4000.0) -> np.ndarray:
        """
        Calculate theoretical spectrum for a peptide sequence.
        
        Args:
            peptide_sequence: Peptide sequence string
            charge: Precursor charge state
            max_mass: Maximum mass to consider
            
        Returns:
            Theoretical spectrum array
        """
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
                    theoretical_spectrum[bin_idx] = 50.0  # Standard intensity
        
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
                    theoretical_spectrum[bin_idx] = 50.0  # Standard intensity
        
        return theoretical_spectrum
    
    def calculate_xcorr(self, experimental_spectrum: np.ndarray, 
                       theoretical_spectrum: np.ndarray) -> float:
        """
        Calculate the fast xcorr score using dot product.
        
        Args:
            experimental_spectrum: Preprocessed experimental spectrum
            theoretical_spectrum: Theoretical spectrum
            
        Returns:
            Cross-correlation score
        """
        # Ensure both spectra have the same length
        min_len = min(len(experimental_spectrum), len(theoretical_spectrum))
        exp_spec = experimental_spectrum[:min_len]
        theo_spec = theoretical_spectrum[:min_len]
        
        # Calculate xcorr as dot product
        xcorr = np.dot(theo_spec, exp_spec) / 10000
        
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
        exp_spectrum = self.preprocess_spectrum(spectrum, charge, max_mass)
        exp_spectrum_corrected = self.apply_xcorr_preprocessing(exp_spectrum)

        
        # Score each peptide
        scores = []
        for peptide in peptide_sequences:
            # Calculate theoretical spectrum
            theo_spectrum = self.calculate_theoretical_spectrum(peptide, charge, max_mass)
            
            # Calculate xcorr score
            xcorr_score = self.calculate_xcorr(exp_spectrum_corrected, theo_spectrum)
            
            scores.append((peptide, xcorr_score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    

# Example usage and testing
if __name__ == "__main__":
    # Create FastXCorr instance
    xcorr_scorer = FastXCorr()
    
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
        "SGVALADESTLAFNLK",  # Target peptide
        "ALADESTLAFNLK",     # Similar peptide
        "SGVALADESTLAFK",    # Truncated peptide
        "KVLADESLFANK",      # Scrambled peptide
        "PEPTIDER",          # Random peptide
        "DIGSETK",       # Another random peptide
    ]
    
    # Score peptides
    print("Scoring peptides against experimental spectrum...")
    
    # First, let's see what the preprocessed spectrum looks like
    exp_spectrum = xcorr_scorer.preprocess_spectrum(experimental_spectrum, charge=2, max_mass=4000.0)
    
    print(f"\nPreprocessed experimental spectrum (exp_spectrum):")
    print(f"Length: {len(exp_spectrum)}")
    print(f"Non-zero values: {np.count_nonzero(exp_spectrum)}")
    
    # Print non-zero values with their indices (mass bins)
    print("\nNon-zero intensity values:")
    print("Mass (Da)\tBin Index\tIntensity")
    print("-" * 35)
    for i, intensity in enumerate(exp_spectrum):
        if intensity > 0:
            mass = i * xcorr_scorer.bin_width + 1 - xcorr_scorer.bin_offset
            print(f"{mass:.2f}\t\t{i}\t\t{intensity:.4f}")
    
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
        theo_spectrum = xcorr_scorer.calculate_theoretical_spectrum(target_peptide, charge=2, max_mass=4000.0)
        
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
                
                if 0 <= bin_idx < len(exp_spectrum_corrected):
                    xcorr_value = exp_spectrum_corrected[bin_idx]
                    print(f"mass {y_ion_mass:.6f}, binned ion {bin_idx}, fast xcorr at {bin_idx}: {xcorr_value:.6f}")
        
        # Show theoretical spectrum peaks
        print(f"\nTheoretical spectrum peaks for {target_peptide}:")
        for i, intensity in enumerate(theo_spectrum):
            if intensity > 0:
                mass = i * xcorr_scorer.bin_width + xcorr_scorer.bin_offset
                print(f"Bin {i}: mass {mass:.6f}, theoretical intensity {intensity:.1f}")
    
    print("\n" + "="*60)
    
    print("\nResults (sorted by XCorr score):")
    print("Peptide Sequence\t\tXCorr Score")
    print("-" * 40)
    for peptide, score in scores:
        print(f"{peptide:<20}\t{score:.4f}")
    
    if scores:
        top_peptide, top_score = scores[0]
        print(f"\nTop hit: {top_peptide}")
        print(f"XCorr Score: {top_score:.4f}")
