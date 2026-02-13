import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

from core.config import settings
from core.logger import log




def reduce_audio_noise(input_file, output_file=None, method='spectral', **kwargs):
    """
    Dedicated function for noise reduction with multiple methods
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional)
        method: 'spectral', 'stationary', 'non_stationary', or 'gentle'
        **kwargs: Additional parameters for fine-tuning
    
    Returns:
        Path to the cleaned audio file
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_nr.wav"
    
    # Load audio with librosa (handles most formats)
    try:
        y, sr = librosa.load(input_file, sr=None)  # Keep original sample rate
        log.info(f"Loaded audio: {len(y)/sr:.2f}s at {sr}Hz")
    except Exception as e:
        log.error(f"Error loading audio: {e}")
        return None
    
    # Apply noise reduction based on method
    if method == 'spectral':
        # Default spectral subtraction (good for most cases)
        y_cleaned = nr.reduce_noise(
            y=y, 
            sr=sr,
            prop_decrease=kwargs.get('prop_decrease', 0.8),
            stationary=False,
            n_std_thresh_stationary=kwargs.get('n_std_thresh', 1.5),
            time_constant_s=kwargs.get('time_constant_s', 2.0),
            freq_mask_smooth_hz=kwargs.get('freq_mask_smooth_hz', 500),
            time_mask_smooth_ms=kwargs.get('time_mask_smooth_ms', 50)
        )
        log.info("Applied spectral noise reduction")
        
    elif method == 'stationary':
        # For consistent background noise (AC, fan, etc.)
        y_cleaned = nr.reduce_noise(
            y=y, 
            sr=sr,
            stationary=True,
            prop_decrease=kwargs.get('prop_decrease', 0.9),
            n_std_thresh_stationary=kwargs.get('n_std_thresh', 2.0)
        )
        log.info("Applied stationary noise reduction")
        
    elif method == 'non_stationary':
        # For varying background noise
        y_cleaned = nr.reduce_noise(
            y=y, 
            sr=sr,
            stationary=False,
            prop_decrease=kwargs.get('prop_decrease', 0.7),
            n_std_thresh_stationary=kwargs.get('n_std_thresh', 1.25),
            time_constant_s=kwargs.get('time_constant_s', 1.5),
            thresh_n_mult_nonstationary=kwargs.get('thresh_n_mult_nonstationary', 2),
            sigmoid_slope_nonstationary=kwargs.get('sigmoid_slope_nonstationary', 10)
        )
        log.info("Applied non-stationary noise reduction")
        
    elif method == 'gentle':
        # Gentle noise reduction (preserves more original audio)
        y_cleaned = nr.reduce_noise(
            y=y, 
            sr=sr,
            prop_decrease=kwargs.get('prop_decrease', 0.55),
            stationary=False,
            n_std_thresh_stationary=kwargs.get('n_std_thresh', 1.0),
            time_constant_s=kwargs.get('time_constant_s', 1.0),
            freq_mask_smooth_hz=kwargs.get('freq_mask_smooth_hz', 300),
            time_mask_smooth_ms=kwargs.get('time_mask_smooth_ms', 25)
        )
        log.info("Applied gentle noise reduction")
    
    else:
        raise ValueError("Method must be 'spectral', 'stationary', 'non_stationary', or 'gentle'")
    
    # Save the cleaned audio
    sf.write(output_file, y_cleaned, sr)
    log.info(f"Saved denoised audio to: {output_file}")
    
    return output_file

def adaptive_noise_reduction(input_file, output_file=None):
    """
    Automatically choose the best noise reduction approach
    Analyzes audio characteristics to select optimal method
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_adaptive_denoised.wav"
    
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    
    # Analyze audio characteristics
    # Calculate spectral centroid variation to detect noise type
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_variation = np.std(spectral_centroids) / np.mean(spectral_centroids)
    
    # Calculate RMS energy variation
    rms = librosa.feature.rms(y=y)[0]
    rms_variation = np.std(rms) / np.mean(rms)
    
    log.info(f"Centroid variation: {centroid_variation:.3f}")
    log.info(f"RMS variation: {rms_variation:.3f}")
    
    # Choose method based on characteristics
    if centroid_variation < 0.2 and rms_variation < 0.3:
        # Likely stationary noise
        method = 'stationary'
        log.info("Detected stationary noise - using stationary reduction")
    elif centroid_variation > 0.4 or rms_variation > 0.5:
        # Likely non-stationary noise
        method = 'non_stationary'
        log.info("Detected non-stationary noise - using adaptive reduction")
    else:
        # Default to spectral method
        method = 'spectral'
        log.info("Using default spectral reduction")
    
    return reduce_audio_noise(input_file, output_file, method=method)

def noise_profile_reduction(input_file, noise_sample_start=0, noise_sample_duration=1.0, output_file=None):
    """
    Use a specific part of the audio as noise profile for reduction
    
    Args:
        input_file: Path to input audio
        noise_sample_start: Start time (seconds) of noise-only sample
        noise_sample_duration: Duration (seconds) of noise-only sample
        output_file: Output file path
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_profile_denoised.wav"
    
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    
    # Extract noise sample
    noise_start_sample = int(noise_sample_start * sr)
    noise_end_sample = int((noise_sample_start + noise_sample_duration) * sr)
    noise_sample = y[noise_start_sample:noise_end_sample]
    
    log.info(f"Using noise profile from {noise_sample_start}s to {noise_sample_start + noise_sample_duration}s")
    
    # Apply noise reduction using the noise sample
    y_cleaned = nr.reduce_noise(
        y=y, 
        sr=sr,
        y_noise=noise_sample,
        prop_decrease=0.8,
        stationary=False
    )
    
    # Save result
    sf.write(output_file, y_cleaned, sr)
    log.info(f"Saved profile-based denoised audio to: {output_file}")
    
    return output_file

# Example usage:
# Basic usage:
# cleaned = reduce_audio_noise("recording.wav")
# cleaned = reduce_audio_noise("recording.wav", method='gentle')

# Adaptive (automatic method selection):
# cleaned = adaptive_noise_reduction("recording.wav")

# Using noise profile from first second of recording:
# cleaned = noise_profile_reduction("recording.wav", noise_sample_start=0, noise_sample_duration=1.0)

def simple_noise_reduction(input_file, output_file=None, strength=0.8):
    """
    Simple, compatible noise reduction that works with any noisereduce version
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional)
        strength: Noise reduction strength (0.1 to 1.0)
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_simple_denoised.wav"
    
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    log.info(f"Loaded audio: {len(y)/sr:.2f}s at {sr}Hz")
    
    # Simple noise reduction with minimal parameters
    y_cleaned = nr.reduce_noise(y=y, sr=sr, prop_decrease=strength)
    
    # Save result
    sf.write(output_file, y_cleaned, sr)
    log.info(f"Saved denoised audio to: {output_file}")
    
    return output_file