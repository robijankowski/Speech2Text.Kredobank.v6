import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


from app.core.config import settings
from app.core.logger import log



def remove_long_silences(input_file, output_file=None, method='pydub', **kwargs):
    """
    Remove long silent pauses from audio file using different methods
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional)
        method: 'pydub', 'librosa', or 'hybrid'
        **kwargs: Additional parameters for fine-tuning
    
    Returns:
        Path to the processed audio file
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_sr.wav"
    
    if method == 'pydub':
        return _remove_silences_pydub(input_file, output_file, **kwargs)
    elif method == 'librosa':
        return _remove_silences_librosa(input_file, output_file, **kwargs)
    elif method == 'hybrid':
        return _remove_silences_hybrid(input_file, output_file, **kwargs)
    else:
        raise ValueError("Method must be 'pydub', 'librosa', or 'hybrid'")


def _remove_silences_pydub(input_file, output_file, **kwargs):
    """
    Remove silences using pydub (best for most use cases)
    """
    # Load audio
    audio = AudioSegment.from_file(input_file)
    log.info(f"Original duration: {len(audio)/1000:.2f}s")

    # Parameters
    silence_thresh = kwargs.get('silence_thresh', audio.dBFS - 16)  # 16dB below average
    min_silence_len = kwargs.get('min_silence_len', 1000)  # 1 second minimum
    keep_silence = kwargs.get('keep_silence', 200)  # Keep 200ms of silence
    
    log.info(f"Silence threshold: {silence_thresh:.1f} dBFS")
    log.info(f"Minimum silence length: {min_silence_len}ms")
    
    # Split on silence and rejoin with shorter pauses
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    if not chunks:
        log.info("No audio segments found, saving original")
        audio.export(output_file, format="wav")
        return output_file
    
    # Rejoin chunks
    processed_audio = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        processed_audio += chunk
        # Add small pause between chunks (except for last chunk)
        if i < len(chunks) - 1:
            pause_duration = kwargs.get('pause_duration', 300)  # 300ms pause
            processed_audio += AudioSegment.silent(duration=pause_duration)
    
    log.info(f"Processed duration: {len(processed_audio)/1000:.2f}s")
    log.info(f"Time saved: {(len(audio) - len(processed_audio))/1000:.2f}s")
    
    # Export
    processed_audio.export(output_file, format="wav")
    log.info(f"Saved silence-removed audio to: {output_file}")
    
    return output_file


def _remove_silences_librosa(input_file, output_file, **kwargs):
    """
    Remove silences using librosa (more precise, better for speech)
    """
    # Load audio
    y, sr = librosa.load(input_file, sr=None)
    log.info(f"Original duration: {len(y)/sr:.2f}s at {sr}Hz")
    
    # Parameters
    top_db = kwargs.get('top_db', 20)  # Silence threshold in dB
    frame_length = kwargs.get('frame_length', 2048)
    hop_length = kwargs.get('hop_length', 512)
    
    # Trim silence from beginning and end
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    # Split audio into non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    if len(intervals) == 0:
        log.info("No non-silent intervals found, saving trimmed audio")
        sf.write(output_file, y_trimmed, sr)
        return output_file
    
    # Parameters for rejoining
    min_interval_length = kwargs.get('min_interval_length', sr * 0.1)  # 100ms minimum
    pause_samples = int(kwargs.get('pause_duration', 0.3) * sr)  # 300ms pause
    
    # Filter out very short intervals and rejoin with pauses
    processed_segments = []
    for start, end in intervals:
        if end - start >= min_interval_length:
            processed_segments.append(y[start:end])
    
    if not processed_segments:
        log.info("No segments long enough, saving trimmed audio")
        sf.write(output_file, y_trimmed, sr)
        return output_file
    
    # Join segments with pauses
    y_processed = processed_segments[0]
    for segment in processed_segments[1:]:
        # Add pause
        y_processed = np.concatenate([y_processed, np.zeros(pause_samples), segment])
    
    log.info(f"Processed duration: {len(y_processed)/sr:.2f}s")
    log.info(f"Time saved: {(len(y) - len(y_processed))/sr:.2f}s")
    
    # Save
    sf.write(output_file, y_processed, sr)
    log.info(f"Saved silence-removed audio to: {output_file}")
    
    return output_file


def _remove_silences_hybrid(input_file, output_file, **kwargs):
    """
    Hybrid approach: Use librosa for detection, pydub for processing
    """
    # Load with librosa for analysis
    y, sr = librosa.load(input_file, sr=None)
    
    # Detect non-silent intervals with librosa
    top_db = kwargs.get('top_db', 20)
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # Convert to time intervals
    time_intervals = [(start/sr, end/sr) for start, end in intervals]
    
    # Load with pydub for processing
    audio = AudioSegment.from_file(input_file)
    
    # Extract non-silent segments
    segments = []
    min_segment_duration = kwargs.get('min_segment_duration', 0.1) * 1000  # Convert to ms
    
    for start_time, end_time in time_intervals:
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        if end_ms - start_ms >= min_segment_duration:
            segments.append(audio[start_ms:end_ms])
    
    if not segments:
        log.info("No segments found, saving original")
        audio.export(output_file, format="wav")
        return output_file
    
    # Rejoin with controlled pauses
    processed_audio = segments[0]
    pause_duration = kwargs.get('pause_duration', 300)  # 300ms
    
    for segment in segments[1:]:
        processed_audio += AudioSegment.silent(duration=pause_duration)
        processed_audio += segment
    
    log.info(f"Original duration: {len(audio)/1000:.2f}s")
    log.info(f"Processed duration: {len(processed_audio)/1000:.2f}s")
    log.info(f"Time saved: {(len(audio) - len(processed_audio))/1000:.2f}s")
    
    processed_audio.export(output_file, format="wav")
    log.info(f"Saved silence-removed audio to: {output_file}")

    return output_file


def aggressive_silence_removal(input_file, output_file=None, **kwargs):
    """
    Aggressive silence removal for files with lots of dead air
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_aggressive_trimmed.wav"
    
    audio = AudioSegment.from_file(input_file)
    
    # Very aggressive parameters
    silence_thresh = kwargs.get('silence_thresh', audio.dBFS - 14)  # Higher threshold
    min_silence_len = kwargs.get('min_silence_len', 500)  # Shorter minimum
    keep_silence = kwargs.get('keep_silence', 100)  # Keep less silence
    
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    if not chunks:
        audio.export(output_file, format="wav")
        return output_file
    
    # Join with minimal pauses
    processed_audio = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        processed_audio += chunk
        if i < len(chunks) - 1:
            processed_audio += AudioSegment.silent(duration=150)  # Very short pause
    
    log.info(f"Aggressive removal - Original: {len(audio)/1000:.2f}s, Processed: {len(processed_audio)/1000:.2f}s")
    log.info(f"Removed: {(len(audio) - len(processed_audio))/1000:.2f}s ({((len(audio) - len(processed_audio))/len(audio)*100):.1f}%)")
    
    processed_audio.export(output_file, format="wav")
    return output_file


def gentle_silence_removal(input_file, output_file=None, **kwargs):
    """
    Gentle silence removal that preserves natural speech patterns
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_gentle_trimmed.wav"
    
    audio = AudioSegment.from_file(input_file)
    
    # Conservative parameters
    silence_thresh = kwargs.get('silence_thresh', audio.dBFS - 20)  # Lower threshold
    min_silence_len = kwargs.get('min_silence_len', 2000)  # Longer minimum (2 seconds)
    keep_silence = kwargs.get('keep_silence', 500)  # Keep more silence
    
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    
    if not chunks:
        audio.export(output_file, format="wav")
        return output_file
    
    # Join with natural pauses
    processed_audio = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        processed_audio += chunk
        if i < len(chunks) - 1:
            processed_audio += AudioSegment.silent(duration=800)  # Natural pause
    
    log.info(f"Gentle removal - Original: {len(audio)/1000:.2f}s, Processed: {len(processed_audio)/1000:.2f}s")
    log.info(f"Removed: {(len(audio) - len(processed_audio))/1000:.2f}s")
    
    processed_audio.export(output_file, format="wav")
    return output_file


def analyze_silence_patterns(input_file):
    """
    Analyze the silence patterns in an audio file to help choose parameters
    """
    log.info(f"Analyzing silence patterns in: {input_file}")
    
    audio = AudioSegment.from_file(input_file)
    
    # Test different thresholds
    thresholds = [audio.dBFS - 10, audio.dBFS - 15, audio.dBFS - 20, audio.dBFS - 25]
    
    for thresh in thresholds:
        silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=thresh)
        if silent_ranges:
            total_speech = sum(end - start for start, end in silent_ranges)
            silence_time = len(audio) - total_speech
            log.info(f"Threshold {thresh:.1f}dB: {len(silent_ranges)} segments, "
                  f"{silence_time/1000:.1f}s silence ({silence_time/len(audio)*100:.1f}%)")
        else:
            log.info(f"Threshold {thresh:.1f}dB: No segments detected")
    
    return audio.dBFS


def batch_remove_silences(input_folder, output_folder=None, method='pydub', **kwargs):
    """
    Process multiple audio files in a folder
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, "silence_removed")
    
    os.makedirs(output_folder, exist_ok=True)
    
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    processed_files = []
    
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in audio_extensions):
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_silence_removed.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            log.info(f"\nProcessing: {filename}")
            try:
                result = remove_long_silences(input_path, output_path, method=method, **kwargs)
                processed_files.append(result)
            except Exception as e:
                log.error(f"Error processing {filename}: {e}")

    log.info(f"\nProcessed {len(processed_files)} files")
    return processed_files

