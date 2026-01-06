import librosa


def concatenate_segments_to_phrases(transcription_obj, max_gap=2.0):
    """
    Concatenates transcription segments into phrases based on time gaps and sentence boundaries.
    
    Args:
        transcription_obj: Object with 'segments' attribute containing TranscriptionSegment objects
        max_gap (float): Maximum time gap in seconds between segments to concatenate them (default: 0.5)
    
    Returns:
        list: List of dictionaries with format {"text": str, "start": float}
    
    Note:
        - Segments are concatenated if time gap <= max_gap AND previous segment doesn't end with '.'
        - If previous segment ends with '.', a new phrase is always started regardless of time gap
    """
    if not hasattr(transcription_obj, 'segments') or not transcription_obj.segments:
        return []
    
    segments = transcription_obj.segments
    phrases = []
    
    # Start with the first segment
    current_phrase_text = segments[0].text.strip()
    current_phrase_start = segments[0].start
    
    for i in range(1, len(segments)):
        current_segment = segments[i]
        previous_segment = segments[i-1]
        
        # Calculate time gap between end of previous segment and start of current segment
        time_gap = current_segment.start - previous_segment.end
        
        # Check if previous segment ends with a period (indicates sentence end)
        previous_text_ends_with_period = previous_segment.text.strip().endswith('.')
        
        if time_gap <= max_gap and not previous_text_ends_with_period:
            # Concatenate to current phrase
            current_phrase_text += " " + current_segment.text.strip()
        else:
            # Save current phrase and start a new one
            phrases.append({
                "text": current_phrase_text,
                "start": current_phrase_start
            })
            
            # Start new phrase
            current_phrase_text = current_segment.text.strip()
            current_phrase_start = current_segment.start
    
    # Don't forget to add the last phrase
    phrases.append({
        "text": current_phrase_text,
        "start": current_phrase_start
    })
    
    return phrases



def combine_phrase_tables(table1, table2):
    """
    Combines two phrase tables and sorts them by start time.
    
    Args:
        table1 (list): First table - list of dictionaries with format {"text": str, "start": float}
        table2 (list): Second table - list of dictionaries with format {"text": str, "start": float}
    
    Returns:
        list: Combined and sorted list of dictionaries with format {"text": str, "start": float, "role": str}
              Records from table1 get role="AG", records from table2 get role="CL"
              Consecutive records with same role are merged into single records
    """
    # Handle edge cases
    if not table1 and not table2:
        return []
    if not table1:
        sorted_table = sorted([{**phrase, "role": "CL"} for phrase in table2], key=lambda x: x['start'])
        return merge_consecutive_same_role(sorted_table)
    if not table2:
        sorted_table = sorted([{**phrase, "role": "AG"} for phrase in table1], key=lambda x: x['start'])
        return merge_consecutive_same_role(sorted_table)
    
    # Add role attribute to each table
    table1_with_role = [{**phrase, "role": "AG"} for phrase in table1]
    table2_with_role = [{**phrase, "role": "CL"} for phrase in table2]
    
    # Combine both tables
    combined_table = table1_with_role + table2_with_role
    
    # Sort by start time
    sorted_table = sorted(combined_table, key=lambda phrase: phrase['start'])
    
    # Merge consecutive records with same role
    merged_table = merge_consecutive_same_role(sorted_table)
    
    return merged_table

def merge_consecutive_same_role(sorted_table):
    """
    Merge consecutive records that have the same role.
    
    Args:
        sorted_table (list): List of dictionaries sorted by start time
        
    Returns:
        list: Table with consecutive same-role records merged
    """
    if not sorted_table:
        return []
    
    merged_table = []
    current_record = sorted_table[0].copy()
    
    for i in range(1, len(sorted_table)):
        next_record = sorted_table[i]
        
        # If same role, merge the texts
        if current_record["role"] == next_record["role"]:
            # Combine texts with a space
            current_record["text"] = current_record["text"].strip() + " " + next_record["text"].strip()
        else:
            # Different role, save current record and start new one
            merged_table.append(current_record)
            current_record = next_record.copy()
    
    # Don't forget to add the last record
    merged_table.append(current_record)
    
    return merged_table


def format_conversation_text(combined_table):
    """
    Converts a combined phrase table into formatted conversation text.
    
    Args:
        combined_table (list): List of dictionaries with format {"text": str, "start": float, "role": str}
    
    Returns:
        str: Formatted conversation text in format:
             AG: text
             CL: text
             AG: text
             ...
    """
    if not combined_table:
        return ""
    
    formatted_lines = []
    for phrase in combined_table:
        role = phrase.get('role', 'UNKNOWN')
        text = phrase.get('text', '').strip()
        start = phrase.get('start', 0.0)
        if text:  # Only add non-empty text
            formatted_lines.append(f"{role}: {text}.")

    return '\n'.join(formatted_lines)




def fix_whisper_timestamp_drift_method1(left_transcription, right_transcription, left_file, right_file):
    """
    Method 1: Scale timestamps proportionally based on actual audio duration.
    """
    print("\n=== FIXING DRIFT: METHOD 1 (Proportional Scaling) ===")
    
    # Get actual audio durations
    left_audio, sr_left = librosa.load(left_file, sr=None)
    right_audio, sr_right = librosa.load(right_file, sr=None)
    
    actual_left_duration = len(left_audio) / sr_left
    actual_right_duration = len(right_audio) / sr_right
    
    # print(f"Actual audio durations:")
    # print(f"Left:  {actual_left_duration:.3f}s")
    # print(f"Right: {actual_right_duration:.3f}s")
    
    # Get Whisper's perceived durations
    left_whisper_duration = left_transcription.segments[-1].end if left_transcription.segments else 0
    right_whisper_duration = right_transcription.segments[-1].end if right_transcription.segments else 0
    
    # print(f"Whisper transcription durations:")
    # print(f"Left:  {left_whisper_duration:.3f}s")
    # print(f"Right: {right_whisper_duration:.3f}s")
    
    # Calculate scaling factors
    left_scale = actual_left_duration / left_whisper_duration if left_whisper_duration > 0 else 1.0
    right_scale = actual_right_duration / right_whisper_duration if right_whisper_duration > 0 else 1.0
    
    # print(f"Scaling factors:")
    # print(f"Left:  {left_scale:.6f}")
    # print(f"Right: {right_scale:.6f}")
    
    # Apply scaling to timestamps
    def scale_transcription_timestamps(transcription, scale_factor):
        if not hasattr(transcription, 'segments'):
            return transcription
            
        for segment in transcription.segments:
            segment.start *= scale_factor
            segment.end *= scale_factor
            
            # Scale word-level timestamps if present
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    if hasattr(word, 'start'):
                        word.start *= scale_factor
                    if hasattr(word, 'end'):
                        word.end *= scale_factor
        
        return transcription
    
    # Only scale if there's significant drift
    if abs(left_scale - 1.0) > 0.01:
        print(f"Scaling left timestamps by {left_scale:.6f}")
        left_transcription = scale_transcription_timestamps(left_transcription, left_scale)
    
    if abs(right_scale - 1.0) > 0.01:
        print(f"Scaling right timestamps by {right_scale:.6f}")
        right_transcription = scale_transcription_timestamps(right_transcription, right_scale)
    
    return left_transcription, right_transcription