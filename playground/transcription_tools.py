
from transcribe.utilities.transcribe_stereo_tools import async_transcript_audio_file_verbose_o4_stereo, async_transcript_audio_file_verbose_o4_single_channel


def process_stereo_with_o4(left_file_cleaned: str, 
                           right_file_cleaned: str, 
                           org_file_cleaned: str, 
                           metadata_text: str =""):
    """
    Process audio files with O4 transcription
    """


    # Transcribe left channel
    print("\n\n\n" + "="*30 + f" Transcribe O4 AGENT left wav " + "="*30)
    left_cleaned_transcription = async_transcript_audio_file_verbose_o4_single_channel(left_file_cleaned, 
                                                                  o4_metadata_text=metadata_text)
    # if left_cleaned_transcription:      
    #     print("\n" + str(left_cleaned_transcription))

    # Transcribe right channel
    print("\n\n\n" + "="*30 + f" Transcribe O4 CLIENT right wav as " + "="*30)
    right_cleaned_transcription = async_transcript_audio_file_verbose_o4_single_channel(right_file_cleaned, 
                                                                   o4_metadata_text=metadata_text)
    # if right_cleaned_transcription:
    #     print("\n" + str(right_cleaned_transcription))

    print("\n\n\n" + "="*30 + f" Transcribe O4 ORIGINAL wav " + "="*30)
    org_cleaned_transcription = async_transcript_audio_file_verbose_o4_stereo(org_file_cleaned, 
                                                                     o4_metadata_text=metadata_text)
    # if o4_mono_cleaned_transcription:
    #     print("\n" + str(o4_mono_cleaned_transcription))

    return left_cleaned_transcription, right_cleaned_transcription, org_cleaned_transcription


   