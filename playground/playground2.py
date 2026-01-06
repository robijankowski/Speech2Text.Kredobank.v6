# -*- coding: utf-8 -*-

#Import the openai Library
from openai import OpenAI
from utilities.whisper_correctors_utilities import split_transcription_into_roles
from utilities.whisper_correctors_utilities import find_unclear_words, transcript_audio_file

# Create an api client


AUDIO_FILES = [
    "./Sources/2 380976776096 Кобець Т. 06.02.wav",
    "./Sources/380975854673 Люція Гродзицька 08.05.wav",
    "./Sources/5 380972851516 Ковальчук М. 21.06.wav",
    "./Sources/6 380671202008 Воробець І. 23.03.wav",
    "./Sources/6 380974345718 Гасяк А. 27.06.wav" 
]

# Load audio file
audio_file= open(AUDIO_FILES[0], "rb")

# Transcribe
transcription = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1",
    prompt="Алло, Кобець, Гродзицька, Ковальчук, Воробець, Гетьман, Кредобанку, до вас телефонують з Кредобанку, мене звати Анна, телефонна розмова записується, ідентификації, як клієнта банку, за номером угоди, кредитного договору, кількість, на все добре, ЗСУ, станом, дуже дякую, платіж, ставте, затримують, готівковому, готівкових, місяці",
    response_format="verbose_json",
    # response_format="srt"
    # response_format="vtt"
    # response_format="json"
    language="uk",
    temperature=0.2,
    # response_format="text"
)

print(transcription)



# Print the transcribed text
# print(f"\n*************************************\n{transcription}\n*************************************")

# Print dialogue by roles AG and CL
# text = split_transcription_into_roles(transcription)

# Print found "unclear" words
# text = find_unclear_words(transcription)

# print(f"\n*************************************\n{text}\n*************************************")

# text = abstract_summary_extraction(transcription.text)
# print(f"\n*************************************\n{text}\n*************************************")

# text = sentiment_analysis(transcription.text)
# print(f"\n*************************************\n{text}\n*************************************")

# text = action_item_extraction(transcription.text)
# print(f"\n*************************************\n{text}\n*************************************")

