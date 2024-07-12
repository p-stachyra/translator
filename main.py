import whisper

from translator import TranscriptGenerator
from subtitles_aligner import SubtitlesAligner


SAMPLE_PATH = r"E:\projects\translator\snow.mp4"


def main():
    audio = whisper.load_audio(SAMPLE_PATH)
    tg = TranscriptGenerator(model_name="large-v2.pt")
    sa = SubtitlesAligner(filepath=SAMPLE_PATH)

    transcription = tg.translate_to_english(audio=audio)
    # print(transcription)
    # sdf = sa.get_subtitles(transcription_data=transcription)
    # sdf.to_csv("example_subtitles.csv")
    new_video = sa.align_subtitles(transcription_data=transcription)

    
    #print(sa.prepare_subtitles(subtitles_dataframe=sdf))


if __name__ == "__main__":
    main()