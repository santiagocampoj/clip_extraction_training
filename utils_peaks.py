import pandas as pd
import numpy as np
import os
import tqdm

import params as yamnet_params
import yamnet as yamnet_model
import soundfile as sf
import resampy

WINDOW_SIZE = 30  # seconds
PROMINENCE = 1
WIDTH = 1

params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')



def save_clips_from_peaks(df, output_folder, sampling_rate, title, logging):
    csv_output_folder = output_folder
    # if output folder does not exist, create it
    output_folder = os.path.join(output_folder, "peak_clips")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        audio_file = row['filename']
        start_time_seconds = row['start_time']
        end_time_seconds = row['end_time']
        # round duration to 2 decimals
        duration = np.round(end_time_seconds - start_time_seconds, 2)

        # loading audio file
        wav_data, sr = sf.read(audio_file, dtype='int16')
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        logging.info(f"Processing audio file: {audio_file}")
        logging.info(f"Starting time: {start_time_seconds} seconds, Ending time: {end_time_seconds} seconds, Duration: {duration} seconds")
        
        # milliseconds
        start_time_ms = int(start_time_seconds * 1000)
        end_time_ms = int(end_time_seconds * 1000)
        logging.info(f"Starting time milliseconds: {start_time_ms}, Ending time milliseconds: {end_time_ms}")


        clip = waveform[start_time_ms:end_time_ms]
        clip = clip.astype('float32')
        # clip duration in seconds
        clip_duration = len(clip) / sr
        logging.info(f"Clip duration: {clip_duration} seconds")

        if len(clip.shape) > 1:
            clip = np.mean(clip, axis=1)
        if sr != params.sample_rate:
            clip = resampy.resample(clip, sr, params.sample_rate)
        
        scores, embeddings, spectrogram = yamnet(clip)
        prediction = np.mean(scores, axis=0)
        top3_i = np.argsort(prediction)[::-1][:3]

        logging.info(f"Top 3 classes: {', '.join([yamnet_classes[i] for i in top3_i])}")
        logging.info(f"Probabilities: {', '.join([str(prediction[i]) for i in top3_i])}")

        # save a dataframe with the filename, start time, end time, and the top 3 classes
        filename = audio_file.split("\\")[-1]
        df_clip = pd.DataFrame({
            'filename': filename,
            'start_time': [start_time_seconds],
            'end_time': [end_time_seconds],
            'duration': [duration],
            'predictions': [', '.join([yamnet_classes[i] for i in top3_i])],
            'probabilities': [', '.join([str(prediction[i]) for i in top3_i])],
        })

        # save the dataframe to a csv file
        csv_filename = f'peak_predictions_L50_{title}.csv'
        csv_path = os.path.join(csv_output_folder, csv_filename)
        # save to csv with the header only if the file does not exist
        df_clip.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        logging.info(f"Saving predictions to {csv_path}")

        logging.info(f"Filename: {filename}")
        clip_path = os.path.join(output_folder, f"{filename}_peak_{yamnet_classes[top3_i[0]]}_{index}.wav")

        # using sf.write to save the clip
        sf.write(clip_path, clip, sampling_rate)
        logging.info(f"Saving clip {clip_path}")