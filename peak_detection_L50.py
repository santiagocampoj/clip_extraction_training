import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from datetime import datetime
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import soundfile as sf
import resampy
import warnings
import math
import logging

from visualization_peak import *
import argparse

import params as yamnet_params
import yamnet as yamnet_model

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename='peak_detection.log', 
    filemode='w'
    )

# warning messages disabled
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Couldn't find ffmpeg or avconv")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WINDOW_SIZE = 30  # seconds
PROMINENCE = 1
WIDTH = 1
TOP_PREDIC = 3
ADDING_THRESHOLD = 10  # seconds


def make_clip_predictions(waveform, sr):
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    # convert to mono and resample if necessary
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # actual prediction
    scores, embeddings, spectrogram = yamnet(waveform)
    prediction = np.mean(scores, axis=0)
    # top 3 classes
    top_predict_i = np.argsort(prediction)[::-1][:TOP_PREDIC]
    # logging.info the na,e of the class
    logging.info(f"Top 3 classes: {[yamnet_classes[i] for i in top_predict_i]}")
    return [yamnet_classes[i] for i in top_predict_i], [prediction[i] for i in top_predict_i]


def leq(levels):
    levels = levels[~np.isnan(levels)]
    l = np.array(levels)
    return 10 * np.log10(np.mean(np.power(10, l / 10)))


def find_audiomoth_folders(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'AUDIOMOTH' in dirs:
            ## add AUDIOMOTH to the path
            root = os.path.join(root, 'AUDIOMOTH')
            # find the csv file
            for root, dirs, files in os.walk(root):
                for file in files:
                    if file.endswith('.csv'):
                        # add absolute path to each file
                        csv_file_path = os.path.join(root, file)
                        # print(csv_file_path)
                        yield csv_file_path


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=False, help="Path to the csv file")
    return parser.parse_args()


def main():
    # python .\peak_detection_L50.py -p "\\192.168.205.117\AAC_Server\PUERTOS\NOISEPORT\20231211_SANTUR"

    args = argument_parser()
    base_path = args.path
    # collectin all the csv files
    csv_files = list(find_audiomoth_folders(base_path))

    # process each csv file
    for csv_file in tqdm(csv_files, desc='Processing csv files'):
        # reach teh csv file
        df = pd.read_csv(csv_file)

        title = csv_file.split("\\")[-3]
        audiomoth_folder = csv_file.replace("5-Resultados", "3-Medidas").replace("SPL", "AUDIOMOTH")
        audiomoth_folder = "\\".join(audiomoth_folder.split("\\")[:-1])
        output_folder = "\\".join(csv_file.split("\\")[:-1])
        output_folder = output_folder.replace("3-Medidas", "5-Resultados").replace("AUDIOMOTH", "SPL")
        output_folder = os.path.join(output_folder, "Peaks")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        df['filename'] = df['filename'].apply(lambda x: os.path.join(audiomoth_folder, x))
        df['date'] = pd.to_datetime(df['date'])
        # rolling median for the LA values with a window of 30 seconds
        df['LA_median'] = df['LA'].rolling(window=WINDOW_SIZE, min_periods=1).quantile(0.5) + ADDING_THRESHOLD
        above_threshold = df[df['LA'] > df['LA_median']]


        if not above_threshold.empty:
            clip_info = []
            peaks, properties = find_peaks(above_threshold['LA'], prominence=PROMINENCE, width=WIDTH)
            df_peaks = above_threshold.iloc[peaks]
            logging.info(f"Detected {len(df_peaks)} peaks")

            start_points = properties['left_ips'].astype(int)
            end_points = properties['right_ips'].astype(int)
            durations = end_points - start_points
            
            # save the peaks in a csv
            peak_data = []
            for start, end in zip(start_points, end_points):
                peak_LA_values = above_threshold['LA'].iloc[start:end+1].values
                leq_value = leq(peak_LA_values)
                peak_data.append({
                    'filename': above_threshold['filename'].iloc[start],
                    'start_time': above_threshold['date'].iloc[start],
                    'end_time': above_threshold['date'].iloc[end],
                    'duration': end - start,
                    'leq': leq_value.round(2),
                    'LA_values': peak_LA_values.tolist()
                })

            peaks_df = pd.DataFrame(peak_data)
            peaks_df.to_csv(os.path.join(output_folder, f"peaks_detection_{title}.csv"), index=False)
            logging.info(f"Peaks saved at {output_folder} as peaks_detection_{title}.csv")


            mean = np.mean(durations)
            mean_rounded = np.round(mean, 2)
            logging.info("")
            logging.info(f"Average duration: {mean_rounded} seconds")
            logging.info(f"Max duration: {np.max(durations)} seconds")
            logging.info(f"Min duration: {np.min(durations)} seconds")
            
            # process each peak
            num_peaks_processed = 0
            # comment this line to process all the peaks
            # peaks_df = peaks_df.head(10)
            for index, row in tqdm(peaks_df.iterrows(), total=len(peaks_df)):
                logging.info(f"Extracting segment from {row['filename']}")
                try:
                    #read the whole audiofile
                    wav_data, sr = sf.read(row['filename'], dtype=np.int16)
                    logging.info(f"Sample rate: {sr}")
                except Exception as e:
                    logging.warning(f"Failed to read file {row['filename']}. Error: {e}")
                    return None
                
                # start time of the audio file
                start_time_audio = row['filename']
                start_time_audio = start_time_audio.split('\\')[-1].split('_')
                start_time_audio = start_time_audio[0] + start_time_audio[1].split('.')[0]
                start_time_audio = datetime.strptime(start_time_audio, '%Y%m%d%H%M%S')
                logging.info(f"Start time of the audio file: {start_time_audio}")


                # get start and end time of the peak
                start_time = row['start_time']
                end_time = row['end_time']
                duration = row['duration']
                logging.info(f"Start time: {start_time}, End time: {end_time}, Duration: {duration}")


                ##### SLICE AUDIO #####
                start_time = (row['start_time'] - pd.Timestamp(start_time_audio)).total_seconds()
                end_time = (row['end_time'] - pd.Timestamp(start_time_audio)).total_seconds()

                # samples indices, add a secondto the start and the end time
                start_index = int((start_time - 0.25) * sr)
                end_index = int((end_time + 0.25) * sr)

                # extract segment
                segment = wav_data[start_index:end_index]
                actual_segment_duration = len(segment) / sr
                # if actual_segment_duration == 0, skip the peak and the segment
                if actual_segment_duration == 0:
                    logging.warning(f"Segment duration {actual_segment_duration}")
                    logging.warning("Segment duration is 0, skipping the peak, it beloong to two different audio files")
                    continue
                elif actual_segment_duration > duration + 20:
                    logging.warning(f"Segment duration {actual_segment_duration}")
                    logging.warning("Segment duration is more than 10s, skipping the peak, it beloong to two different audio files")
                    continue                
                else:
                    logging.info(f"Actual Segment duration: {actual_segment_duration} seconds")
                

                # START PREDICTION
                wave_form = segment / 32768.0  # Convert to [-1.0, +1.0]
                wave_form = wave_form.astype('float32')
                
                # make predictions
                classes, predictions = make_clip_predictions(wave_form, sr)
                num_peaks_processed += 1


                #### MAKE A CLIP AND SAVE IT ####
                peak_date_str = row['start_time'].strftime('%Y%m%d_%H%M%S')
                clip_filename = f"{peak_date_str}_{classes[0]}.wav"
                logging.info(f"Clip filename: {clip_filename}")
                os.makedirs(os.path.join(output_folder, 'peak_clips'), exist_ok=True)
                clip_path = os.path.join(output_folder, 'peak_clips', clip_filename)
                sf.write(clip_path, segment, sr)
                logging.info(f"Segment {clip_filename} saved to {clip_path}")


                #### SAVE INFO IN A CSV ####
                clip_info.append({
                    'filename': clip_path,
                    'start_time': row['start_time'],
                    'end_time': row['end_time'],
                    'duration': actual_segment_duration,
                    'classes': classes,
                    'predictions': predictions,
                    # adding the peak leq value
                    'leq': row['leq'],  # leq value
                    'LA_values': row['LA_values']  # LA_values
                })
                logging.info(f"Clip info: {clip_info[-1]}")


            # save the info in a csv
            clips_df = pd.DataFrame(clip_info)
            clips_df.to_csv(os.path.join(output_folder, f"peak_prediction_{title}.csv"), index=False)
            logging.info(f"Extracted {num_peaks_processed} clips and saved information at {output_folder}")
            logging.info(f"Actual Processed {num_peaks_processed} peaks")
        

        else:
            logging.error("No peaks detected")



if __name__ == "__main__":
    main()