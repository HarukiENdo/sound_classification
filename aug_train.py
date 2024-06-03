import os
import glob
import shutil
import random
random.seed(888)
import numpy as np
import soundfile as sf
import torchaudio
from scipy.signal import resample
import librosa
import audiomentations

data_dir=""

dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))

def trim_into_dynamic_range(signal):
    for i in range(len(signal)):
        point_integer = int(signal[i])
        if point_integer > 8191:
            point_integer = 8191
        elif point_integer < -8192:
            point_integer = -8192
        signal[i] = point_integer
    return signal

#add_white_noise 平均0,標準偏差signalのノイズを作る
def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    augmented_signal = trim_into_dynamic_range(augmented_signal)
    return augmented_signal.astype(np.int16)

#initial_length*time_stretch_factorの長さに変換する　タイムストレッチ
#new_lengthがinitial_lengthより小さい場合は0padding、大きい場合は16000までをreturn
def time_stretch(audio_data, time_stretch_factor):
    initial_length = len(audio_data) 
    new_length = int(initial_length * time_stretch_factor) # Calculate the new length of the audio
    stretched_audio = resample(audio_data, new_length) # Apply time stretching
    
    if new_length >= initial_length:
        return trim_into_dynamic_range(stretched_audio[:initial_length]).astype(np.int16)
    else:
        filled_stretched_audio = np.zeros(initial_length)
        filled_stretched_audio[:new_length] = stretched_audio
        return trim_into_dynamic_range(filled_stretched_audio).astype(np.int16)

#change_volume #ゲイン調整とは異なり単純にvolumeを何倍かする操作
def change_volume(audio_data,volume_factor):
    augmented_signal=audio_data*volume_factor
    augmented_signal = trim_into_dynamic_range(augmented_signal)
    return augmented_signal

#pitch shift
def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=num_semitones)

data_dir=""
out_dir = ""

min_time_mask_duration_percentage = 0.05
max_time_mask_duration_percentage = 0.20

transform_time_mask = audiomentations.TimeMask(
    min_band_part=min_time_mask_duration_percentage,
    max_band_part=max_time_mask_duration_percentage,
    fade=True, #音声信号の切れ目を自然な感じにする
    p=1.0,
)

min_gain_transition_duration_percentage = 0.05
max_gain_transition_duration_percentage = 0.05
min_gain_transition_db = -3.0
max_gain_transition_db = 3.0
transform_gain_transition = audiomentations.GainTransition(
    min_gain_db = min_gain_transition_db,
    max_gain_db = max_gain_transition_db,
    min_duration = min_gain_transition_duration_percentage,
    max_duration = max_gain_transition_duration_percentage,
    duration_unit = 'fraction',
    p=1.0
)

min_shift_percent=-0.2
max_shift_percent=0.2
transform_time_shift=audiomentations.Shift(
    min_shift=min_shift_percent,
    max_shift=max_shift_percent,
    p=1.0
)

# min_repeat_part_duration_second = 0.01
# max_repeat_part_duration_second = 0.07 #0.05
# transform_repeat_part = audiomentations.RepeatPart(
#     mode = 'replace',
#     min_part_duration = min_repeat_part_duration_second,
#     max_part_duration = max_repeat_part_duration_second,
#     crossfade_duration = 0.005, #crossfade_duration リピートされる部分と元の部分の間の繋ぎ目を滑らかなにするパラメータ
#     p=1.0
# )

def DA(dir_in,out,nb_needed):
    initial_data_paths = sorted(glob.glob(dir_in+'/*'))
    number_data_paths = len(initial_data_paths)
    
    for j in range(nb_needed):
        if j%1000 == 0:
            print("Progress: ", 100*j/nb_needed)
            
        chosen_index = random.randint(0,number_data_paths-1)
        chosen_data_path = initial_data_paths[chosen_index]
        chosen_data_path_name = os.path.basename(chosen_data_path[:-4])
        out_path = out + '/' + chosen_data_path_name + '_oversample_id_' + str(j) +  '.wav'
        
        audio_data, sample_rate = sf.read(chosen_data_path, dtype='int16')
        audio_data = audio_data[:16000]
    
        random_augment_choice = random.randint(0,10)
        if random_augment_choice==0: #white noise
            noise_factor = random.uniform(0.01,0.03) #0.01
            aug = add_white_noise(audio_data, noise_factor)
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==1: #time stretch
            time_stretch_factor = random.uniform(0.7,1.3) #0.9,1.1
            aug = time_stretch(audio_data, time_stretch_factor)
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==2: #pitch change
            pitch_scale_semitones = random.randint(-3,3)
            # CAUTION: don't use soundfile to load. use librosa
            audio_data, sample_rate = librosa.load(chosen_data_path, sr=16000)
            audio_data = audio_data[:16000]
            aug = pitch_scale(audio_data, 16000, pitch_scale_semitones)
            aug = trim_into_dynamic_range(aug*32768).astype(np.int16)
            sf.write(out_path, aug, 16000, 'PCM_16')
        elif random_augment_choice==3: #change volume
            volume_factor=random.uniform(0.8,1.2)
            aug = change_volume(audio_data, volume_factor)
            sf.write(out_path, aug, 16000, 'PCM_16')
        elif random_augment_choice==4: #time shift
            audio_data = audio_data.astype('float32')
            aug=transform_time_shift(audio_data,sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==5: #time mask
            audio_data = audio_data.astype('float32')
            aug = transform_time_mask(audio_data, sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==6: #gain transition
            audio_data = audio_data.astype('float32')
            aug = transform_gain_transition(audio_data, sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==7: #white noise & time_stretch
            noise_factor=random.uniform(0.01,0.03)
            aug = add_white_noise(audio_data, noise_factor)
            time_stretch_factor = random.uniform(0.8,1.2) #0.9,1.1
            aug = time_stretch(aug, time_stretch_factor)
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==8: #pitch_change & time-mask
            pitch_scale_semitones = random.randint(-2,2)
            # CAUTION: don't use soundfile to load. use librosa
            audio_data, sample_rate = librosa.load(chosen_data_path, sr=16000)
            audio_data = audio_data[:16000]
            aug = pitch_scale(audio_data, 16000, pitch_scale_semitones)
            aug = trim_into_dynamic_range(aug*32768).astype(np.int16)
            aug = aug.astype('float32') 
            aug = transform_time_mask(aug, sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==9: # time shift & gain transition
            audio_data = audio_data.astype('float32')
            aug=transform_time_shift(audio_data,sample_rate=16000)
            aug = transform_gain_transition(aug, sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        elif random_augment_choice==10: # whitenoise time mask
            noise_factor = random.uniform(0.01,0.03) #0.01
            aug = add_white_noise(audio_data, noise_factor)
            aug = aug.astype('float32')
            aug = transform_time_mask(aug, sample_rate=16000)
            aug = trim_into_dynamic_range(aug)
            aug = aug.astype('int16')
            sf.write(out_path, aug, sample_rate, 'PCM_16')
        # elif random_augment_choice==7: #repeat part
        #     audio_data = audio_data.astype('float32')
        #     aug = transform_repeat_part(audio_data, sample_rate=16000)
        #     aug = trim_into_dynamic_range(aug)
        #     aug = aug.astype('int16')
        #     sf.write(out_path, aug, sample_rate, 'PCM_16')
        # elif random_augment_choice==8: #change_volume & repeat part
        #     aug=audio_data*random.uniform(0.8,1.2)
        #     aug = transform_repeat_part(aug, sample_rate=16000)
        #     aug = trim_into_dynamic_range(aug)
        #     aug = aug.astype('int16')
        #     sf.write(out_path, aug, sample_rate, 'PCM_16')

def copy_files(dir_in,out):
    file_list=os.listdir(dir_in)
    for file_name in file_list:
        in_path=os.path.join(dir_in,file_name)
        out_path=os.path.join(out,file_name)
        if not os.path.exists(out_path):
            shutil.copy(in_path, out_path)

def copy_files_environ(dir_in,out):
    max_data=300000
    count=0
    file_list=os.listdir(dir_in)
    random.shuffle(file_list)
    
    for file_name in file_list:
        if count>=max_data:
            break
            
        in_path=os.path.join(dir_in,file_name)
        out_path=os.path.join(out,file_name)
        if not os.path.exists(out_path):
            shutil.copy(in_path, out_path)
            count+=1

dir_in =os.path.join(data_dir+'/cut')
out=os.path.join(out_dir+'/cut')
nb_needed = 250000-1129
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("cut DA finish")

dir_in =os.path.join(data_dir+'/car')
out=os.path.join(out_dir+'/car')
nb_needed = 250000-2064
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("car DA finish")

dir_in =os.path.join(data_dir+'/fruit')
out=os.path.join(out_dir+'/fruit')
nb_needed = 250000-3010
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("fruit DA finish")

dir_in =os.path.join(data_dir+'/leaf')
out=os.path.join(out_dir+'/leaf')
nb_needed = 250000-3217
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("leaf DA finish")

dir_in =os.path.join(data_dir+'/truck')
out=os.path.join(out_dir+'/truck')
nb_needed = 250000-89
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("truck DA finish")

dir_in =os.path.join(data_dir+'/unknown')
out=os.path.join(out_dir+'/unknown')
nb_needed = 250000-2589
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("unknown DA finish")

dir_in =os.path.join(data_dir+'/walk')
out=os.path.join(out_dir+'/walk')
nb_needed = 250000-73343
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("walk DA finish")

dir_in =os.path.join(data_dir+'/talk')
out=os.path.join(out_dir+'/talk')
nb_needed = 250000-15689
DA(dir_in,out,nb_needed)
copy_files(dir_in,out)
print("talk DA finish")

dir_in =os.path.join(data_dir+'/environ')
out=os.path.join(out_dir+'/environ')
copy_files(dir_in,out)
print("environ copy finish")

data_dir=""

dir_in = '/car'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/cut'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/environ'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/fruit'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/leaf'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/truck'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/unknown'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/walk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))
dir_in = '/talk'
dir_in =os.path.join(data_dir+dir_in)
print(dir_in, ': ',len( sorted(glob.glob(dir_in+'/*'))))