import os
import numpy as np
import cv2
import torch
from PhysNetModel import PhysNet
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import matplotlib.pyplot as plt

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, sig)
    return y

def hr_fft(sig, fs, harmonics_removal=True):
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig_f.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig_f.shape[0] * fs
    hr2 = f_hr2 * 60

    if harmonics_removal:
        hr = hr2 if np.abs(hr1 - 2 * hr2) < 10 else hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig_f)) / len(sig_f) * fs * 60
    return hr, sig_f_original, x_hr

def load_frames_from_folder(folder_path):
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    return np.array(frames)

def process_video(frames, model, fps, device, clip_length=450):
    """
    Process video frames and extract rPPG using the PhysNet model.
    """
    # Normalize frames to [0, 1]
    frames = frames.astype('float32') / 255.0
    frames = frames.transpose(0, 3, 1, 2)  # Rearrange dimensions to [frames, channels, height, width]

    # Split frames into clips of `clip_length`
    num_clips = len(frames) // clip_length
    remaining_frames = len(frames) % clip_length

    # Process full-length clips
    full_clips = frames[:num_clips * clip_length].reshape(num_clips, clip_length, 3, 128, 128)

    # **Transpose the clips to match PhysNet's expected input shape**
    full_clips = full_clips.transpose(0, 2, 1, 3, 4)  # [batch_size, channels, frames, height, width]

    # Convert to PyTorch tensor
    full_clips = torch.tensor(full_clips).to(device)

    # Extract rPPG for each clip
    rppg_all = []
    with torch.no_grad():
        for clip in full_clips:
            # print("pineapple", clip.shape)
            clip = clip.unsqueeze(0)  # Add batch dimension
            rppg = model(clip)[:, -1, :]  # Extract rPPG signal
            rppg_all.append(rppg[0].detach().cpu().numpy())

    if remaining_frames > 10:
        remaining_clip = frames[-remaining_frames:]
        remaining_clip = remaining_clip.transpose(1, 0, 2, 3)
        remaining_clip = torch.tensor(remaining_clip).to(device)

        remaining_clip = remaining_clip.unsqueeze(0)
        rppg = model(remaining_clip)[:, -1, :]  # Extract rPPG signal
        rppg_all.append(rppg[0].detach().cpu().numpy())

    if remaining_frames <= 10 and remaining_frames > 0:
        rppg_all.append(np.zeros(remaining_frames, dtype=np.float32))

    # Concatenate rPPG signals from all clips
    rppg_all = np.concatenate(rppg_all, axis=-1)
    rppg_all = butter_bandpass(rppg_all, lowcut=0.6, highcut=4, fs=fps)  # Bandpass filter

    assert len(rppg_all) == len(frames), f"Mismatch: rPPG length {len(rppg_all)} != total frames {len(frames)}"

    return rppg_all

def save_results(rppg, fps, output_path):
    hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

    ax1.plot(np.arange(len(rppg)) / fps, rppg)
    ax1.set_xlabel('time (sec)')
    ax1.grid('on')
    ax1.set_title('rPPG waveform')

    ax2.plot(psd_x, psd_y)
    ax2.set_xlabel('heart rate (bpm)')
    ax2.set_xlim([40, 200])
    ax2.grid('on')
    ax2.set_title('PSD')

    plt.savefig(f'{output_path}_results.png')
    plt.close(fig)

    np.savez(f'{output_path}_rppg.npz', rppg=rppg, hr=hr)

    print(f'Processed video: Heart rate = {hr:.2f} bpm')

def process_all_videos(input_root, output_root, model, fps, device):
    for subfolder in sorted(os.listdir(input_root)):
        subfolder_path = os.path.join(input_root, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")

            output_path = os.path.join(output_root, subfolder)
            os.makedirs(output_path, exist_ok=True)

            frames = load_frames_from_folder(subfolder_path)

            if len(frames) == 0:
                print(f"No frames found in folder: {subfolder}")
                continue

            rppg_signal = process_video(frames, model, fps, device)

            save_results(rppg_signal, fps, os.path.join(output_path, 'rppg'))

input_root = 'dataset_new/cropped_aligned'
output_root = 'dataset_new/rppg'
os.makedirs(output_root, exist_ok=True)

fps = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PhysNet(S=2).to(device).eval()
model.load_state_dict(torch.load('models/Contrast-PhysNet/model_weights.pt', map_location=device))

process_all_videos(input_root, output_root, model, fps, device)
