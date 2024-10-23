import subprocess
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import wave
import contextlib

# Full path to yt-dlp
yt_dlp_path = r"C:\Users\xxx\youtube-dl\yt-dlp.exe"
audio_link = "https://www.youtube.com/watch?v=sVx1mJDeUjY"
# Define the command as a list of arguments
command = [
    yt_dlp_path,
    audio_link,
    "-f", "bestaudio",
    "--extract-audio",
    "--audio-format", "wav",
    "--output", "input.wav"
]

subprocess.run(command, check=True)

# Read the audio file
sample_rate, data = wavfile.read('input.wav')

# Normalize the audio data
def normalize(data):
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        return (data.astype(np.float64) - 128) / 128.0
    elif data.dtype in (np.float32, np.float64):
        return data.astype(np.float64)
    else:
        raise ValueError('Unsupported data type')

normalized_data = normalize(data)

# Apply μ-law companding
def mu_law_companding(data, mu=255):
    data_mu = np.sign(data) * np.log1p(mu * np.abs(data)) / np.log1p(mu)
    return data_mu

data_companded = mu_law_companding(normalized_data)

# Add dithering
def add_dither(data, bits):
    dither_amplitude = 1.0 / (2 ** bits)
    dither = np.random.uniform(-dither_amplitude / 2, dither_amplitude / 2, size=data.shape)
    return data + dither

bits = 4  # Desired bit depth (change as needed)
data_dithered = add_dither(data_companded, bits)

# Noise shaping quantization
def noise_shaping_quantization(data, bits):
    levels = 2 ** bits
    quantization_error = np.zeros(data.shape[1]) if data.ndim > 1 else 0.0
    data_quantized = np.zeros_like(data)
    
    data_shifted = (data + 1.0) / 2.0  # Shift data to [0, 1]
    for i in range(len(data)):
        input_sample = data_shifted[i] + quantization_error
        quantized_sample = np.round(input_sample * (levels - 1)) / (levels - 1)
        data_quantized[i] = quantized_sample
        quantization_error = input_sample - quantized_sample  # Update error
    data_quantized = data_quantized * 2.0 - 1.0  # Shift back to [-1, 1]
    return data_quantized

quantized_data = noise_shaping_quantization(data_dithered, bits)

# Apply μ-law expansion
def mu_law_expanding(data, mu=255):
    data_expanded = np.sign(data) * (1 / mu) * (np.expm1(np.abs(data) * np.log1p(mu)))
    return data_expanded

data_expanded = mu_law_expanding(quantized_data)

# Determine target_dtype based on bits
if bits <= 8:
    target_dtype = np.uint8
elif bits <= 16:
    target_dtype = np.int16
elif bits <= 24:
    target_dtype = np.int32  # We'll handle 24-bit data using 32-bit integers
elif bits <= 32:
    target_dtype = np.int32
else:
    raise ValueError('Unsupported bit depth')

# Denormalize the quantized data
def denormalize(data, target_dtype):
    if target_dtype == np.uint8:
        return np.uint8(np.clip((data + 1.0) * 128, 0, 255))
    elif target_dtype == np.int16:
        return np.int16(np.clip(data * 32767, -32768, 32767))
    elif target_dtype == np.int32:
        return np.int32(np.clip(data * 2147483647, -2147483648, 2147483647))
    else:
        raise ValueError('Unsupported target data type')

data_out = denormalize(data_expanded, target_dtype)

# Save the quantized audio file
wavfile.write('output.wav', sample_rate, data_out)

# Display WAV file details
def print_wav_details(filename):
    with contextlib.closing(wave.open(filename, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        bit_depth = sample_width * 8
        duration = num_frames / sample_rate

        print(f"WAV File Details: {filename}")
        print(f"Number of Channels: {num_channels}")
        print(f"Sample Width (bytes): {sample_width}")
        print(f"Sample Rate (Hz): {sample_rate}")
        print(f"Number of Frames: {num_frames}")
        print(f"Bit Depth: {bit_depth}-bit")
        print(f"Duration (seconds): {duration:.2f}")
        print("-" * 30)

print_wav_details('input.wav')
print_wav_details('output.wav')