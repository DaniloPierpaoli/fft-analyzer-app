import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftshift
import pandas as pd

# ========== Your FFT Peak Detector ==========
def find_freqs(V, sample_rate, M):
    N = len(V)
    tN = N / sample_rate
    n = np.arange(-N/2, N/2, 1)
    f = n / tN
    V_fft = fft(V, norm='forward')
    S_n = np.real(V_fft * np.conjugate(V_fft))
    half_freq = int(len(f) / 2)
    positive_f = f[half_freq:]
    positive_Sn = fftshift(S_n)[half_freq:]

    peaks = np.empty((0, 2))
    old_Sn = 0
    index = 0
    while index < len(positive_Sn):
        Sn = positive_Sn[index]
        if Sn > old_Sn:
            if (Sn > positive_Sn[index + 1] and
                Sn > positive_Sn[index + 3] and
                Sn > positive_Sn[index + 7] and
                Sn > positive_Sn[index + 11] and
                Sn > positive_Sn[index + 15]):
                peaks = np.concatenate([peaks, [[Sn, positive_f[index]]]], axis=0)
                old_Sn = 0
                index += 20
                if index > len(positive_Sn) - 23:
                    break
            else:
                index += 1
        else:
            old_Sn = Sn
            index += 1

    df_peaks = pd.DataFrame(peaks, columns=["Sn", "Frequency (Hz)"])
    sorted_df = df_peaks.sort_values(by="Sn", ascending=False)
    return sorted_df["Frequency (Hz)"].head(M)


# ========== Your Note Detector ==========
def find_notes(V, sample_rate, M):
    def note_finder(frequency):
        notes = {
            16.35: 'C', 17.32: 'C#', 18.35: 'D', 19.45: 'D#', 20.60: 'E',
            21.83: 'F', 23.12: 'F#', 24.50: 'G', 25.96: 'G#', 27.50: 'A',
            29.14: 'A#', 30.87: 'B'
        }
        octaves = [0, 1, 2, 3, 4, 5, 6, 7]
        old_note = 16.35
        old_octave = 0
        old_label = 'C'

        for octave in octaves:
            for base_freq, label in notes.items():
                freq_val = base_freq * 2**octave
                if frequency < freq_val:
                    if abs(freq_val - frequency) < abs(old_note - frequency):
                        return label, octave
                    else:
                        return old_label, old_octave
                else:
                    old_note = freq_val
                    old_octave = octave
                    old_label = label

    frequencies = find_freqs(V, sample_rate=sample_rate, M=M)
    notes_array = np.empty((0, 3))
    for freq in frequencies:
        note, octave = note_finder(freq)
        notes_array = np.concatenate([notes_array, [[note, octave, freq]]], axis=0)
    return pd.DataFrame(notes_array, columns=["Note", "Octave", "Frequency (Hz)"])


# ========== Streamlit App ==========
st.set_page_config(page_title="Sound FFT Analyzer", layout="centered")

st.title("🎸Sound FFT Analyzer")
st.markdown("Upload a WAV file to analyze frequency peaks and notes.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file:
    # === Sidebar settings ===
    st.sidebar.subheader("Settings")
    M = st.sidebar.slider("Number of Peaks to Show", min_value=1, max_value=10, value=5)

    rate, data = wavfile.read(uploaded_file)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize if float
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = (data * 32767).astype(np.int16)

    # ===== Notes Table =====
    st.subheader(f"Detected Notes (Top {M})")
    detected_notes = find_notes(data, sample_rate=rate, M=M)
    st.dataframe(detected_notes)

    # ===== Full FFT Plot with Peak Markers =====
    st.subheader("Frequency Spectrum")

    fft_data = np.abs(np.fft.fft(data))
    freqs = np.fft.fftfreq(len(fft_data), d=1 / rate)
    half_N = len(freqs) // 2

    peak_freqs = find_freqs(data, sample_rate=rate, M=M).values

    fig, ax = plt.subplots()
    ax.plot(freqs[:half_N], fft_data[:half_N], color='tomato')

    for f in peak_freqs:
        idx = np.argmin(np.abs(freqs[:half_N] - f))
        amp = fft_data[idx]
        ax.scatter(f, amp, color='yellow', s=50, zorder=5)
        ax.text(f, amp, f"{f:.1f} Hz", fontsize=8, ha='center', va='bottom', color='black')

    f_min = max(0, min(peak_freqs) - 30)
    f_max = max(peak_freqs) + 30
    ax.set_xlim(f_min, f_max)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT Spectrum with Peaks")
    ax.grid(True)
    st.pyplot(fig)
