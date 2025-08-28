import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import pandas as pd

# ========== Your FFT Peak Detector ==========
def find_freqs(V, samplerate, M):
    """
    Returns the M frequencies corresponding to the largest peaks of the ESD.
    Inputs:
        V: discrete signal
        samplerate: Hz
        M: number of frequencies to find
    Output:
        freq_peaks: array of M frequencies
        esd_peaks: array of corresponding ESD values
    """
    N = len(V)

    # calculate the Fourier transform of V
    V_fft = fft(V, norm='forward')

    # Calculate energy spectral density
    esd = np.abs(V_fft[:N // 2])**2

    # Returns array of frequencies of the FFT 
    freqs = fftfreq(N, 1 / samplerate)[:N // 2]

    data = pd.DataFrame({
        'frequency': freqs,
        'esd': esd
    })

    # Returns top M peaks
    top_peaks = data.sort_values(by='esd', ascending=False).head(M).sort_values(by='frequency')

    return top_peaks['frequency'].values, top_peaks['esd'].values


# ========== Your Note Detector ==========
def find_notes(V, samplerate, M):
    """
    Returns the M musical notes closest to the M largest ESD peaks.
    Inputs:
        V: discrete signal
        samplerate: Hz
        M: number of peaks to find
    Outputs:
        notes: list of note names
        octaves: list of octave numbers
        frequencies: list of corresponding frequencies
    """
    note_base_freqs = {
        'C': 261.6, 'C#': 277.2, 'D': 293.7, 'D#': 311.1, 'E': 329.6,
        'F': 349.2, 'F#': 370.0, 'G': 392.0, 'G#': 415.3, 'A': 440.0,
        'A#': 466.2, 'B': 493.9
    }

    note_names = []
    octaves = []
    peak_freqs, _ = find_freqs(V, samplerate, M)

    for f in peak_freqs:
        closest_note = None
        closest_octave = None
        min_diff = float('inf')

        for octave in range(0, 8):
            for note, base in note_base_freqs.items():
                freq = base * (2 ** (octave - 4))
                if abs(f - freq) < min_diff:
                    min_diff = abs(f - freq)
                    closest_note = note
                    closest_octave = octave

        note_names.append(closest_note)
        octaves.append(closest_octave)

    return note_names, octaves, peak_freqs


# ========== Streamlit App ==========
st.set_page_config(page_title="Guitar FFT Analyzer", layout="centered")

st.title("Frenquencies Analyzer using Fast Fourier Transforms")
st.markdown("Upload a WAV file to analyze frequency peaks and musical notes.")

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
        data = (data * 32767).astype(np.float)

    # ===== Notes Table =====
    st.subheader(f"Detected Notes (Top {M})")
    note_names, octaves, freqs = find_notes(data, samplerate=rate, M=M)
    df_notes = pd.DataFrame({
        "Note": note_names,
        "Octave": octaves,
        "Frequency (Hz)": [f"{f:.2f}" for f in freqs]
    })
    st.dataframe(df_notes)

    # ===== Full FFT Plot with Peak Markers =====
    st.subheader("Frequency Spectrum")

    fft_data = np.abs(np.fft.fft(data))
    freqs_full = np.fft.fftfreq(len(fft_data), d=1 / rate)
    half_N = len(freqs_full) // 2

    peak_freqs, _ = find_freqs(data, samplerate=rate, M=M)

    fig, ax = plt.subplots()
    ax.plot(freqs_full[:half_N], fft_data[:half_N], color='tomato')

    for f in peak_freqs:
        idx = np.argmin(np.abs(freqs_full[:half_N] - f))
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

