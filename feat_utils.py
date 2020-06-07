#############################################################
#                                                           #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################

import numpy as np
import librosa
import config as cfg
from scipy import signal
def additive_mixing(speech_file, noise_file, snr):
    """Mix normalized source1 and source2.
    Args:
      speech_file: source1.
      noise_file: source2.
      snr: Signal to noise ratio

    Returns:
      mix_audio: ndarray, mixed audio.
      s: ndarray, pad or truncated and scalered source1.
      n: ndarray, scaled source2.
      alpha: float, normalize coefficient.
    """

    speech_audio,_ = librosa.load(speech_file, sr=cfg.sr)
    noise_audio, _ = librosa.load(noise_file, sr=cfg.sr)

    # If noise length is less than speech lengh, repeat the noise
    if len(noise_audio) < len(speech_audio):
        n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
        noise_audio = np.tile(noise_audio, n_repeat)

    noise_audio = noise_audio[0: len(speech_audio)]

    # Scale speech to given snr.
    scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
    speech_audio *= scaler

    mixed_audio = speech_audio + noise_audio

    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    speech_audio *= alpha
    noise_audio *= alpha
    return mixed_audio, speech_audio, noise_audio, alpha

def rms(y):
    """Root mean square.
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr.

    Args:
      s: ndarray, source1.
      n: ndarray, source2.
      snr: float, SNR.
      method: 'rms'.

    Outputs:
      float, scaler.
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def calc_sp(wav,mode='magnitude'):
    
    X=librosa.stft(wav, n_fft=cfg.n_fft, 
            win_length=cfg.win_length,hop_length=cfg.hop_length)
#    _,_,X=signal.spectrogram(wav, window=np.hamming(cfg.win_length),
#            nperseg=cfg.win_length, noverlap=cfg.win_length-cfg.hop_length,
#            mode=mode)
    if mode == 'magnitude':
        X=np.abs(X)
    
    return X.T

def log_sp(x):
    return np.log(x + 1e-08)

def pad_with_border(x):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * cfg.n_pad + [x] + [x[-1:]] * cfg.n_pad
    return np.concatenate(x_pad_list, axis=0)

def mat_2d_to_3d(x, hop=1):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    x=pad_with_border(x)      
    len_x, n_in = x.shape
    if (len_x < cfg.n_context):
        x = np.concatenate((x, np.zeros((cfg.n_context - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + cfg.n_context <= len_x):
        x3d.append(x[i1 : i1 + cfg.n_context])
        i1 += hop
    return np.array(x3d)

