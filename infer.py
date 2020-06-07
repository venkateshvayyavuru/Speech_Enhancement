#############################################################
#                                                           #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################
import time
start1 = time.perf_counter()
import config as cfg
import feat_utils as fe
import numpy as np
from keras.models import load_model
import os
import pickle
import librosa
from librosa import istft, stft, output
import keras.losses
from lassoloss import cep_loss
keras.losses.cep_loss = cep_loss


#start_time = time.time()

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


scaler=pickle.load(open('./scaler_akm.p','rb'))
model=load_model('resnet128_CepLoss_lstm.h5')
def inference(mixed_audio):
	start = time.perf_counter()
	print("Time taken for all imports:", (time.perf_counter()-start1))
	mixed_complex_x = fe.calc_sp(mixed_audio,mode='complex')
	mixed_x = fe.log_sp(np.abs(mixed_complex_x))
	start2 = time.perf_counter()
	print("Time taken for calculating stft:",(time.perf_counter()-start))
	mixed_x=scaler.transform(mixed_x)
	start3 = time.perf_counter()
	print("Time taken for scaling:",(time.perf_counter()-start2))
	mixed_x_3d=fe.mat_2d_to_3d(mixed_x, hop=1)
	start4 = time.perf_counter()
	print("Time taken for converting 2d to 3d:",(time.perf_counter()-start3))
	pred=model.predict(mixed_x_3d)
	start5 = time.perf_counter()
	print("Time taken for prediction:",(time.perf_counter()-start4))
	pred=scaler.inverse_transform(pred)
	start6 = time.perf_counter()
	print("Time taken for inverse scaling:",(time.perf_counter()-start5))
	pred_sp = np.exp(pred)
	s=istft((pred_sp * np.exp(1j*np.angle(mixed_complex_x))).T, win_length=cfg.win_length, hop_length=cfg.hop_length)
	enh_file='enh_wav_file.wav'
	output.write_wav(enh_file, s/max(abs(s)), 16000)
	print("Time taken for synthesizing the enhanced wave file:",(time.perf_counter()-start6))

mixed_audio,_ = librosa.load( "Noisy_Files/example_5min.wav", sr = 16000)     
inference(mixed_audio)   
print("Total_Time_Taken:",(time.perf_counter()-start1))
