#############################################################
#                                                           #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################

from tensorflow import spectral
from keras.losses import mean_squared_error as mse
from keras.losses import mean_absolute_error as mae
import window as w
import  tensorflow as tf

def cep_loss(y_true,y_pred):
    y_pred=tf.cast(y_pred, tf.complex64)
    y_true=tf.cast(y_true, tf.complex64)
    y_pred_cep=spectral.irfft(y_pred)
    y_true_cep=spectral.irfft(y_true)  
    y_pred=tf.cast(y_pred, tf.float32)
    y_true=tf.cast(y_true, tf.float32)
    y_true_cep_win=w.win(y_true_cep[:,:257])
    y_pred_cep_win=w.win(y_pred_cep[:,:257])   
    return 0.9967*mae(y_true_cep_win,y_pred_cep_win)+0.0033*mse(y_true,y_pred)
