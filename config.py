#############################################################
#                                                           #
#   Summary:  Config file.                                  #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################

sr = int(16000)
win_length = int(512)            # frame size 
hop_length = int(win_length/2)   # frame Shift
n_fft=int(512)                   # FFT Length 
n_freq = int(n_fft/2 + 1)

n_context = int(21)
n_pad= int(n_context/2 )
n_hop = int(n_context/4+1)
