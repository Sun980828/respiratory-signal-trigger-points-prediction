import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import TP_dect, real_time_sig_process, cut_sig

pred_path = './Dataset/prediction/'
rawdata_path = './Dataset/test_set.csv'
pred_list  = os.listdir(pred_path)
input_win = 50
output_win = 15
filter_win = 9
half_filter_win = int((filter_win - 1) / 2)
latency,num_slope,num_label = 6,5,output_win
num_input = input_win

df = pd.read_csv(rawdata_path, header=None).to_numpy()
processed_origin = [real_time_sig_process(line)[half_filter_win:-half_filter_win] for line in df]
origin = processed_origin[0]

Y_arr = [origin[i_input + num_input : i_input  + num_input + num_label] for i_input in range(len(origin)-num_input-num_label+1)]
TP_det_list,TP_det_index_list = [],[]
InEx_label,last_i_arr = -1, -2
iterate_Pred = 1
# detect all TP of preprocessed signal
for i_arr in range(len(Y_arr)):
    if i_arr % iterate_Pred != 0:
        continue
    else:
        #start_time = time.time()
        
        ref_TP,label_TP = False,False
        ref_TP,label_TP = TP_dect(Y_arr[i_arr],latency,num_slope,num_label)
        if ref_TP and label_TP * InEx_label == -1:
            TP_pot_index = ref_TP + i_arr + num_input
            TP_pot       = Y_arr[i_arr][ref_TP]
            # save the TP potential
            last_pot_index,last_pot,last_pot_label,last_i_arr = TP_pot_index, TP_pot, label_TP, i_arr
            
        elif (ref_TP and i_arr - last_i_arr == iterate_Pred and label_TP * InEx_label == 1):
            TP_det_index, TP_det, InEx_label = last_pot_index, last_pot, -InEx_label
            TP_det_index_list.append(TP_det_index)
            TP_det_list.append(TP_det)
        
        # if last time has TP, this time no TP, the TP is the 'TP'
        elif (i_arr - last_i_arr == iterate_Pred and InEx_label * last_pot_label == -1):
            TP_det_index, TP_det, InEx_label = last_pot_index, last_pot, -InEx_label
            TP_det_index_list.append(TP_det_index)
            TP_det_list.append(TP_det)
# plot the all detected TP in processed signal
#'''            
# plt.rcParams["figure.figsize"] = (20,5)
# plt.rcParams['figure.dpi'] = 100
# plt.plot(TP_det_index_list,TP_det_list,'x')
# plt.title('Detected TP in the signal',fontsize = 20)
# plt.plot(origin)
# plt.show()
#'''

origin_tp, origin_tp_idx = TP_det_list, TP_det_index_list

trash_points = np.array(TP_det_index_list[1:])[np.diff(TP_det_index_list)>100]
cut_points = []
for i in trash_points:
    trash_point_idx = TP_det_index_list.index(i)
    cut_points.append((trash_point_idx-1, trash_point_idx))

best_sig = cut_sig(cut_points, origin)

plt.plot(best_sig[4])
plt.show()



