from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import os
import logging

def TP_detect(norm_sensor):
    CC = 0
    FC_peak = 0
    FC_tough = 0

    CC_index,FC_peak_index,FC_tough_index =[],[],[]
    for i_point in range(len(norm_sensor)-6):
        slope_1 = norm_sensor[i_point+1] - norm_sensor[i_point]
        slope_2 = norm_sensor[i_point+2] - norm_sensor[i_point+1]
    
    # change point detection
        if slope_1*slope_2 <= 0:
            CC = CC + 1
            CC_index.append(i_point+1)
        
            slope_3 = norm_sensor[i_point+3] - norm_sensor[i_point+2]
            slope_4 = norm_sensor[i_point+4] - norm_sensor[i_point+3]
            slope_5 = norm_sensor[i_point+5] - norm_sensor[i_point+4]
            slope_6 = norm_sensor[i_point+6] - norm_sensor[i_point+5]
        
        # bulge
            if slope_1 >= 0 and slope_3 <0 and slope_4 <0 and slope_5 <0  and slope_6 <0 and norm_sensor[i_point+1]> 30:
                FC_peak = FC_peak+1
                FC_peak_index.append(i_point+1)
            
            # incase two peaks are too close
#            if len(FC_peak_index)>1 and FC_peak_index[-1] - FC_peak_index[-2] < 15:
#                FC_peak = FC_peak - 1
#                FC_peak_index.remove(i_point+1)

        # sunk
            if slope_1 <= 0 and slope_3 >0 and slope_4 >0 and slope_5 >0 and slope_6 >0 and norm_sensor[i_point+1]< -30:
                FC_tough = FC_tough + 1
                FC_tough_index.append(i_point+1)

    return FC_peak_index,FC_tough_index,FC_peak,FC_tough,CC


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def listcsv(rootpath):
    dir_list = os.listdir(rootpath)
    csv_list = []
    for cur_file in dir_list:    
        path = os.path.join(rootpath,cur_file)
        
        if os.path.splitext(path)[1] == '.csv':
            csv_list.append(path)
    return csv_list


def TrainTestSplit(num_In,num_Label,num_data,normSensor):
    numTrain = int((num_data-num_In-num_Label+1)*0.7)
    numTest =  int((num_data-num_In-num_Label+1)*0.3)
    
    train_X,train_Y = [],[]
    for i_train in range(numTrain):
        train_X.append(normSensor[i_train       :i_train+num_In])
        train_Y.append(normSensor[i_train+num_In:i_train+num_In+num_Label])
    
    test_X,test_Y = [],[]
    for i_test in range(numTest):
        test_X.append(normSensor[numTrain+i_test       :numTrain+i_test+num_In])
        test_Y.append(normSensor[numTrain+i_test+num_In:numTrain+i_test+num_In+num_Label])
        
    return train_X,train_Y,test_X,test_Y

# trigger point detection
# tuning: how many continuous slope in the prediction (lengtn of out put label), will no TP missed
def TP_dect(label,latency,num_slope,num_label):
    
    # 1. change point find
    CC_slope = np.diff(label[latency-1:num_label-num_slope+1])
    
    # 2. if change point, continuous change
    for i_ccslope in range(len(CC_slope)-1):
        if CC_slope[-i_ccslope-1] >= 0 and CC_slope[-i_ccslope-2] <=0:    # vally
            TP_slope = np.diff(label[num_label - num_slope - i_ccslope : num_label - i_ccslope])
            if all(TP_slope > 0) and label[num_label - num_slope - i_ccslope -1] <0.4:
                return (num_label - num_slope - i_ccslope -1),-1
                
        elif CC_slope[-i_ccslope-1] <= 0 and CC_slope[-i_ccslope-2] >=0:   # peak
            TP_slope = np.diff(label[num_label - num_slope - i_ccslope : num_label - i_ccslope])
            if all(TP_slope < 0) and label[num_label - num_slope - i_ccslope -1] >0.6:
                return (num_label - num_slope - i_ccslope -1),1
                
        else:
            continue
    return False, False

def err_analysis(Det_index,Pre_index,Pre_miss,Det_miss):  # error unit in index
    if (len(Det_index)-len(Pre_miss)) == (len(Pre_index)-len(Det_miss)):
        print(len(Det_index),len(Pre_index))
        print(len(Det_index)-len(Pre_miss),len(Pre_index)-len(Det_miss))
        Det_index_ = [j for i,j in enumerate(Det_index) if i not in Pre_miss]
        Pre_index_ = [j for i,j in enumerate(Pre_index) if i not in Det_miss]
        Diff =[(i-j)/38 for i,j in zip(Pre_index_,Det_index_)]  # pre - det: if predict before, negative
        '''
        MAE = sum([abs(_) for _ in Diff])/len(Diff)
        RMSE = math.sqrt(sum([_*_ for _ in Diff])/len(Diff))
        '''
        return Diff
    else:
        print('Wrong with lenth:')
        print(len(Det_index),len(Pre_index))
        return False

def init_logger(log_path):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:'
                          '%(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

def extract_datasets(path, input_dim, output_dim):
    Train_x, Train_y, Test_x, Test_y = np.array([]), np.array([]), np.array([]), np.array([])
    df = pd.read_csv(path, header=None).to_numpy()
    # print(df.shape)
    for row in df:
        single_sensor_signal = row[~np.isnan(row)]
        signal_length = len(single_sensor_signal)
        train_x, train_y, test_x, test_y = TrainTestSplit(input_dim, output_dim, signal_length, single_sensor_signal)
        Train_x = np.append(Train_x, train_x)
        Train_y = np.append(Train_y, train_y)
        Test_x = np.append(Test_x, test_x)
        Test_y = np.append(Test_y, test_y)
    Train_x = Train_x.reshape(-1, input_dim, 1).astype('float32')
    Train_y = Train_y.reshape(-1, output_dim, 1).astype('float32')
    Test_x = Test_x.reshape(-1, input_dim, 1).astype('float32')
    Test_y = Test_y.reshape(-1, output_dim, 1).astype('float32')

    print(f'Datasets successfully extracted, with:\
            \nTrain_x: {Train_x.shape}\
            \nTrain_y: {Train_y.shape}\
            \nTest_x: {Test_x.shape}\
            \nTest_y: {Test_y.shape}')
    return Train_x, Train_y, Test_x, Test_y

def real_time_sig_process(sensor_value, win_len=9, sav_order=5, mode='nearest', lam=10e3, p=0.5):
    sav_filter_win = savgol_filter(sensor_value, win_len, sav_order, mode=mode)
    baseline_sensor = baseline_als(sav_filter_win, lam, p)
    norm_sensor = sav_filter_win - baseline_sensor
    low_samp = np.quantile(norm_sensor, 0.25)
    up_samp = np.quantile(norm_sensor, 0.75)
    IQR = up_samp - low_samp
    samp_norm = (norm_sensor + 2*IQR) / 4 / IQR
    samp_norm = np.clip(samp_norm, 0, 1)
    return samp_norm

def find_cutpoints(TP_det_index_list, TP_distance=50):
    trash_points = np.array(TP_det_index_list[1:])[np.diff(TP_det_index_list)>TP_distance]
    cut_points = []
    for i in trash_points:
        trash_point_idx = TP_det_index_list.index(i)
        cut_points.append((TP_det_index_list[trash_point_idx-1], TP_det_index_list[trash_point_idx]))
    return cut_points

def cut_sig(cut_points, res_sig, least_sig_len=73):

    best_sig = []
    last_end = 0
    for i, (start, end) in enumerate(cut_points):
        current_start = start - last_end
        current_end = end - last_end
        last_end = end
        cutted = res_sig[:current_start]
        res_sig = res_sig[current_end:]
        best_sig.append(cutted)
        if i == len(cut_points) - 1:
            best_sig.append(res_sig)
        
    best_sig = [sig for sig in best_sig if len(sig) >= least_sig_len]
    
    return best_sig

def trig_point_detect(Y_arr, latency=6, num_slope=5, num_label=15, num_input=50):
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

    return TP_det_index_list, TP_det_list

