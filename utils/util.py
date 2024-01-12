import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import os
import math
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
import seaborn as sns
from sklearn import metrics
import time
import pickle
import metric_learn
from sklearn.decomposition import PCA  
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import LocalOutlierFactor
import functools
import tempfile
import datetime
import tensorflow as tf
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())
import tensorflow as tf
tf.config.list_physical_devices('GPU')

plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"
def make_data(x, str,w):
    x=pd.DataFrame(x)
    window=w
    for i in range(1, window):
        column=f'{str}_lag{i}'
        x[column]=x[str].shift(i)
    return x

def make_data_rev(x, str,w):
    x=pd.DataFrame(x)
    window=w
    for i in range(1, window):
        column=f'{str}_next{i}'
        x[column]=x[str].shift(-i)
    return x

def make_label(x, w):
    window=w
    x['label_all']=x['label']
    for i in range(1, window):
        x.loc[(x['label_all']==1)|(x['label'].shift(i)==1), 'label_all']=1
        x.loc[(x['label_all']==0)&(x['label'].shift(i)==0), 'label_all']=0
    
    x['label']=x['label_all']
    return x[['label']]

def calc_leq(df, unit):
    df.reset_index(inplace=True, drop=True)
    for i in range(int(len(df)/unit)+1):
        hour_df=df.loc[i*unit:(i+1)*unit, 'original'].copy()
        N=len(hour_df)
        Leq=10*np.log10(np.sum(np.power(10, hour_df/10)))-10*np.log10(N)
        df.loc[i*unit:(i+1)*unit, 'leq']=Leq
    return df

def leq_filter(df):
    df=calc_leq(df, 18000)
    df.loc[(df['original']<df['leq']), 'd']=0
    return df

def validate(test_v, anorm, thr=0.2):
    test_v['z']=np.where(anorm>=thr, 1, 0)
    test_v.reset_index(inplace=True, drop=True)

    #     適合率
    tp=test_v[(test_v['label']==1)&(test_v['z']==1)]
    z_p=test_v[test_v['z']==1]
    pre_score=len(tp)/len(z_p)

    #     再現率
    df_anorm=[]
    search= 1 if test_v.loc[0, 'label']==0 else 0
    for num in range(len(test_v)):
        if search==1 and test_v.loc[num, 'label']==search:
            start=num
            search=0
        elif search==0 and test_v.loc[num, 'label']==search:
            stop=num-1
            anorm_range=test_v.loc[start:stop].copy()
            df_anorm.append(anorm_range)
            search=1
            
    count=[]
    for i in range(len(df_anorm)):
        if len(df_anorm[i].loc[df_anorm[i]['z']==1])>=1:
               count.append(i)    

    re_score=len(count)/len(df_anorm)

    return pre_score, re_score

def figure(df_test, d, thr):
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    df_test=df_test.copy()
    df_test['z']=np.where(d>=thr, 1, 0)
    
    z=df_test['z'].values*100
    test_plot=df_test['original'].values
    num_ax=math.ceil(len(test_plot)/17999)
    label=df_test['label'].values*100
    label_index=range(len(label))
    time_unit=60
    time=[t*0.2/60 for t in range(len(d))]
    time_unit_data=18000
    fig, ax=plt.subplots(num_ax, 1, figsize=(35, 16*num_ax))
    plt.subplots_adjust(hspace=0.35)
    for i in range(num_ax):
        ax[i].plot(time[i*time_unit_data:time_unit_data*(i+1)], d[i*time_unit_data:time_unit_data*(i+1)], '-r',linewidth = 1, label='異常度')
        ax[i].fill_between(time[i*time_unit_data:time_unit_data*(i+1)], label[i*time_unit_data:time_unit_data*(i+1)], facecolor='lime', label='異常ラベル' )

#         #異常と判別したところを強調したい場合
#         ax[i].fill_between(time[i*time_unit_data:time_unit_data*(i+1)], z[i*time_unit_data:time_unit_data*(i+1)], facecolor='steelblue' )
        
        ax2=ax[i].twinx()
        ax2.plot(time[i*time_unit_data:time_unit_data*(i+1)], test_plot[i*time_unit_data:time_unit_data*(i+1)], '-k',linewidth = 2, label='騒音レベル')

        ax[i].set_xticks(np.arange(60*i, 60*(i+1),3))
        ax[i].set_xticklabels(np.arange(60*i, 60*(i+1),3), fontsize=40)
        ax[i].xaxis.set_tick_params(rotation=30)
        ax[i].set_xlim(i*time_unit, time_unit*(i+1))
        
        ax[i].set_yticks(np.arange(0, 12, 2)/10)
        ax[i].set_yticklabels(np.arange(0, 12, 2)/10, fontsize=40)
        
        ax2.set_yticks(np.arange(40, 90, 10))
        ax2.set_yticklabels(np.arange(40, 90, 10), fontsize=40)
        
        ax[i].set_ylim(0, 2)
        ax2.set_ylim(10, 80)
        h1, l1 = ax[i].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[i].legend(h2+h1, l2+l1, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, prop={'family':"MS Mincho", 'size':40})
        ax[i].set_xlabel('時点(分)', fontfamily="MS Mincho",fontsize=40)
        ax[i].set_ylabel('　　　異常度', fontfamily="MS Mincho",fontsize=40, loc='bottom')
        ax2.set_ylabel('　騒音レベル値(dB)', fontfamily="MS Mincho",fontsize=40, loc='top')
    plt.show()
    return df_test

def figure_detail(df_test, d_test, start=0, stop=60):
    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Times New Roman"
    
    start_data=int(math.floor(start)*60/0.2)
    stop_data=int(math.ceil(stop)*60/0.2)

    test_plot=df_test['original'].values[start_data:stop_data]
    d=d_test[start_data:stop_data]
    label=df_test['label'].values[start_data:stop_data]*100
    label_index=range(len(label))
    time=[t*0.2/60 for t in range(start_data, stop_data)]
    fig, ax=plt.subplots(1, 1, figsize=(35, 6))
    ax.plot(time, test_plot, color='#6687AF' ,linewidth = 3, label='Noise Level')
    ax2=ax.twinx()
    ax2.plot(time, d, color='#F17B51',linewidth = 3, label='Anomaly Score')
    ax.fill_between(time, label,  facecolor='#D9D9D9', label='Abnormal period' )
    ax.set_xticks(np.arange(math.floor(start), math.ceil(stop), 5))
    ax.set_xticklabels(np.arange(math.floor(start), math.ceil(stop), 5), fontsize=50)
#     ax.xaxis.set_tick_params(rotation=30)
    ax.set_xlim(start, stop)
    ax.set_yticks(np.arange(40, 90, 10))
    ax.set_yticklabels(np.arange(40, 90, 10), fontsize=60)        
    ax.set_ylim(40, 80)
    ax2.set_yticks(np.arange(0, 12, 2)/10)
    ax2.set_yticklabels(np.arange(0, 12, 2)/10, fontsize=60)
    ax2.set_ylim(0, 1)
    ax.set_ylim(30, 80)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h2+h1, l2+l1, loc='upper center', bbox_to_anchor=(.5, 1.4), ncol=3, prop={'family':"MS Mincho", 'size':60})
    ax.set_xlabel('Time(min)', fontfamily="MS Mincho",fontsize=60)
    ax.set_ylabel('Noise Level', fontfamily="MS Mincho",fontsize=60, loc='bottom')
    ax2.set_ylabel('Anomaly Score', fontfamily="MS Mincho",fontsize=60, loc='bottom')
#     plt.grid(True)
    plt.show()    

# Xは１つor2つの時系列データを持つ配列
def plot_timeseries(X, label, d=None, thr=None, mlflow=None):
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    test_plot=X[0]
    if len(X)==2:
        test_plot_1=X[1]
    
    num_ax=math.ceil(len(test_plot)/17999)
    time_unit=60
    time=[t*0.2/60 for t in range(len(test_plot))]
    time_unit_data=18000
    fig, ax=plt.subplots(num_ax, 1, figsize=(35, 13*num_ax))
    plt.subplots_adjust(hspace=0.35)
    for i in range(num_ax):
        ax[i].plot(time[i*time_unit_data:time_unit_data*(i+1)], test_plot[i*time_unit_data:time_unit_data*(i+1)],color='#6687AF',linewidth = 2, label='騒音レベル')
        if len(X)==2:
            ax[i].plot(time[i*time_unit_data:time_unit_data*(i+1)], test_plot_1[i*time_unit_data:time_unit_data*(i+1)], '-y',linewidth = 2, label='再構成')
                   
        ax[i].fill_between(time[i*time_unit_data:time_unit_data*(i+1)], label[i*time_unit_data:time_unit_data*(i+1)]*100, facecolor='#D9D9D9', label='異常ラベル' )
        ax[i].fill_between(time[i*time_unit_data:time_unit_data*(i+1)], label[i*time_unit_data:time_unit_data*(i+1)]*-100, facecolor='#D9D9D9' )
        ax[i].set_ylim(test_plot.min()-(test_plot.max()-test_plot.min()), test_plot.max())
        h1, l1 = ax[i].get_legend_handles_labels()
        ax[i].set_ylabel('　騒音レベル値(dB)', fontfamily="MS Mincho",fontsize=40, loc='top')
        ax[i].set_xlim(i*time_unit, time_unit*(i+1))
        ax[i].set_xticks(np.arange(60*i, 60*(i+1),3))
        ax[i].set_xticklabels(np.arange(60*i, 60*(i+1),3), fontsize=40)
        ax[i].xaxis.set_tick_params(rotation=30)
        ax[i].set_xlabel('時点(分)', fontfamily="MS Mincho",fontsize=40)
        
        if not isinstance(d, type(None)):
            ax2=ax[i].twinx()
            ax2.plot(time[i*time_unit_data:time_unit_data*(i+1)], d[i*time_unit_data:time_unit_data*(i+1)], color='#F17B51',linewidth = 1, label='異常度')
            # 異常と判別したところを強調したい場合
            if not isinstance(thr, type(None)):
                z=np.where(d>=thr, 1, 0)
                ax2.fill_between(time[i*time_unit_data:time_unit_data*(i+1)], z[i*time_unit_data:time_unit_data*(i+1)]*1.5, facecolor='black')
                ax2.fill_between(time[i*time_unit_data:time_unit_data*(i+1)], z[i*time_unit_data:time_unit_data*(i+1)]*-1.5, facecolor='black')
            ax2.set_yticks(np.arange(0, 12, 2)/10)
            ax2.set_yticklabels(np.arange(0, 12, 2)/10, fontsize=40)
            h2, l2 = ax2.get_legend_handles_labels()
            if i==0:
                ax2.legend(h2+h1, l2+l1, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, prop={'family':"MS Mincho", 'size':40})
            ax2.set_ylabel('　　　異常度', fontfamily="MS Mincho",fontsize=40, loc='bottom')
            ax2.set_ylim(0, 2)
    if not isinstance(mlflow, type(None)):
        print(mlflow)
        plt.savefig(mlflow)
    plt.show()

def fig_pr(test_v, d, bins):
    plt.rcParams["font.size"] = 15
    plt.rcParams["font.family"] = "Times New Roman"
    bins_1=int(bins*0.8)
    thr_1=np.linspace(d.min(), 0.6, bins_1)
    thr_2=np.linspace(0.6, d.max(), bins-bins_1)
    thresholds=np.concatenate([thr_1, thr_2])
    precision=np.array([])
    recall=np.array([])
    for i in thresholds:
        p, r=validate(test_v, d, i)
        precision=np.append(precision, p)
        recall=np.append(recall, r)

    auc = metrics.auc(recall, precision)
    f_score=(2*precision*recall)/(precision+recall)
    thr=thresholds[np.argmax(f_score)]

#     #F値が最大になる点を明示したい場合
#     plt.plot(recall, precision, marker='o', markevery=[np.argmax(f_score)], label='PR curve (AUC = %.2f)'%auc)
    #明示しなくていい場合
    plt.plot(recall, precision, '-k',linewidth = 2, label='PR曲線')
    
#     plt.legend(prop={'family':"MS Mincho"}, loc="lower left")
    plt.xlabel('再現率', fontfamily="MS Mincho")
    plt.ylabel('適合率', fontfamily="MS Mincho")
    plt.grid(True)
    plt.show()

    return precision, recall, f_score, thresholds, thr, auc

def auc_gs(test_v, d, bins):
    bins_1=int(bins*0.8)
    thr_1=np.linspace(d.min(), 0.6, bins_1)
    thr_2=np.linspace(0.6, d.max(), bins-bins_1)
    thresholds=np.concatenate([thr_1, thr_2])
    precision=np.array([])
    recall=np.array([])
    for i in thresholds:
        p, r=validate(test_v, d, i)
        precision=np.append(precision, p)
        recall=np.append(recall, r)

    auc = metrics.auc(recall, precision)

    return auc

def fig_th_f(thresholds, f_score):
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(thresholds, f_score, marker="o", markevery=[np.argmax(f_score)])
    plt.xlabel('thresholds')
    plt.ylabel('f_score')
    plt.grid(True)
    plt.show()
    
    return thresholds[np.argmax(f_score)], f_score.max()