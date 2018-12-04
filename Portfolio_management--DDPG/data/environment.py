# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 05:41:28 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
from pykalman import KalmanFilter

eps=10e-8

def fill_zeros(x):
    return '0'*(6-len(x))+x

class Environment:
    def rolling_window(self,a, step):
        shape   = a.shape[:-1] + (a.shape[-1] - step + 1, step)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def get_kf_value(self,y_values):
        kf = KalmanFilter()
        Kc, Ke = kf.em(y_values, n_iter=1).smooth(0)
        return Kc
    def __init__(self,start_date,end_date,codes,features,window_length,market,mode):
        #,test_data,assets_list,M,L,N,start_date,end_date
        self.features = features
        #preprocess parameters
        self.cost=0.0025

        #read all data
        data=pd.read_csv(r'./data/'+market+'.csv',index_col=0,parse_dates=True,dtype=object)
        data["code"]=data["code"].astype(str)
        if market=='China':
            data["code"]=data["code"].apply(fill_zeros)

        data=data.loc[data["code"].isin(codes)]
        data[features]=data[features].astype(float)

        # 生成有效时间
        print(pd.to_datetime(start_date))
        print(data.index[1])
        start_date = [date for date in data.index if date > pd.to_datetime(start_date)][0]
        end_date = [date for date in data.index if date < pd.to_datetime(end_date)][-1]
        #data=data[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
        #TO DO:REFINE YOUR DATA
        
        if mode == "train":
            print("applying kalman filter")
            print(data['close'])
            
            wsize = 3
            arr = self.rolling_window(data['close'].values, wsize)
            zero_padding = np.zeros(shape=(wsize-1,wsize))
            arrst = np.concatenate((zero_padding, arr))
            arrkalman = np.zeros(shape=(len(arrst),1))

            for i in range(len(arrst)):
                arrkalman[i] = self.get_kf_value(arrst[i])

            print(type(arrkalman))
            #data['close'] = pd.DataFrame({"close":arrkalman})
            
            data['close'] = arrkalman
            
            print(data['close'])
            #kalmandf = pd.DataFrame(arrkalman, columns=['close_kalman'])
            #data = pd.concat([data,kalmandf], axis=1)
            print(("applied kalman filter on close"))
            
        #Initialize parameters
        self.M=len(codes)+1
        self.N=len(features)
        self.L=window_length

        #为每一个资产生成数据
        asset_dict=dict()#每一个资产的数据
        datee=data.index.unique()
        self.date_len=len(datee)
        for asset in codes:
            asset_data=data[data["code"]==asset].reindex(datee).sort_index()#加入时间的并集，会产生缺失值pd.to_datetime(self.date_list)
            asset_data['close']=asset_data['close'].fillna(method='pad')
            base_price = asset_data.ix[end_date, 'close']
            asset_dict[str(asset)]= asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low']=asset_dict[str(asset)]['low']/base_price

            if 'open' in features:
                asset_dict[str(asset)]['open']=asset_dict[str(asset)]['open']/base_price
            
            if 'sentimentRange' in features:
                asset_dict[str(asset)]['sentimentRange']=asset_dict[str(asset)]['sentimentRange']


            asset_data=asset_data.fillna(method='bfill',axis=1)
            asset_data=asset_data.fillna(method='ffill',axis=1)#根据收盘价填充其他值
            #***********************open as preclose*******************#
            #asset_data=asset_data.dropna(axis=0,how='any')
            asset_data=asset_data.drop(columns=['code'])
            asset_dict[str(asset)]=asset_data


        #开始生成tensor
        self.states=[]
        self.price_history=[]
        print("*-------------Now Begin To Generate Tensor---------------*")
        t =self.L+1
        while t<self.date_len:
            V_close = np.ones(self.L)
            if 'high' in features:
                V_high=np.ones(self.L)
            if 'open' in features:
                V_open=np.ones(self.L)
            if 'low' in features:
                V_low=np.ones(self.L)
            if 'sentimentRange' in features:
                V_sentiment = np.ones(self.L)

            y=np.ones(1)
            for asset in codes:
                asset_data=asset_dict[str(asset)]
                V_close = np.vstack((V_close, asset_data.ix[t - self.L - 1:t - 1, 'close']))
                if 'high' in features:
                    V_high=np.vstack((V_high,asset_data.ix[t-self.L-1:t-1,'high']))
                if 'low' in features:
                    V_low=np.vstack((V_low,asset_data.ix[t-self.L-1:t-1,'low']))
                if 'open' in features:
                    V_open=np.vstack((V_open,asset_data.ix[t-self.L-1:t-1,'open']))
                if 'sentimentRange' in features:
                    V_sentiment=np.vstack((V_sentiment,asset_data.ix[t-self.L-1:t-1,'sentimentRange']))

                y=np.vstack((y,asset_data.ix[t,'close']/asset_data.ix[t-1,'close']))
            state = V_close

            if 'high' in features:
                state = np.stack((state,V_high), axis=2)
            if 'low' in features:
                state = np.stack((state,V_low), axis=2)
            if 'open' in features:
                state = np.stack((state,V_open), axis=2)
            if 'sentimentRange' in features:
                state = np.stack((state,V_sentiment), axis=2)


            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)
            
            self.price_history.append(y)
            t=t+1
        self.reset()


    def first_ob(self):
        return self.states[self.t]

    def step(self,w1,w2):
        if self.FLAG:
            not_terminal = 1
            price = self.price_history[self.t]
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            std = self.states[self.t - 1][0].std(axis=0, ddof=0)
            w2[np.isnan(w2)] = 1.0
            price[np.isnan(price)] = 1.0
            w2_std = np.zeros(len(self.features))
            gamma = [0.99,0.9]
            for i in range(len(self.features)):
                a = w2[0].reshape(1,w2[0].shape[0])
                b = std[:,i].reshape(std[:,i].shape[0],1)
                w2_std[i] = sum(sum(a*b)) *gamma[i]

            risk = np.sum(w2_std)/len(self.features)

            r = (np.dot(w2, price)[0] - mu)[0]


            reward = np.log(r + eps)

            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == len(self.states) - 1:
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward, 'continue': not_terminal, 'next state': self.states[self.t],
                    'weight vector': w2, 'price': price,'risk':risk}
            return info
        else:
            info = {'reward': 0, 'continue': 1, 'next state': self.states[self.L + 1],
                    'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),
                    'price': self.price_history[self.L + 1],'risk':0}

            self.FLAG=True
            return info

    def reset(self):
        self.t=self.L+1
        self.FLAG = False




        