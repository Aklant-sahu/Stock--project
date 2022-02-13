from distutils.log import debug
from logging import exception
from urllib import response

from flask import Flask, render_template,request

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
def cleansort(data):
       
    data['Date']=pd.to_datetime(data['Date'])
    data=data.sort_values(by='Date')
    data.set_index(pd.DatetimeIndex(data['Date']), inplace=True)
    try:
        data=data.drop(['Date'],axis=1)
    except:
        pass
    
    
    return data

def tech_indicators(data):
    data.ta.rsi(close='Close', length=5, append=True)
    data.ta.ema(close='Close', length=5, append=True)
    
    data=data[6:]
    #data.ta.cdl_pattern(name="all",append=True)
    #data.ta.strategy("momentum")
    return data

def datashift(data):
    
    col=data.columns
    
    ### ltp== last traded price
    ### vwap= volume weighted average price
    data=data.reset_index(drop=True)
    for j in col:
        x=[]
        for i in range(len(data)):
            if (i==0):
                x.append(0)
            else:
                x.append(data[j][i-1])
        nam='prev_'+j
        data[nam]=x
    data['openingprice_diff']=data['Open']-data['prev_Open']
    data['prev_h_l_diff']=data['prev_High']-data['prev_Low']
    data['open_close_diff']=data['prev_Open']-data['prev_Close']
    
    data['marketopeningprice_diff']=data['Market_Open']-data['prev_Market_Open']
    data['marketprev_h_l_diff']=data['prev_Market_High']-data['prev_Market_Low']
    data['marketopen_close_diff']=data['prev_Market_Open']-data['prev_Market_Close']
    col= [e for e in col if e not in ('Close')]
    data=data.drop(col,axis=1)
    data=data[1:]
    data=data.reset_index(drop=True)
    #data=data.drop(['PREV. CLOSE '],axis=1)
    return data


def xy(data):
    X=data.drop('Close',axis=1)
    y=data['Close']
    return X,y
def intratcs():
    df=pd.read_csv('static\dataset\TCS.NS.csv')
    df[df['Open'].isnull()==True]
    df2=pd.read_csv('static\dataset\^NSEI.csv')
    df2=df2[['Date','Close','Open','High','Low']]

    df2.columns=['Date', 'Market_Close','Market_Open','Market_High','Market_Low']
    df=pd.merge(df,df2,on=['Date'])


    df=df.dropna()
    a=cleansort(df.copy())
    b=tech_indicators(a)
    c=datashift(b)
    '''c=c.drop([ 'prev_Open', 'prev_High', 'prev_Low', 'prev_Close',
        'prev_Adj Close', 'prev_Market_Close',
        'prev_Market_Open', 'prev_Market_High', 'prev_Market_Low'
        ],axis=1)'''
    X,y=xy(c)

    #scale=StandardScaler()
    scale=MinMaxScaler(feature_range = (0, 1))

    X_train, X_test, y_train, y_test = X.iloc[:2700],X.iloc[2700:],y[:2700],y[2700:]

    X_train=scale.fit_transform(X_train)
    X_test=scale.transform(X_test)
    filename = 'tcs-intraday.pklsav'
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X_test)
    ans=pd.DataFrame(scale.inverse_transform(X_test),columns=X.columns)
    ans.reset_index(inplace=True)



    ans['actual_close']=y_test.values
    
    ans['pred_close']=y_pred
    ans=ans.drop('actual_close',axis=1)
    val=[]
    for i in range(len(ans)):
        if ans['pred_close'][i]>ans['prev_Close'][i]:
            val.append("BUY")
        else:
            val.append("SELL")
    ans['action']=val
        
    return ans.iloc[-1]
def intrareliance():
    df=pd.read_csv('static\dataset\RELIANCE.NS (1).csv')
    df[df['Open'].isnull()==True]
    df2=pd.read_csv('static\dataset\^NSEI.csv')
    df2=df2[['Date','Close','Open','High','Low']]

    df2.columns=['Date', 'Market_Close','Market_Open','Market_High','Market_Low']
    df=pd.merge(df,df2,on=['Date'])


    df=df.dropna()
    a=cleansort(df.copy())
    b=tech_indicators(a)
    c=datashift(b)
    '''c=c.drop([ 'prev_Open', 'prev_High', 'prev_Low', 'prev_Close',
        'prev_Adj Close', 'prev_Market_Close',
        'prev_Market_Open', 'prev_Market_High', 'prev_Market_Low'
        ],axis=1)'''
    X,y=xy(c)
    
    #scale=StandardScaler()
    scale=MinMaxScaler(feature_range = (0, 1))

    X_train, X_test, y_train, y_test = X.iloc[:2700],X.iloc[2700:],y[:2700],y[2700:]

    X_train=scale.fit_transform(X_train)
    X_test=scale.transform(X_test)
    filename = 'reliance-intraday.pklsav'
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X_test)
    ans=pd.DataFrame(scale.inverse_transform(X_test),columns=X.columns)
    ans.reset_index(inplace=True)



    ans['actual_close']=y_test.values

    ans['pred_close']=y_pred
    ans=ans.drop('actual_close',axis=1)

    val=[]
    for i in range(len(ans)):
        if ans['pred_close'][i]>ans['prev_Close'][i]:
            val.append("BUY")
        else:
            val.append("SELL")
    ans['action']=val
        
    return ans.iloc[-1]
def longtcs():

    df=pd.read_csv('static\dataset\TCS.NS.csv')
    df[df['Open'].isnull()==True]
    df2=pd.read_csv('static\dataset\^NSEI.csv')
    df2=df2[['Date','Close','Open','High','Low']]

    df2.columns=['Date', 'Market_Close','Market_Open','Market_High','Market_Low']
    df=pd.merge(df,df2,on=['Date'])


    df=df.dropna()
    a=cleansort(df.copy())
    b=tech_indicators(a)
    c=datashift(b)
    '''c=c.drop([ 'prev_Open', 'prev_High', 'prev_Low', 'prev_Close',
        'prev_Adj Close', 'prev_Market_Close',
        'prev_Market_Open', 'prev_Market_High', 'prev_Market_Low'
        ],axis=1)'''
    X,y=xy(c)
    ltsm_data=y.copy()
    X_train, X_test, y_train, y_test = X.iloc[:2700],X.iloc[2700:],y[:2700],y[2700:]
    g=y_test



    day=30
    sc = MinMaxScaler(feature_range = (0, 1))
    train=ltsm_data.values[:2700]
    test=ltsm_data.values[2700-day:]
    #sc=StandardScaler()
    dataset=ltsm_data.values
    #scaled_data= sc.fit_transform(np.reshape(dataset,(len(dataset),1)))
    scaled_train= sc.fit_transform(np.reshape(train,(len(train),1)))
    scaled_test= sc.transform(np.reshape(test,(len(test),1)))
    
    test_data=scaled_test
    fut_data=test_data[-30:,:]
    prediction_days=20
    
    X_test=[]
    y_test=dataset[2700:]
    for i in range(day,len(test_data)):
        X_test.append(test_data[i-day:i, 0])



    X_test = np.array(X_test)
    X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    file=open('./jsonmodel.json','r')
    json_model=file.read()
    file.close()
    model=tf.keras.models.model_from_json(json_model)
    model.load_weights('./json_weigts.h5')
    
#y_test=dataset[2500:]
    future=[]
    new_data=[]
    data=np.array(fut_data)
    print(data.shape)
    pred=model.predict(np.reshape(data,(1,day, 1)))

    future.append(pred)
    for i in range(1,prediction_days):
        try:
            data=fut_data[0+i:, 0]
            data=np.append(data,future[:])
            #print(data.shape)
            pred=model.predict(np.reshape(data,(1,X_test.shape[1], 1)))
        except:
            #print(len(future))
            ex=future[i-day:]
            pred=model.predict(np.reshape(ex,(1,X_test.shape[1], 1)))
        
        future.append(pred)
    fut=sc.inverse_transform(np.array(future).reshape(-1,1))
    
    plt.plot(range(0,233),g,'y',label='Actual')
    
    plt.plot(range(233,233+len(fut)),fut,'r',label=f'Future-{fut.shape[0]} days')
    plt.legend()
    plt.savefig(r'static/images/tcs.png')
    plt.close()


def longreliance():
    df=pd.read_csv('static\dataset\RELIANCE.NS (1).csv')
    df[df['Open'].isnull()==True]
    df2=pd.read_csv('static\dataset\^NSEI.csv')
    df2=df2[['Date','Close','Open','High','Low']]

    df2.columns=['Date', 'Market_Close','Market_Open','Market_High','Market_Low']
    df=pd.merge(df,df2,on=['Date'])


    df=df.dropna()
    a=cleansort(df.copy())
    b=tech_indicators(a)
    c=datashift(b)
    '''c=c.drop([ 'prev_Open', 'prev_High', 'prev_Low', 'prev_Close',
        'prev_Adj Close', 'prev_Market_Close',
        'prev_Market_Open', 'prev_Market_High', 'prev_Market_Low'
        ],axis=1)'''
    X,y=xy(c)
    ltsm_data=y.copy()
    X_train, X_test, y_train, y_test = X.iloc[:2700],X.iloc[2700:],y[:2700],y[2700:]
    g=y_test



    day=30
    sc = MinMaxScaler(feature_range = (0, 1))
    train=ltsm_data.values[:2700]
    test=ltsm_data.values[2700-day:]
    #sc=StandardScaler()
    dataset=ltsm_data.values
    #scaled_data= sc.fit_transform(np.reshape(dataset,(len(dataset),1)))
    scaled_train= sc.fit_transform(np.reshape(train,(len(train),1)))
    scaled_test= sc.transform(np.reshape(test,(len(test),1)))
    
    test_data=scaled_test
    fut_data=test_data[-30:,:]
    prediction_days=20
    
    X_test=[]
    y_test=dataset[2700:]
    for i in range(day,len(test_data)):
        X_test.append(test_data[i-day:i, 0])



    X_test = np.array(X_test)
    X_test= np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    file=open('jsonmodel-reliance.json','r')
    json_model=file.read()
    file.close()
    model=tf.keras.models.model_from_json(json_model)
    model.load_weights('./json-weights-reliance.h5')
    
#y_test=dataset[2500:]
    future=[]
    new_data=[]
    data=np.array(fut_data)
    print(data.shape)
    pred=model.predict(np.reshape(data,(1,day, 1)))

    future.append(pred)
    for i in range(1,prediction_days):
        try:
            data=fut_data[0+i:, 0]
            data=np.append(data,future[:])
            #print(data.shape)
            pred=model.predict(np.reshape(data,(1,X_test.shape[1], 1)))
        except:
            #print(len(future))
            ex=future[i-day:]
            pred=model.predict(np.reshape(ex,(1,X_test.shape[1], 1)))
        
        future.append(pred)
    fut=sc.inverse_transform(np.array(future).reshape(-1,1))
    
    plt.plot(range(0,233),g,'y',label='Actual')
    
    plt.plot(range(233,233+len(fut)),fut,'r',label=f'Future-{fut.shape[0]} days')
    plt.legend()
    
    plt.savefig('static/images/reliance.png')
    plt.close()








app = Flask(__name__)
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template("index.html")

@app.route('/intra_day', methods=["POST","GET"])
# ‘/’ URL is bound with hello_world() function.
def intra_day():
    
    if request.method=="GET":
        return render_template("intra_day.html")
    
         
@app.route('/view_stock', methods=["POST","GET"])
# ‘/’ URL is bound with hello_world() function.
def view_stock():
    
    
    if request.method=="POST":
        sname=request.form['stock']
        if sname=="tcs":
            
            stock=intratcs()
            print(stock)
            return render_template("view_stock.html",stock=stock)
        elif sname=="reliance":
             stock=intrareliance()
             print(stock)
             return render_template("view_stock.html",stock=stock)
         


            

            
        
        
@app.route('/long_term', methods=["POST","GET"])
# ‘/’ URL is bound with hello_world() function.
def long_term():
    if request.method=="POST":
        sname=request.form['stock']
        if sname=="tcs":
            longtcs()
            return render_template("long_term.html",image="static/images/tcs.png")
        else:
            longreliance()
            return render_template("long_term.html",image="static/images/reliance.png")
    elif request.method=="GET": 
        return render_template("long_term.html")
        
        
    
    
        


if __name__ == '__main__':
      
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)