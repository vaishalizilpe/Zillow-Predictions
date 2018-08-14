
# coding: utf-8

# In[73]:


import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

air=pd.read_csv("C:\Users\vaishali\Desktop\AirBnb-CapOne\listings.csv",low_memory=False)
zil=pd.read_csv("C:\Users\vaishali\Desktop\AirBnb-CapOne\Zip_Zhvi_2bedroom.csv")

zil_subset=zil[zil.City=='New York']
zil_subset=zil_subset.drop(['RegionID','City','State','Metro','CountyName','SizeRank'],axis=1)
zil_subset=zil_subset.drop(zil_subset.columns[1:142], axis=1)
zil_subset_trans=pd.melt(zil_subset, id_vars='RegionName', value_vars=zil_subset.columns[1:115])
zil_subset_trans=zil_subset_trans.rename(columns={'Variable':'PropertyDate','Value':'Price'})
zil_subset_trans.columns=['RegionName','PropertyDate','Price']

series = zil_subset_trans
series.Price=series.Price.astype(float)

d = dict(tuple(series.groupby('RegionName')))
final_df=[]
for key,value in d.iteritems():
    sample_data=d[key]
    sample_data=sample_data.set_index('PropertyDate')
    sample_data.plot()
    pyplot.show()
    sample_data.drop('RegionName',axis=1,inplace=True)
    X = sample_data.values
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    
    history = [x for x in X]
    predictions = list()
    for t in range(60):
        model = ARIMA(history, order=(5,1,0))
        try:
            model_fit = model.fit(disp=0)
        except:
            print('did not converge')
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        #obs = test[t]
        history.append(yhat)
        
    print('Completed modelling for zipcode {}\n'.format(key))
    df=pd.DataFrame()
    df['date_time']=pd.date_range('2008-01-01', periods=174, freq='M')
    df=df.date_time.map(lambda x: x.strftime('%Y-%m'))
    df1=pd.DataFrame(history,columns=['Price'])
    df1['RegionName']=key
    df2=pd.concat([df,df1],axis=1)
    final_df.append(df2)
    print df2.head()
df3=pd.concat(final_df)
df3.to_csv("C:\Users\RAHUL\Desktop\AirBnb-CapOne\zillow_predictions.csv",index=False)

