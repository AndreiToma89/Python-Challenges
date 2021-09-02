#!/usr/bin/env python
# coding: utf-8




get_ipython().run_line_magic('cd', '/Users/andrei.toma/Downloads/DontGetKicked/')





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv('training.csv',)





plt.figure(figsize=(15,5))
sns.heatmap(df.isnull(),annot = False,yticklabels=False,cmap='viridis')





df['PurchDate'] = pd.to_datetime(df['PurchDate'])




df['Year'] = df['PurchDate'].apply(lambda date: date.year)
df['Month'] = df['PurchDate'].apply(lambda date: date.month)




df.head()




df = df.drop(['RefId','VNZIP1','BYRNO','Model','Trim',
             'SubModel','VNST','WheelTypeID','AUCGUART','PRIMEUNIT',
             'PurchDate'],axis = 1)




df.head()





sns.countplot(x = df['IsBadBuy'])





print(str(((df['IsBadBuy'].value_counts()[1]/df['IsBadBuy'].value_counts().sum()))*100).format("{:.2f}"))
print(((df['IsBadBuy'].value_counts()[0]/df['IsBadBuy'].value_counts().sum()))*100)



df['IsBadBuy'].value_counts()



df.isnull().sum()



for i in df.columns:
    if (df[i].dtype =='object') & ((df[i].isnull().sum()) > 0):
        df[i].fillna('Unknown',inplace=True)
        

df = df.dropna()



df.reset_index(inplace  = True)



cols = [x for x in df.columns if df[x].dtypes == 'object']
     


X = pd.get_dummies(data=df[cols], drop_first=True)




numeric_cols = [x for x in df.columns if df[x].dtype != 'object' ]
df_numeric = df[numeric_cols[2:]]
# df_numeric.head()



X = pd.concat([X,df_numeric],axis = 1)
y = df['IsBadBuy']



#!pip install imblearn



from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)



from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

def Random_Forest(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=50)
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix,classification_report
    
    def type(y):
        if y.value_counts()[0] == y.value_counts()[1]:
            return "SMOTE data"
        else:
            return "Raw data"
    type_s = type(y)
    print('\n')
    print("""=========== Random Forest's results - {} ========================""".format(type_s))
    

    print(f'Train : {model.score(X_train, y_train):.3f}')
    print(f'Test : {model.score(X_test, y_test):.3f}')
    print('\n')

    
    print(confusion_matrix(predictions,y_test))
    print('\n')
    print(classification_report(predictions,y_test))



Random_Forest(X_res, y_res)
Random_Forest(X,y)




# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


def Logistic_reg(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=150)
    
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix,classification_report
    
    def type(y):
        if y.value_counts()[0] == y.value_counts()[1]:
            return "SMOTE data"
        else:
            return "Raw data"
    type_s = type(y)
    print('\n')
    print("""=========== Logistic Regressions's results - {} ===================""".format(type_s))
    
    print(confusion_matrix(predictions,y_test))
    print('\n')
    print(classification_report(predictions,y_test)) 


# In[ ]:


Logistic_reg(X_res, y_res)
Logistic_reg(X, y)


# In[ ]:


# !pip install tensorflow==2.6.0


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42)
    
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    
model = Sequential()
    
model.add(Dense(X_train.shape[1],activation = 'relu'))
    
model.add(Dense((X_train.shape[1])/2,activation = 'relu'))
    
model.add(Dense(1,activation = 'sigmoid'))
              
model.compile(loss = 'binary_crossentropy',optimizer = 'adam')
              
model.fit(x = X_train,y = y_train,epochs=100,validation_data=(X_test,y_test))


losses = pd.DataFrame(model.history.history)
plt.figure(figsize = (12,6))
losses.plot()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:





# In[ ]:


model = Sequential()
    
model.add(Dense(X_train.shape[1],activation = 'relu'))
model.add(Dropout(rate = 0.5))
    
model.add(Dense((X_train.shape[1])/2,activation = 'relu'))
model.add(Dropout(rate = 0.5))
    
model.add(Dense(1,activation = 'sigmoid'))
              
model.compile(loss = 'binary_crossentropy',optimizer = 'adam')

early_stop = EarlyStopping(monitor='val_loss',mode = 'min',verbose = 1,patience = 5)

model.fit(x = X_train,y = y_train,epochs=100,validation_data=(X_test,y_test),
         callbacks = [early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()


# In[ ]:


predictions = (model.predict(X_test)>0.5).astype('int32')


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predictions))


# In[ ]:


def NN(X,y,n_layers = 3,verbose = 0,epochs=100):
    
    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = Sequential()
    
    model.add(Dense(X_train.shape[1],activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    
    for i in range(n_layers):
        model.add(Dense((X_train.shape[1])/2,activation = 'relu'))
        model.add(Dropout(rate = 0.5))
        
        

    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',optimizer = 'adam')

    early_stop = EarlyStopping(monitor='val_loss',mode = 'min',verbose = verbose,patience = 5)

    model.fit(x = X_train,y = y_train,epochs=epochs,validation_data=(X_test,y_test),
             callbacks = [early_stop],verbose = verbose)

    losses = pd.DataFrame(model.history.history)
    losses.plot()
    predictions = (model.predict(X_test)>0.5).astype('int32')
    print(classification_report(y_test,predictions))


# In[ ]:


NN(X_res,y_res,verbose=1)


# In[ ]:


NN(X,y,verbose=1)

