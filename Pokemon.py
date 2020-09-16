#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', 'C:/Users/atoma/Pokemon play')


# In[2]:


#import the libraries and the data


# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df  =pd.read_csv('pokemon.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


#selecting the desired columns


# In[8]:


df = df[['against_bug', 'against_dark', 'against_dragon',
       'against_electric', 'against_fairy', 'against_fight', 'against_fire',
       'against_flying', 'against_ghost', 'against_grass', 'against_ground',
       'against_ice', 'against_normal', 'against_poison', 'against_psychic',
       'against_rock', 'against_steel', 'against_water', 'attack',
       'base_egg_steps', 'base_happiness', 'base_total', 'capture_rate',
       'classfication', 'defense', 'experience_growth', 'height_m', 'hp',
        'name', 'percentage_male', 
       'sp_attack', 'sp_defense', 'speed', 'type1', 'weight_kg',
       'generation', 'is_legendary']]


# In[9]:


#drop the line where capture_rate is not numeric


# In[10]:


df.drop(df[df['capture_rate']=='30 (Meteorite)255 (Core)'].index,inplace = True)


# In[11]:


df['capture_rate']=df['capture_rate'].apply(lambda x:int(x))


# In[12]:


sns.boxplot(df['capture_rate'])


# In[13]:


#redefine classification column


# In[14]:


df['classfication'].value_counts()


# In[15]:


df['classfication'] = df['classfication'].apply(lambda x:x[:-8])


# In[16]:


#countplot for the dependent variable


# In[17]:


plt.figure(figsize=(12,6))
sns.countplot('type1', data = df)


# In[18]:


sns.distplot(df['height_m'])


# In[19]:


sns.distplot(df['weight_kg'])


# In[20]:


sns.jointplot('height_m','weight_kg',data = df)


# In[21]:


#df.drop(df[(df['weight_kg']>300) | (df['height_m']>4)].index,inplace = True)


# In[22]:


#deal with missing data


# In[23]:


plt.figure(figsize=(14,6))
sns.heatmap(df.isna(),yticklabels=False,cmap='viridis')


# In[24]:


#drop the percentage_male since it has many missing data
df.drop('percentage_male',axis = 1,inplace = True)


# In[25]:


df[np.isnan(df['height_m'])]


# In[26]:


#creating 2 data frames with the means for  height/weight by Type
means_height=df.groupby('type1')['height_m'].mean()
means_weight=df.groupby('type1')['weight_kg'].mean()


# In[27]:


df.info()


# In[28]:


#creating 2 functions to fill in the missing data
def fill_missing_height(type1,height):

    if np.isnan(height):
        return means_height.loc[type1]
    else:
        return height
    
def fill_missing_weight(type1,height):

    if np.isnan(height):
        return means_weight.loc[type1]
    else:
        return height


# In[29]:


df['height_m'] = df.apply(lambda x: fill_missing_height(x['type1'], x['height_m']), axis=1)
df['weight_kg'] = df.apply(lambda x: fill_missing_weight(x['type1'], x['weight_kg']), axis=1)


# In[30]:


df.drop(['name'],axis = 1, inplace=True)


# In[31]:


df['type1'].value_counts()


# In[32]:


df.info()


# In[33]:


#creating dummies for classification column
dummies = pd.get_dummies(df['classfication'],drop_first=True)


# In[34]:


df = pd.concat([df,dummies],axis = 1)


# In[35]:


df.drop('classfication',axis=1,inplace=True)


# In[36]:


#define X and y
X = df.drop('type1',axis = 1).values
y = df['type1'].values


# In[37]:


#encode and tranform y variable to be categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# In[38]:


encoder = LabelEncoder()


# In[39]:


y = encoder.fit_transform(y)


# In[40]:


y = to_categorical(y,18)


# In[41]:


y[0]


# In[42]:


#scale X_train and X_test
from sklearn.preprocessing import MinMaxScaler


# In[43]:


scaler = MinMaxScaler()


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[45]:


X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# In[46]:


#build the NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[47]:


model = Sequential()


# In[48]:


model.add(Dense(549,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(250,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(150,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(50,activation='relu'))
model.add(Dropout(0.3))


#Binary classification  - sigmoid function
model.add(Dense(18,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[49]:


from tensorflow.keras.callbacks import EarlyStopping


# In[50]:


#adding an early stop
early_stop = EarlyStopping(monitor='val_loss',mode = 'min',verbose=1,patience = 15)


# In[51]:


#fir the model
model.fit(X_train,y_train,epochs=600, validation_data = (X_test,y_test),
         callbacks=[early_stop])


# In[52]:


#plt the training evolution
pd.DataFrame(model.history.history).plot()


# In[53]:


#evaluate the model
model.evaluate(X_test,y_test)


# In[54]:


model.metrics_names


# In[55]:


#predict the model
predictions = model.predict_classes(X_test)


# In[56]:


pred = model.predict(X_train[150:151])


# In[57]:


pred.argmax()


# In[58]:


y_train[150:151].argmax()

