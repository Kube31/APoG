import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

def build_tokenizer(myList):
    token = tf.keras.preprocessing.text.Tokenizer(21641, filters='', lower=True, split='\n',
                                                  char_level=False, oov_token='OOV')
    token.fit_on_texts(myList)
    return token

def tokenizer(myList, token):
    #unique_token_comp_count = len(token.word_counts) + 2
    padded_seq = token.texts_to_sequences(myList)
    new_seq = []
    for tok in padded_seq:
        new_seq.append(tok[0])
    return np.asarray(new_seq).astype('float32')#,unique_token_comp_count

def bool_to_int(List):
    my_list = []
    for elem in List:
        if elem:
            my_list.append(1)
        else:
            my_list.append(2)
    return np.asarray(my_list).astype('float32')

def date_splitter(myList):
    liste  =list(map(lambda x: x.split('/'),myList))
    liste1,liste2,liste3 = [],[],[]
    for tuple in liste:
        liste1.append(int(tuple[0]))
        liste2.append(int(tuple[1]))
        liste3.append(int(tuple[2]))
    return np.asarray(liste1).astype('float32'),np.asarray(liste2).astype('float32'),np.asarray(liste3).astype('float32')

def build_range_normalizer(dataframe: pd.DataFrame):
    min = []
    max = []
    columns = []
    for col in dataframe:
        columns.append(col)
        min.append(dataframe[col].min())
        max.append(dataframe[col].max())
    range = pd.DataFrame(columns = columns)
    range.loc[len(range)] = min
    range.loc[len(range)] = max
    return range

def apply_normalizer(df,ranges_df):
    i = 0
    for col in df:
        df[col] = list(map(lambda x: (x-ranges_df[i][0])/(ranges_df[i][1] - ranges_df[i][0]) ,df[col]))
        i = i + 1
    return df

data = pd.read_csv("D:\Master_ESGI\dataset5.6.csv", sep=';', encoding='ISO-8859-1')

tokenizer_competitor = build_tokenizer(data["Competitor"])
tokenizer_nationality = build_tokenizer(data["Nationality"])
tokenizer_venue = build_tokenizer(data["Venue"])
tokenizer_DRIC = build_tokenizer(data["Daily_Race_ID_in_a_Competition"])

for token in ['tokenizer_competitor','token_nationality', 'token_venue', 'token_DRIC']:
    with open('AI_API\\app\\' + token + '.pickle', 'wb') as handle:
        pickle.dump(tokenizer_competitor, handle, protocol=pickle.HIGHEST_PROTOCOL)


data["Competitor"] = tokenizer(data["Competitor"],tokenizer_competitor)
data["Nationality"] = tokenizer(data["Nationality"],tokenizer_nationality)
data["Venue"] = tokenizer(data["Venue"], tokenizer_venue)
data["Daily_Race_ID_in_a_Competition"] = tokenizer(data["Daily_Race_ID_in_a_Competition"], tokenizer_DRIC)

data["Birth_Day"], data["Birth_Month"], data["Birth_Year"] = date_splitter(data["Date_of_birthday"])
data["Venue_Day"], data["Venue_Month"], data["Venue_Year"] = date_splitter(data["Date_venue"])
data = data.drop(columns=["Date_of_birthday", "Date_venue"])

for col in data.columns:
    data[col] = data[col].fillna(0)
    if col in ["First in last 3 runs", "Top 3 in last 3 runs"]:
        data[col] = bool_to_int(data[col])
    elif col in ["Mark_time","WIND","IMC","totalbodyfate(kg)","leanbodyweight(kg)","FFMI","FFMI_Ajusted","Temp",
                 "Feels_like","Dew_point", "Wind_speed"]:
        data[col] = np.asarray(list(map(lambda x : str(x).replace(',','.'),data[col]))).astype('float32')
    data[col] = np.asarray(list(map(lambda x : str(x).replace(',,,,,,,,,','0'),data[col]))).astype('float32')


'''
Data = pd.read_csv("D:\\Dataset_Normalized.csv",sep=";")
Data.columns = ['Mark_time', 'WIND', 'Competitor', 'Nationality', 'Position', 'Venue',
       'Results_score', 'Age_Competitor_at_run', 'weight(kg)', 'height(cm)',
       'IMC', 'totalbodyfate(kg)', 'leanbodyweight(kg)', 'FFMI',
       'FFMI_Ajusted', 'Sunrise', 'Sunset', 'Temp', 'Feels_like', 'Pressure',
       'Humidity', 'Dew_point', 'Clouds', 'Wind_speed', 'Wind_deg',
       'race_type', 'Daily_Race_ID_in_a_Competition', 'First in last 3 runs',
       'Top 3 in last 3 runs', 'Birth_Day', 'Birth_Month', 'Birth_Year',
       'Venue_Day', 'Venue_Month', 'Venue_Year']
'''
inputs = ["Competitor","Nationality","Venue","Age_Competitor_at_run","weight(kg)","height(cm)",
          "IMC","totalbodyfate(kg)","leanbodyweight(kg)","FFMI","FFMI_Ajusted","Sunrise","Sunset","Temp","Feels_like",
          "Pressure","Humidity","Dew_point","Clouds","Wind_speed","Wind_deg","race_type","First in last 3 runs","Top 3 in last 3 runs","Birth_Day","Birth_Month",
          "Birth_Year","Venue_Day","Venue_Month","Venue_Year"]

target = ["Mark_time", "Position"]

# Here the code has changed to put the normalization layer in the neural netword
x = data[inputs].values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x)
X_ranges = build_range_normalizer(X)
X_ranges.to_csv("AI_API\\app\\normalization_ranges.csv", index=False, sep=";")
X = apply_normalizer(X,X_ranges)

y = data[target].values
Y = pd.DataFrame(y)

x_train,x_test, y_train, y_test = train_test_split(X.values,Y,test_size=0.2)
#(X_train,Y_train),(X_test, Y_test) = tf.keras.datasets.boston_housing.load_data()
'''
df_X = pd.DataFrame(x_train)
df_y = pd.DataFrame(y_train)
x_test = pd.DataFrame(x_test)
y_test = pd.DataFrame(y_test)
'''
# Il faut faire attention à ne pas utiliser un model trop performant car grâce ( ou à cause...) de l'active learning
# le model peut se retrouver facilement dans une situation d'overfitting
# cela peut etre dû à la stratégie de sélection des points

def NN_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model

model = NN_model()
model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=200, batch_size=50)

res = model.evaluate(x_test, y_test,batch_size=50)
print("test loss, test acc:", res)

prediction = model.predict(x_test[:2])
print("prediction shape:", prediction.shape)
model.save("AI_API\\app\\model")

# do not normalize target reverse conversion is a waste of time and not as accurate as if we do not normalize it
