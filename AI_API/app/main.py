import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

def charge_tokenizer(tokenizer_file_path):
    with open(tokenizer_file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def API_tokenizer(elem, token):
    return token.texts_to_sequences([elem])[0][0]

def API_bool_to_int(elem):
    if elem:
        return 1
    else:
        return 2

def date_splitter(elem):
    return elem.split('/')

def normalize(x:float,range):
    return ((x - range[0]) / (range[1] - range[0]))

model = tf.keras.models.load_model("app/model")
normalization_range = pd.read_csv("app/normalization_ranges.csv", sep=";")
tokenizer_competitor = charge_tokenizer("app/tokenizer_competitor.pickle")
tokenizer_nationality = charge_tokenizer("app/token_nationality.pickle")
tokenizer_venue = charge_tokenizer("app/token_venue.pickle")
tokenizer_DRIC = charge_tokenizer("app/token_DRIC.pickle")

app = FastAPI()

class Item(BaseModel):
    #WIND: str
    Competitor: str
    Birth_day: str
    Nationality: str
    Venue: str
    Date_Venue: str
    #Results_score: str
    Age_Competitor_at_run: str
    weight: str
    height: str
    IMC: str
    totalbodyfate: str
    leanbodyweight: str
    FFMI: str
    FFMI_Ajusted: str
    Sunrise: str
    Sunset: str
    Temp: str
    Feels_like: str
    Pressure: str
    Humidity: str
    Dew_point: str
    Clouds: str
    Wind_speed: str
    Wind_deg: str
    race_type: str
    #Daily_Race_ID_in_a_Competition: str
    First_in_last_3_runs: str
    Top_3_in_last_3_runs: str


@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post("/predict")
async def create_item(item: Item):
    birth_D = date_splitter(item.Birth_day)
    venue_D = date_splitter(item.Date_Venue)
    data = [[
        #float(item.WIND.replace(",",".")),
        normalize(float(API_tokenizer([item.Competitor],tokenizer_competitor)),normalization_range['0']),
        normalize(float(API_tokenizer([item.Nationality],tokenizer_nationality)),normalization_range['1']),
        normalize(float(API_tokenizer([item.Venue],tokenizer_venue)),normalization_range['2']),
        #float(item.Results_score.replace(",",".")),
        normalize(float(item.Age_Competitor_at_run.replace(",",".")),normalization_range['3']),
        normalize(float(item.weight.replace(",",".")),normalization_range['4']),
        normalize(float(item.height.replace(",",".")),normalization_range['5']),
        normalize(float(item.IMC.replace(",",".")),normalization_range['6']),
        normalize(float(item.totalbodyfate.replace(",",".")),normalization_range['7']),
        normalize(float(item.leanbodyweight.replace(",",".")),normalization_range['8']),
        normalize(float(item.FFMI.replace(",",".")),normalization_range['9']),
        normalize(float(item.FFMI_Ajusted.replace(",",".")),normalization_range['10']),
        normalize(float(item.Sunrise.replace(",",".")),normalization_range['11']),
        normalize(float(item.Sunset.replace(",",".")),normalization_range['12']),
        normalize(float(item.Temp.replace(",",".")),normalization_range['13']),
        normalize(float(item.Feels_like.replace(",",".")),normalization_range['14']),
        normalize(float(item.Pressure.replace(",",".")),normalization_range['15']),
        normalize(float(item.Humidity.replace(",",".")),normalization_range['16']),
        normalize(float(item.Dew_point.replace(",",".")),normalization_range['17']),
        normalize(float(item.Clouds.replace(",",".")),normalization_range['18']),
        normalize(float(item.Wind_speed.replace(",",".")),normalization_range['19']),
        normalize(float(item.Wind_deg.replace(",",".")),normalization_range['20']),
        normalize(float(item.race_type.replace(",",".")),normalization_range['21']),
        #float(API_tokenizer([item.Daily_Race_ID_in_a_Competition],tokenizer_DRIC)),
        normalize(float(API_bool_to_int(bool(item.First_in_last_3_runs))),normalization_range['22']),
        normalize(float(API_bool_to_int(bool(item.Top_3_in_last_3_runs))),normalization_range['23']),
        normalize(float(birth_D[0]),normalization_range['24']),  #day
        normalize(float(birth_D[1]),normalization_range['25']),  #month
        normalize(float(birth_D[2]),normalization_range['26']),  #year
        normalize(float(venue_D[0]),normalization_range['27']),  #day
        normalize(float(venue_D[1]),normalization_range['28']),  #month
        normalize(float(venue_D[2]),normalization_range['29']),  #year
    ]]
    pred = model.predict(data)
    return str(pred.tolist()[0][0])

