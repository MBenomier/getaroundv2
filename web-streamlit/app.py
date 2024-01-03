import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


### Config
st.set_page_config(
    page_title="GetAround Analysis",
    page_icon="",
    layout="wide"
)


### App
#
st.title("GetAround  Analysis")

st.image("prep/getaround.png")

st.markdown("""
    Hello and welcome you are on Getaroundand : 
             
    for more information read the following:    
            
    When our users rent a car, they are required to go through a check-in process at the start of the rental and a check-out process at the end to:

      * Evaluate the car's condition and inform relevant parties of any existing or rental-incurred damages.
      * Mileage traveled during car rental.       
      * Check and compare fuel levels.
    
    The check-in and check-out processes for our rentals can be completed through three separate workflows:

      * Mobile rental agreement facilitated through a mobile apps, the driver and owner meet, and both parties sign the rental agreement using the owner's smartphone.
      * Connect option, the driver doesn't meet the owner and unlocks the car using their smartphone.
      * Traditional paper contract " Not available at the moment".
  
  """)



st.markdown("---")


# Use `st.cache` when loading data is extremely useful
# because it will cache your data so that your app
# won't have to reload it each time you refresh your app

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("prep/get_around_delay_analysis.csv",sep=";")
    return data

st.text('Loading in progress...')

data_load_state = st.text('Loading data ...')
data = load_data()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show data'):
    st.header('Datast')
    st.write(data)

data['delay']=data["delay_at_checkout_in_minutes"].apply(lambda x: 1 if x>10 else 0)
                                                                    

st.header('Dataset Overview')

st.markdown("At first sight  is that the delay is not the main reason to cancel the reservation..")

fig1 = px.pie(data, values='delay', 
    names='state', 
    title='Impact of the delay on the reservation,can the reservation be cancelled or not?',
    color_discrete_sequence=[ "#AA336A", "darkcyan"])

st.markdown("""Regarding the check-in type and the time-delta-with the 
                previous rental in minutes, is there a correlation? Should we enable this
               functionality for all cars or only for Connect cars?")""")

fig2 = px.histogram(data[data["time_delta_with_previous_rental_in_minutes"]>0.0], x="time_delta_with_previous_rental_in_minutes", color="checkin_type",
    labels={"value":'time_delta_with_previous_rental_in_minutes'},
    title='Frequency of car rent by agreement :',
    color_discrete_sequence=[ "#AA336A", "darkcyan"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Impact of the delay on the reservation, can the reservation be cancelled or not?")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
   st.subheader("Frequency of car rent by agreement")
   st.plotly_chart(fig2, use_container_width=True)

st.markdown("Here, we can see that owners prefer Mobile rental agreement because they are more frequent rather than Connect.")
st.subheader("Maybe is it more secure to meet the driver with Mobile agreement or are the Mobile agreement more frequent?")

fig3 = px.pie(data, values="time_delta_with_previous_rental_in_minutes",
     names='checkin_type',
     color_discrete_sequence=[ "#AA336A", "darkcyan"])

st.plotly_chart(fig3, use_container_width=True)

st.markdown("Here , we see that Mobile are more are more frequent rather than Connect. We have only 45% of Connect !")
st.markdown("Alright , you are maybe wondering... ")
st.subheader("How does delay impact check-out timing and which agreement has more impact in the check-out delay?")

fig4 = px.histogram(data_frame=data[data["time_delta_with_previous_rental_in_minutes"]>0.0], x='delay_at_checkout_in_minutes', color='checkin_type', histnorm='percent', 
    barmode='overlay',range_x=(-400,400),labels={"value":'Delay at checkout per minute'},
    title='How does delay impact check-out timing? :',
    color_discrete_sequence=[ "#AA336A", "darkcyan"])

st.markdown("Well, there is no surprise, the Mobile agreement will be slower than the Connect agreement, so the check-out delay will be higher")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("")


st.header('How long should the minimum delay be ?')

data = data.dropna(subset=["time_delta_with_previous_rental_in_minutes", "delay_at_checkout_in_minutes"])
data_test = pd.melt(data, id_vars=['car_id', 'rental_id', 'state', 'checkin_type'], value_vars=['time_delta_with_previous_rental_in_minutes', 'delay_at_checkout_in_minutes'])

st.metric(label="car fleet", value=data_test['car_id'].nunique())


fig6 = px.ecdf(
    data_test[data_test['checkin_type']=='mobile'],
    x='value',
    color='variable',
    ecdfnorm= 'percent',
    range_x=(0, 600),
    color_discrete_sequence=[ "#AA336A", "darkcyan"],
    labels={"value":'threshold (minutes)', "percent":'proportion of users (%)'}
    )

fig7 = px.ecdf(
    data_test[data_test['checkin_type']=='connect'],
    x='value',
    color='variable',
    ecdfnorm= 'percent',
    range_x=(0, 600),
    color_discrete_sequence=[ "#AA336A", "darkcyan"],
    labels={"value":'threshold (minutes)', "percent":'proportion of users (%)'}
    )
  

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mobile")
    st.plotly_chart(fig6, use_container_width=True)

with col2:
   st.subheader("Connect")
   st.plotly_chart(fig7, use_container_width=True)

st.markdown(" * These plots represent Cumulative Distribution Functions (ECDF), allowing us to visualize the percentage of users affected by the introduction of a minimum time delay threshold.")
st.markdown(" * We observe that 48% of users return their car on time with the mobile version, compared to 66% with the Connect version. The minimum delay threshold for both versions is set at 30 minutes."
            " * The delay in returns has a proportional impact on pick-up times, gradually accumulating throughout the day."
            " * It could be feasible to reduce the minimum threshold to 20 minutes for the Connect version. However, one piece of information is lacking: the number of cancellations avoided by the owner as a result of increasing the threshold.")
st.markdown(" * The threshold should be lower for Connect cars as there are significantly fewer instances of late returns.")
