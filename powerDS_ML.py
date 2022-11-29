# import libraries
import sklearn
import tensorflow
import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import hiplot as hip
import matplotlib.pyplot as plt

# setting the layout for the seaborn plot
sns.set(style="darkgrid")

# The story I want to tell is the interaction or relationship between the power
# consumption in hot weather or cold climate.

# load DATA:- The data is loaded from the UCI machine learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00616/Tetuan%20City%20power%20consumption.csv"
data = pd.read_csv(url)

# The dataset can only perform regressional tasks because most of the columns
# contains continuous data. In order to play with the data more I added a 
# classification column by creating a function based on the temperature ranges.

def warmth_condition(df):
    if df['Temperature'] >= 30:
        return '4-Hot'
    
    elif df['Temperature'] >= 20:  
        return '3-Mild'
    
    elif df['Temperature'] >= 10:
        return '2-Cold'
    
    return '1-Very Cold'

data['Warmth Level'] = data.apply(warmth_condition, axis=1)


# Designing the Visuals on the App
# --------------------------------

# Partitioning the Web App to accommodate the Visualization of the Dataset and ML algorithm
st.sidebar.write("""
    ### The Functionality of this App is divided into two sections""")

main_opt = st.sidebar.selectbox('Targeted Action: ', ["Data Visualization", "Run Machine Learning Algorithms"])

if(main_opt == "Data Visualization"):

    # The title of the web page
    st.write("""
    ## Do Weather Influence the rate of Power Consumption? 
    Using the Tetuan City Power Consumption Dataset as a case study.
    """)

    # show the dataset description
    if st.checkbox('Show written description of the Dataset'):
        st.write("""
    The Dataset originally constists of Nine columns namely: DateTime, temperature (°C), humidity, wind speed of Tetuan City, power supply 
    (general diffuse flows and diffuse flows) and the consumption for three different locations in the City (Zone 1, Zone 2, and Zone 3). 

    An extra column (Warmth Level) was added to the dataset to aid categorizing the data. The values of each feature (column) were collected 
    after every 10 minutes leading to a total of 52416 observations present in the data.
    """)



    zone_option = st.selectbox(
        "The values shows the maximum (Upper) and minimum (Lower) for the selected columns.  Select Zone",
        ('Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption')) 

    # Displaying a basic summary of the dataset
    # 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", f"{data['Temperature'].max():.2f} ", f"{data['Temperature'].min():.2f}", delta_color="off")
    col2.metric("Humidity", f"{data['Humidity'].max():.2f}", f"{data['Humidity'].min():.2f}", delta_color="off")
    col3.metric("Wind Speed", f"{data['Wind Speed'].max():.2f}", f"{data['Wind Speed'].min():.2f}", delta_color="off")
    col4.metric(zone_option, f"{data[zone_option].max():.2f}", f"{data[zone_option].min():.2f}", delta_color="off")


    # Constraining the Data to output the values of the corresponding columns based on the selected column
    constrain_option = st.selectbox(
        "Constrain the corresponding Max and Min values based on the selected Column in the dataset.",
        ('Temperature', 'Humidity', 'Wind Speed', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption')) 

    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    ccol1.metric("Temperature", f"{data['Temperature'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f} ", 
        f"{data['Temperature'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol2.metric("Humidity", f"{data['Humidity'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data['Humidity'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol3.metric("Wind Speed", f"{data['Wind Speed'].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data['Wind Speed'].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")
    ccol4.metric(zone_option, f"{data[zone_option].where(data[constrain_option] == data[constrain_option].max()).dropna().iloc[0]:.2f}", 
        f"{data[zone_option].where(data[constrain_option] == data[constrain_option].min()).dropna().iloc[0]:.2f}", delta_color="off")


    # Button that allows the user to see the entire table
    if st.checkbox('Show the Dataset'):
        st.dataframe(data=data)


    # Plotting some visuals
    st.write(
        "#### Resampling provides a wide scale of exploration of the Dataset")

    # resampling the data demands taking the mean of the dataset given the number of How
    hour = st.slider('Select the number of hours (e.g.,  1: Hourly,  24: Daily,  168: Weekly,  730: Monthly)', 0, 731, 1)

    # resampling the Tetuan DataFrame.
    if hour != 0:
        data_RS = pd.read_csv(url, parse_dates=['DateTime'], index_col=['DateTime'])
        df = data_RS.resample(f'{hour}H').mean()  # Calculates the mean of the group
        df.reset_index(inplace=True)  # Resets the index to the original so that DateTime can be a column in the dateaframe
        df['Warmth Level'] = df.apply(warmth_condition, axis=1)  # Adds the Warmth level' column to the newly creaeted dataframe.



    if st.checkbox('Show the Resampled Dataset'):
        if hour == 0:
            st.write("""The data is not resampled, you must select atleast 1 on the slide bar!""")

        else:
            st.dataframe(data=df)


    # Select which of the dataframe to plot either the resampled or the actual dataset based on the value of the hour
    if hour == 0:
        df_plot = data  # saves the original plot to be plotted below
        
    else:
        df_plot = df  # saves the resampled plot to be plotted below



    # plotting Capabilities.
    st.write("""
        ### Select the different options to visualize the dataset below.
        """)

    # Plotting options
    plot_opt = st.selectbox("",
        ('Altair Interactive Plot', 'HiPlot', 'Joint Plot')) 

    # Selecting the different options for plotting

    if(plot_opt == 'Altair Interactive Plot'):

        opt1, opt2, opt3 = st.columns(3)
        df_sidebar = df_plot.drop(columns="Warmth Level")  # I don't want 'Warmth Level' to be in the options.
        with opt1:
            x_sb = st.selectbox('x axis: ', df_sidebar.columns)

        with opt2:
            y_sb = st.selectbox('y axis: ', df_sidebar.drop(columns=[x_sb]).columns)

        with opt3:
            color = st.selectbox('hue: ', ["Warmth Level"])

        # Making some interactive plots

        alt.data_transformers.disable_max_rows()  # To enable altair plot more than 5000 rows

        domain = ['1-Very Cold', '2-Cold', '3-Mild', '4-Hot']
        range_ = ['navy', 'royalblue', 'deepskyblue', 'crimson']
        chart = alt.Chart(df_plot).mark_point().encode(
            alt.X(x_sb, title= f'{x_sb}'),
            alt.Y(y_sb, title=f'{y_sb}'),
            color=alt.Color('Warmth Level', scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['DateTime', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 
                         'diffuse flows', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 
                     'Zone 3  Power Consumption']
        ).properties(
            width=600,
            height=400
        ).interactive()

        # plotting Altair with streamlit
        st.altair_chart(chart)


    elif(plot_opt == 'HiPlot'):

        feat_opt = st.multiselect(
            'Add or remove features to visualize in the plot below',
            ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows',
            'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption'], 
            ['Temperature', 'Humidity', 'Wind Speed',  'Zone 1 Power Consumption'])

        # Incase the user makes a mistake by deleting the columns by mistake
        if(len(feat_opt) == 0):
            st.write("""
                ### You cannot leave the field empty, Please select one or more columns!
                """)

        else:
            # New Dataset created based on the column selection
            Hiplot = df_plot[feat_opt]

            # Plotting the data using Hiplot
            xp = hip.Experiment.from_dataframe(Hiplot)
            xp.to_streamlit(key="hip").display()


    elif(plot_opt == 'Joint Plot'):
        J_opt1, J_opt2 = st.columns(2)
        df_sidebar = df_plot.drop(columns=["Warmth Level", "DateTime"])  # I don't want 'Warmth Level' to be in the options.
        with J_opt1:
            x_jp = st.selectbox('x axis: ', df_sidebar.columns)

        with J_opt2:
            y_jp = st.selectbox('y axis: ', df_sidebar.drop(columns=[x_jp]).columns)

        # Plotting with the jointplot
        sns.jointplot(x=x_jp, y= y_jp, data =df_plot)

        # Displaying the Plot using the streamlit command
        st.pyplot(plt.gcf())


else:
    st.write("""
        ### SURPRISE! 😂😉😉

        Stay Tuned, Currently an ongoing project.""")