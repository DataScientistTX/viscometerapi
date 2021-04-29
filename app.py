# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:01:23 2021

@author: sercan
"""
#Import libraries--------------------------------------------------------------
import streamlit as st
import numpy as np
from scipy.optimize import curve_fit   
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import xlrd
import base64
from io import BytesIO

plt.style.use('default')

#Define curve fitting functions------------------------------------------------
#y-shear rate
#K-consistency index
#n-flow behavior index
#ty-yield stress

def YPLfunction(y, ty, K, n):
    return ty + K*y**n

def PLfunction(y, K, n):
    return  K*y**n

def BPfunction(y,PV,YP):
    return YP + PV*y

#Perform curve fitting and calculate r2----------------------------------------
#PL - power law
#YPL - yield power law
#BP - bingham plastic

def r2(residuals,shear_stress,shear_rate):
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((shear_stress-np.mean(shear_stress))**2)
    r_squared = 1 - (ss_res / ss_tot) 
    return r_squared

def PL(shear_stress,shear_rate):
    popt, pcov = curve_fit(PLfunction,shear_rate,shear_stress)
    K,m =popt[0],popt[1]
    residuals = shear_stress- PLfunction(shear_rate, popt[0],popt[1])
    r_squared = r2(residuals,shear_stress,shear_rate)   
    return K,m,r_squared

def YPL(shear_stress,shear_rate):          
    popt, pcov = curve_fit(YPLfunction,shear_rate,shear_stress)
    ty,K,m = popt[0],popt[1],popt[2]
    residuals = shear_stress- YPLfunction(shear_rate, popt[0],popt[1],popt[2])
    r_squared = r2(residuals,shear_stress,shear_rate)  
    
    if popt[0]<0:
        K,m,r_squared = PL(shear_stress,shear_rate)
        ty = 0
    return ty,K,m,r_squared
  
def BP(shear_stress,shear_rate):
    PV = (shear_stress[0] - shear_stress[1])/511
    YP = (2*shear_stress[1] - shear_stress[0])
    residuals = shear_stress- BPfunction(shear_rate, PV, YP)
    r_squared = r2(residuals,shear_stress,shear_rate) 
        
    #Calculate equivalent sigma600 (DR)
    sigma600  = (YP + PV*600*1.7) / (1.066 * 0.4788)

    #Calculate equivalent sigma300 (DR)
    sigma300  = (YP + PV*300*1.7) / (1.066 * 0.4788)
    return r_squared,PV, YP, sigma600, sigma300

#Define functions for download links, and xlsx conversions---------------------
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="example_data.xlsx">Please click here to download an example dataset for this app as an excel file.</a>' # decode b'abc' => abc

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_excel(file)
    return df

st.header("Drilling Fluid Rheological Model Parameters")
st.write("This web-app is used to analyze API rotational viscometer data by comparing various rheological models.")
st.write("The rheological constants for Yield Power-law (YPL - also called Herschel-Bulkley), Power-law, and Bingham-Plastic models are calculated and compared.")
st.write("Please upload the data using the file uploader on the left side. Please make sure that the data is in excel (.xlsx) format, where the first column is the RPM values and the second column is shear stress values (as viscometer dial readings) for each corresponding RPM.")
st.write("Below link can be used to download an example dataset for this web-app.")
st.write("NOTE: If you are using a 6-speed viscometer, you might be more interested in apiviscometer.herokuapp.com")
d = {'RPM': [300,200,100,60,30,6,3], 'Viscometer Dial Readings (DR)': [105.1,90.8,71.7,63.4,55.3,45.8,44]}
df_template = pd.DataFrame(data=d)
st.markdown(get_table_download_link(df_template), unsafe_allow_html=True)
  
st.sidebar.title("Upload data")    
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    df.columns = ["Viscometer RPM","'Viscometer Dial Readings (DR)"]
    df = df.sort_values(by="Viscometer RPM", ascending=False)
    df = df.reset_index(drop=True)
    
    dial_readings = df["'Viscometer Dial Readings (DR)"]
    shear_rate = df["Viscometer RPM"]*1.7011

#Entries: if higher RPM value entered is lower than next one, make corrections-
    for i in range(5):
        if dial_readings[i]<dial_readings[i+1]:
            dial_readings[i] = dial_readings[i+1]
    
#Perform curve fitting for all three rheological models------------------------
    shear_stress = np.asarray(dial_readings) * 1.066 * 0.4788 #unit conversion from DR to Pascal
    
    ty_YPL,K_YPL,n_YPL,r2_YPL = YPL(shear_stress,shear_rate)
    K_PL,n_PL,r2_PL = PL(shear_stress,shear_rate)
    r2_BP,PV,YP,DR600,DR300 = BP(shear_stress,shear_rate)
    
#Denoised values for visuals---------------------------------------------------
    shear_stress_calc_YPL = YPLfunction(shear_rate, ty_YPL, K_YPL, n_YPL)
    shear_stress_calc_PL = PLfunction(shear_rate, K_PL, n_PL)
    shear_stress_calc_BP = BPfunction(shear_rate, PV, YP)
    
#Printing out the rheological constants for each model-------------------------
    st.subheader ("Herschel Bulkley (Yield Power Law) Model Rheological Constants")
    st.write("Yield stress ($t_{y}$) is", round(ty_YPL,2), "$Pa$")
    st.write("Consistency index (K) is", round(K_YPL,4), "$Pa.s^{n}$")
    st.write("Flow index (n) is", round(n_YPL,2))
    st.write("Coefficient of determination ($R^2$) is", round(r2_YPL,3))
     
    st.subheader ("Power-Law Model Rheological Constants")
    st.write("Consistency index (K) is", round(K_PL,4), "$Pa.s^{n}$")
    st.write("Flow index (n) is", round(n_PL,2))
    st.write("Coefficient of determination ($R^2$) is", round(r2_PL,3))
    
    st.subheader ("Bingham Plastic Model Rheological Constants")
    st.write("Plastic viscosity (PV) is", round(DR600-DR300,2) , "$cp$") 
    st.write("Yield point (YP) is", round(2*DR300-DR600,2), "$lb/100ft^2$")
    st.write("Coefficient of determination ($R^2$) is", round(r2_BP,3))
    
#Visuazalization---------------------------------------------------------------
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    
    ax.scatter(x=shear_rate,y=shear_stress,
               label="Measured viscometer data", color="red")
    
    ax.plot(shear_rate,shear_stress_calc_YPL,
            label="Yield Power-law model fit",color="blue")
    
    ax.plot(shear_rate,shear_stress_calc_PL,
            label="Power-law model fit",color="orange")
    
    ax.plot(shear_rate,shear_stress_calc_BP,
            label="Bingham Plastic model Fit",color="green")    
    
    ax.set_xlabel("Shear Rate (1/s)")
    ax.set_ylabel("Shear Stress (Pa)")
    ax.set_xlim(0,round(max(shear_rate)+40,0))
    ax.set_ylim(0,round(max(shear_stress)+10,0))
    ax.legend()
    
#Write the figure to streamlit-------------------------------------------------
    st.write(fig)
    
    #Deciding the best fit to the data---------------------------------------------
    if round(r2_BP,3) >= round(r2_PL,3) and round(r2_BP,3) >=round(r2_YPL,3):
        st.subheader("Bingham plastic (BP) model provides the best fit to the data.")
    
    elif round(r2_PL,3) >= round(r2_BP,3) and round(r2_PL,3) >= round(r2_YPL,3):
        st.subheader("Power law (PL) model provides the best fit to the data.")
    
    else:
        st.subheader("Yield power law (YPL) model provides the best fit to the data.")
    
    PV = DR600 - DR300
    YP = 2*DR300 - DR600
     
# initialize list of lists 
    data = [['Ty - YPL', ty_YPL], 
            ['K - YPL', K_YPL], 
            ['n - YPL', n_YPL],
            ['R2 - YPL', r2_YPL], 
            ['K - PL', K_PL], 
            ['n - PL', n_PL], 
            ['R2 - PL', r2_PL],
            ['PV', PV], 
            ['YP', YP], 
            ['R2 - BP', r2_BP],                 ] 
  
# Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Description', 'Number'])
    st.write(df)

    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(df, 'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
            
else:
    st.write("Please upload the data first")
    
st.write("Developer: Sercan Gul (sercan.gul@gmail.com, https://github.com/sercangul)")
st.write("Source code: https://github.com/sercangul/viscometerapi")
