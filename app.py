# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 01:50:49 2020

@author: Admin
"""
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

def main():
    
    def YPLfunction(y, tauy, K, m):
        return tauy + K*y**m
    
    def PLfunction(y, K2, n):
        return  K2*y**n

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


    def rheology_PL(sigma,shearrate):
        shearstress = np.asarray(sigma) * 1.066 * 0.4788 #unit conversion   
        popt, pcov = curve_fit(PLfunction,shearrate,shearstress)
        K,m =popt[0],popt[1]
        residuals = shearstress- PLfunction(shearrate, popt[0],popt[1])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((shearstress-np.mean(shearstress))**2)
        r_squared = 1 - (ss_res / ss_tot)       
        
        return K,m,r_squared

    def rheology_YPL(sigma,shearrate):
        tauy =[]
        K = []
        m = []
        
        #Trying the fit for YPL model
        shearstress = np.asarray(sigma) * 1.066 * 0.4788 #unit conversion         
        popt, pcov = curve_fit(YPLfunction,shearrate,shearstress)
        tauy,K,m = popt[0],popt[1],popt[2]
        residuals = shearstress- YPLfunction(shearrate, popt[0],popt[1],popt[2])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((shearstress-np.mean(shearstress))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if tauy<0:
            K,m,r_squared = rheology_PL(sigma,shearrate)
            tauy = 0
        return tauy,K,m,r_squared
      
        
    def BPr2(stressmeasured,stresscalculated,shearrate):
        residuals = stressmeasured- stresscalculated
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((stressmeasured-np.mean(stressmeasured))**2)
        r_squared = 1 - (ss_res / ss_tot)   
        return r_squared
  
    st.header("Drilling Fluid Rheological Model Parameters")
    st.write("This web-app is used to analyze API rotational viscometer data by comparing various rheological models.")
    st.write("The rheological constants for Yield Power-law (YPL - also called Herschel-Bulkley), Power-law, and Bingham-Plastic models are calculated and compared.")
    st.write("Please upload the data using the file uploader on the left side. Please make sure that the data is in excel (.xlsx) format, where the first column is the RPM values and the second column is shear stress values (as viscometer dial readings) for each corresponding RPM.")
    st.write("Below link can be used to download an example dataset for this web-app.")
    
    d = {'RPM': [300,200,100,60,30,6,3], 'Shear Stress (DR)': [105.1,90.8,71.7,63.4,55.3,45.8,44]}
    df_template = pd.DataFrame(data=d)
    
    st.markdown(get_table_download_link(df_template), unsafe_allow_html=True)


    @st.cache(allow_output_mutation=True)
    def load_data(file):
        df = pd.read_excel(file)
        return df
        
    st.sidebar.title("Upload data")    
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df.columns = ["Viscometer RPM","Shear Stress (DR)"]
        df = df.sort_values(by="Viscometer RPM", ascending=False)
        df = df.reset_index(drop=True)
        
        sigma = df["Shear Stress (DR)"]
        shearrate = df["Viscometer RPM"]*1.7
        df["Shear Rate (1/s)"] = round(shearrate,1)
        shearstress = np.asarray(sigma) * 1.066 * 0.4788 #unit conversion    
        df["Shear Stress (Pa)"] = shearstress
        df = df.round(1)
        st.dataframe(df.style.format("{:.1f}"))

        slope = (shearstress[0] - shearstress[1])/(shearrate[0]-shearrate[1])
        intercept = shearstress[1] - slope*shearrate[1]
        
        tauy_YPL,K_YPL,m_YPL,r2_YPL = rheology_YPL(sigma,shearrate)
        K_PL,m_PL,r2_PL = rheology_PL(sigma,shearrate)

        sigmacalcBP = []
        for i in shearrate:
            calculation = intercept + slope*i
            sigmacalcBP.append(calculation)
        r2_BP = BPr2(shearstress,sigmacalcBP,shearrate)
    
        sigmacalcYPL = tauy_YPL + K_YPL*shearrate**m_YPL
        sigmacalcPL = K_PL*shearrate**m_PL
    
        st.subheader ("Herschel Bulkley (Yield Power Law) Model Rheological Constants")
        st.write("Yield stress ($t_{y}$) is", round(tauy_YPL,2), "$Pa$")
        st.write("Consistency index (K) is", round(K_YPL,4), "$Pa.s^{n}$")
        st.write("Flow index (n) is", round(m_YPL,2))
        st.write("Coefficient of determination ($R^2$) is", round(r2_YPL,3))
         
        st.subheader ("Power-Law Model Rheological Constants")
        st.write("Consistency index (K) is", round(K_PL,4), "$Pa.s^{n}$")
        st.write("Flow index (n) is", round(m_PL,2))
        st.write("Coefficient of determination ($R^2$) is", round(r2_PL,3))
        
        #Calculate equivalent sigma600 (DR)
        sigma600 = calculation = intercept + slope*600*1.7

        #Calculate equivalent sigma300 (DR)
        sigma300 = calculation = intercept + slope*300*1.7
                
        st.subheader ("Bingham Plastic Model Rheological Constants")
        st.write("Plastic viscosity (PV) is", round(sigma600-sigma300,2) , "$cp$") 
        st.write("Yield point (YP) is", round(2*sigma300-sigma600,2), "$lb/100ft^2$")
        st.write("Coefficient of determination ($R^2$) is", round(r2_BP,3))
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        
        ax.scatter(x=shearrate,y=shearstress,label="Measured viscometer data", color="red")
    
        ax.plot(shearrate,sigmacalcYPL,label="Yield Power-law model fit",color="blue")
        ax.plot(shearrate,sigmacalcPL,label="Power-law model fit",color="orange")
        ax.plot(shearrate,sigmacalcBP,label="Bingham Plastic model Fit",color="green")    
    
        ax.set_xlabel("Shear Rate (1/s)")
        ax.set_ylabel("Shear Stress (Pa)")
        ax.set_xlim(0,round(max(shearrate)+10,0))
        ax.set_ylim(0,round(max(shearstress)+10,0))
        ax.legend()
        st.write(fig)
        
        if r2_YPL > r2_BP and r2_YPL>r2_PL:
            st.subheader("Yield power law (YPL) model provides the best fit to the data.")
            
        if r2_PL >= r2_BP and r2_PL>=r2_YPL:
            st.subheader("Power law (PL) model provides the best fit to the data.")
        
        if r2_BP >= r2_PL and r2_BP>=r2_YPL:
            st.subheader("Bingham plastic (BP) model provides the best fit to the data.")
        
    else:
        st.write("Please upload the data first")
    st.write("Developer: Sercan Gul (sercan.gul@gmail.com, https://github.com/sercangul)")
    st.write("Source code: https://github.com/sercangul/viscometer_streamlitapp")
if __name__ == "__main__":
    main()