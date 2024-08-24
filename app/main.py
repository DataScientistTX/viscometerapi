import streamlit as st
import numpy as np
import pandas as pd
from models import YPL, PL, BP
from utils import load_data, get_table_download_link, download_link
from visualizations import create_visualization

st.header("Drilling Fluid Rheological Model Parameters")
st.write("This web-app is used to analyze API rotational viscometer data by comparing various rheological models.")
st.write("The rheological constants for Yield Power-law (YPL - also called Herschel-Bulkley), Power-law, and Bingham-Plastic models are calculated and compared.")
st.write("Please upload the data using the file uploader on the left side. Please make sure that the data is in excel (.xlsx) format, where the first column is the RPM values and the second column is shear stress values (as viscometer dial readings) for each corresponding RPM.")
st.write("Below link can be used to download an example dataset for this web-app.")
st.write("NOTE: If you are using a 6-speed viscometer, you might be more interested in apiviscometer.herokuapp.com")
d = {'RPM': [300, 200, 100, 60, 30, 6, 3], 'Viscometer Dial Readings (DR)': [105.1, 90.8, 71.7, 63.4, 55.3, 45.8, 44]}
df_template = pd.DataFrame(data=d)
st.markdown(get_table_download_link(df_template), unsafe_allow_html=True)

st.sidebar.title("Upload data")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    df.columns = ["Viscometer RPM", "'Viscometer Dial Readings (DR)"]
    df = df.sort_values(by="Viscometer RPM", ascending=False)
    df = df.reset_index(drop=True)

    dial_readings = df["'Viscometer Dial Readings (DR)"]
    shear_rate = df["Viscometer RPM"] * 1.7011

    # Entries: if higher RPM value entered is lower than next one, make corrections
    for i in range(5):
        if dial_readings[i] < dial_readings[i + 1]:
            dial_readings[i] = dial_readings[i + 1]

    shear_stress = np.asarray(dial_readings) * 1.066 * 0.4788  # unit conversion from DR to Pascal

    ty_YPL, K_YPL, n_YPL, r2_YPL = YPL(shear_stress, shear_rate)
    K_PL, n_PL, r2_PL = PL(shear_stress, shear_rate)
    r2_BP, PV, YP, DR600, DR300 = BP(shear_stress, shear_rate)

    # Denoised values for visuals
    shear_stress_calc_YPL = YPLfunction(shear_rate, ty_YPL, K_YPL, n_YPL)
    shear_stress_calc_PL = PLfunction(shear_rate, K_PL, n_PL)
    shear_stress_calc_BP = BPfunction(shear_rate, PV, YP)

    st.subheader("Herschel Bulkley (Yield Power Law) Model Rheological Constants")
    st.write("Yield stress ($t_{y}$) is", round(ty_YPL, 2), "$Pa$")
    st.write("Consistency index (K) is", round(K_YPL, 4), "$Pa.s^{n}$")
    st.write("Flow index (n) is", round(n_YPL, 2))
    st.write("Coefficient of determination ($R^2$) is", round(r2_YPL, 3))

    st.subheader("Power-Law Model Rheological Constants")
    st.write("Consistency index (K) is", round(K_PL, 4), "$Pa.s^{n}$")
    st.write("Flow index (n) is", round(n_PL, 2))
    st.write("Coefficient of determination ($R^2$) is", round(r2_PL, 3))

    st.subheader("Bingham Plastic Model Rheological Constants")
    st.write("Plastic viscosity (PV) is", round(DR600 - DR300, 2), "$cp$")
    st.write("Yield point (YP) is", round(2 * DR300 - DR600, 2), "$lb/100ft^2$")
    st.write("Coefficient of determination ($R^2$) is", round(r2_BP, 3))

    fig = create_visualization(shear_rate, shear_stress, ty_YPL, K_YPL, n_YPL, K_PL, n_PL, PV, YP)
    st.write(fig)

    # Deciding the best fit to the data
    if round(r2_BP, 3) >= round(r2_PL, 3) and round(r2_BP, 3) >= round(r2_YPL, 3):
        st.subheader("Bingham plastic (BP) model provides the best fit to the data.")
    elif round(r2_PL, 3) >= round(r2_BP, 3) and round(r2_PL, 3) >= round(r2_YPL, 3):
        st.subheader("Power law (PL) model provides the best fit to the data.")
    else:
        st.subheader("Yield power law (YPL) model provides the best fit to the data.")

    PV = DR600 - DR300
    YP = 2 * DR300 - DR600

    data = [['Ty - YPL', ty_YPL],
            ['K - YPL', K_YPL],
            ['n - YPL', n_YPL],
            ['R2 - YPL', r2_YPL],
            ['K - PL', K_PL],
            ['n - PL', n_PL],
            ['R2 - PL', r2_PL],
            ['PV', PV],
            ['YP', YP],
            ['R2 - BP', r2_BP]]

    df = pd.DataFrame(data, columns=['Description', 'Number'])
    st.write(df)

    if st.button('Download Dataframe as CSV'):
        tmp_download_link = download_link(df, 'YOUR_DF.csv', 'Click here to download your data!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

else:
    st.write("Please upload the data first")

st.write("Developer: Sercan Gul (sercan.gul@gmail.com, https://github.com/sercangul)")
st.write("Source code: https://github.com/sercangul/viscometerapi")