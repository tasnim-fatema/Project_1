# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
df = pd.read_csv("data/housing.csv")

print (df.columns)

x_columns = ['longitude', 'latitude', 'housing_median_age',
             'total_rooms', 'population', 'households', 'median_income',
             'ocean_proximity'
             ]

y_coulmn = ['median_household_value']
