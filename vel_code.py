
# -*- coding: utf-8 -*-
"""
Spyder Editor


This is a temporary script file.
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sympy import symbols, diff 
from tabulate import tabulate 

vel_data = pd.read_csv("new_vel.csv")

#Putting the necessary data into variables
initial_vel = vel_data["Vi"]
final_vel = vel_data["Vf"]
sigma_vi = vel_data["OVi"]
sigma_vf = vel_data["OVf"]
# Equation needed in order to calculate partial derivatives
x, y = symbols('x y', real = True) 
F = x/y

# This is where we calculate for the coeffcient of Restitution
# there are 10 values for the 10 trials we did
ej = abs(final_vel/initial_vel)



# Here we are calculating for the partial derivative of the equation
Fx_list = []
Fy_list = []
for trial in range(len(initial_vel)):

    Fx = F.diff(x)
    #partial = Fx(final_vel,initial_vel)
    partial_x = Fx.evalf(subs = {x:list(final_vel)[trial], y:list(initial_vel)[trial]})
    #New_Fx = Fx.evalf(subs = {a:final_vel, b:initial_vel})
    Fx_list.append(partial_x)
    
    Fy = F.diff(y)
    #partial = Fx(final_vel,initial_vel)
    partial_y = Fy.evalf(subs = {x:list(final_vel)[trial], y:list(initial_vel)[trial]})
    #New_Fx = Fx.evalf(subs = {a:final_vel, b:initial_vel})
    Fy_list.append(partial_y)
  
# Calculating the unertainty by propogating the errors sigma vi and sigma vf by using
# the partial derivatives made and their calculations
sigma_ej_list = []
for trial in range(len(initial_vel)):
    sigma_ej = math.sqrt((Fx_list[trial]**2)*(sigma_vi[trial]**2)+(Fy_list[trial]**2)*(sigma_vf[trial]**2))
    sigma_ej_list.append(sigma_ej)
    
# Calculating the unweighted mean
unweighted_mean = sum(ej)/len(ej)

# Calculating the sigma of unweighted mean
for trial in range(len(ej)):
    sigma_uwm = math.sqrt((sum((ej-unweighted_mean)**2))/(len(ej)-1))
 # Another way to calculate the sigma of unweighted mean  
#sum_ej_uwm = sum(ej-unweighted_mean)**2
#sigma_uwm = math.sqrt((sum_ej_uwm)/(len(ej)-1))

# Calculating the standard error on the mean
std_error = (sigma_uwm)/math.sqrt(len(ej))
std=pd.DataFrame(ej).sem()

#print(std)
#print(std_error)


# Now we are going to calculate the weighted mean (ew)
list =[]
#for trial in range(len(ej)):
   # ew = ej[trial]
   # list.append(ew)
   # print(ew)


fir = sum((ej/(np.array(sigma_ej_list))**2))
ran = sum(1/(np.array(sigma_ej_list))**2)
weighted_mean = fir/ran

def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

def w_avg_e(values, weights):
    d = values
    w = weights
    return (d * w).sum() / w.sum()

w_m = w_avg_e(ej,1/(np.array(sigma_ej_list))**2)
print(weighted_mean)
print(w_m)

# Calculating standard error of weighted mean
new_sigma_wm = (sum(1/(np.array(sigma_ej_list))**2))**(-1/2)


# This is the summary of all the stats that were calculated
print(tabulate({'Statistic for Coefficient of Restitution': ['Unweighted Mean', 'Weighted Mean'], 'Values': [unweighted_mean, w_m], 'Standard Error': [std,new_sigma_wm]}, headers="keys", tablefmt='fancy_grid'))
print(tabulate({'Statistic for Coefficient of Restitution': ['Unweighted Mean', 'Weighted Mean'], 'Values': [str(round(unweighted_mean,3)) + u" \u00B1 " + str(round(std[0],3)), str(round(w_m,3))+u" \u00B1 "+ str(round(new_sigma_wm,3))]}, headers="keys", tablefmt='fancy_grid'))

# This is to round to the amount of sig figs needed
print(round(std,3))


# Plotting e against Vi
print(ej)
print(initial_vel)

plt.plot(initial_vel, ej, color='green', linestyle='solid', linewidth = 1,
         marker='o', markerfacecolor='blue', markersize=12)
 









