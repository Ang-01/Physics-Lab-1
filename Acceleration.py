import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def line(x,a,b):
    return a*x+b
#Opening up the files
#Make sure in the paranthesese, you put the names of your csv files.
acc_1 = pd.read_csv("acc1.csv")
acc_2 = pd.read_csv("acc2.csv")
acc_3 = pd.read_csv("acc3.csv")
acc_4 = pd.read_csv("acc4.csv")
acc_5 = pd.read_csv("acc5.csv")
print(acc_1)
print(acc_1['ax'])
dataBank = [acc_1,acc_2,acc_3,acc_4,acc_5]
#Building the lists that contain slope of track (x axis), acceleration (y axis), and error on acceleration (y_err)
x=[0.001/1.5,0.002/1.5,0.003/1.5,0.004/1.5,0.005/1.5]
y=[]
y_err=[]
for i in dataBank:
    y+=[float(pd.DataFrame(i["A"]).mean())]
    y_err+=[float(pd.DataFrame(i["A"]).sem())]

#Graphing the data points with errors
plt.xlim(xmin=0,xmax=0.006/1.5)
plt.ylim(ymin=0,ymax=0.12/1.5)
plt.errorbar(x,y,yerr=y_err,fmt='o',ecolor='blue',color='purple',capsize=5)
#Fitting the line of best fit to the plot
popt, pcov = curve_fit(line,x,y, sigma=y_err)
print("m =", popt[0], "+/-", pcov[0,0]**0.5)
print("b =", popt[1], "+/-", pcov[1,1]**0.5)
#Showing the line of best fit
xfine = np.linspace(0,0.005/1.5,6)
plt.plot(xfine, line(xfine, popt[0], popt[1]), color="red")

#Estimating loss due to Friction
def friction(v,a,a_err,l):
    return [(v**2)/(2*a*l)-1,((v**2)/(2*a*l)-1)*a_err/a]

# for i in dataBank:
#     print(friction(i['v1'],i['ax'],i['Oax'],i['l2']))