import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from tabulate import tabulate 

def line(x,a,b):
    return a*x+b
#Opening up the files
#Make sure in the paranthesese, you put the names of YOUR csv files.
acc_1 = pd.read_csv("acc1.csv")
acc_2 = pd.read_csv("acc2.csv")
acc_3 = pd.read_csv("acc3.csv")
acc_4 = pd.read_csv("acc4.csv")
acc_5 = pd.read_csv("acc5.csv")
# print(acc_1)
# print(acc_1['ax'])
dataBank = [acc_1,acc_2,acc_3,acc_4,acc_5]
#Building the lists that contain slope of track (x axis), acceleration (y axis), and error on acceleration (y_err)
x=[0.001/1.5,0.002/1.5,0.003/1.5,0.004/1.5,0.005/1.5]
y=[]
y_err=[]
for i in dataBank:
    y+=[float(pd.DataFrame(i["A"]).mean())]
    # i["A"], i refers to the name of the file, "A" is the name of the column in our data, that contains the A values.
    y_err+=[float(pd.DataFrame(i["A"]).sem())]

#Graphing the data points with errors
plt.xlim(xmin=0,xmax=0.006/1.5)
plt.ylim(ymin=0,ymax=0.06/1.5)
plt.errorbar(x,y,yerr=y_err,fmt='o',ecolor='blue',color='purple',capsize=5)
#Fitting the line of best fit to the plot
popt, pcov = curve_fit(line,x,y, sigma=y_err)
#Summary of results for slope (g) and y-intercept
print("g =", popt[0], "+/-", pcov[0,0]**0.5)
print("intercept =", popt[1], "+/-", pcov[1,1]**0.5)
#Showing the line of best fit
xfine = np.linspace(0,0.006/1.5,6)
plt.title("Best Fit Curve of Acceleration with respect to sin(θ)")
plt.xlabel("sin(θ)")
plt.ylabel("Acceleration (m/s)")
plt.plot(xfine, line(xfine, popt[0], popt[1]), color="red")

#Estimating loss due to Friction
def friction(v,a,a_err,l):
    return (v**2)/(2*a*l)-1,abs(((v**2)/(2*a*l)-1)*a_err/a)
fricdata=[]
for i in dataBank:
      fricdata+=[friction(i['v1'],i['ax'],i['Oax'],i['l2'])]
# The line below will print a huge table of friction losses and errors. To see it, uncomment it, and set font size to 7, and make the terminal bigger.      
# print(tabulate({'Trial Number': [i for i in range(1,11)], '1 shim': list(fricdata[0][0]), 'Error on 1 shim': list(fricdata[0][1]), '2 shim': list(fricdata[1][0]), 'Error on 2 shim': list(fricdata[1][1]), '3 shim': list(fricdata[2][0]), 'Error on 3 shim': list(fricdata[2][1]), '4 shim': list(fricdata[3][0]), 'Error on 4 shim': list(fricdata[3][1]), '5 shim': list(fricdata[4][0]), 'Error on 5 shim': list(fricdata[4][1])}, headers="keys", tablefmt='fancy_grid'))
