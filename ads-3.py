
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.cluster import KMeans


# Read the dataset
dt = pd.read_csv('Electricity.csv')
data = dt.tail(20)
electricity = data['Electricity from hydro (TWh)'].values
e1 = data['Electricity from solar (TWh)'].values
countries = data['Entity'].values

# Define the error range function
def err_range(p, x, cov):
    return p[0]*x + p[1] + np.sqrt(np.diag(cov))

# Perform clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(data[['Electricity from hydro (TWh)', 'Electricity from solar (TWh)']])

# Plot the clusters
plt.figure(figsize=(10, 5))
plt.scatter( e1,electricity,c=kmeans.labels_)
plt.xlabel('Electricity from solar energy')
plt.ylabel('Electricity from hydro energy')
plt.title('Clusters of electricity produced from different sources',fontsize=15)
plt.legend(bbox_to_anchor=(0.5, 1.05), loc='left', ncol=3)
plt.show()


# Read the dataset
dt = pd.read_csv('Electricity.csv')
data1 = dt.head(30)

# Define the fitting function
def fit_func(x, a, b):
    return a*x + b

x = data1['Year'].values
y = data1['Electricity from wind (TWh)'].values

# Perform the fit
popt, pcov = curve_fit(fit_func, x, y)

# Calculate the error ranges
err = sem(y)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=err, fmt='o', label='Data')
x_fit = np.linspace(min(x), max(x), 100)
plt.plot(x_fit, fit_func(x_fit, *popt), label='Fitted line')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [err_range(popt, x[i], pcov)[0], y[i]], 'k--')
plt.xlabel('Year')
plt.ylabel('Electricity produced from wind energy')
plt.title('Electricity produced by wind energy from 1985-2020',fontsize=15)
plt.legend()



# Define the error range function
def err_range(p, x, cov):
    return p[0]*x + p[1] + np.sqrt(np.diag(cov))

# Read the dataset
dt = pd.read_csv('Electricity.csv')
data2 = dt.head(30)
x = data2['Year'].values
y = data2['Electricity from nuclear (TWh)'].values

# Perform the fit
popt, pcov = curve_fit(fit_func, x, y)

# Calculate the error ranges
err = sem(y)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=err, fmt='o', label='Data',color='red')
x_fit = np.linspace(min(x), max(x), 100)
plt.plot(x_fit, fit_func(x_fit, *popt), label='Fitted line',color = 'green')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [err_range(popt, x[i], pcov)[0], y[i]], 'k--')
plt.xlabel('Year',fontsize=12)
plt.ylabel('Electricity produced from nuclear energy',fontsize=12)
plt.title('Electricity produced by nuclear energy from 1985-2020',fontsize=15)
plt.legend()
