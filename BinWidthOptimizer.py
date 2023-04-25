import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Generate random non-normal data
# data = np.random.gamma(3, 2, size=1000)
# data = np.random.normal(3, 1, size=500)

# Use real data
data = np.genfromtxt('PugetSoundButterClams_LengthWidthRatio.csv')

# Set the title of the figure
data_title = 'Puget Sound Butter Clams Length/Width Ratio (n=88) with Gaussian Fit'

# Set the number of histograms
num_histograms = 16

# Set the initial number of bins
initial_bins = 5
bin_delta = 3

# Set the number of rows and columns
rows = 5
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.5]})

def optimal_bins_freedman_diaconis(data):
    n = len(data)
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * IQR * (n ** (-1/3))
    num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))

    return num_bins

def gaussian_fit_and_plot(ax, data, num_bins):
    n, bins, patches = ax.hist(data, bins=num_bins, density=True, alpha=0.6)
    mu, std = norm.fit(data)

    x = np.linspace(min(bins), max(bins), 1000)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)

    # Calculate R-squared value
    y = norm.pdf(bins[:-1] + np.diff(bins)/2, mu, std)
    r_squared = r2_score(n, y)

    return r_squared

r_squared_values = np.zeros((rows-1, cols))
bin_numbers = np.zeros((rows-1, cols))

for i in range(rows-1):
    for j in range(cols):
        index = i * cols + j
        num_bins = initial_bins + index * bin_delta
        r_squared_values[i, j] = gaussian_fit_and_plot(axes[i, j], data, num_bins)
        bin_numbers[i, j] = num_bins
        axes[i, j].set_title(f'{num_bins} bins, R\u00B2: {r_squared_values[i, j]:.4f}', fontsize=8)

# Find the position of the highest R-squared value
best_fit_pos = np.unravel_index(np.argmax(r_squared_values, axis=None), r_squared_values.shape)

# Highlight the graph with the highest R-squared value by changing its title color
axes[best_fit_pos].set_title(f'{initial_bins + np.ravel_multi_index(best_fit_pos, (rows-1, cols)) * 3} bins, R\u00B2: {r_squared_values[best_fit_pos]:.4f}', color='red', fontsize=10)

# Add some space between the histograms
plt.tight_layout()

# Scatterplot of R-squared values vs bin numbers
scatter_ax = axes[-1, :]
scatter_ax = plt.subplot2grid((rows, cols), (4, 0), colspan=cols, rowspan=1)
scatter_ax.scatter(bin_numbers, r_squared_values, label='R\u00B2 values')
scatter_ax.set_xlabel('Number of bins')
scatter_ax.set_ylabel('R\u00B2 value')
scatter_ax.axvline(x=optimal_bins_freedman_diaconis(data), color='blue', linewidth=2, label='FD predicted optimal')

# Fit a curve to the scatterplot data
def fit_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Use initial parameter guesses to avoid overflows
initial_guess = [1, 0.01, 0]

popt, _ = curve_fit(fit_func, bin_numbers.flatten(), r_squared_values.flatten(), p0=initial_guess)

x = np.linspace(bin_numbers.min(), bin_numbers.max(), 100)
y = fit_func(x, *popt)
scatter_ax.plot(x, y, 'r', label='Fitted curve')

scatter_ax.set_title('R-squared values vs Bin numbers')
scatter_ax.legend()

# Set the overall title for the whole figure
optimized_FD = optimal_bins_freedman_diaconis(data)
fig.suptitle(data_title, fontsize=15, y=0.995)

# Adjust the spacing and layout
plt.subplots_adjust(hspace=0.5, wspace=1, top=0.9, bottom=0.1)
plt.tight_layout()


# Display the histograms and scatterplot in the same window
plt.show()