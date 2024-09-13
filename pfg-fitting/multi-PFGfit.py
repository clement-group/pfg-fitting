# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:31:53 2024


"""
#After fitting PFG data in the Topspin save it as .txt file.
#For any other .txt file, should provide n1 and n2 values in the main function
#It has option for method and model choice to fit the data
#One can use weigted average for the data, True or False option
#color for model or method controlled by plot option
#Intensity weight is preffered
#10-4 is multiplied to convert gauss/cm to T/m
#D and d are big and small delta in seconds
#v is gyromagentic ratio


import os
import re
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from scipy.integrate import quad
from scipy.optimize import least_squares, dual_annealing
from scipy.odr import ODR, Model as ODRModel, RealData

try:
    from pyswarm import pso
except ImportError:
    pso = None

# Model functions
def integrand(q, x, Din, Dout):
    DQ = Din * np.sin(q)**2 + Dout * np.cos(q)**2
    return np.exp(-DQ * x) * np.sin(q)

def model_integral(x, y0, Din, Dout):
    integral_values = np.array([quad(integrand, 0, np.pi/2, args=(xi, Din, Dout))[0] for xi in x])
    return y0 * integral_values

def model_integral_Dout0(x, y0, Din):
    return model_integral(x, y0, Din, 0)

def model_3D_monoexp(x, y0, D):
    return y0 * np.exp(-x * D)

def model_3D_biexp(x, y0, D1, D2, f):
    return y0 * (f * np.exp(-x * D1) + (1 - f) * np.exp(-x * D2))

def model_3D_stretchexp(x, y0, D, b):
    return y0 * np.exp(-(x * D)**b)

def model_combo(x, y0, Din, Dout, D1, D2, f1, f2):
    return y0 * np.array([(1-f1-f2) * quad(integrand, 0, np.pi/2, args=(xi, Din, Dout))[0]
            + f1 * np.exp(-xi * D1) + f2 * np.exp(-xi * D2) for xi in x])

# Data handling functions

def PFG_data_extract(data_dir, exp_no, nucleus):
    #Inputs:
        # exp_no (string): experiment no. to extract data (measurement folder specified in data_dir above)
        # nucleus (string): identity of observed nucleus (1H, 7Li, 19F are accepted here)

    #Returns:
        # diff_decay (dataframe): dataframe with 3 columns ('grad_strength [G/cm]','diff_ints','norm_diff_ints')
        # gamma (float): gyromagnetic ratio of observed nucleus [given in Hz/T]
        # delta (float): little delta time (duration of gradient pulse) [ms]
        # DELTA (float): big delta time (diffusion time) [ms]

    #Extract dataframe of decay curve from folder
    exp_path = os.path.join(data_dir, str(exp_no))

    #Find t1ints.txt file
    t1_ints = pd.read_csv(exp_path + '/pdata/1/t1ints.txt', names=['no1', 'ints', 'no3'], delim_whitespace=True,skiprows=1)

    #Extract integral values from t1ints.txt file
    diff_ints = t1_ints[t1_ints['ints'] != 0]['ints']
    diff_ints = diff_ints.reset_index(drop=True) #Reset df index so that the values now go from 0 to number of pts
    norm_diff_ints = diff_ints/(diff_ints.max())

    #Extract gradient strengths from difflist file
    grad_strength = pd.read_csv(exp_path +'/difflist',names=['grad_strength'],header=None)  #No header as first line is first point, not header

    #Create diff_decay df with
    diff_decay = pd.concat([grad_strength, diff_ints, norm_diff_ints], axis=1)
    diff_decay.columns = ['grad_strength', 'diff_ints', 'norm_diff_ints']
    diff_decay = diff_decay.dropna()


    #Obtain delta and DELTA values from diff.xml
    with open(exp_path + '/diff.xml') as myfile:
        content = myfile.read()
    delta = re.search('<delta>(.*)</delta>', content).group(1)
    delta = float(delta)                                        # Convert to a float from a string

    DELTA = re.search('<DELTA>(.*)</DELTA>', content).group(1)
    DELTA = float(DELTA)                                        # Convert to a float from a string

    #Define gamma according to observed nucleus
    if nucleus == '7Li':
        # gamma = 16.546e6 # MHz/T
        gamma = 10.39677e7   # gyromagnetic ratio in rad⋅s−1⋅T−1
    # elif nucleus == '1H':
    #     gamma = 42.58e6
    # elif nucleus == '19F':
    #     gamma = 40.08e6
    else:
        raise Exception("Sorry, the gamma of this nucleus is not included in this")

    # return diff_decay, gamma, delta, DELTA
    v = gamma
    d = delta * 1e-3 # change from ms to s
    D = DELTA * 1e-3 # change from ms to s

    x_data = (diff_decay['grad_strength']**2)*(v ** 2) * (d ** 2) * (D - d / 3) * 1e-4
    y_data = diff_decay['norm_diff_ints']

    mask = np.array(~(np.isnan(x_data) | np.isnan(y_data)))
    x_data = x_data[mask].values
    y_data = y_data[mask].values

    print(f'x_data is {x_data.shape}')

    if len(x_data) == 0 or len(y_data) == 0:
        print(f"Error: No valid numeric data found in columns of {exp_path}")
        return None, None

    print(f"Successfully read {len(x_data)} data points from columns of {exp_path}")
    return x_data, y_data


def read_data(file_path, n1=0, n2=1, v=None, d=None, D=None):
    try:
        # Determine file type
        _, file_extension = os.path.splitext(file_path)
        
        # Read the file based on its extension
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path, header=None)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=None)
        elif file_extension.lower() == '.txt':
            # First, try to read as a structured text file
            with open(file_path, 'r') as file:
                content = file.read()
            data_section = re.search(r'Point\s+Gradient\s+Expt\s+Calc\s+Difference\n([\s\S]+)', content)

            if data_section:
                data_lines = data_section.group(1).strip().split('\n')
                data = [line.split() for line in data_lines]
                df = pd.DataFrame(data, columns=['Point', 'Gradient', 'Expt', 'Calc', 'Difference'])
                n1 = df.columns.get_loc('Gradient')
                n2 = df.columns.get_loc('Expt')
            else:
                # If structured format not found, read as space-separated
                df = pd.read_csv(file_path, sep=r'\s+', header=None)
        else:
            print(f"Unsupported file format: {file_extension}")
            return None, None

        # Check if the file has enough columns
        if df.shape[1] <= max(n1, n2):
            print(f"Error: File {file_path} does not have enough columns. It has {df.shape[1]} column(s), but n1={n1} and n2={n2} were requested.")
            return None, None

        # Extract data from specified columns
        column_n1 = pd.to_numeric(df.iloc[:, n1], errors='coerce')
        y_data = pd.to_numeric(df.iloc[:, n2], errors='coerce')

        # Calculate x_data as per the new formula
        if v is not None and d is not None and D is not None:
            x_data = (column_n1**2)*(v ** 2) * (d ** 2) * (D - d / 3)*1E-4
        else:
            x_data = column_n1  # Fallback if any parameters are not provided

        # Remove any rows with NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask].values
        y_data = y_data[mask].values

        if len(x_data) == 0 or len(y_data) == 0:
            print(f"Error: No valid numeric data found in columns {n1} and {n2} of {file_path}")
            return None, None

        print(f"Successfully read {len(x_data)} data points from columns {n1} and {n2} of {file_path}")
        return x_data, y_data
    
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None, None



def save_results(output_file_path, file_name, x_data, y_data, fits, model_names, method_used, r_squared_values, D_values, errors):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        headers = ["X", "Experimental Data"] + model_names
        f.write("\t".join(headers) + "\n")
        for i in range(len(x_data)):
            line = [f"{x_data[i]:.6E}", f"{y_data[i]:.6E}"] + [f"{fit[i]:.6E}" for fit in fits]
            f.write("\t".join(line) + "\n")
        f.write("\nSummary:\n")
        for model, method, r2, D, error in zip(model_names, method_used, r_squared_values, D_values, errors):
            f.write(f"Model: {model} ({method})\n")
            f.write(f"File Name: {file_name}\n")
            f.write("Optimized parameters:\n")
            for param, value in D.items():
                err = error[param].stderr
                if isinstance(value, (int, float)) and isinstance(err, (int, float)):
                    f.write(f"{param}: {value:.4E} ± {err:.4E}\n")
                else:
                    f.write(f"{param}: {value} ± {err}\n")
            f.write(f"R² = {r2:.4g}\n")
            f.write("\n")

# Optimization helper functions
def pso_wrapper(func, bounds, args):
    def wrapper(x):
        return np.sum((func(args[0], *x) - args[1])**2)
    lb, ub = zip(*bounds)
    xopt, _ = pso(wrapper, lb, ub)
    return xopt

def sa_wrapper(func, bounds, args):
    def wrapper(x):
        return np.sum((func(args[0], *x) - args[1])**2)
    result = dual_annealing(wrapper, bounds=bounds)
    return result.x

def odr_wrapper(func, params, x_data, y_data):
    def model_func(beta, x):
        return func(x, *beta)
    model = ODRModel(model_func)
    data = RealData(x_data, y_data)
    odr = ODR(data, model, beta0=[params[p].value for p in params])
    output = odr.run()
    return output.beta

def trf_wrapper(func, bounds, args):
    def wrapper(x, *args):
        return func(args[0], *x) - args[1]
    lb, ub = zip(*bounds)
    result = least_squares(wrapper, x0=[(l+u)/2 for l,u in zip(lb, ub)], bounds=(lb, ub), args=args, method='trf')
    return result.x


# Fitting and plotting function
def fit_and_plot(x_data, y_data, chosen_models, chosen_methods, plot_options, file_name=None, use_inverse_variance_weighting=False, use_squared_intensity_weighting=False, use_weighted_least_squares=False):
    fits, model_names, r_squared_values = [], [], []

    fig, ax = plt.subplots(figsize=(plot_options['fig_width'], plot_options['fig_height']))

    models = {
        # model_name, model_func, line_style, param_names, transparency
        1: ("2D model (Dout=0)", model_integral_Dout0, '--', ['Din'], 0.7),
        2: ("2D model (Dout ≠ 0)", model_integral, '--', ['Din', 'Dout'], 0.5),
        3: ("3D Monoexponential", model_3D_monoexp, '--', ['D'], 0.9),
        4: ("3D Biexponential", model_3D_biexp, '--', ['D1', 'D2', 'f'], 0.8),
        5: ("3D Stretched Exponential", model_3D_stretchexp, '--', ['D', 'b'], 0.6),
        6: ("2D + 3D Biexponential", model_combo, '--', ['Din', 'Dout', 'D1', 'D2', 'f1', 'f2'], 0.7)
    }
    method_used = []
    D_values = []
    errors = []

    # Allow customization of model and method colors
    model_colors = plot_options.get('model_colors', {1: 'red', 2: 'blue', 3: 'yellow', 4: 'green', 5: 'magenta', 6: 'purple'})
    method_colors = plot_options.get('method_colors', {'de': 'blue', 'pso': 'green', 'sa': 'red', 'odr': 'purple', 'trf': 'orange'})
    use_model_color = plot_options.get('model_color', True)  # Default to True if not specified

    method_transparency = {
        'de': 0.6, 'pso': 0.7, 'sa': 0.6, 'odr': 0.6, 'trf': 0.5
    }

    weights = None

    if use_inverse_variance_weighting:
        # Estimate variance based on y-values
        # Assume a constant relative error (e.g., 5%)
        relative_error = 0.03
        estimated_variance = (relative_error * y_data)**2

        epsilon = 1e-16  # Small constant to avoid division by zero
        weights = 1 / (estimated_variance + epsilon)

        print("Using estimated inverse variance weighting based on constant relative error.")

    elif use_squared_intensity_weighting:
        weights = y_data**2
    elif use_weighted_least_squares:
        weights = 1 / y_data
    else:
        weights = None

    for model_num in chosen_models:
        model_name, model_func, line_style, param_names, transparency = models[model_num]
        for method in chosen_methods:
            print(f"\nFitting {model_name} using {method}...")
            lmfit_model = Model(model_func)
            params = setup_parameters(model_num)

            label = ""  # Initialize label variable

            try:
                if method == 'de':
                    result = lmfit_model.fit(y_data, params, x=x_data, method='differential_evolution', weights=weights)
                elif method == 'pso':
                    if pso:
                        bounds = [(p.min, p.max) for p in params.values()]
                        pso_result = pso_wrapper(lmfit_model.func, bounds, (x_data, y_data))
                        for i, p in enumerate(params):
                            params[p].value = pso_result[i]
                        result = lmfit_model.fit(y_data, params, x=x_data, weights=weights)
                    else:
                        raise ModuleNotFoundError("pyswarm.pso not found")
                elif method == 'sa':
                    bounds = [(p.min, p.max) for p in params.values()]
                    sa_result = sa_wrapper(lmfit_model.func, bounds, (x_data, y_data))
                    for i, p in enumerate(params):
                        params[p].value = sa_result[i]
                    result = lmfit_model.fit(y_data, params, x=x_data, weights=weights)
                elif method == 'odr':
                    odr_result = odr_wrapper(lmfit_model.func, params, x_data, y_data)
                    for i, p in enumerate(params):
                        params[p].value = odr_result[i]
                    result = lmfit_model.fit(y_data, params, x=x_data, weights=weights)
                elif method == 'trf':
                    bounds = [(p.min, p.max) for p in params.values()]
                    trf_result = trf_wrapper(lmfit_model.func, bounds, (x_data, y_data))
                    for i, p in enumerate(params):
                        params[p].value = trf_result[i]
                    result = lmfit_model.fit(y_data, params, x=x_data, weights=weights)
                else:
                    raise ValueError("Invalid optimization method")

                result = lmfit_model.fit(y_data, result.params, x=x_data, method='leastsq', weights=weights)
                x_fit = np.linspace(min(x_data), max(x_data), 1000)
                y_fit = result.eval(x=x_fit)

                # Calculate R-squared
                y_pred = result.eval(x=x_data)
                if weights is not None:
                    # Weighted R-squared calculation
                    y_mean = np.average(y_data, weights=weights)
                    total_sum_of_squares = np.sum(weights * (y_data - y_mean)**2)
                    residual_sum_of_squares = np.sum(weights * (y_data - y_pred)**2)
                    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
                else:
                    # Unweighted R-squared calculation
                    y_mean = np.mean(y_data)
                    total_sum_of_squares = np.sum((y_data - y_mean)**2)
                    residual_sum_of_squares = np.sum((y_data - y_pred)**2)
                    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

                # Calculate adjusted R-squared
                n = len(y_data)
                p = len(result.var_names)
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

                print(f"R-squared: {r_squared:.4g}")
                print(f"Adjusted R-squared: {adjusted_r_squared:.4g}")

                if r_squared < 0 or adjusted_r_squared < 0:
                    print("Warning: Negative R-squared or Adjusted R-squared detected.")

                # Calculate sum of squared residuals
                sum_squared_residuals = np.sum((y_data - y_pred)**2)
                print(f"Sum of Squared Residuals: {sum_squared_residuals:.6f}")

                label = f'{model_name} ({method}), ' + ', '.join([f'{p} = {result.best_values[p]:.2E}' for p in param_names if p.startswith('D')])
                label += f', R² = {r_squared:.4g}'

                # Determine the color based on model_color setting
                if use_model_color:
                    color = model_colors.get(model_num, 'black')
                else:
                    color = method_colors.get(method, 'black')

                ax.plot(x_fit, y_fit, label=label, color=color, linestyle=line_style, linewidth=plot_options['line_width'], alpha=method_transparency[method])
                fits.append(y_pred)
                model_names.append(label)
                r_squared_values.append(r_squared)
                method_used.append(method)
                D_values.append(result.best_values)
                errors.append(result.params)
                print_results(result, r_squared, adjusted_r_squared)
            except Exception as e:
                print(f"Error fitting {model_name} using {method}: {e}")

    ax.scatter(x_data, y_data, label='Experimental Data', color=plot_options.get('data_color', 'black'), s=48)

    # Customize plot
    customize_plot(ax, plot_options, file_name)
    if file_name:
        plot_and_save(plot_options['output_folder'], file_name, fig, plot_options)
    return fig, fits, model_names, method_used, r_squared_values, D_values, errors

def setup_parameters(model_num):
    params = Parameters()
    if model_num == 1:  # 2D model (Dout=0)
        params.add('y0', value=1.01, min=0.95, max=1.0)
        params.add('Din', value=5.0e-13, min=1e-15, max=1e-10)
    elif model_num == 2:  # 2D model (Dout ≠ 0)
        params.add('y0', value=1.0, min=0.95, max=1.00)
        params.add('Din', value=1.0e-13, min=1e-14, max=1e-11)
        params.add('Dout', value=1.0e-12, min=1e-13, max=1e-11)
    elif model_num == 3:  # 3D Monoexponential
        params.add('y0', value=1.0, min=0.99, max=1.02)
        params.add('D', value=2.0e-13, min=1e-16, max=9e-10)
    elif model_num == 4:  # 3D Biexponential
        params.add('y0', value=1.0, min=0.95, max = 1.0)
        params.add('D1', value=1.00e-13, min=1e-14, max=1e-11)
        params.add('D2', value=1.00e-12, min=1e-14, max=1e-11)
        params.add('f', value=0.5, min=0.0, max=1.0)
    elif model_num == 5:  # 3D Stretched Exponential
        params.add('y0', value=1.0, min=0.97, max=1.0)
        params.add('D', value=1.0e-13, min=1e-15, max=9e-10)
        params.add('b', value=0.8, min=0.1, max=1.0)
    elif model_num == 6:  # 3D Stretched Exponential
        params.add('y0', value=1.0, min=0.97, max=1.0)
        params.add('Din', value=1.0e-13, min=5e-15, max=1e-12)
        params.add('Dout', value=1.0e-13, min=5e-15, max=1e-12)
        params.add('D1', value=5.00e-12, min=1e-14, max=1e-10)
        params.add('D2', value=1.00e-12, min=1e-14, max=1e-10)
        params.add('f1', value=0.3, min=0.0, max=1.0)
        params.add('f2', value=0.3, min=0.0, max=1.0)
    return params



def plot_and_save(output_folder, file_name, fig, plot_options):
    # Ensure the output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Construct the output file path
    plot_output_file_path = Path(output_folder) / f"plot_{Path(file_name).stem}.png"

    # Save the figure
    fig.savefig(plot_output_file_path, dpi=plot_options['dpi_value'])

# Disable LaTeX rendering
plt.rcParams['text.usetex'] = False

def scientific_formatter(x, pos):
    if x == 0:
        return "$0$"
    elif abs(x) < 1e-16:  # Very small numbers
        return "$0$"
    else:
        exp = int(np.floor(np.log10(abs(x))))
        coef = x / 10**exp
        return f'${coef:.1f}\\times10^{{{exp}}}$'

def customize_plot(ax, plot_options, file_name):
    font_name = plot_options.get('font_name', 'Arial')

    # Set x-axis label
    ax.set_xlabel(plot_options.get('x_label', 'X Axis Label'),
                  fontdict={'family': font_name,
                            'size': plot_options.get('x_label_fontsize', 12),
                            'style': plot_options.get('x_label_fontstyle', 'normal'),
                            'weight': plot_options.get('x_label_fontweight', 'normal')},
                  color=plot_options.get('x_label_color', 'black'))

    # Set y-axis label
    ax.set_ylabel(plot_options.get('y_label', 'Y Axis Label'),
                  fontdict={'family': font_name,
                            'size': plot_options.get('y_label_fontsize', 12),
                            'style': plot_options.get('y_label_fontstyle', 'normal'),
                            'weight': plot_options.get('y_label_fontweight', 'normal')},
                  color=plot_options.get('y_label_color', 'black'))

    # Set title
    if plot_options.get('auto_title', True):
        ax.set_title(f"Fit using {Path(file_name).stem}",
                     fontdict={'family': font_name,
                               'size': plot_options.get('title_fontsize', 14),
                               'style': plot_options.get('title_fontstyle', 'normal'),
                               'weight': plot_options.get('title_fontweight', 'bold')},
                     color=plot_options.get('title_color', 'black'))

    # Set legend
    ax.legend(loc=plot_options['legend_loc'],
              prop={'family': font_name, 'size': plot_options.get('legend_fontsize', 10)})

    # Set tick label font and size
    tick_font = fm.FontProperties(family=font_name,
                                  size=plot_options.get('tick_fontsize', 10))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(tick_font)

    # Scientific notation for x-axis
    if plot_options.get('x_scientific', False):
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Scientific notation for y-axis
    if plot_options.get('y_scientific', False):
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Grid
    ax.grid(plot_options['grid'])

    # Set axis limits
    if plot_options['x_limit']:
        ax.set_xlim(plot_options['x_limit'])
    if plot_options['y_limit']:
        ax.set_ylim(plot_options['y_limit'])

    # Customize ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(plot_options['x_major_ticks']))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(plot_options['x_minor_ticks']))
    ax.yaxis.set_major_locator(plt.MaxNLocator(plot_options['y_major_ticks']))
    ax.yaxis.set_minor_locator(plt.MaxNLocator(plot_options['y_minor_ticks']))
    ax.tick_params(axis='x', which='major', length=plot_options['x_major_tick_length'],
                   width=plot_options['x_major_tick_width'], colors=plot_options.get('x_tick_color', 'black'),
                   labelsize=plot_options.get('tick_fontsize', 10))
    ax.tick_params(axis='x', which='minor', length=plot_options['x_minor_tick_length'],
                   width=plot_options['x_minor_tick_width'], colors=plot_options.get('x_tick_color', 'black'))
    ax.tick_params(axis='y', which='major', length=plot_options['y_major_tick_length'],
                   width=plot_options['y_major_tick_width'], colors=plot_options.get('y_tick_color', 'black'),
                   labelsize=plot_options.get('tick_fontsize', 10))
    ax.tick_params(axis='y', which='minor', length=plot_options['y_minor_tick_length'],
                   width=plot_options['y_minor_tick_width'], colors=plot_options.get('y_tick_color', 'black'))

    # Customize axis visibility and thickness
    for spine in ax.spines.values():
        spine.set_linewidth(plot_options.get('axis_thickness', 1))
    ax.spines['top'].set_visible(plot_options.get('show_top_axis', True))
    ax.spines['bottom'].set_visible(plot_options.get('show_bottom_axis', True))
    ax.spines['left'].set_visible(plot_options.get('show_left_axis', True))
    ax.spines['right'].set_visible(plot_options.get('show_right_axis', True))
    ax.spines['top'].set_linewidth(plot_options['top_axis_thickness'])
    ax.spines['bottom'].set_linewidth(plot_options['bottom_axis_thickness'])
    ax.spines['left'].set_linewidth(plot_options['left_axis_thickness'])
    ax.spines['right'].set_linewidth(plot_options['right_axis_thickness'])

    # Set scaling
    if plot_options['x_scale'] == 'log':
        ax.set_xscale('log')
    if plot_options['y_scale'] == 'log':
        ax.set_yscale('log')


def print_results(result, r_squared, adjusted_r_squared):
    print("Fit Parameters:")
    for key, value in result.best_values.items():
        print(f"{key} = {value:.4E}", end=", ")
    print(f"\nR-squared: {r_squared:.4g}")
    print(f"Adjusted R-squared: {adjusted_r_squared:.4g}")

def main():
    exp_path = '/Users/tylerpennebaker/BoxSync/structural_stability/300_test'
    folder_path = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Structural_stability/NMR data/PFG_fits/anisotropic_analyis/t1ints'  # Path to folder containing data files
    output_folder = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Structural_stability/NMR data/PFG_fits/anisotropic_analyis/out'  # Output folder for results

    # Get all files in the folder
    # files = [file for file in os.listdir(folder_path) if file.endswith(('.xlsx', '.csv', '.txt'))]
    dirs = sorted([file for file in os.listdir(exp_path)])

    # Prompt user to select files
    # print("Select files to process (enter numbers separated by spaces):")
    # for i, file in enumerate(files, start=1):
    #     print(f"{i}. {file}")
    # try:
    #     file_selection = list(map(int, input("Enter selection: ").split()))
    #     selected_files = [files[i-1] for i in file_selection]
    print("Select exp_nos to process (enter numbers separated by spaces):")
    print(dirs)
    try:
        exp_selection = list(map(int, input("Enter selection: ").split()))
    except (IndexError, ValueError):
        print("Invalid selection. Please enter numbers separated by spaces.")
        return

    # Choose model and optimization method
    chosen_models = [1, 2, 3, 4, 5, 6]  # Try all models
    # chosen_models = [4]  # Choose models to fit
    chosen_methods = ["de"]  # Choose methods to use [de, sa, pso, odr, trf]

    # Plot options (you can modify these as needed)
    plot_options = {
        'x_limit': None, 'y_limit': None, 'legend_fontsize': 20,
        'x_scientific': True, 'y_scientific': True, 'x_decimal_places': 0, 'y_decimal_places': 3,
        'x_label': "B (s/m2)", 'y_label': 'Normalized intensity (a.u.)',
        'x_label_fontsize': 20, 'y_label_fontsize': 20,
        'tick_fontsize': 15, 'x_major_ticks': 5, 'x_minor_ticks': 20,
        'y_major_ticks': 5, 'y_minor_ticks': 20,
        'x_major_tick_length': 7, 'x_major_tick_width': 1.5, 'x_minor_tick_length': 4,
        'x_minor_tick_width': 1, 'y_major_tick_length': 7, 'y_major_tick_width': 1.5,
        'y_minor_tick_length': 4, 'y_minor_tick_width': 1,
        'top_axis_thickness': 1.5, 'bottom_axis_thickness': 1.5, 'left_axis_thickness': 1.5, 'right_axis_thickness': 1.5,
        'model_color': True, 'model_colors': {1: 'red', 2: 'blue', 3: 'yellow', 4: 'green', 5: 'magenta'},
        'method_colors': {'de': 'cyan', 'pso': 'magenta', 'sa': 'yellow', 'odr': 'brown', 'trf': 'pink'},
        'title_fontsize': 14,
        'x_label_fontstyle': 'italic', 'legend_loc': 'best', 'y_label_fontstyle': 'normal',
        'font_name': "Arial", 'fig_width': 10, 'fig_height': 11, 'line_width': 2,
        'grid': False, 'x_scale': 'linear', 'y_scale': 'linear', 'dpi_value': 300,
        'output_folder': output_folder
    }


    # v = 10.39677E7   # Gamma,gyromagnetic ratio in rad⋅s−1⋅T−1
    # d = 0.003 # small delta,, gradient duration in second
    # D = 0.020 # Big delta, diffusion time in second

    # Specify column numbers directly in the code
    # n1, n2 = 0, 1
    # for file_name in selected_files:

    for exp_no in exp_selection:
        file_name = str(exp_no)
        # file_path = os.path.join(folder_path, file_name)
        # print(f"\nProcessing file: {file_name}")
        x_data, y_data = PFG_data_extract(exp_path, exp_no, nucleus='7Li')  # Pass parameters to read_data

        # x_data, y_data = read_data(file_path, n1, n2, v=v, d=d, D=D)  # Pass parameters to read_data
        if x_data is None or y_data is None:
            print(f"Error reading data from {file_name}. Skipping this file.")
            continue

        try:
            # Fit, plot and weighting methods
            fig, fits, model_names, method_used, r_squared_values, D_values, errors = fit_and_plot(
                x_data, y_data, chosen_models, chosen_methods, plot_options, file_name=file_name,
                use_inverse_variance_weighting=False,
                use_squared_intensity_weighting=False,
                use_weighted_least_squares=False
            )

            # Save results
            output_file_path = os.path.join(output_folder, f"results_{Path(file_name).stem}.txt")
            save_results(output_file_path, file_name, x_data, y_data, fits, model_names, method_used, r_squared_values, D_values, errors)

            print(f"Results saved to {output_file_path}")

            # Save the plot
            plot_and_save(output_folder, file_name, fig, plot_options)

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    main()
