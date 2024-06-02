import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import requests
from bs4 import BeautifulSoup
# from googletrans import Translator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


##### VISUALIZATION

# PROPERTIES OF PLOTS
def set_plot_properties(x_label, y_label, y_lim=[], ax=None):
    """
    Set properties of a plot axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].

    Returns:
        None
    """
    plt.xlabel(x_label)  # Set the label for the x-axis
    plt.ylabel(y_label)  # Set the label for the y-axis
    if len(y_lim) != 0:
        plt.ylim(y_lim)  # Set the limits for the y-axis if provided

    if ax:
        ax.set_xlabel(x_label)  # Set the label for the x-axis
        ax.set_ylabel(y_label)  # Set the label for the y-axis
        if len(y_lim) != 0:
            ax.set_ylim(y_lim)  # Set the limits for the y-axis if provided


# BAR CHART
def plot_bar_chart(data, variable, x_label=None, y_label='Count', y_lim=[], legend=[], color='cadetblue', annotate=False, top=None, vertical=False):
    """
    Plot a bar chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].
        legend (list, optional): The legend labels. Defaults to [].
        color (str, optional): The color of the bars. Defaults to 'cadetblue'.
        annotate (bool, optional): Flag to annotate the bars with their values. Defaults to False.
        top (int or None, optional): The top value for plotting. Defaults to None.
        vertical (bool, optional): Flag to rotate x-axis labels vertically. Defaults to False.

    Returns:
        None
    """
    # Count the occurrences of each value in the variable
    counts = data[variable].value_counts()[:top] if top else data[variable].value_counts()
    x = counts.index  # Get x-axis values
    y = counts.values  # Get y-axis values
    
    # Sort x and y values together
    x, y = zip(*sorted(zip(x, y)))

    # Plot the bar chart with specified color
    plt.bar(x, y, color=color)
    
    # Set the x-axis tick positions and labels, rotate if vertical flag is True
    plt.xticks(rotation=90 if vertical else 0)

    # Annotate the bars with their values if annotate flag is True
    if annotate:
        for i, v in enumerate(y):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

    if x_label == None:
        x_label = variable

    set_plot_properties(x_label, y_label, y_lim) # Set plot properties using helper function

    plt.show()


# PIE CHART
def plot_pie_chart(data, variable, colors, labels=None, legend=[], autopct='%1.1f%%'):
    '''
    Plot a pie chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        colors (list): The colors for each pie slice.
        labels (list, optional): The labels for each pie slice. Defaults to None.
        legend (list, optional): The legend labels. Defaults to [].
        autopct (str, optional): The format for autopct labels. Defaults to '%1.1f%%'.

    Returns:
        None
    '''
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable

    # Plot the pie chart with specified parameters
    plt.pie(counts, colors=colors, labels=labels, startangle=90, autopct=autopct, textprops={'fontsize': 21})
    plt.legend(legend if len(legend) > 0 else counts.index, 
               fontsize=16, bbox_to_anchor=(0.7, 0.9))  # Add a legend if provided
    
    plt.show()  # Display the pie chart


# HISTOGRAM
def plot_histogram(data, variable, x_label=None, y_label='Count', color='rosybrown'):
    '''
    Plot a histogram based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the histogram bars. Defaults to 'cadetblue'.

    Returns:
        None
    '''
    plt.hist(data[variable], bins=50, color=color)  # Plot the histogram using 50 bins

    if x_label == None:
        x_label = variable

    set_plot_properties(x_label, y_label)  # Set plot properties using helper function

    plt.show()


# BOXPLOT
def plot_box(data, grouped_variable, by_variable, vertical=False):
    # Generate the boxplot
    data[[grouped_variable, by_variable]].boxplot(by=by_variable, color='#5F9EA0')

    # Remove the grid lines
    plt.grid(visible=None)

    # Remove the title
    plt.title(None)

    # Set the x-axis tick positions and labels, rotate if vertical flag is True
    plt.xticks(rotation=90 if vertical else 0)

    # Set xlabel and ylabel
    set_plot_properties(by_variable, grouped_variable)

    # Display the plot
    plt.show()


# SCATTER
def plot_scatter(data, variable1, variable2, color='cadetblue'):
    """
    Plot a scatter plot between two variables in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variables.
        variable1 (str): The name of the first variable.
        variable2 (str): The name of the second variable.
        color (str, optional): The color of the scatter plot. Defaults to 'cadetblue'.

    Returns:
        None
    """
    plt.scatter(data[variable1], data[variable2], color=color, alpha=0.5)  # Plot the scatter plot

    set_plot_properties(variable1, variable2)  # Set plot properties using helper function


# KDE
def plot_kde(data, variables, colors):
    # Create KDE plots for the scaled numerical columns
    for i, var in enumerate(variables):
        sns.kdeplot(data[var], color=colors[i])

    # Set the legend and label the x-axis
    plt.legend(variables, fontsize=12)
    plt.gca().set_xticks([])
    plt.xlabel('')

    # Display the plot
    plt.show()


# CORRELATION MATRIX
def plot_correlation_matrix(data, method):
    '''
    Plot a correlation matrix heatmap based on the given data.

    Args:
        data (pandas.DataFrame): The input data for calculating correlations.
        method (str): The correlation method to use.

    Returns:
        None
    '''
    corr = data.corr(method=method)  # Calculate the correlation matrix using the specified method

    mask = np.tri(*corr.shape, k=0, dtype=bool)  # Create a mask to hide the upper triangle of the matrix
    corr.where(mask, np.NaN, inplace=True)  # Set the upper triangle values to NaN

    plt.figure(figsize=(30, 15))  # Adjust the width and height of the heatmap as desired

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                annot=True,
                vmin=-1, vmax=1,
                cmap=sns.diverging_palette(220, 10, n=20))  # Plot the correlation matrix heatmap

# CONFUSION MATRIX
def plot_confusion_matrix(ax, matrix, title, color_map='Blues'):
    sns.heatmap(matrix, annot=True, fmt='d', cmap=color_map, ax=ax)
    ax.set_title('{} Confusion Matrix'.format(title))
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')


# # UMAP
# def visualize_dimensionality_reduction(transformation, targets, predictions=None):
#     '''
#     Visualize the dimensionality reduction results using a scatter plot.

#     Args:
#         transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
#         targets (numpy.ndarray or list): The target labels or cluster assignments.
#         predictions (list): List of True or False values indicating if each observation was well predicted.

#     Returns:
#         None
#     '''
#     if predictions is not None:
#         fig, (ax1, ax2) = plt.subplots(1, 2)

#         # Create a scatter plot of the t-SNE output for predictions
#         colors = ['lightgrey' if pred else 'indianred' for pred in predictions]
#         ax2.scatter(transformation[:, 0], transformation[:, 1], c=colors)
#         yes = plt.scatter([], [], c='lightgrey', label='Yes')
#         no = plt.scatter([], [], c='indianred', label='No')
#         ax2.legend(handles=[yes, no], title='Predicted')

#     else:
#         ax1 = plt.plot()

#     # Convert object labels to categorical variables
#     labels, targets_categorical = np.unique(targets, return_inverse=True)

#     # Create a scatter plot of the t-SNE output
#     cmap = plt.cm.tab20
#     norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
#     ax1.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

#     # Create a legend with the class labels and corresponding colors
#     handles = [ax1.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
#     ax1.legend(handles=handles, title='Success')

#     plt.show()


##### WEB SCRAPING
def fetch_soup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


def web_scraping_dges_areas(url):
    # Read DGES area/courses website html
    soup = fetch_soup(url)

    # Select the areas tab
    area_tab = soup.select_one('div.noprint')

    # Create lists with areas' names and links
    open_area = area_tab.find('li').text
    remaining_areas = [area.text for area in area_tab.find_all('a')]
    areas = [open_area] + remaining_areas
    links = [url + link.get('href')[-8:] for link in area_tab.find_all('a')]

    # Remove the last item from both lists
    areas.pop()
    links.pop()

    # Create an empty dictionary
    areas_dict = {}

    # Get the courses from each area
    for i, area in enumerate(areas):
        if i != 0:
            soup = fetch_soup(links[i-1])

        courses = soup.find_all('div', class_='lin-area-c2')
        courses = [course.text for course in courses]
        areas_dict[area] = courses

        time.sleep(1)

    return areas_dict


##### DICTIONARY TREATMENT

# # TRANSLATOR
# def dictionary_translator(dictionary, from_lang='pt', to_lang='en'):
#     # Initialize a Translator object
#     translator = Translator()

#     # Initialize an empty dictionary to store translated key-value pairs
#     translated_dictionary = {}

#     # Iterate through key-value pairs in the input dictionary
#     for key, values in dictionary.items():
#         # Translate the key
#         translated_key = translator.translate(key, src=from_lang, dest=to_lang).text
#         # Initialize an empty list to store translated values for the key
#         translated_dictionary[translated_key] = []

#         # Iterate through values for the key
#         for value in values:
#             # Translate each value
#             translated_value = translator.translate(value, src=from_lang, dest=to_lang).text
#             # Append the translated value to the list of translated values
#             translated_dictionary[translated_key].append(translated_value)
            
#             # Pause for a while to avoid hitting rate limits
#             time.sleep(2)

#         # Pause for a while to avoid hitting rate limits
#         time.sleep(5)

#     return translated_dictionary


# CAPITALIZER
def dictionary_capitalizer(dictionary):
    # Initialize a new dictionary to store capitalized keys and values
    capitalized_dictionary = {}

    # Iterate through key-value pairs in the input dictionary
    for key, values in dictionary.items():
        # Capitalize the key
        capitalized_key = key.capitalize()
        # Capitalize each value in the list of values
        capitalized_values = [value.capitalize() for value in values]
        # Add the capitalized key and values to the new dictionary
        capitalized_dictionary[capitalized_key] = capitalized_values

    return capitalized_dictionary



##### DATA PREPROCESSING

# DATA TYPES
def transform_variables_to_boolean(train, test=None):
    # Iterate through columns in train data
    for col in train.columns:
        # Get unique non-null values in the column
        unique_values = train[col].dropna().unique()
        # Count the number of unique values
        n_unique_values = len(unique_values)

        # If there are only two unique values, convert the column to boolean type
        if n_unique_values == 2:
            train[col] = train[col].astype(bool)

            # If test data is provided, convert the column to boolean type in test data as well
            if test is not None:
                test[col] = test[col].astype(bool)

    return train, test


def datatype_distinction(data):
    '''
    Distinguishes between the numerical and categorical columns in a DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    numerical : pandas.DataFrame
        DataFrame containing only numerical columns.

    categorical : pandas.DataFrame
        DataFrame containing only categorical columns.
    '''
    # Select numerical columns using select_dtypes with np.number
    numerical = data.select_dtypes(include=np.number).copy()
    
    # Select categorical columns by excluding numerical types
    categorical = data.select_dtypes(exclude=np.number).copy()
    
    return numerical, categorical


# DATA TRANSFORMATION
def transformation(technique, data, column_transformer=False):
    '''
    Applies the specified transformation technique to the DataFrame.

    Parameters:
    -----------
    technique : object
        The transformation technique (e.g., from Scikit-learn) to be applied.

    data : pandas.DataFrame
        The input DataFrame to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    data_transformed : pandas.DataFrame
        Transformed DataFrame.

    Notes:
    ------
    - If column_transformer is False, the columns in the transformed DataFrame
      will retain the original column names.
    - If column_transformer is True, the method assumes that technique has a
      get_feature_names_out() method and uses it to get feature names for the
      transformed data, otherwise retains the original column names.
    '''
    # Apply the specified transformation technique to the data
    data_transformed = technique.transform(data)
    
    # Create a DataFrame from the transformed data
    data_transformed = pd.DataFrame(
        data_transformed,
        index=data.index,
        columns=technique.get_feature_names_out() if column_transformer else data.columns
    )
    
    return data_transformed


def data_transform(technique, X_train, X_val=None, column_transformer=False):
    '''
    Fits a data transformation technique on the training data and applies the transformation 
    to both the training and validation data.

    Parameters:
    -----------
    technique : object
        The data transformation technique (e.g., from Scikit-learn) to be applied.

    X_train : pandas.DataFrame or array-like
        The training data to fit the transformation technique and transform.

    X_val : pandas.DataFrame or array-like, optional (default=None)
        The validation data to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    X_train_transformed : pandas.DataFrame
        Transformed training data.

    X_val_transformed : pandas.DataFrame or None
        Transformed validation data. None if X_val is None.

    Notes:
    ------
    - Fits the transformation technique on the training data (X_train).
    - Applies the fitted transformation to X_train and optionally to X_val if provided.
    '''
    # Fit the transformation technique on the training data
    technique.fit(X_train)
    
    # Apply transformation to the training data
    X_train_transformed = transformation(technique, X_train, column_transformer)

    # Apply transformation to the validation data if provided
    X_val_transformed = None
    if X_val is not None:
        X_val_transformed = transformation(technique, X_val, column_transformer)

    return X_train_transformed, X_val_transformed


# MISSING VALUES
def drop_missing_values(ax, train, test=None, drop_perc=50):
    # Calculate the number of missing values along the specified axis
    axis_nulls = train.isnull().sum(axis=ax)

    # Calculate the size of the train data along the specified axis
    if ax == 0:
        size = len(train.index)
    else:
        size = len(train.columns)

    # Calculate the percentage of missing values
    nulls_percentage = round(100 * axis_nulls / size, 1)
    
    # Initialize a list to store columns with high missing percentage
    to_drop = []
    count = 0
    
    # Print columns to remove
    print('REMOVE')
    for obj, perc in nulls_percentage.items():
        if perc > drop_perc:
            print(f'{obj}: {perc}%')
            to_drop.append(obj)
            count += 1
    
    # Remove columns with high missing percentage
    train.drop(to_drop, axis=abs(ax-1), inplace=True)
    
    # Remove the same columns from the test data if ax is 0
    if test and ax == 0:
        test.drop(to_drop, axis=abs(ax-1), inplace=True)

    print('Total:', count)

    return train, test



# ENCODING
def one_hot_encoding(train, test=None, target=None):
    if target:
        # Define X and y
        train = train.drop(columns=[target])

    # Filter the dataset with only the object data type columns
    train_obj = train.select_dtypes(include=['object'])

    # Get the number of unique values from the filtered dataset
    train_obj_nu = train_obj.nunique()

    # Get the columns with more than 2 unique values
    columns_to_encode = train_obj_nu.index[train_obj_nu > 2]

    # One-Hot
    ct = ColumnTransformer([
        ('oneHot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), columns_to_encode)
        ], remainder='passthrough')
    train, test = data_transform(ct, train, test, column_transformer=True)
    train.columns = train.columns.str.replace(r'(oneHot|remainder)__', '', regex=True)
    
    if test:
        test.columns = test.columns.str.replace(r'(oneHot|remainder)__', '', regex=True)

    # Tranform variables with only two unique value to boolean
    train, test = transform_variables_to_boolean(train, test)

    return train, test


##### FEATURE ENGINEERING

# FEATURE CREATION
def calculate_mean_difference_inner(data, grouped_variable, by_variables, new_column, means):
    # Merge data with mean values based on 'by_variables'
    merged_data = data.merge(means, on=by_variables, how='left')
    
    # Calculate mean difference
    merged_data[new_column] = merged_data[grouped_variable] - merged_data['mean_' + grouped_variable]
    
    # Drop redundant columns
    return merged_data.drop(columns=['mean_' + grouped_variable])


def calculate_mean_difference(train, grouped_variable, by_variables, new_column, test=None):
    train_index = train.index
    if test:
        test_index = test.index

    # Calculate mean values for 'grouped_variable' grouped by 'by_variables'
    means = train.groupby(by_variables)[grouped_variable].mean().reset_index()
    means.columns = by_variables + ['mean_' + grouped_variable]

    # Apply mean difference calculation function to train and test sets
    train = calculate_mean_difference_inner(train, grouped_variable, by_variables, new_column, means)
    train.index = train_index
    
    if test:
        test = calculate_mean_difference_inner(test, grouped_variable, by_variables, new_column, means)
        test.index = test_index

    return train, test



# RECLASSIFYING
# # Option 1: finished cycle [2nd cycle, 3rd cycle, high school, higher education] (ordinal)
# def get_ordinal_qualification(qualification):
#     if qualification == 'No school':
#         return 0    # No education
    
#     elif qualification in [f'{i}th grade' for i in range(3, 12)]:
#         return 1    # Basic education
    
#     elif '12' in qualification or qualification == 'Incomplete Bachelor\'s':
#         return 2    # Intermidiate education
    
#     else:
#         return 3    # Advanced education


# Option 2: year of education (numerical)
def get_years_of_education(qualification):
    years = re.findall(r'\d+', qualification)

    if qualification == 'No school':
        return 0    # No education
    
    elif years:
        return int(years[0])    # Total years
    
    elif qualification == 'Incomplete Bachelor\'s':
        return 13   # Considering 1 year in university
    
    elif qualification == 'Bachelor degree':
        return 15   # Bachelor's duration general rule is 3 years
    
    elif qualification == 'Post-Graduation':
        return 16   # Plus 1 year after Bachelor degree
    
    elif qualification == 'Master degree':
        return 17   # Plus 2 years after Bachelor degree
    
    elif qualification == 'PhD':
        return 21   # Plus 4 years after Master degree