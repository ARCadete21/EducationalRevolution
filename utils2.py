import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from utils1 import set_plot_properties, data_transform


##### IMPUTATION

# K-NEAREST NEIGHBORS IMPUTER
def apply_optimal_knnimputer(data, k_min, k_max, show=False, export_name=None):
    """
    Find the best value of K for KNN imputation by evaluating the average silhouette score,
    selecting the best performing weight type between 'uniform' and 'distance'.

    Args:
        data (pandas.DataFrame): The input data for KNN imputation and clustering.
        k_min (int): The minimum value of K to evaluate.
        k_max (int): The maximum value of K to evaluate.

    Returns:
        KNNImputer: The best KNNImputer with the selected weight type.
    """
    # Define the range of K values to evaluate
    k_values = range(k_min, k_max + 1)

    # 
    best_avg_silhouette_score = float('-inf')

    # Iterate over each weight type
    for weights in ['uniform', 'distance']:
        # Initialize an empty list to store the average silhouette scores
        avg_silhouette_scores = []

        # Define KFold with 10 splits
        kf = KFold(n_splits=10)

        # Iterate over each K value
        for k in k_values:
            # Create the KNN imputer with the current K value and weight type
            knn_imputer = KNNImputer(n_neighbors=k, weights=weights)

            # Perform KNN imputation and clustering for each fold in cross-validation
            silhouette_scores = []
            for train_index, test_index in kf.split(data):
                # Get the indexes of the observations assigned for each partition
                train, test = data.iloc[train_index], data.iloc[test_index]

                # Perform KNN imputation on the training and test sets
                train_imputed, test_imputed = data_transform(knn_imputer, train, test)
                
                # Cluster the imputed data using KMeans or other clustering algorithm
                kmeans = KMeans(n_clusters=k, random_state=16)
                kmeans.fit(train_imputed)

                # Evaluate the clustering performance using silhouette score
                labels = kmeans.predict(test_imputed)
                silhouette = silhouette_score(test_imputed, labels)
                silhouette_scores.append(silhouette)

            # Calculate the average silhouette score for the current K value
            avg_silhouette_score = np.mean(silhouette_scores)
            avg_silhouette_scores.append(avg_silhouette_score)

        # Find the index of the K value with the highest average silhouette score
        max_avg_silhouette_score = max(avg_silhouette_scores)
        
        # If the current weight type has a higher silhouette score, update the best weight type and imputer
        if max_avg_silhouette_score > best_avg_silhouette_score:
            best_avg_silhouette_score = max_avg_silhouette_score
            best_k = k_values[np.argmax(avg_silhouette_scores)]
            best_imputer = KNNImputer(n_neighbors=best_k, weights=weights)

    data = data_transform(best_imputer, data)[0]

    if show:
        print(best_imputer)

    if export_name:
        data.to_csv(f'temp\\imputed_data\\{export_name}.csv', index=True)

    return data


##### PRINCIPAL COMPONENT ANALYSIS
def pc_analysis(X_train, var_threshold=0.8, show=False):
    '''
    Perform Principal Component Analysis (PCA) and retain components based on variance threshold.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training data for PCA.

    var_threshold : float, optional (default=0.8)
        Variance threshold to retain principal components.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing principal components that satisfy the variance threshold.

    Notes:
    ------
    - Computes principal components from the input training data.
    - Plots the cumulative explained variance ratio and a threshold line.
    - Retains principal components that explain variance above the specified threshold.
    '''
    n_columns = len(X_train.columns)
    
    # Perform PCA
    pca = PCA(n_components=n_columns)
    components = pca.fit_transform(X_train)

    # Determine the number of components that reach the variance threshold
    num_components = (pca.explained_variance_ratio_.cumsum() >= var_threshold).argmax() + 1

    # Create DataFrame containing principal components meeting the variance threshold
    pc = pd.DataFrame(components, 
                  index=X_train.index, 
                  columns=[f'PC{i}' for i in range(n_columns)]
                  ).iloc[:, :num_components]
    
    if show:
        print('Number of columns:', pc.shape[1])
        # Plot cumulative explained variance ratio
        plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o')
        plt.axhline(y=var_threshold, color='r', linestyle='-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
        plt.show()

    return pc


##### EVALUATION

### KMEANS

# INERTIA & SILHOUETTE
def plot_inertia_and_silhouette(data, k_min=2, k_max=30):
    """
    Plot the inertia (dispersion) and silhouette score for different numbers of clusters.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input data for clustering.
        k_min (int, optional): The minimum number of clusters to evaluate. Defaults to 2.
        k_max (int, optional): The maximum number of clusters to evaluate. Defaults to 15.

    Returns:
        None
    """
    dispersions = []
    scores = []

    k_clusters = range(k_min, k_max + 1)

    for k in k_clusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersions.append(kmeans.inertia_)  # Calculate the dispersion (inertia) for each number of clusters
        kmeans.predict(data)
        scores.append(silhouette_score(data, kmeans.labels_, metric='euclidean'))  # Calculate the silhouette score

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(k_clusters, dispersions, marker='o')  # Plot the inertia (dispersion)
    set_plot_properties('Number of clusters', 'Dispersion (inertia)', ax=ax1)
    ax2.plot(k_clusters, scores, marker='o')  # Plot the silhouette score
    set_plot_properties('Number of clusters', 'Silhouette score', ax=ax2)

    plt.show()


### HIERARCHICAL CLUSTERING

# R2
def get_r2_hc(df, link_method, max_nclus=6, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt.

    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".

    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares

    r2 = []  # where we will store the R2 metrics for each cluster solution

    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)

        # get cluster labels
        hclabels = cluster.fit_predict(df)

        # concat df with labels
        df_concat = pd.concat((df, pd.Series(hclabels, name='labels', index=df.index)), axis=1)

        # compute ssw for each cluster labels
        ssw_labels = df_concat.groupby(by='labels').apply(get_ss)

        # remember: SST = SSW + SSB
        ssb = sst - np.sum(ssw_labels)

        r2.append(ssb / sst)  # save the R2 of the given cluster solution

    return np.array(r2)


def plot_r2_hc(data, max_nclus=6, min_nclus=1, dist="euclidean"):
    # Prepare input
    hc_methods = ["ward", "complete", "average", "single"]

    # Call function defined above to obtain the R2 statistic for each hc_method
    r2_hc_methods = np.vstack(
        [
            get_r2_hc(df=data, link_method=link, max_nclus=max_nclus, min_nclus=min_nclus, dist=dist)
            for link in hc_methods
        ]
    ).T
    r2_hc_methods = pd.DataFrame(r2_hc_methods, index=range(1, max_nclus + 1), columns=hc_methods)

    # Plot data
    fig = plt.figure()
    sns.lineplot(data=r2_hc_methods, linewidth=2.5, markers=["o"]*4)

    # Finalize the plot
    fig.suptitle("R2 plot for various hierarchical methods", fontsize=21)
    plt.gca().invert_xaxis()  # invert x axis
    plt.legend(title="HC methods", title_fontsize=11)
    plt.xticks(range(1, max_nclus + 1))
    plt.xlabel("Number of clusters", fontsize=13)
    plt.ylabel("R2 metric", fontsize=13)

    plt.show()


# DENDROGRAM
def plot_dendrogram(data, linkage_method, cut_line=None):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input data for clustering.
        linkage_method (str): The linkage method used for clustering.
        cut_line (float, optional): The threshold value to cut the dendrogram. Defaults to None.

    Returns:
        None
    """
    # Fit the AgglomerativeClustering model
    model = AgglomerativeClustering(linkage=linkage_method, distance_threshold=0, n_clusters=None).fit(data)

    # Create the plot
    fig, ax = plt.subplots()
    plt.title('Hierarchical Clustering Dendrogram')

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create the linkage matrix
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the dendrogram
    dendrogram(linkage_matrix, truncate_mode='level', p=1000)

    # Add a cut line if provided
    if cut_line is not None:
        plt.axhline(y=cut_line, color='black', linestyle='-')

    # Display the plot
    plt.show()


##### ANALYSIS

# OBSERVATIONS MATCHES BETWEEN SOLUTIONS
def clusters_comparison(data, solution1, solution2):
    """
    Compare the clusters of two solutions using a confusion matrix.

    Args:
        data (pandas.DataFrame): The input data containing the cluster assignments for both solutions.
        solution1 (str): The column name representing the cluster assignments of the first solution.
        solution2 (str): The column name representing the cluster assignments of the second solution.

    Returns:
        pandas.DataFrame: The confusion matrix comparing the clusters of the two solutions.
    """
    # Determine the number of unique clusters in each solution
    length1, length2 = len(data[solution1].unique()), len(data[solution2].unique())

    # Determine the maximum number of clusters
    n = max(length1, length2)

    # Compute the confusion matrix
    confusion = confusion_matrix(data[solution1], data[solution2])

    # Create a DataFrame for the confusion matrix with appropriate row and column labels
    df = pd.DataFrame(
        confusion,
        index=['{} {} Cluster'.format(solution1, i) for i in np.arange(0, n)],
        columns=['{} {} Cluster'.format(solution2, i) for i in np.arange(0, n)]
    )

    # Return the subset of the confusion matrix corresponding to the number of clusters in each solution
    return df.iloc[:length1, :length2]


# FEATURES MEANS
def groupby_mean(data, variable, features_means=False, n_features=45):
    """
    Group the data by a variable and calculate the mean for each group.

    Args:
        data (pandas.DataFrame): The input data.
        variable (str): The variable used for grouping.
        n_features (int, optional): The number of features to include in the result. Defaults to 30.

    Returns:
        pandas.DataFrame: The transposed DataFrame containing the mean values for each group.
    """
    # Group the data by the specified variable and calculate the mean for each group
    grouped_data = data.groupby(variable).mean()

    # Select the first n_features + 1 columns (including the variable column) and transpose the DataFrame
    result = grouped_data.iloc[:, :n_features + 1].T
    
    if features_means:
        overall_mean = data.mean()
        overall_mean = {column: '{:.6f}'.format(value) 
                        for column, value in overall_mean.items()}
        result['MEAN'] = overall_mean
        result['MEAN'] = result['MEAN'].astype(float)
            
    # Return the transposed DataFrame
    return result


##### DIMENSIONALITY REDUCTION VISUALIZATION
def visualize_dimensionality_reduction(transformation, targets):
    '''
    Visualize the dimensionality reduction results using a scatter plot.

    Args:
        transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
        targets (numpy.ndarray or list): The target labels or cluster assignments.
        predictions (list): List of True or False values indicating if each observation was well predicted.

    Returns:
        None
    '''
    # Convert object labels to categorical variables
    labels, targets_categorical = np.unique(targets, return_inverse=True)

    # Create a scatter plot of the t-SNE output
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
    plt.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

    # Create a legend with the class labels and corresponding colors
    handles = [plt.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Cluster')

    plt.show()


##### PROFILING
def cluster_profiles(df, label_columns, compar_titles=None, cluster_names=None):
    if compar_titles is None:
        compar_titles = [""] * len(label_columns)

    fig, axes = plt.subplots(nrows=len(label_columns), ncols=2, squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compar_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i != label]
        dfax = df.drop(drop_cols, axis=1)

        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:, [0, 1]]
        counts.columns = [label, "counts"]
        color = sns.color_palette('Dark2')

        # Setting Data
        pd.plotting.parallel_coordinates(centroids, label, color=color, ax=ax[0])
        sns.barplot(x=label, y="counts", data=counts, ax=ax[1], palette=color)

        # Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        if cluster_names:
            cluster_labels = [cluster_names.get(int(handle.get_label()), handle.get_label()) for handle in handles]
        else:
            cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        
        ax[0].annotate(text=titl, xy=(0.95, 1.1), xycoords='axes fraction', fontsize=16, fontweight='heavy')
        ax[0].legend(handles, cluster_labels)  # Adaptable to number of clusters
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=16)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=16)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

    plt.subplots_adjust(hspace=0.4, top=0.90, bottom=0.2)
    plt.suptitle("Cluster Profiling", fontsize=23)
    plt.show()
