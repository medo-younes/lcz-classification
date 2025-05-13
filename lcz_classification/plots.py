import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.metrics import confusion_matrix
import pandas as pd

import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_spectral_signature(band_stats, x_col, class_col, color_dict, title, xlabel, stat='median', out_file=None):
    """Plot spectral signature using a band statistics DataFrame 
    
    Args:
        band_stats (DataFrame): Pandas DataFrame of band statistics, retrieved from rasterstats.zonal_stats function
        x_col (str): Column for x-axis
        class_col (str): Column including class names
        color_dict (dict): Mapping between class name and color
        title (str): Plot title
        xlabel (str): Text label for x-axis
        stat (str): Name of statistic to plot, median is default, other options include mean, std, min and max
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Plot of spectral signature of all classess in the band_stats DataFrame
    """

    class_order = band_stats.drop_duplicates(class_col).sort_values(class_col)[class_col].to_list()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in class_order:
        class_stats=band_stats.set_index(class_col).loc[name]
        x = class_stats[x_col].to_list()
        y = class_stats[stat].to_list()
        ax.plot(x, y, label=name, color=color_dict[name]) # plot line for class

    ax.set_ylabel(f'{stat.title()} Surface Reflectance', weight = 'bold')
    ax.set_xlabel(xlabel, weight = 'bold')
    ax.set_title( title ,weight='bold')

    # Hide specific spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Legend formatting
    plt.legend(bbox_to_anchor =(0.5,-0.4), loc='lower center', ncol = 4, frameon=False)

    if out_file:
        plt.savefig(out_file)

    


def plot_pairwise_jm(df, class1,class2, dist_col, title, figsize, cbar=True,out_file=None):
    """Pairwise Jeffries-Matuista Distance Plot 
    
    Args:
        df (DataFrame): Jeffries-Matuista Distance between each class combination
        class1 (str): Column name of first classess
        class2 (str): Column name of second classes
        dist_col (str): Colum name including Jeffries-Matuista Distance values
        title (str): Plot title
        figsize (tuple): Desired figure Size
        cbar (boolean): Show color bar if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Plot of Jeffries-Matuista Distances between all classes
    """
    # Get Unique list of classes
    classes=df[class1].unique()

    arrays=[df.set_index(class1)[dist_col].loc[[cl]].values.T for cl in df[class1].unique()]

    max_len = max(len(arr) for arr in arrays)

    # Pad at the beginning (before index 0)
    result = [np.concatenate([np.zeros([max_len - len(arr) + 1]),  arr])  for arr in arrays]

    matrix = np.vstack(result).T


    if out_file:
        plt.savefig(out_file)

    
    # Make pairwise distance plot
    plt.figure(figsize=figsize)
    plot_mask =  matrix == 0.0

    sns.heatmap(matrix, 
                annot=True, 
                fmt=".1f", 
                cmap='RdYlGn',
                xticklabels=classes, 
                yticklabels=classes, 
                linewidths=1, 
                linecolor='white', 
                mask=plot_mask, 
                cbar=cbar
                
                )

    plt.title(title, fontweight='bold',fontdict=dict(size = 15), loc='left', x=0.0)

    if out_file:
        plt.savefig(out_file,bbox_inches='tight', dpi=300, pad_inches=0.2)



def plot_pixel_counts(pixel_count_df, count_col,class_col, color_col, title, as_percent=False, out_file=None):
    """Bar plot of pixel counts for each class
    
    Args:
        df (DataFrame): Pixel counts of each class in a Pandas DataFrame
        count_col (str): Name of column including pixel counts
        class_col (str): Name of column including class names
        color_col (str): Name of column including color values
        title (str): Plot title
        as_percent (boolean): Display values as percentage of total pixels if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Bar plot of pixel counts for each class
    """


    fig, ax = plt.subplots(figsize=(8, 6))
    
    pixel_count_df=pixel_count_df.sort_values(count_col,ascending=False) 
    y=pixel_count_df[count_col]  
    classes=pixel_count_df[class_col]
    colors=pixel_count_df[color_col]
    if as_percent:
        y = y / y.sum() * 100
        ylabel = 'Percentage of Pixels'
        plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0)) 
    else:
        
        ylabel = "Number of Pixels"

    

    ax.bar(x=classes,height=y, color=colors, linewidth=1, edgecolor='black')

    plt.xticks(rotation=90)
    plt.ylabel(ylabel, fontdict=dict(size=15))
    plt.title(title, fontdict=dict(size=15,weight='bold'))

    # Hide specific spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if out_file:
        plt.savefig(out_file)


def plot_confusion_matrix(y_true,y_pred, title,labels, figsize,cmap='Blues', as_percent=False, out_file=None):
    """Confusion matrix heat map using Seaborn 
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels 
        title (str): Plot title
        labels (list): Label names mapped to numeric label values
        figsize (tuple): Desired figure dimensions
        cmap (str): Desired color palette
        as_percent (boolean): Display values as percentage of total pixels if True
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Confusion matrix heat map using Seaborn 
    """

    cm=confusion_matrix(y_true,y_pred)
    mask = cm == 0.0

    plt.figure(figsize=figsize)
    fmt=".0f"

    if as_percent:
        cm=(cm / len(y_true)) * 100
        fmt=".1f"
    
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                cbar=False,
                xticklabels=labels, 
                yticklabels=labels, 
                mask=mask,
                linewidths=1,
                linecolor='white'
             
                )
    plt.xlabel("True",  fontdict=dict(size = 15, weight ='bold'))
    plt.ylabel("Predicted",  fontdict=dict(size = 15, weight ='bold'))
    plt.title(title, fontdict=dict(size = 12, weight ='bold'), loc='center', pad=20, x= 0.4)
    plt.legend([],[], frameon=False)

    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', )

    plt.show()





def plot_feature_importances(rf,features, title, out_file):
    """Plot Feature Importances of a Random Forest Classifier as a horizontal bar plot
    
    Args:
        rf (RandomForestClassifer): Trained RandomForestClassifer from sklearn
        features (list): Name of predictors use in the Random Forest Model
        title (str): Plot title
        out_file (str): Output file path of saved figure, only saves figure if not None
 
    Returns:
        matplotlib.figure.Figure: Feature Importances of a Random Forest Classifier as a horizontal bar plot
    """

    rf_i = pd.Series(rf.feature_importances_, index=features).sort_values()

    # Choose a colormap (e.g., 'viridis', 'plasma', 'coolwarm', etc.)
    cmap = cm.get_cmap('RdYlGn')

    # Normalize the series to the range 0-1
    norm = colors.Normalize(vmin=rf_i.min(), vmax=rf_i.max())

    # Map each value to a color
    rgba_colors = [cmap(norm(val)) for val in rf_i]
    hex_colors = [colors.to_hex(c) for c in rgba_colors]

    fig, ax = plt.subplots()
    rf_i.plot.barh(ax=ax, color=hex_colors)
    ax.set_title(title)
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()

    if out_file:
        plt.savefig(out_file)

    plt.show()
