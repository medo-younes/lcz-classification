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

    """
    Plot spectral signature using a band statistics DataFrame 
    
    Parameters:
    -----------

    band_stats
    x_col
    class_col
    color_dict
    title
    xlabel
    stat='median'

    Returns:
    --------
    Plot
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

    


def pairwise_plot(df, class1,class2, dist_col, title, out_file=None):
    # Get Unique list of classes
    classes=df[class1].unique()
    n_classes=len(classes)

    # Format data into Array with shape (n_classes, n_classes)
    n=1
    matrix=list()
    mask = list()

    # 
    for cl in classes: 
        jxy = df.set_index(class1).loc[cl]
        jxy=jxy.set_index(class2)[dist_col].T[classes].values
        matrix.append(jxy)
        
        # Mask for duplicate (redundant) pairs in the display
        f = [False for i in range(0, n)]
        t = [True for i in range(0, n_classes - n)]
        m=f.copy()
        m.extend(t)
        mask.append(m)
        n +=1

    mask=np.array(mask)
    matrix=np.array(matrix)

    if out_file:
        plt.savefig(out_file)


    # Make pairwise distance plot
    plt.figure(figsize=(10, 8))
    plot_mask = (mask == True) | (matrix == 0.0)
    sns.heatmap(matrix, 
                annot=True, 
                fmt=".1f", 
                cmap='RdYlGn',
                xticklabels=classes, 
                yticklabels=classes, 
                linewidths=1, 
                linecolor='white', 
                mask=plot_mask
                )

    plt.title(title, fontweight='bold',fontdict=dict(size = 20))

    if out_file:
        plt.savefig(out_file)



def plot_pixel_counts(pixel_count_df, count_col,class_col, color_col, title, out_file=None, as_percent=False, ):


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



def map_training_areas(train,test,boundary=None):
    

    m = train.explore(
                    color='blue', 
                    legend=True, 
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri World Imagery'
                    )


    test.explore(
        fill=True,
        color='red',
        m=m
    )

    boundary.explore(style={
                    "fill": False,
                    "color": "red"
                    },
                    m=m
    )
    return m

def plot_confusion_matrix(y_test,y_pred, title,labels, cmap='Blues', as_percent=False, out_file=None):

    cm=confusion_matrix(y_test,y_pred)
    mask = cm == 0.0

    plt.figure(figsize=(5,5))
    fmt=".0f"

    if as_percent:
        cm=(cm / len(y_test))
        fmt=".1%"
    
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                cbar=False,
                xticklabels=labels, 
                yticklabels=labels, 
                mask=mask,
                linewidths=2, 
                linecolor='black', 
             
                )
    plt.xlabel("True",  fontdict=dict(size = 15, weight ='bold'))
    plt.ylabel("Predicted",  fontdict=dict(size = 15, weight ='bold'))
    plt.title(title, fontdict=dict(size = 12, weight ='bold'), loc='center', pad=20, x= 0.4)
    plt.legend([],[], frameon=False)

    if out_file:
        plt.savefig(out_file)

    plt.show()





def plot_feature_importances(rf,features, title, out_file):

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


# def plot_training_samples(training, cmm, legend):

#     cmm_gdf = gpd.read_file(cmm)
#     training = gpd.read_file(training)

#     training['LCZ'] = training['LCZ'].astype(int)
#     training = training.sort_values('LCZ')

#     # add a column with the correspondence between LCZ class and its name
#     training['LCZ_name'] = training['LCZ'].map(legend).str[0]

#     lcz_list = [value[0] for value in legend.values()]

#     cmap_colors = [value[1] for value in legend.values()]

#     print(f'List of LCZ: {lcz_list}')
#     print(f'List of colors: {cmap_colors}')

#     m = cmm_gdf.explore(
#         style_kwds = {'fillOpacity': 0},
#         marker_kwds=dict(radius=10, fill=True),
#         tooltip_kwds=dict(labels=False),
#         tooltip = False,
#         popup = False,
#         highlight = False,
#         name="cmm"
#     )

#     training.explore(m=m,
#         column="LCZ_name",
#         tooltip="LCZ_name",
#         popup=True,
#         tiles="CartoDB positron",
#         style_kwds=dict(color="black"),
#         categories=lcz_list,
#         cmap=cmap_colors
#     )

#     # create a dictionary (shapes) containing the geometries of the training samples
#     # the dictionary keys are the LCZ classes
#     shapes = {}
#     LCZ_class = training['LCZ'].unique()
#     for LCZ in LCZ_class:
#         shapes[LCZ] = training.loc[training['LCZ'] == LCZ].geometry

#     return training, m, shapes


# def plot_ucl(imperv, perc_build, svf, canopy_height, buildings):

#     # Display the classification layers
#     fig, axs = plt.subplots(2, 3, figsize=(14, 8))

#     im1 = axs[0, 0].imshow(svf)
#     axs[0, 0].set_title('Sky View Factor [0-1]')
#     cbar1 = fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)

#     im2 = axs[0, 1].imshow(imperv)
#     axs[0, 1].set_title('Impervious Surface Fraction [0-1]')
#     cbar2 = fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)

#     im3 = axs[0, 2].imshow(perc_build)
#     axs[0, 2].set_title('Building Surface Fraction [0-1]')
#     cbar3 = fig.colorbar(im3, ax=axs[0, 2], shrink=0.8)

#     im4 = axs[1, 0].imshow(canopy_height)
#     axs[1, 0].set_title('Tree Canopy Height [0-1]')
#     cbar4 = fig.colorbar(im4, ax=axs[1, 0], shrink=0.8)

#     im5 = axs[1, 1].imshow(buildings, vmax=0.2)
#     axs[1, 1].set_title('Buildings [0-1]')
#     cbar5 = fig.colorbar(im5, ax=axs[1, 1], shrink=0.8)

#     # Remove the axis for the blank subplot
#     axs[1, 2].axis('off')

#     # Remove x and y ticks from every subplot
#     for ax in axs.flat:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(ax.get_title(), fontsize=10)

#     # Adjust the spacing between subplots
#     plt.tight_layout()


#     plt.show()


# def plot_dynamic_map(map, legend):
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Extract class keys and corresponding colors in order
#     class_keys = sorted(legend.keys())
#     class_colors = [legend[key][1] for key in class_keys]
#     cmap = ListedColormap(class_colors, name='LCZ classes colormap')

#     # Create a mapping from class values to the colormap indices
#     class_to_cmap_index = {class_key: idx for idx, class_key in enumerate(class_keys)}
#     mapped_array = np.vectorize(class_to_cmap_index.get)(map)

#     # Plot the map array using the colormap
#     im = ax.imshow(mapped_array, cmap=cmap)

#     # Generate the legend only for the classes present in the data
#     unique_classes = np.unique(map)
#     legend_labels = []
#     legend_colors = []

#     for class_val in unique_classes:
#         if class_val in legend:
#             legend_labels.append(legend[class_val][0])
#             legend_colors.append(legend[class_val][1])

#     # Create patches for the legend
#     patches = [mpatches.Patch(color=color, label=label)
#                for label, color in zip(legend_labels, legend_colors)]
#     ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#     # Add the title
#     ax.set_title("Classified map")

#     # Remove axes for cleaner visualization
#     ax.axis('off')

#     # Display the plot
#     plt.tight_layout()
#     plt.show()