from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def tsne(
        X, 
        y, 
        title = '', 
        figsize = (8,6), 
        annotate = [],
        show_plt = True,
        save_plt_path = ''
        ):
    if not show_plt: plt.ioff

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    # split the embedding into glaucoma vs healthy
    num_class = 2
    colours = ['orangered', 'dodgerblue']
    Y_idx, embeddings, fn = [], [], []
    for i in range(num_class): Y_idx.append(np.where(np.array(y) == i)[0])

    for i in range(num_class): 
        embeddings.append(X_embedded[ Y_idx[i] ])
        if len(annotate) > 0: fn.append( np.array(annotate)[Y_idx[i]] )

    # annotate ref: https://stackoverflow.com/questions/14432557/scatter-plot-with-different-text-at-each-data-point
    fig, ax = plt.subplots(figsize= figsize)
    ax.set_title(title)
    for i in range(num_class): 
        ax.scatter(embeddings[i][:, 0], embeddings[i][:, 1], color = colours[i])
        
        if len(annotate) > 0:
            # annotate each pt
            for j, txt in enumerate(fn[i] ): 
                ax.annotate(txt, (embeddings[i][j,0], embeddings[i][j,1] ))
    
    if save_plt_path: plt.savefig(save_plt_path)
    if show_plt: plt.show()
    else: plt.close()


from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def tsne_seaborn(
        X, 
        y, 
        num_class = 2,
        title='', 
        x_label = '',
        y_label = '',
        figsize=(8, 6), 
        annotate=[],
        show_plt=True,
        save_plt_path='',
        dpi = 1200
        ):
    if not show_plt:
        plt.ioff()

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    colours = ['orangered', 'dodgerblue', 'forestgreen']
    Y_idx, embeddings, fn = [], [], []
    
    for i in range(num_class):
        Y_idx.append(np.where(np.array(y) == i)[0])

    for i in range(num_class): 
        embeddings.append(X_embedded[Y_idx[i]])
        if len(annotate) > 0: 
            fn.append(np.array(annotate)[Y_idx[i]])

    # Create a DataFrame for Seaborn
    data = []
    labels = []
    for i in range(num_class):
        data.extend(embeddings[i])
        labels.extend([i] * len(embeddings[i]))
    
    df = pd.DataFrame(data, columns=['Dimension 1', 'Dimension 2'])
    df['Class'] = labels

    # Plot using Seaborn
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
    ax.set_title(title)
    # Set custom axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Class', data=df, palette=sns.color_palette(colours[:num_class]))
    
    # Create a custom legend with specific labels and colors
    legend_labels = ['0', '1', '2'] # ie ['Healthy', 'Glaucoma']  
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colours[i], markersize=10) for i in range(num_class)]
    ax.legend(legend_handles, legend_labels, loc='upper right')

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    if len(annotate) > 0:
        for i in range(num_class):
            for j, txt in enumerate(fn[i]):
                ax.annotate(txt, (embeddings[i][j, 0], embeddings[i][j, 1]))

    if save_plt_path: 
        plt.savefig(save_plt_path)

    if show_plt: 
        plt.show()
    else: 
        plt.close()

