import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt

def run_umap(
            y_seen, 
            activations, 
            num_class = 2, 
            title = '', 
            colours = ['red', 'blue'],
            annotate = [],
            figsize = (16,9),
            show_plt = False,
            save_plt_path = ''
            ):
    '''
    y_seen[i] corrspound to activations[i]

    @params:
        y_seen: list or numpy true class idx, make sure moved to cpu or typecasted to numpy array
        activations: list of activations
        colours: ensure len(colours) == num_class
        save_plt_path: if not none, then save to path
    '''
    if not show_plt: plt.ioff


    projector = umap.UMAP(n_neighbors=5, random_state=133)
    embedding = projector.fit_transform(activations)

    # split the embedding into glaucoma vs healthy
    Y_idx, embeddings, fn = [], [], []
    for i in range(num_class): Y_idx.append(np.where(y_seen == i)[0])
    for i in range(num_class): 
        embeddings.append(embedding[ Y_idx[i] ])
        if len(annotate) != 0: fn.append( annotate[Y_idx[i]] )

    


    # annotate ref: https://stackoverflow.com/questions/14432557/scatter-plot-with-different-text-at-each-data-point
    fig, ax = plt.subplots(figsize= figsize)
    ax.set_title(title)
    for i in range(num_class): 
        ax.scatter(embeddings[i][:, 0], embeddings[i][:, 1], color = colours[i])
        
        # annotate each pt
        if len(annotate) == 0: continue
        for j, txt in enumerate(fn[i] ): 
            ax.annotate(txt, (embeddings[i][j,0], embeddings[i][j,1] ))
    #ax.gca().set_aspect('equal', 'datalim')

    if save_plt_path: plt.savefig(save_plt_path)

    if show_plt: plt.show()
    else: plt.close()
