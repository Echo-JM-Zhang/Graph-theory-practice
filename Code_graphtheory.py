import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import zscore
from time import time
import os

# Transfer raw data to correlation data
data_path = r"D:\JingminZhang\rutsuko's lab\DataAnalysis"
savepath = r"D:\JingminZhang\rutsuko's lab\DataAnalysis"
if not os.path.exists(savepath):
    os.mkdir(savepath)

name_list = ["APPROACH", "AVOID", "AACONFLICT", "Homecage"]
fig, axes = plt.subplots(1, 4, figsize=(25, 5))
for i, item in enumerate(name_list):
    df = pd.read_csv(os.path.join(data_path, 'Network_%s.csv' % item))
    corrMatrix = df.corr()
    sns.heatmap(ax=axes[i], data=corrMatrix, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    axes[i].set_title(item)


    corrMatrix.insert(0, '', corrMatrix.columns)
    filename = "Corr_%s.csv" % item
    with open(os.path.join(savepath, filename), "w") as f:
        f.write(corrMatrix.to_csv(index=False))
plt.savefig(os.path.join(savepath,"Correlation_matrix.png"))
plt.close()
plt.show()

# Use Graph Theory to build brain network
avg_clustering_result = []
for group in name_list:
    # read-in original data
    df = pd.read_csv(os.path.join(savepath,"Corr_%s.csv" % group))
    Names_All = df[df.columns[0]].to_list()
    for i, item in enumerate(Names_All):
        if item[-1] == " ":
            item = item[0:-1]
        if " " in item:
            Names_All[i] = item.replace(" ", "\n")

    df = df.drop(columns=[df.columns[0]])
    r_matrix = df.to_numpy()

    # diag to np.nan, then get threshold
    for i in range(len(r_matrix)):
        r_matrix[i][i] = np.nan
    # threshold_max = np.nanpercentile(r_matrix, 95)
    threshold_max = 0.56
    # N=7, pearson r p = 0.05 single tail

    # get index and names pairs
    r_matrix[np.isnan(r_matrix)] = 0
    r_matrix[r_matrix < threshold_max] = np.nan
    idx = np.where(~np.isnan(r_matrix))
    idx = np.asarray(idx).T
    all_edges = []
    for i in range(len(idx)):
        all_edges.append((Names_All[idx[i][0]], Names_All[idx[i][1]],r_matrix[idx[i][0],idx[i][1]]))

    # create graph class
    G = nx.Graph()
    G.add_weighted_edges_from(all_edges)

    # Some basic analyses
    centrality_result = nx.degree_centrality(G)
    betweeness_result = nx.betweenness_centrality(G)
    clustering_result = nx.clustering(G, weight="weight")
    avg_clustering = nx.average_clustering(G, weight="weight")
    avg_clustering_result.append(avg_clustering)

    # Get results for each brain region (label)
    all_labels = centrality_result.keys()
    centrality_y, clustering_y, betweeness_y = [], [], []
    for label in all_labels:
        centrality_y.append(centrality_result[label])
        clustering_y.append(clustering_result[label])
        betweeness_y.append(betweeness_result[label])

    # prepare data to plot
    centrality_y = np.asarray(centrality_y)
    clustering_y = np.asarray(clustering_y)
    betweeness_y = np.asarray(betweeness_y)
    x = np.arange(len(all_labels))

    # add sort, all sorted by node_y, descending
    idx = np.argsort(centrality_y)[::-1]
    centrality_y, clustering_y, betweeness_y= centrality_y[idx], clustering_y[idx], betweeness_y[idx]
    all_labels = np.asarray(list(all_labels))
    all_labels = all_labels[idx]
    for i, item in enumerate(all_labels):
        if "\n" in item:
            all_labels[i] = item.replace("\n", "")

    # make plot
    fig = plt.figure(figsize=[8, 4], dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.bar(x, centrality_y)
    ax1.set_ylabel("Degree Centrality")
    plt.xticks(x, all_labels, rotation=80, fontsize=6)

    ax2 = fig.add_subplot(1, 3, 2)
    plt.bar(x, betweeness_y)
    ax2.set_ylabel("Betweeness Centrality")
    plt.xticks(x, all_labels, rotation=80, fontsize=6)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_ylabel("Clustering Coefficient")
    plt.bar(x, clustering_y)
    plt.xticks(x, all_labels, rotation=80, fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(savepath,"Group%s_CC_weight.png" % group))
    plt.close()
    plt.show()

    fig = plt.figure(figsize=[5, 5], dpi=200)
    node_sizes = [centrality_result[node]*20*400 for node in G.nodes]
    edge_sizes= [G.edges[pair]['weight']*3.6-1.6 for pair in G.edges]

    # nx.draw(G, pos=nx.kamada_kawai_layout(G),
    #         alpha=0.5, edgecolors="k",
    #         width=edge_sizes,
    #         node_color="r", edge_color="b",
    #         with_labels=True,
    #         font_size=8, font_color="white")
    nx.draw_networkx_nodes(G, pos=nx.kamada_kawai_layout(G),
                           alpha=0.7, node_size=node_sizes,
                           node_color="r", edgecolors="#8B0000")
    nx.draw_networkx_edges(G, pos=nx.kamada_kawai_layout(G),
                           width=edge_sizes, edge_color="0.4")
    nx.draw_networkx_labels(G, pos=nx.kamada_kawai_layout(G),
                            font_size=7, font_color="0.3")
    plt.savefig(os.path.join(savepath, "Group%s_weight.png" % group))
    plt.close()
    plt.show()
