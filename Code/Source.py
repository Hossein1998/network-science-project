"""
Author: Hossein Rafiee zade
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import random 
import pandas as pd


ax = ox.graph_from_place('Rafsanjan', network_type='drive')
ax = ax.to_undirected()
G=ox.project_graph(ax)
ox.plot.plot_graph(G,filepath="image2.png" , save=True)


nx.info(G)

k = sum([v for k, v in G.degree()]) / len(G)

G = nx.Graph(G)
nx.info(G)

k = sum([v for k, v in G.degree()]) / len(G)

between =  nx.betweenness_centrality(G)

# plot it
df = pd.DataFrame(data=pd.Series(between).sort_values(), columns=['cc'])
df['colors'] = ox.plot.get_colors(n=len(df), cmap='inferno', start=0.2)
df = df.reindex(G.nodes())
nc = df['colors'].tolist()
fig, ax = ox.plot_graph(ax, bgcolor='k', node_size=5, node_color=nc, node_edgecolor='none', node_zorder=2,
                        edge_color='#555555', edge_linewidth=1.5, edge_alpha=1 ,filepath="between.png" , save=True)

close =  nx.closeness_centrality(G)

# plot it
df = pd.DataFrame(data=pd.Series(close).sort_values(), columns=['cc'])
df['colors'] = ox.plot.get_colors(n=len(df), cmap='inferno', start=0.2)
df = df.reindex(G.nodes())
nc = df['colors'].tolist()
fig, ax = ox.plot_graph(ax, bgcolor='k', node_size=5, node_color=nc, node_edgecolor='none', node_zorder=2,
                        edge_color='#555555', edge_linewidth=1.5, edge_alpha=1,filepath="close.png" , save=True)

degree =  nx.degree_centrality(G)

# plot it
df = pd.DataFrame(data=pd.Series(degree).sort_values(), columns=['cc'])
df['colors'] = ox.plot.get_colors(n=len(df), cmap='inferno', start=0.2)
df = df.reindex(G.nodes())
nc = df['colors'].tolist()
fig, ax = ox.plot_graph(ax, bgcolor='k', node_size=5, node_color=nc, node_edgecolor='none', node_zorder=2,
                        edge_color='#555555', edge_linewidth=1.5, edge_alpha=1,filepath="degree.png" , save=True)

PageRank=nx.pagerank(G)

# plot it
df = pd.DataFrame(data=pd.Series(PageRank).sort_values(), columns=['cc'])
df['colors'] = ox.plot.get_colors(n=len(df), cmap='inferno', start=0.2)
df = df.reindex(G.nodes())
nc = df['colors'].tolist()
fig, ax = ox.plot_graph(ax, bgcolor='k', node_size=5, node_color=nc, node_edgecolor='none', node_zorder=2,
                        edge_color='#555555', edge_linewidth=1.5, edge_alpha=1,filepath="PageRank.png" , save=True)

ox.plot_graph(ox.graph_from_place('Rafsanjan', network_type='drive'), bgcolor='k',filepath="image5.png", node_size = 4, 
                                node_color='#999999', node_edgecolor='none', node_zorder=1,
                                edge_color='#555555', edge_linewidth = 0.3, edge_alpha=1,save=True)

ox.io.save_graphml(ox.graph_from_place('Rafsanjan', network_type='drive'), filepath=None, gephi=True, encoding='utf-8')

NumnerOfNodes=G.number_of_nodes()
NumnerOfedges=G.number_of_edges()
print(NumnerOfNodes)
print(NumnerOfedges)

degrees = [val for (node, val) in G.degree()]

Prob_list=[]
for i in range(1,7):
  Prob_list.append(degrees.count(i)/NumnerOfNodes)

Deg_list=[1,2,3,4,5,6]

plt.figure()
plt.plot(Deg_list,Prob_list)
plt.xlabel("k", fontsize=15)
plt.ylabel("P(k)", fontsize=15)
plt.title("Degree distribution", fontsize=15)
plt.savefig('fig00.png')
plt.grid(True)
plt.show(True)

plt.loglog(Deg_list,Prob_list)
plt.xlabel("k", fontsize=15)
plt.ylabel("P(k)", fontsize=15)
plt.title("Degree distribution", fontsize=15)
plt.savefig('fig2.png')
plt.show(True)

degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

import powerlaw
fit = powerlaw.Fit(degree_sequence,xmin=1)
fit.alpha

fig2 = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig2)
plt.savefig('fig3.png')

R, p = fit.distribution_compare('exponential','power_law',normalized_ratio=True)

R, p = fit.distribution_compare('exponential','lognormal_positive',normalized_ratio=True)

diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(G)])

degree_dic = Counter(dict(G.degree()).values())

degree_hist = pd.DataFrame({"degree": list(degree_dic.values()),
                            "Number of Nodes": list(degree_dic.keys())})
plt.figure(figsize=(20,10))
sns.barplot(y = 'degree', x = 'Number of Nodes', 
              data = degree_hist, 
              color = 'darkblue')
plt.xlabel('Node Degree', fontsize=30)
plt.ylabel('Number of Nodes', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

G2 = G.to_undirected()
nx.draw(G2, pos=nx.spiral_layout(G2,scale=4, center=None, dim=2, resolution=0.5, equidistant=False), node_size=0.03, width=0.1, node_color='lightblue' , edge_color='k')
plt.savefig('fig5.png')

G2 = G.to_undirected()
nx.draw(G2, pos=nx.spring_layout(G2,scale=2), node_size=0.03, width=0.1)
plt.savefig('fig6.png')

g = G.to_undirected()
pos=nx.spring_layout(g)
fig, ax = plt.subplots()
fig.set_facecolor('black')
nx.draw(g,pos,node_color='purple',edge_color='lime',width=0.1,edge_cmap=plt.cm.Blues,node_size=1, node_shape='^')
ax.axis('off')
fig.set_facecolor('black')
plt.savefig("all_graph.pdf")
plt.savefig("all_graph.png")
print(nx.info(g))

nx.node_connectivity(G)

density=nx.density(G)

sorted_x = sorted(between.items(), key=operator.itemgetter(1), reverse=True)
rand_x = list(range(0,4426 ))

random.shuffle(rand_x)
between_giant = []
between_rand = []
avg_degs = []
G_simple = nx.Graph(G)
G_simple2 = nx.Graph(G)

for x in range(3000):
 
        remove = sorted_x[x]      
        remove2 = sorted_x[rand_x[x]]
        G_simple.remove_nodes_from(remove)
        G_simple2.remove_nodes_from(remove2)
        
        connected_component_subgraphs1 = (G_simple.subgraph(c) for c in nx.connected_components(G_simple))
        connected_component_subgraphs2 = (G_simple2.subgraph(c) for c in nx.connected_components(G_simple2))

        giant = len(max(connected_component_subgraphs1, key=len))
        giant2 = len(max(connected_component_subgraphs2, key=len))

        between_giant.append(giant)
        between_rand.append(giant2)

y1 = between_giant
y2 = between_giant

y1= y1[ :-1]
y2= y2[1: ]

perc = np.linspace(0,100,len(between_giant))
fig = plt.figure(1, (12,8))
ax = fig.add_subplot(1,1,1)

ax.plot(perc, between_giant)
ax.plot(perc, between_rand)

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)
ax.set_xlabel('Fraction of Nodes Removed',fontsize=25)
ax.set_ylabel('Giant Component Size',fontsize=25)
ax.legend(['betweenness','random'],fontsize=20)
plt.savefig('fig7.png')
plt.show()

import folium
import osmnx as ox
import networkx as nx

ox.config(use_cache=True, log_console=True)

G = ox.graph_from_point((30.40631, 55.9821), dist=3000, network_type='drive')

G = ox.speed.add_edge_speeds(G)
G = ox.speed.add_edge_travel_times(G)

orig = ox.get_nearest_node(G, (30.40631, 55.9821))
dest = ox.get_nearest_node(G, (30.39866, 55.98592))
route = nx.shortest_path(G, orig, dest, 'travel_time')

route_map = ox.plot_route_folium(G, route)
route_map.save('test2.html')