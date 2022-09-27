#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')

## Network
import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import pylab as plt 
from itertools import count 
from operator import itemgetter 
from networkx.drawing.nx_agraph import graphviz_layout 
import pylab


# In[2]:



df = pd.read_csv('C:\MXH_code\EDGE4.csv',encoding='utf-8')
print(len(df))


# In[3]:


pd.set_option('precision',10)
G = nx.from_pandas_edgelist(df, 'Source', 'Target', create_using = nx.Graph())

nodes = G.nodes()
degree = G.degree()
colors = [degree[n] for n in nodes]
#size = [(degree[n]) for n in nodes]

pos = nx.kamada_kawai_layout(G)
#pos = nx.spring_layout(G, k = 0.2)
cmap = plt.cm.viridis_r
cmap = plt.cm.Greys

vmin = min(colors)
vmax = max(colors)

fig = plt.figure(figsize = (9,12), dpi=100)

nx.draw(G,pos,alpha = 0.8, nodelist = nodes, node_color = 'r', node_size = 10, with_labels= False,font_size = 6, width = 0.2, cmap = cmap, edge_color ='#808080')
fig.set_facecolor('#FFFFCC')

plt.legend()
plt.show()


# In[4]:


for i in sorted(G.nodes()):
    G.nodes[i]['Degree'] = G.degree(i)
    
nodes_data = pd.DataFrame([i[1] for i in G.nodes(data=True)], index=[i[0] for i in G.nodes(data=True)])
nodes_data = nodes_data.sort_index(ascending= False)
nodes_data.index.names=['ID']
nodes_data.reset_index(level=0, inplace=True)

print(nodes_data)


# In[5]:


nodes_data.sort_values(by='Degree', ascending=False)


# In[6]:


#betweenness_centrality
bet_cen = nx.betweenness_centrality(G)
df_bet_cen = pd.DataFrame.from_dict(bet_cen, orient='index')
df_bet_cen.columns = ['betweenness_centrality']
df_bet_cen.index.names = ['ID']
df_bet_cen.reset_index(level=0, inplace=True)
Betweenness = pd.merge(nodes_data,df_bet_cen, on = ['ID'])


# In[9]:


print(Betweenness)


# In[11]:


Betweenness.sort_values(by='Degree', ascending=False)


# In[18]:


clust_coefficients = nx.clustering(G)
df_clust = pd.DataFrame.from_dict(clust_coefficients, orient='index')
df_clust.columns = ['clust_coefficient']
df_clust.index.names = ['ID']
df_clust.reset_index(level=0, inplace=True)
Clustering = pd.merge(nodes_data, df_clust, on = ['ID'])


# In[19]:


print(Clustering)


# In[20]:


Clustering.sort_values(by='Degree', ascending=False)


# In[21]:


# Closeness centrality
clo_cen = nx.closeness_centrality(G)
df_clo = pd.DataFrame.from_dict(clo_cen, orient='index')
df_clo.columns = ['closeness_centrality']
df_clo.index.names = ['ID']
df_clo.reset_index(level=0, inplace=True)
Closeness = pd.merge(nodes_data, df_clo, on = ['ID'])


# In[24]:


Closeness.sort_values(by='Degree', ascending=False)


# In[27]:


#Harmonic Centrality
har = nx.harmonic_centrality(G)
df_har = pd.DataFrame.from_dict(har, orient='index')
df_har.columns = ['harmonic_centrality']
df_har.index.names = ['ID']
df_har.reset_index(level=0, inplace=True)
Harmonic = pd.merge(nodes_data, df_har, on = ['ID'])


# In[28]:


Harmonic.sort_values(by='Degree', ascending=False)


# In[29]:


eig_cen = nx.eigenvector_centrality_numpy(G)
df_eig = pd.DataFrame.from_dict(eig_cen, orient='index')
df_eig.columns = ['eigenvector_centrality']
df_eig.index.names = ['ID']
df_eig.reset_index(level=0, inplace=True)
Eigenvector = pd.merge(nodes_data, df_eig, on = ['ID'])


# In[30]:


Eigenvector.sort_values(by='Degree', ascending=False)


# In[31]:


page_rank = nx.pagerank(G)
df_page_rank = pd.DataFrame.from_dict(page_rank, orient='index')
df_page_rank.columns = ['page_rank']
df_page_rank.index.names = ['ID']
df_page_rank.reset_index(level=0, inplace=True)
Page = pd.merge(nodes_data,df_page_rank, on = ['ID'])


# In[32]:


Page.sort_values(by='Degree', ascending=False)


# In[34]:


def girvan_newman(graph):
	# find number of connected components
	sg = nx.connected_components(graph)
	sg_count = nx.number_connected_components(graph)

	while(sg_count == 1):
		graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
		sg = nx.connected_components(graph)
		sg_count = nx.number_connected_components(graph)

	return sg


# In[35]:


def edge_to_remove(graph):
  G_dict = nx.edge_betweenness_centrality(graph)
  edge = ()

  # extract the edge with highest edge betweenness centrality score
  for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse = True):
      edge = key
      break

  return edge


# In[37]:


# find communities in the graph
c = girvan_newman(G.copy())

# find the nodes forming the communities
node_groups = []

for i in c:
node_groups.append(list(i))


# In[38]:


node_groups


# In[39]:


color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else: 
        color_map.append('green')  

nx.draw(G, node_color=color_map, with_labels=True)
plt.show()


# In[40]:


len(G.nodes)


# In[ ]:




