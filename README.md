# 1. Introduction

COSMOS is a computational tool crafted to overcome the challenges associated 
with integrating spatially resolved multi-omics data. This software harnesses a 
graph neural network algorithm to deliver cutting-edge solutions for analyzing 
biological data that encompasses various omics types within a spatial framework. 
Key features of COSMOS include domain segmentation, effective visualization, and 
the creation of spatiotemporal maps. These capabilities empower researchers to 
gain a deeper understanding of the spatial and temporal dynamics within 
biological samples, distinguishing COSMOS from other tools that may only support 
single omics types or lack comprehensive spatial integration. The proven 
superior performance of COSMOS underscores its value as an essential resource in 
the realm of spatial omics.

Paper: Cooperative Integration of Spatially Resolved Multi-Omics Data with 
COSMOS

# 2. Result

Below is an example to show modality weights of two omics in COSMOS.

![Fig](/images/modality_weights_of_two_omics_in_COSMOS.png)

Below is an example of the domain segmentation by COSMOS integration.

![Fig](/images/domain_segmentation_by_COSMOS_integration_result.png)

Below is an example of UMAP visualization of COSMOS integration.

![Fig](/images/UMAP_visualization_of_COSMOS_integration.png)

Below is an example of pseudo-spatiotemporal map (pSM) from COSMOS integration.

![Fig](/images/pseudo_spatiotemporal_map_from_COSMOS_integration.png)
    
# 3. Environment setup and code compilation

__3.1. Download the package__

The package can be downloaded by running the following command in the terminal:
```
git clone https://github.com/Lin-Xu-lab/COSMOS.git
```
Then, use
```
cd COSMOS
```
to access the downloaded folder. 

If the "git clone" command does not work with your system, you can download the 
zip file from the website 
https://github.com/Lin-Xu-lab/COSMOS.git and decompress it. Then, the folder 
that you need to access is COSMOS-main. 

__3.2. Environment setup__

The package has been successuflly tested in a Linux environment of python 
version 3.8.8, pandas version 1.5.2, and so on. An option to set up 
the environment is to use Conda 
(https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

You can use the following command to create an environment for SpaSNE:
```
conda create -n cosmos python=3.8 pandas=1.5.2 numpy=1.22.4 scanpy=1.8.2 matplotlib=3.6.2 umap-learn=0.5.2 scikit-learn=0.24.1 seaborn=0.11.1 torch=1.13.1 networkx=3.0 gudhi=3.7.1 anndata=0.8.0 cmcrameri=1.5 pytorch-geometric=2.3.0
```

After the environment is created, you can use the following command to activate 
it:
```
conda activate cosmos
```

Please install Jupyter Notebook from https://jupyter.org/install. For example, 
you can run
```
pip install notebook
```
in the terminal to install the classic Jupyter Notebook.  

__3.3. Import spasne in different directories (optional)__

If you would like to import spasne in different directories, there is an option 
to make it work. Please run
```
python setup.py install --user &> log
```
in the terminal.

After doing these successfully, you are supposed to be able to import COSMOS 
when you are using Python or Jupyter Notebook in other folders:
```
import COSMOS
```

# 4. Example

Below is the notebook script for the Mouse Visual Cortex example. First, please 
type
```
cd COSMOS-demos
```
in the terminal to enter the "COSMOS-demos" folder.

Then, type
```
jupyter notebook &
```
to open the Jupyter Notebook. Left click the 
cosmos_mouseVisualCortex_example.ipynb file to open it. 

Run the code below to import packages and set random seed:
```
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from umap import UMAP
import sklearn
import seaborn as sns
from COSMOS import cosmos
from COSMOS.pyWNN import pyWNN 
import warnings
warnings.filterwarnings('ignore')
random_seed = 20
```
Loading data and transforming it to AnnData object
```
# Importing mouse visual cortex STARMap data
df_data = pd.read_csv('./MVC_counts.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_meta = pd.read_csv('./MVC_meta.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_pixels = df_meta.iloc[:,2:4]
df_labels = list(df_meta.iloc[:,1])
adata = sc.AnnData(X = df_data)
adata.obs['LayerName'] = df_labels
adata.obs['LayerName_2'] = list(df_meta.iloc[:,4])

# Spatial positions
adata.obsm['spatial'] = np.array(df_pixels)
adata.obs['x_pos'] = adata.obsm['spatial'][:,0]
adata.obs['y_pos'] = adata.obsm['spatial'][:,1]
label_type = ['L1','L2/3','L4','L5','L6','HPC/CC']
```
Generating synthetic spatially resolved paired multi-omics data
```
# Shuffling L4/L5 and L5/L6 of the original data, respectively.
index_all = [np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[0]])]
for k in range(1,len(label_type)):
    temp_idx = np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[k]])
    index_all.append(temp_idx)
index_int1 = np.array(list(index_all[2]) + list(index_all[3]))
index_int2 = np.array(list(index_all[4]) + list(index_all[3]))

# Adding Gaussian noise to each omics
adata2 = adata.copy()
np.random.seed(random_seed)
data_noise_1 = 1 + np.random.normal(0,0.05,adata.shape)
adata2.X[index_int1,:] = np.multiply(adata.X,data_noise_1)[np.random.permutation(index_int1),:]

adata3 = adata.copy()
np.random.seed(random_seed+1)
data_noise_2 = 1 + np.random.normal(0,0.05,adata.shape)
adata3.X[index_int2,:] = np.multiply(adata.X,data_noise_2)[np.random.permutation(index_int2),:]
```

Applying COSMOS to integrate two omics
```
# COSMOS integration
cosmos_ebdg = cosmos.Cosmos(adata1=adata2,adata2=adata3)
cosmos_ebdg.preprocessing_data(n_neighbors = 10)
cosmos_ebdg.train(spatial_regularization_strength=0.05, z_dim=50, 
         lr=1e-3, wnn_iter = 200, epochs=1000, max_patience=30, min_stop=200, 
         random_seed=random_seed, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)
```
Showing modality weights of two omics in COSMOS
```
def plot_weight_value(alpha, label, modality1='omics1', modality2='omics2',order = None):
    df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
    df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
    df['label'] = label
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_COSMOS', 'Modality', 'Weight value']
    matplotlib.rcParams['font.size'] = 8.0
    fig, axes = plt.subplots(1, 1, figsize=(5,3))
    ax = sns.violinplot(data=df, x='label_COSMOS', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False, orient = 'v', order=order)
    ax.set_title(modality1 + ' vs ' + modality2) 
    plt.tight_layout(w_pad=0.05)

weights = cosmos_ebdg.weights
df_wghts = pd.DataFrame(weights,columns = ['w1','w2'])
weights = np.array(df_wghts)
for k in range(1,len(label_type)):
    wghts_mean = np.mean(weights[index_all[0],:],0)
for k in range(1,len(label_type)):
    wghts_mean_temp = np.mean(weights[index_all[k],:],0)
    wghts_mean = np.vstack([wghts_mean, wghts_mean_temp])
df_wghts_mean = pd.DataFrame(wghts_mean,columns = ['w1','w2'],index = label_type)
df_sort_mean = df_wghts_mean.sort_values(by=['w1'])
plot_weight_value(np.array(df_wghts), np.array(adata.obs['LayerName']), order = list(df_sort_mean.index))

```
![Fig](/images/modality_weights_of_two_omics_in_COSMOS.png)

Domain segmentation by COSMOS integration
```
# Searching the optimal clustering resolution by "leiden" to give the best ARI
def opt_resolution(df_embedding, labels, res_s = 0.1, res_e = 1.0, step = 0.1,n_cluster = None):
    max_ari = 0
    opt_res = 0
    while max_ari == 0:
        for res in np.arange(res_s,res_e,step):
            embedding_adata = sc.AnnData(df_embedding)
            sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')
            sc.tl.leiden(embedding_adata, resolution=float(res))
            clusters = list(embedding_adata.obs["leiden"])
            ARI_score = sklearn.metrics.adjusted_rand_score(labels, clusters)
            ARI_score = round(ARI_score, 2)
            cluster_num = len(np.unique(clusters))
            print('res = ' + str(round(res, 2)) + ', Cluster# = ' + str(cluster_num))
            if n_cluster:
                if ARI_score > max_ari and len(np.unique(clusters)) == n_cluster:
                    max_ari = ARI_score
                    opt_res = res
                    opt_clusters = clusters
                    print('res = ' + str(round(res, 2)) + ', ARI = ' + str(ARI_score))
                if res > opt_res and cluster_num > n_cluster:
                    break
            else:   
                if ARI_score > max_ari:
                    max_ari = ARI_score
                    opt_res = res
                    opt_clusters = clusters
                    print('res = ' + str(res) + ', ARI = ' + str(ARI_score))
        if max_ari == 0:
            n_cluster = n_cluster - 1
    return opt_res, max_ari, opt_clusters

# Obtaining the optimal domain segmentation
df_embedding = pd.DataFrame(cosmos_ebdg.embedding)
opt_res, max_ari, opt_clusters = opt_resolution(df_embedding,list(adata.obs['LayerName']),res_s = 0.2, res_e = 1, step = 0.05,n_cluster = 6)

adata_new = adata.copy()
adata_new.obs['Cluster'] = opt_clusters
adata_new.obs["Cluster"]=adata_new.obs["Cluster"].astype('category')

matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(2, 1, figsize=(6,8))
sz = 80
plot_color=['#D1D1D1','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']

domains="LayerName"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'Manual annotation' 
ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes[0])
ax.axis('off')

domains="Cluster"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'COSMOS, ARI = ' + str(max_ari)
ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes[1])
ax.axis('off')

```
res = 0.2, Cluster# = 5
res = 0.25, Cluster# = 5
res = 0.3, Cluster# = 5
res = 0.35, Cluster# = 6
res = 0.35, ARI = 0.79
res = 0.4, Cluster# = 6
res = 0.45, Cluster# = 6
res = 0.5, Cluster# = 6
res = 0.55, Cluster# = 7
(-282.4831346765602, 6759.627661884987, -630.5286230210683, 14490.623535171215)
![Fig](/images/domain_segmentation_by_COSMOS_integration_result.png)

UMAP visualization of COSMOS integration
```
adata_pt = sc.AnnData(df_embedding)
umap_2d = UMAP(n_components=2, init='random', random_state=random_seed, min_dist = 0.3,n_neighbors=30,metric = "cosine")
umap_pos = umap_2d.fit_transform(df_embedding)
adata_pt.obs['x_pos'] = list(adata_new.obs['x_pos'])
adata_pt.obs['y_pos'] = list(adata_new.obs['y_pos'])
adata_pt.obs['LayerName'] = list(adata_new.obs['LayerName'])
adata_pt.obs['cosmos_umap_pos_x'] = umap_pos[:,0]
adata_pt.obs['cosmos_umap_pos_y'] = umap_pos[:,1]

matplotlib.rcParams['font.size'] = 12.0
sz = 20
fig, axes = plt.subplots(1, 1, figsize=(3,3))

plot_color=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']

domains="LayerName"
num_celltype=len(adata_pt.obs[domains].unique())
adata_pt.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'UMAP by COSMOS' 
ax=sc.pl.scatter(adata_pt,alpha=1,x="cosmos_umap_pos_x",y="cosmos_umap_pos_y",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')

```
(-4.159879457950592,
 14.219783318042754,
 -7.029114389419556,
 15.366010332107544)

![Fig](/images/UMAP_visualization_of_COSMOS_integration.png)

Pseudo-spatiotemporal map (pSM) from COSMOS integration
```
sc.pp.neighbors(adata_pt, n_neighbors=20, use_rep='X')
# Setting the root to be the first cell in 'HPC' cells
adata_pt.uns['iroot'] = np.flatnonzero(adata.obs['LayerName_2'] == 'HPC')[0]
# Diffusion map
sc.tl.diffmap(adata_pt)
# Diffusion pseudotime
sc.tl.dpt(adata_pt)
pSM_values = adata_pt.obs['dpt_pseudotime'].to_numpy()

matplotlib.rcParams['font.size'] = 12.0
sz = 20
fig, axes = plt.subplots(1, 2, figsize=(7,3))

x = np.array(adata_pt.obs['cosmos_umap_pos_x'])
y = np.array(adata_pt.obs['cosmos_umap_pos_y'])
ax_temp = axes[0]
im = ax_temp.scatter(x, y, s=sz, c=pSM_values, marker='.', cmap='coolwarm',alpha = 1)
ax_temp.axis('off')
ax_temp.set_title('pSM in UMAP')
fig.colorbar(im, ax = ax_temp,orientation="vertical", pad=-0.01)

x = np.array(adata_pt.obs['y_pos'])
y = np.array(adata_pt.obs['x_pos'])
ax_temp = axes[1]
im = ax_temp.scatter(x, y, s=sz, c=pSM_values, marker='.', cmap='coolwarm',alpha = 1)
ax_temp.axis('off')
ax_temp.set_title('pSM in image')
fig.colorbar(im, ax = ax_temp,orientation="vertical", pad=-0.01)

plt.tight_layout()

```
![Fig](/images/pseudo_spatiotemporal_map_from_COSMOS_integration.png)

# 5. Contact information

Please contact our team if you have any questions:

Yuansheng Zhou (Yuansheng.Zhou@UTSouthwestern.edu)

Xue Xiao (Xiao.Xue@UTSouthwestern.edu)

Lei Dong (Lei.Dong@UTSouthwestern.edu)

Chen Tang (Chen.Tang@UTSouthwestern.edu)

Lin Xu (Lin.Xu@UTSouthwestern.edu)

Please contact Chen Tang for questions related to environment setting, software 
installation, and this GitHub page.

# 6. Copyright information 

The COSMOS software uses the BSD 3-clause license. Please see the "LICENSE" file
for the copyright information.
