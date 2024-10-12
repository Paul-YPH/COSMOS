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
COSMOS, Zhou Y., X. Xiao, L. Dong, C. Tang, G. Xiao*, and L Xu*, 2024.  

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

You can use the following command to create an environment for COSMOS:
```
conda create -n cosmos python pandas numpy scanpy matplotlib umap-learn scikit-learn seaborn torch networkx gudhi anndata cmcrameri pytorch-geometric
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

__3.3. Import COSMOS in different directories (optional)__

If you would like to import COSMOS in different directories, there is an option 
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
Show list of software versions:
```
!pip list
```
Loading data and transforming it to AnnData object
```
# Importing mouse visual cortex STARMap data
df_data = pd.read_csv('./MVC_counts.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_meta = pd.read_csv('./MVC_meta.csv',sep=",",header=0,na_filter=False,index_col=0) 
df_pixels = df_meta.iloc[:,2:4]
df_labels = list(df_meta.iloc[:,1])
adata = sc.AnnData(X = df_data)
adata.obs['LayerName'] = df_labels # Combining HPC and CC
adata.obs['LayerName_2'] = list(df_meta.iloc[:,4]) # Separating HPC and CC

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
adata1 = adata.copy()
np.random.seed(random_seed)
data_noise_1 = 1 + np.random.normal(0,0.05,adata.shape)
adata1.X[index_int1,:] = np.multiply(adata.X,data_noise_1)[np.random.permutation(index_int1),:]

adata2 = adata.copy()
np.random.seed(random_seed+1)
data_noise_2 = 1 + np.random.normal(0,0.05,adata.shape)
adata2.X[index_int2,:] = np.multiply(adata.X,data_noise_2)[np.random.permutation(index_int2),:]
```

Applying COSMOS to integrate two omics
```
# COSMOS integration
cosmos_comb = cosmos.Cosmos(adata1=adata1,adata2=adata2)
cosmos_comb.preprocessing_data(n_neighbors = 10)
cosmos_comb.train(spatial_regularization_strength=0, z_dim=50, 
         lr=1e-3, wnn_epoch = 500, total_epoch=1000, max_patience_bef=10, max_patience_aft=30, min_stop=200, 
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

weights = cosmos_comb.weights
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
def screen_resolution(df_embedding, labels, res_s = 0.1, res_e = 1.0, step = 0.05, methods = 'leiden'):
    max_ari = 0
    opt_res = 0
    rec_ari = []
    rec_res = []
    rec_cluster_num = []
    for res in np.arange(res_s,res_e,step):
        embedding_adata = sc.AnnData(df_embedding)
        sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')
        if methods == 'leiden':
            sc.tl.leiden(embedding_adata, resolution=float(res))
            clusters = list(embedding_adata.obs["leiden"])
        else:
            sc.tl.louvain(embedding_adata, resolution=float(res))
            clusters = list(embedding_adata.obs["louvain"])
        ARI_score = sklearn.metrics.adjusted_rand_score(labels, clusters)
        ARI_score = round(ARI_score, 2)
        cluster_num = len(np.unique(clusters))
        print('res = ' + str(round(res, 2)) + ', ARI = ' + str(ARI_score) + ', Cluster# = ' + str(cluster_num))
        rec_ari.append(ARI_score)
        rec_res.append(res)
        rec_cluster_num.append(cluster_num)
    print('Maximal ARI = ' + str(max(rec_ari)) + ' with res = ' + str(round(rec_res[np.argmax(rec_ari)],2)))
    return rec_ari, rec_res, rec_cluster_num

# Obtaining the optimal domain segmentation

adata_new = adata1.copy()
label_annotations = list(adata.obs['LayerName'])

alpha = 0
res_s = 0.2
res_e = 0.5
step = 0.01
methods = 'leiden'
df_embedding = pd.DataFrame(cosmos_comb.embedding)
rec_ari, rec_res, rec_cluster_num = screen_resolution(df_embedding, label_annotations, res_s = res_s, res_e = res_e, step = step, methods = methods)

opt_ari_cosmos = max(rec_ari)
opt_res_cosmos = rec_res[np.argmax(rec_ari)]
embedding_adata = sc.AnnData(df_embedding)
sc.pp.neighbors(embedding_adata, n_neighbors=50, use_rep='X')
sc.tl.leiden(embedding_adata, resolution=float(opt_res_cosmos))
opt_clusters_cosmos = list(embedding_adata.obs["leiden"])
adata_new.obs['Cluster_cosmos'] = opt_clusters_cosmos
adata_new.obs["Cluster_cosmos"]=adata_new.obs["Cluster_cosmos"].astype('category')

matplotlib.rcParams['font.size'] = 12.0
fig, axes = plt.subplots(2, 1, figsize=(4,5))
sz = 40
plot_color=['#D1D1D1','#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']

domains="LayerName"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'Manual annotation' 
ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes[0])
ax.axis('off')

domains="Cluster_cosmos"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'COSMOS, ARI = ' + str(opt_ari_cosmos)
ax=sc.pl.scatter(adata_new,alpha=1,x="x_pos",y="y_pos",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes[1])
ax.axis('off')

```
res = 0.2, ARI = 0.79, Cluster# = 5
res = 0.21, ARI = 0.79, Cluster# = 5
res = 0.22, ARI = 0.79, Cluster# = 5
res = 0.23, ARI = 0.79, Cluster# = 5
res = 0.24, ARI = 0.8, Cluster# = 5
res = 0.25, ARI = 0.8, Cluster# = 5
res = 0.26, ARI = 0.8, Cluster# = 5
res = 0.27, ARI = 0.8, Cluster# = 5
res = 0.28, ARI = 0.8, Cluster# = 5
res = 0.29, ARI = 0.84, Cluster# = 6
res = 0.3, ARI = 0.84, Cluster# = 6
res = 0.31, ARI = 0.74, Cluster# = 6
res = 0.32, ARI = 0.74, Cluster# = 6
res = 0.33, ARI = 0.74, Cluster# = 6
res = 0.34, ARI = 0.74, Cluster# = 6
res = 0.35, ARI = 0.74, Cluster# = 6
res = 0.36, ARI = 0.74, Cluster# = 6
res = 0.37, ARI = 0.7, Cluster# = 6
res = 0.38, ARI = 0.74, Cluster# = 6
res = 0.39, ARI = 0.76, Cluster# = 6
res = 0.4, ARI = 0.75, Cluster# = 6
res = 0.41, ARI = 0.75, Cluster# = 6
res = 0.42, ARI = 0.75, Cluster# = 6
res = 0.43, ARI = 0.75, Cluster# = 6
res = 0.44, ARI = 0.75, Cluster# = 6
res = 0.45, ARI = 0.75, Cluster# = 6
res = 0.46, ARI = 0.75, Cluster# = 6
res = 0.47, ARI = 0.66, Cluster# = 8
res = 0.48, ARI = 0.76, Cluster# = 7
res = 0.49, ARI = 0.75, Cluster# = 7
Maximal ARI = 0.84 with res = 0.29
(-282.4831346765602, 6759.627661884987, -630.5286230210683, 14490.623535171215)
![Fig](/images/domain_segmentation_by_COSMOS_integration_result.png)

UMAP visualization of COSMOS integration
```
umap_2d = UMAP(n_components=2, init='random', random_state=random_seed, min_dist = 0.3,n_neighbors=30)
umap_pos = umap_2d.fit_transform(df_embedding)
adata_new.obs['cosmos_umap_pos_x'] = umap_pos[:,0]
adata_new.obs['cosmos_umap_pos_y'] = umap_pos[:,1]

matplotlib.rcParams['font.size'] = 12.0
sz = 20
fig, axes = plt.subplots(1, 1, figsize=(3,3))

plot_color=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', \
            '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#ffd8b1', '#800000', '#aaffc3', '#808000', '#000075', '#000000', '#808080', '#ffffff', '#fffac8']

domains="LayerName"
num_celltype=len(adata_new.obs[domains].unique())
adata_new.uns[domains+"_colors"]=list(plot_color[:num_celltype])
titles = 'UMAP by COSMOS' 
ax=sc.pl.scatter(adata_new,alpha=1,x="cosmos_umap_pos_x",y="cosmos_umap_pos_y",color=domains,title=titles ,color_map=plot_color,show=False,size=sz,ax = axes)
ax.axis('off')

```
(-8.883453440666198, 14.081739974021911, -6.550229287147522, 8.842858052253723)

![Fig](/images/UMAP_visualization_of_COSMOS_integration.png)

Pseudo-spatiotemporal map (pSM) from COSMOS integration
```
sc.pp.neighbors(embedding_adata, n_neighbors=20, use_rep='X')
# Setting the root to be the first cell in 'HPC' cells
embedding_adata.uns['iroot'] = np.flatnonzero(adata.obs['LayerName_2'] == 'HPC')[0]
# Diffusion map
sc.tl.diffmap(embedding_adata)
# Diffusion pseudotime
sc.tl.dpt(embedding_adata)
pSM_values = embedding_adata.obs['dpt_pseudotime'].to_numpy()

matplotlib.rcParams['font.size'] = 12.0
sz = 20
fig, axes = plt.subplots(1, 2, figsize=(7,3))

x = np.array(adata_new.obs['cosmos_umap_pos_x'])
y = np.array(adata_new.obs['cosmos_umap_pos_y'])
ax_temp = axes[0]
im = ax_temp.scatter(x, y, s=sz, c=pSM_values, marker='.', cmap='coolwarm',alpha = 1)
ax_temp.axis('off')
ax_temp.set_title('pSM in UMAP')
fig.colorbar(im, ax = ax_temp,orientation="vertical", pad=-0.01)

x = np.array(adata_new.obs['x_pos'])
y = np.array(adata_new.obs['y_pos'])
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

# 6. Copyright information 

The COSMOS software uses the BSD 3-clause license. Please see the "LICENSE" file
for the copyright information.
