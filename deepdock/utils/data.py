import torch
from plyfile import PlyData
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import FaceToEdge, Cartesian

from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
from rdkit import Chem
import pandas as pd
import numpy as np
import glob
import os

from deepdock.utils import mol2graph


def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if axis.name not in ['nx', 'ny', 'nz'] ])
    pos = torch.stack(features[:3], dim=-1)
    features = torch.stack(features[3:], dim=-1)

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = Data(x=features, pos=pos, face=face)

    return data
    
      
class PDBbind_protsurf_dataset(Dataset):
    def __init__(self, pdb_IDs, root, transform=None, pre_transform=None):
        super(PDBbind_protsurf_dataset, self).__init__()
        
        self.files = [os.path.join(root, i, i+'_pocket.ply') for i in pdb_IDs]
        
    def len(self):
        return len(self.files)

    def get(self, idx):
        mesh = read_ply(self.files[idx])
        data = FaceToEdge()(mesh)
        data = Cartesian()(data)
        return data
    

class PDBbind_complex_dataset(Dataset):
    def __init__(self, data_path, transform=None, pre_transform=None, 
                 min_target_nodes=None, max_target_nodes=None,
                 min_ligand_nodes=None, max_ligand_nodes=None):
        super(PDBbind_complex_dataset, self).__init__()
        
        self.data = torch.load(data_path)
        self.data = [i for i in self.data if not np.isnan(i[1].x.numpy().min())]
        if min_target_nodes:
            self.data = [i for i in self.data if len(i[1].x.numpy()) >= min_target_nodes]
        if max_target_nodes:
            self.data = [i for i in self.data if len(i[1].x.numpy()) <= max_target_nodes]
        if min_ligand_nodes:
            self.data = [i for i in self.data if len(i[0].x.numpy()) >= min_ligand_nodes]
        if max_ligand_nodes:
            self.data = [i for i in self.data if len(i[0].x.numpy()) <= max_ligand_nodes]
        
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
   
    
def compute_clusters(data, n_clusters):
    X = data[1].pos.numpy()
    connectivity = torch_geometric.utils.to_scipy_sparse_matrix(data[1].edge_index)
    
    if len(X) > n_clusters:
        ward = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward').fit(X)
        data[1].clus = torch.tensor(ward.labels_)
    else:
        data[1].clus = torch.tensor(range(len(X)))
        
    return data


def compute_cluster_batch_index(cluster, batch):
    max_prev_batch = 0
    for i in range(batch.max().item()+1):
        cluster[batch == i] += max_prev_batch
        max_prev_batch = cluster[batch == i].max().item() + 1
    return cluster
   

def Mol2MolSupplier (file=None, sanitize=True, cleanupSubstructures=True):
    # Taken from https://chem-workflows.com/articles/2020/03/23/building-a-multi-molecule-mol2-reader-for-rdkit-v2/
    mols=[]
    with open(file, 'r') as f:
        doc=[line for line in f.readlines()]

    start=[index for (index,p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
    finish=[index-1 for (index,p) in enumerate(doc) if '@<TRIPOS>MOLECULE' in p]
    finish.append(len(doc))
    
    try:
        name=[doc[index].rstrip().split('\t')[-1] for (index,p) in enumerate(doc) if 'Name' in p]
    except:
        pass
    
    interval=list(zip(start,finish[1:]))
    for n, i in enumerate(interval):
        block = ",".join(doc[i[0]:i[1]]).replace(',','')
        m = Chem.MolFromMol2Block(block, sanitize=sanitize, cleanupSubstructures=cleanupSubstructures)
        if m is not None:
            if name:
                m.SetProp('Name',name[n])
            mols.append(m)

    return(mols)
    
     
      
