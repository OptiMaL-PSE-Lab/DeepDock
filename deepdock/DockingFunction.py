import copy
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
from torch_scatter import scatter_add
from torch.distributions import Normal
from torch_geometric.data import Batch
from scipy.optimize import differential_evolution

from deepdock.utils.data import *


def score_compound(ligand, target, model, dist_threshold=3., seed=None, device='cpu'):
    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if not isinstance(ligand, Batch):
        if isinstance(ligand, Chem.Mol):
            # Check if ligand it has 3D coordinates
            try:
                ligand.GetConformer()
            except ValueError:
                print("Ligand must have 3D conformation. Check using mol.GetConformer()")
            
            # Prepare ligand and target to be used by pytorch geometric
            ligand = from_networkx(mol2graph.mol_to_nx(ligand))
            ligand = Batch.from_data_list([ligand])
            
        else:
            raise Exception('mol should be an RDKIT molecule or a Batch instance')

    if not isinstance(target, Batch):
        if isinstance(target, str):
            # Prepare ligand and target to be used by pytorch geometric
            target = Cartesian()(FaceToEdge()(read_ply(target)))
            target = Batch.from_data_list([target])
        else:
            raise Exception('target should be a string with the ply file paht or a Batch instance')

    # Use the model to score conformations
    model.eval()
    ligand, target = ligand.to(device), target.to(device)
    pi, sigma, mu, dist, atom_types, bond_types, batch = model(ligand, target)
    prob = calculate_probablity(pi, sigma, mu, dist)
    if dist_threshold is not None:
        prob[torch.where(dist > dist_threshold)[0]] = 0.
    prob = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
    
    return prob.cpu().detach().numpy()


def dock_compound(mol, target_ply, model, dist_threshold=3., popsize=150, maxiter=500, seed=None, mutation=(0.5, 1), recombination=0.8, device='cpu'):
    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
            mol=Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            
        # Prepare ligand and target to be used by pytorch geometric
        ligand = from_networkx(mol2graph.mol_to_nx(mol))
        ligand = Batch.from_data_list([ligand])
            
    else:
        raise Exception('mol should be an RDKIT molecule')

    if not isinstance(target_ply, Batch):
        if isinstance(target_ply, str):
            # Prepare ligand and target to be used by pytorch geometric
            target = Cartesian()(FaceToEdge()(read_ply(target_ply)))
            target = Batch.from_data_list([target])
        else:
            raise Exception('target should be a string with the ply file paht or a Batch instance')

    # Use the model to generate distance probability distributions
    model.eval()
    ligand, target = ligand.to(device), target.to(device)
    pi, sigma, mu, dist, atom_types, bond_types, batch = model(ligand, target)

    #Set optimization function
    opt = optimze_conformation(mol=mol, target_coords=target.pos.cpu(), n_particles=1,
                               pi=pi.cpu(), mu=mu.cpu(), sigma=sigma.cpu(), dist_threshold=dist_threshold, seed=seed)

    #Define bounds for optimization
    max_bound = np.concatenate([[np.pi]*3, target.pos.cpu().max(0)[0].numpy(), [np.pi]*len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi]*3, target.pos.cpu().min(0)[0].numpy(), [-np.pi]*len(opt.rotable_bonds)], axis=0)
    bounds = (min_bound, max_bound)

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, list(zip(bounds[0],bounds[1])), maxiter=maxiter,
                                    popsize=int(np.ceil(popsize/(len(opt.rotable_bonds)+6))),
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)

    # Get optimized molecule
    starting_mol = opt.mol
    opt_mol = apply_changes(starting_mol, result['x'], opt.rotable_bonds)
    ligCoords = torch.stack([torch.tensor(m.GetConformer().GetPositions()[opt.noHidx]) for m in [opt_mol]])
    dist = compute_euclidean_distances_matrix(ligCoords, opt.targetCoords).flatten().unsqueeze(1)
    result['num_MixOfGauss'] = torch.where(dist <= dist_threshold)[0].size(0)
    result['num_atoms'] = opt_mol.GetNumHeavyAtoms()
    result['num_rotbonds'] = len(opt.rotable_bonds)
    result['rotbonds'] = opt.rotable_bonds

    return opt_mol, starting_mol, result



class optimze_conformation():
    def __init__(self, mol, target_coords, n_particles, pi, mu, sigma, save_molecules=False, dist_threshold=1000, seed=None):
        super(optimze_conformation, self).__init__()
        if seed:
            np.random.seed(seed)
            
        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([mol])
        self.save_molecules = save_molecules
        self.dist_threshold = dist_threshold
        self.mol =get_random_conformation(mol, rotable_bonds=self.rotable_bonds, seed=seed)
        
        self.targetCoords = torch.stack([target_coords for _ in range(n_particles)]).double()
        self.pi = torch.cat([pi for _ in range(n_particles)], axis=0)
        self.sigma = torch.cat([sigma for _ in range(n_particles)], axis=0)
        self.mu = torch.cat([mu for _ in range(n_particles)], axis=0)
        self.noHidx = [idx for idx in range(self.mol.GetNumAtoms()) if self.mol.GetAtomWithIdx(idx).GetAtomicNum()is not 1]
        
    def score_conformation(self, values):
        """
        Parameters
        ----------
        values : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`
        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        if len(values.shape) < 2: values = np.expand_dims(values, axis=0)
        mols = [copy.copy(self.mol) for _ in range(self.n_particles)]
        
        # Apply changes to molecules
        # apply rotations
        [SetDihedral(mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6+r]) for r in range(len(self.rotable_bonds)) for m in range(self.n_particles)]
        
        # apply transformation matrix
        [rdMolTransforms.TransformConformer(mols[m].GetConformer(), GetTransformationMatrix(values[m, :6])) for m in range(self.n_particles)]
        
        # Calcualte distances between ligand conformation and target
        ligCoords_list = [torch.tensor(m.GetConformer().GetPositions()[self.noHidx]) for m in mols]
        ligCoords = torch.stack(ligCoords_list).double()
        dist = torch.cdist(ligCoords, self.targetCoords, 2).flatten().unsqueeze(1)
        
        # Calculate probability for each ligand-target pair
        prob = calculate_probablity(self.pi, self.sigma, self.mu, dist)
        prob[torch.where(dist > self.dist_threshold)[0]] = 0.
        
        # Reshape and sum probabilities
        prob = prob.reshape(self.n_particles, -1).sum(1)
        if self.save_molecules: self.opt_mols.append(mols[torch.argmax(prob)])
        
        # Delete useless tensors to free memory
        del ligCoords_list, ligCoords, dist, mols
        
        return -prob.detach().numpy()
    
        
def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
           
def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])
    
def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat= np.array([[np.cos(z)*np.cos(y), (np.cos(z)*np.sin(y)*np.sin(x))-(np.sin(z)*np.cos(x)), (np.cos(z)*np.sin(y)*np.cos(x))+(np.sin(z)*np.sin(x)), disp_x],
                        [np.sin(z)*np.cos(y), (np.sin(z)*np.sin(y)*np.sin(x))+(np.cos(z)*np.cos(x)), (np.sin(z)*np.sin(y)*np.cos(x))-(np.cos(z)*np.sin(x)), disp_y],
                        [-np.sin(y),           np.cos(y)*np.sin(x),                                   np.cos(y)*np.cos(x),                                  disp_z],
                        [0,                    0,                                                     0,                                                    1]], dtype=np.double)
    return transMat
    
def compute_euclidean_distances_matrix(X, Y):
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    X = X.double()
    Y = Y.double()
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    return dists**0.5
        
def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)

    return prob
    
def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)
        
    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6+r]) for r in range(len(rotable_bonds))]
        
    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))
        
    return opt_mol
    
def get_torsions(mol_list):
    atom_counter=0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                        or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                        or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                        continue    
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append((idx4+atom_counter, idx3+atom_counter, idx2+atom_counter, idx1+atom_counter))
                        break
                    else:
                        torsionList.append((idx1+atom_counter, idx2+atom_counter, idx3+atom_counter, idx4+atom_counter))
                        break
                break
                    
        atom_counter += m.GetNumAtoms()
    return torsionList

def get_random_conformation(mol, rotable_bonds=None, seed=None):
    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
                mol=Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
    else:
        raise Exception('mol should be an RDKIT molecule')
    if seed:
            np.random.seed(seed)
    if rotable_bonds is None:        
        rotable_bonds = get_torsions([mol])
    new_conf = apply_changes(mol, np.random.rand(len(rotable_bonds, )+6)*10, rotable_bonds)
    Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf

def atom_scores(ligand, target, probabilities, batch):
    atom_contributions=[]
    for i in torch.arange(0, len(batch.unique())):
        num_atoms = len(ligand.x[ligand.batch==i])
        num_target_nodes = len(target.x[target.batch==i])
        contribution = probabilities[batch==i].reshape((num_atoms, num_target_nodes))
        atom_contributions.append(contribution.sum(1).cpu().detach().numpy())
    return atom_contributions
    
def calculate_atom_contribution(ligand, target, model, dist_threshold=3., seed=None, device='cpu'):
    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    if not isinstance(ligand, Batch):
        if isinstance(ligand, Chem.Mol):
            # Check if ligand it has 3D coordinates
            try:
                ligand.GetConformer()
            except ValueError:
                print("Ligand must have 3D conformation. Check using mol.GetConformer()")
            
            # Prepare ligand and target to be used by pytorch geometric
            ligand = from_networkx(mol2graph.mol_to_nx(ligand))
            ligand = Batch.from_data_list([ligand])
            
        else:
            raise Exception('mol should be an RDKIT molecule or a Batch instance')

    if not isinstance(target, Batch):
        if isinstance(target, str):
            # Prepare ligand and target to be used by pytorch geometric
            target = Cartesian()(FaceToEdge()(read_ply(target)))
            target = Batch.from_data_list([target])
        else:
            raise Exception('target should be a string with the ply file paht or a Batch instance')

    # Use the model to score conformations
    model.eval()
    ligand, target = ligand.to(device), target.to(device)
    pi, sigma, mu, dist, atom_types, bond_types, batch = model(ligand, target)
    prob = calculate_probablity(pi, sigma, mu, dist)
    if dist_threshold is not None:
        prob[torch.where(dist > dist_threshold)[0]] = 0.
    atom_contributions = atom_scores(ligand, target, prob, batch)
    prob = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
    
    return prob.cpu().detach().numpy(), atom_contributions