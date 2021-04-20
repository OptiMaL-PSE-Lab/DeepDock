import copy
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from torch.distributions import Normal



class optimze_conformation():
    def __init__(self, mol, target_coords, n_particles, pi, mu, sigma, save_molecules=False, dist_threshold=1000):
        super(optimze_conformation, self).__init__()
        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = self.get_torsions([mol])
        self.save_molecules = save_molecules
        self.dist_threshold = dist_threshold
        np.random.seed(123)
        self.mol = self.apply_changes(mol, np.random.rand(len(self.rotable_bonds, )+6)*10)
        Chem.rdMolTransforms.CanonicalizeConformer(self.mol.GetConformer())
        
        
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
        [self.SetDihedral(mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6+r]) for r in range(len(self.rotable_bonds)) for m in range(self.n_particles)]
        
        # apply transformation matrix
        [rdMolTransforms.TransformConformer(mols[m].GetConformer(), self.GetTransformationMatrix(values[m, :6])) for m in range(self.n_particles)]
        
        # Calcualte distances between ligand conformation and target
        ligCoords_list = [torch.tensor(m.GetConformer().GetPositions()[self.noHidx]) for m in mols]
        ligCoords = torch.stack(ligCoords_list).double()
        dist = torch.cdist(ligCoords, self.targetCoords, 2).flatten().unsqueeze(1)
        
        # Calculate probability for each ligand-target pair
        prob = self.calculate_probablity(self.pi, self.sigma, self.mu, dist)
        prob[torch.where(dist > self.dist_threshold)[0]] = 0.
        
        # Reshape and sum probabilities
        prob = prob.reshape(self.n_particles, -1).sum(1)
        if self.save_molecules: self.opt_mols.append(mols[torch.argmax(prob)])
        
        # Delete useless tensors to free memory
        del ligCoords_list, ligCoords, dist, mols
        
        return -prob.detach().numpy()
    
        
    def SetDihedral(self, conf, atom_idx, new_vale):
        rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
           
    def GetDihedral(self, conf, atom_idx):
        return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])
    
    def GetTransformationMatrix(self, transformations):
        x, y, z, disp_x, disp_y, disp_z = transformations
        transMat= np.array([[np.cos(z)*np.cos(y), (np.cos(z)*np.sin(y)*np.sin(x))-(np.sin(z)*np.cos(x)), (np.cos(z)*np.sin(y)*np.cos(x))+(np.sin(z)*np.sin(x)), disp_x],
                            [np.sin(z)*np.cos(y), (np.sin(z)*np.sin(y)*np.sin(x))+(np.cos(z)*np.cos(x)), (np.sin(z)*np.sin(y)*np.cos(x))-(np.cos(z)*np.sin(x)), disp_y],
                            [-np.sin(y),           np.cos(y)*np.sin(x),                                   np.cos(y)*np.cos(x),                                  disp_z],
                            [0,                    0,                                                     0,                                                    1]], dtype=np.double)
        return transMat
    
    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5
        
    def calculate_probablity(self, pi, sigma, mu, y):
        normal = Normal(mu, sigma)
        logprob = normal.log_prob(y.expand_as(normal.loc))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)

        return prob
    
    def apply_changes(self, mol, values):
        opt_mol = copy.copy(mol)
        
        # apply rotations
        [self.SetDihedral(opt_mol.GetConformer(), self.rotable_bonds[r], values[6+r]) for r in range(len(self.rotable_bonds))]
        
        # apply transformation matrix
        rdMolTransforms.TransformConformer(opt_mol.GetConformer(), self.GetTransformationMatrix(values[:6]))
        
        return opt_mol
    
    def get_torsions(self, mol_list):
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
    
    
