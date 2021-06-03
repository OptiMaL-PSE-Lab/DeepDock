import os
import sys
import numpy as np
import shutil
import pymesh
import Bio.PDB
from Bio.PDB import * 
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
from IPython.utils import io
from sklearn.neighbors import KDTree
from scipy.spatial import distance

import deepdock
sys.path.insert(0, deepdock.__path__[0]+'/masif/source')

from default_config.masif_opts import masif_opts
from deepdock.prepare_target.compute_normal import compute_normal
from deepdock.prepare_target.computeAPBS import computeAPBS
from deepdock.prepare_target.computeCharges import computeCharges, assignChargesToNewMesh
from deepdock.prepare_target.computeHydrophobicity import computeHydrophobicity
from deepdock.prepare_target.computeMSMS import computeMSMS
from deepdock.prepare_target.fixmesh import fix_mesh
from deepdock.prepare_target.save_ply import save_ply
from deepdock.utils.mol2graph import *


def compute_inp_surface(target_filename, ligand_filename, dist_threshold=10):
    sufix = '_'+str(dist_threshold+5)+'A.pdb'
    out_filename = os.path.splitext(target_filename)[0]
    
    # Get atom coordinates
    mol = Chem.MolFromMol2File(ligand_filename, sanitize=False, cleanupSubstructures=False)
    g = mol_to_nx(mol)
    atomCoords = np.array([g.nodes[i]['pos'].tolist() for i in g.nodes])
    
    # Read protein and select aminino acids in the binding pocket
    parser = Bio.PDB.PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.

    structures = parser.get_structure('target', out_filename+'.pdb')
    structure = structures[0] # 'structures' may contain several proteins in this case only one.

    atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')
    ns = Bio.PDB.NeighborSearch(atoms)
    
    close_residues= []
    for a in atomCoords:  
        close_residues.extend(ns.search(a, dist_threshold+5, level='R'))
    close_residues = Bio.PDB.Selection.uniqueify(close_residues)

    class SelectNeighbors(Select):
        def accept_residue(self, residue):
            if residue in close_residues:
                if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                    return True
                else:
                    return False
            else:
                return False

    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(out_filename+sufix, SelectNeighbors())
    
    # Identify closes atom to the ligand
    structures = parser.get_structure('target', out_filename+sufix)
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    atoms  = Bio.PDB.Selection.unfold_entities(structure, 'A')

    #dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
    #atom_idx = np.argmin(dist)
    #dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
    #atom_idx = np.argsort(np.min(dist, axis=1))[0]
    
    # Compute MSMS of surface w/hydrogens, 
    try:
        dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
        atom_idx = np.argmin(dist)
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                  protonate=True, one_cavity=atom_idx)
        
        # Find the distance between every vertex in binding site surface and each atom in the ligand.
        kdt = KDTree(atomCoords)
        d, r = kdt.query(vertices1)
        assert(len(d) == len(vertices1))
        iface_v = np.where(d <= dist_threshold)[0]
        faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
        # Compute "charged" vertices
        if masif_opts['use_hbond']:
            vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
        # For each surface residue, assign the hydrophobicity of its amino acid. 
        if masif_opts['use_hphob']:
            vertex_hphobicity = computeHydrophobicity(names1)    
        
        # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
        vertices2 = vertices1
        faces2 = faces1
    
        # Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)
        with io.capture_output() as captured:
            regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        
    except:
        try:
            dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
            atom_idx = np.argsort(np.min(dist, axis=1))[0]
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                      protonate=True, one_cavity=atom_idx)

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
            # Compute "charged" vertices
            if masif_opts['use_hbond']:
                vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if masif_opts['use_hphob']:
                vertex_hphobicity = computeHydrophobicity(names1)    
        
            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1
    
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
                
        except:
            vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename+sufix,\
                                                                      protonate=True, one_cavity=None)

            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
    
            # Compute "charged" vertices
            if masif_opts['use_hbond']:
                vertex_hbond = computeCharges(out_filename, vertices1, names1)    
    
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if masif_opts['use_hphob']:
                vertex_hphobicity = computeHydrophobicity(names1)    
        
            # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
            vertices2 = vertices1
            faces2 = faces1
    
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        
    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Assign charges on new vertices based on charges of old vertices (nearest
    # neighbor)
    
    if masif_opts['use_hbond']:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hbond, masif_opts)
    
    if masif_opts['use_hphob']:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hphobicity, masif_opts)
    
    if masif_opts['use_apbs']:
        vertex_charges = computeAPBS(regular_mesh.vertices, out_filename+sufix, out_filename+"_temp")
        
    # Compute the principal curvature components for the shape index. 
    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)
    
    # Convert to ply and save.
    save_ply(out_filename+".ply", regular_mesh.vertices,\
             regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
             normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
             si=si)