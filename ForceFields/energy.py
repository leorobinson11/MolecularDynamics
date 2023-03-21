from ForceFields.system import System
from ForceFields.geometry import Geometry

import numpy as np
import pandas as pd
import os, sys
from functools import reduce
from operator import and_


"""
This file contains:
    energy calculations 

term in the calculations use aproximations instead of paramaters
"""

class Energy(Geometry):
    def __init__(self,system:System)->None:
        Geometry.__init__(self)
        self.System = system
        self.load_constants()

    def GETenergyPairs(self)->None:
        #recalling all the atom combinations needed in energy calculation 
        #(!!! file reading + graph building done before)
        self.System.all_distances()
        self.System.all_angles()
        self.System.all_torsions()

    def load_constants(self)->dict:
        #reading in all the tables of constants
        self.pd_constants = {}
        for file in os.listdir(os.path.join('Data','Constants')):
            self.pd_constants.update({
                file.replace('.csv',''):pd.read_csv(os.path.join('Data','Constants',file))
            })
        return self.pd_constants

    def paramaterQuery(self,file:str,atoms:tuple)->pd.core.frame.DataFrame:
        #returns a row of the dataframe from a flexable number of conditions (the number of atoms)
        #runs through the diffenet permutations of atoms, in the case of them being 
        #saved in a different order in the csv file
        df = self.pd_constants[file]
        if type(atoms) == tuple:
            #if there are multiple atoms
            for arrangement in [atoms,atoms[::-1]]:
                conditions = (getattr(df,name) == I.symbol for (name,I) in zip(df.columns, arrangement))
                result = df[reduce(and_,conditions)]
                if len(result) > 0: return result
        else:
            #if there is only one atom (Van der Walls)
            return df[df.I == atoms.symbol]
        #if there is no existing paramater for the input atoms
        sys.exit(f"[-] System exit due to missing {file} paramaters for atoms: {[atom.symbol for atom in atoms]}")

    def GETbondPotential(self, atoms:tuple)->float:
        I,J = atoms
        result = self.paramaterQuery('bonds', atoms)
        r0, k = result.R0.item(), result.K.item()
        distance = self.GETdistance(I,J)
        return k*(distance-r0)

    def GETanglePotential(self, atoms:tuple)->float:
        I,J,K = atoms
        result = self.paramaterQuery('angles',atoms)
        r0, k = result.R0.item(), result.Ke.item()
        r0 = (r0*np.pi)/180
        angle = self.GETangle(I,J,K)
        #angle = (angle*180)/np.pi
        return 0.5*k*np.square(angle-r0)

    def GETtorsionPotential(self,atoms:tuple)->float:
        I,J,K,L = atoms
        result = self.paramaterQuery('torsions',atoms)
        k, n = result.Ke.item(), result.N.item()
        angle = self.GETtorsion(I,J,K,L)
        angle = (angle*180)/np.pi
        return 0.5 * k * (1 - np.cos(n*angle))
        
    def GETvanderwallPotential(self,atoms:tuple)->float:
        I, J = atoms
        result_I = self.paramaterQuery('vanderwaals',I)
        result_J = self.paramaterQuery('vanderwaals',J)
        ei, ri = result_I.E.item(), result_I.R.item()
        ej, rj = result_J.E.item(), result_J.R.item()
        rij = (ri + rj)*0.5
        eij = np.sqrt(ei*ej)
        distance = self.GETdistance(I,J)
        return 4 * eij * ((rij/distance)**12 - (rij/distance)**6)
        
    def SumGraph(self,graph:list,formular)->float:
        #calculates one type of potential energy for all atoms
        total = 0
        for atoms in graph:
            total += abs(formular(atoms))*2 #as Fij = -Fji
        return total
    
