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
        self.System.all_strechbend()
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
        r0, a, d = result.R0.item(), result.A.item(), result.D.item()
        distance = self.GETdistance(I,J)
        return d*(np.square(
            1 - np.exp(-a*(distance-r0))
        ))

    def GETanglePotential(self, atoms:tuple)->float:
        I,J,K = atoms
        result = self.paramaterQuery('angles',atoms)
        r0, k = result.R0.item(), result.Ke.item()
        """ r0 to radians """
        r0 = (r0*np.pi)/180
        angle = self.GETangle(I,J,K)
        #angle = (angle*180)/np.pi
        return 0.5*k*np.square(angle-r0)

    def GETstrechbendPotential(self, atoms:tuple)->float:
        I, J, K = atoms
        #for bond component
        result = self.paramaterQuery('bonds', (J,K))
        r0 = result.R0.item()
        distance = self.GETdistance(K,J)
        #for angle component
        result = self.paramaterQuery('angles', atoms)
        a0 = result.R0.item()
        a0 = (a0*np.pi)/180
        angle = self.GETangle(I,J,K)
        #for strech bend component
        result = self.paramaterQuery('strechbend',(J, K))
        k = result.K.item()
        return  0.5*k*(distance - r0) * (angle - a0)

    def GETtorsionPotential(self,atoms:tuple)->float:
        I,J,K,L = atoms
        result = self.paramaterQuery('torsions',atoms)
        V = result.V1.item(), result.V2.item(), result.V3.item()
        angle = self.GETtorsion(I,J,K,L)
        angle = (angle*180)/np.pi
        return 0.5*sum(
            v*(1-(-1)**n*np.cos(n*angle)) for (v,n) in zip(V, range(1,4))
        )
        
    def GETvanderwallPotential(self,atoms:tuple)->float:
        I, J = atoms
        result_I = self.paramaterQuery('vanderwaals',I)
        result_J = self.paramaterQuery('vanderwaals',J)
        ei, ri, bi = result_I.E.item(), result_I.R.item(), result_I.B.item()
        ej, rj, bj = result_J.E.item(), result_J.R.item(), result_J.B.item()

        rij = ri + rj
        eij = np.sqrt(ei*ej)

        bij = np.sqrt(bi*bj)
        aij = 6*np.exp(bij)/(bij-6)
        cij = bij/(bij-6)

        distance = self.GETdistance(I,J)
        return eij * (
            -aij * np.power(rij/distance, 6)
            +
            bij * np.exp(cij*(distance/rij))
        )
        
    def SumGraph(self,graph:list,formular)->float:
        #calculates one type of potential energy for all atoms
        total = 0
        for atoms in graph:
            total += abs(formular(atoms))*2 #as Fij = -Fji
        return total
    
