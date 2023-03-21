from ForceFields.atom import Atom
from ForceFields.geometry import Geometry

import sys
import numpy as np

"""
This file contains:
    Reading of files
    Graph building

"""

class System(Geometry):
    def __init__(self)->None:
        Geometry.__init__(System)

    def readXYZFile(self,filepath:str)->None:
        #reads the file and returns the list of atom object and the number of atoms, !!! for xyz only !!!
        self.atoms = []
        try:
            with open(filepath) as f:
                for ln, line in enumerate(f.readlines()):
                    if ln == 0:
                        self.atom_number = int(line)
                    elif ln >= 2:
                        formated = [word.replace('\n','') for word in line.split(' ') if word != '']
                        symbol = formated[0]
                        cartesians = [float(cor) for cor in formated[1:]]
                        self.atoms.append(
                            Atom(symbol,cartesians)
                        )
            print(f'[+] File {filepath} successfully loaded')
        except FileNotFoundError:
                sys.exit(f'[-] File {filepath} could not be found')

    def getcenterofMass(self)->np.array:
        #center of mass - used as sanity test as it should stay constant
        center = np.array([0.0, 0.0, 0.0])
        for atom in self.atoms:
            center += atom.cartesians * atom.mass
        return center
        
    def resetForce(self)->None:
        for atom in self.atoms:
            atom.force = np.array([0.0,0.0,0.0])

    def are_bonded(self,I:Atom,J:Atom,threshhold:float=1)->bool:
        #checks weather two atoms are bonded with the condition that the displacement between the two 
        #is smaller than the sum of the atomic radiu times some threshhold factor
        if self.GETdistance(I,J) < (I.radius+J.radius)*(I.electronegativity+J.electronegativity)*threshhold:
            #Hydrogen atoms aren't bonded to eachother
            if not (I.symbol == 'H' and J.symbol == 'H'):
                return True
        return False

    def Build_basic_Graph(self, bond_thresh:float=1, cutoff:float=3)->dict:
        #builds a graph of all the bonded atoms and the non bonded atoms (cutoff factor for the non bonded)
        self.bonded_graph = {}
        self.non_bonded_graph = {}
        for I in self.atoms:
            for J in self.atoms:
                if I != J and self.are_bonded(I,J, bond_thresh):
                    if I in self.bonded_graph:
                        self.bonded_graph[I].append(J)
                    else:
                        self.bonded_graph.update({I:[J]})
                elif I != J and cutoff > self.GETdistance(I,J):
                    if I in self.non_bonded_graph:
                        self.non_bonded_graph[I].append(J)
                    else:
                        self.non_bonded_graph.update({I:[J]})
        return self.bonded_graph,self.non_bonded_graph

    def check_bonded_Graph(self)->tuple:
        #checks the graph for weather any of atoms has more or to few bonds than that atom could have, 
        #if so the user could readjust the threshold factor for different results
        self.to_many_bonds = []
        for (key,items) in self.bonded_graph.items():
            if len(items) > key.GETmax_bonds():
                self.to_many_bonds.append(key)
        return self.to_many_bonds

    def find_distances(self,graph:dict)->list:
        #returns a graph of every distance between each pair of atoms (for each pair only once)
        #unlike the functions for obtainig all other paramates this doesn't save directly to an object property
        distance_graph = []
        for (key,items) in graph.items(): #atom0
            for item in items: #atom2
                if (key,item) not in distance_graph and (item,key) not in distance_graph:
                    distance_graph.append((key,item))
        return distance_graph
    
    def all_distances(self)->tuple:
        #performs the find_distances function for the bonded and non bonded graphs
        self.bonddistance_graph = self.find_distances(self.bonded_graph)
        self.non_bonddistance_graph = self.find_distances(self.non_bonded_graph)
        return self.bonddistance_graph, self.non_bonddistance_graph

    def all_angles(self)->list:
        #retrurns all angles - functions by calculating the posible combinations of the angles 
        #for atoms bonded to or more
        self.angle_graph = []
        for (key,items) in self.bonded_graph.items(): #atom0
            if len(items) >= 2:
                for I in items: #atom1
                    for J in items: #atom2
                        if I != J and (J,key,I) not in self.angle_graph:
                            self.angle_graph.append((I,key,J))
        return self.angle_graph

    def all_torsions(self)->list:
        #returns all posible torsion angles by iterating through the graph and checking weather the atoms
        #have the required number of bonds (1,not including the previous atom in the chain)
        self.torsion_graph = []
        for (key,items) in self.bonded_graph.items(): #atom0
                for I in items: #atom1
                    if len(self.bonded_graph[I]) >= 2:
                        for J in self.bonded_graph[I]: #atom2
                            if J != key and len(self.bonded_graph[J]) >= 2:
                                for K in self.bonded_graph[J]: #atom3
                                    if K not in (key,I) and (K,J,I,key) not in self.torsion_graph:
                                        self.torsion_graph.append((key,I,J,K))
        return self.torsion_graph
    
    def DrawGraph(self,graph:dict)->None:
        #prints out the bond graph
        for (key,item) in graph.items():
            print(f'          {key.symbol} --> {[subitem.symbol for subitem in item]}')