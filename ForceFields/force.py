from ForceFields.system import System
from ForceFields.geometry import Geometry
from ForceFields.atom import Atom
from ForceFields.energy import Energy

import numpy as np
import math


"""
This file contains:
    calculations for the vector forces
    
    Force Field Paramaters from: https://sci-hub.mksa.top/10.1039/ft9949002881
"""

class Force(Energy, Geometry):
    def __init__(self,system:System)->None:
        Energy.__init__(self, system)
        Geometry.__init__(self)
        self.System = system
        self.GETenergyPairs()

    def TimesUnitVect(self, I:Atom, J:Atom, force_mag:float)->np.array:
        #multipling the magnitude of the force with the normilized vector of the two attoms to percserve direction
        return np.multiply(
                self.normalize(
                    np.subtract(I.cartesians,J.cartesians)
                ),
                force_mag
            )

    def GETbondForce(self, atoms:tuple)->np.array:
        I,J = atoms
        result = self.paramaterQuery('bonds', atoms)
        r0, k = result.R0.item(), result.K.item()
        distance = self.GETdistance(I,J)
        force_mag = -k*(distance-r0)
        return self.TimesUnitVect(I, J, force_mag)

    def GETangleForce(self,atoms:tuple)->np.array:
        I,J,K = atoms
        result = self.paramaterQuery('angles',atoms)
        r0, k = result.R0.item(), result.Ke.item()
        angle = self.GETangle(I,J,K)
        r0 = (r0*np.pi)/180
        #angle = (angle*180)/np.pi
        force_mag = -k*(angle-r0)
        return self.TimesUnitVect(I, K, force_mag)

    def GETtorsionForce(self,atoms:tuple)->float:
        I,J,K,L = atoms
        result = self.paramaterQuery('torsions',atoms)
        k, n = result.Ke.item(), result.N.item()
        angle = self.GETtorsion(I,J,K,L)
        angle = (angle*180)/np.pi
        force_mag = -0.5 * k * n * np.sin(n*angle)
        #unit vector tangent of the bond radius, perpendicular to the axis
        radial_v = I.cartesians - J.cartesians
        axis_v = J.cartesians - K.cartesians
        direction = np.cross(radial_v,axis_v)
        return np.multiply(self.normalize(direction), force_mag)

    def GETvanderwallForce(self, atoms:tuple)->np.array:
        #returns the vector of the force between an atom pair exurted by bonds
        I, J = atoms
        distance = self.GETdistance(I,J)
        result_I = self.paramaterQuery('vanderwaals',I)
        result_J = self.paramaterQuery('vanderwaals',J)
        ei, ri = result_I.E.item(), result_I.R.item()
        ej, rj = result_J.E.item(), result_J.R.item()
        rij = (ri + rj)*0.5
        eij = np.sqrt(ei*ej)
        force_mag = 48 * eij * ((rij**12/distance**13) - 0.5*(rij**6/distance**7))
        return self.TimesUnitVect(I,J, force_mag)

    def GETelectrostatic(self,atoms)->np.array:
        I,J = atoms
        #Coulombs constant
        k = 8.9875517923
        distance = self.GETdistance(I,J)
        force_mag = (k*I.charge*J.charge)/np.square(distance)
        return self.TimesUnitVect(I,J, force_mag)

    def SumGraph(self,graph:str,formular)->None:
        #suming the forces of a specific type on the atoms
        for atoms in graph:
            newforce = formular(atoms)
            #as Fij = -Fji
            atoms[0].force += newforce
            atoms[-1].force -= newforce

    def ApplyAll(self)->None:
        #iterates throught all the lists and functions for force calculations
        graphs = [
            self.System.bonddistance_graph,
            self.System.angle_graph,
            self.System.torsion_graph,
            self.System.non_bonddistance_graph
        ]
        formulars = [
            self.GETbondForce,
            self.GETangleForce,
            self.GETtorsionForce,
            self.GETvanderwallForce
        ]
        for (graph, formular) in zip(graphs,formulars):
            self.SumGraph(graph, formular)