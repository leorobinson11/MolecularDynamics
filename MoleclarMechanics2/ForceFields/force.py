from ForceFields.system import System
from ForceFields.geometry import Geometry
from ForceFields.atom import Atom
from ForceFields.energy import Energy

import numpy as np


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
        #Morse Potential
        I,J = atoms
        result = self.paramaterQuery('bonds', atoms)
        r0, a, d = result.R0.item(), result.A.item(), result.D.item()
        distance = self.GETdistance(I,J)
        pt = a*(distance-r0)
        force_mag = 2*d*(1-np.exp(pt))*a*np.exp(pt)
        return self.TimesUnitVect(I,J, force_mag)

    def GETangleForce(self,atoms:tuple)->np.array:
        I,J,K = atoms
        result = self.paramaterQuery('angles',atoms)
        r0, k = result.R0.item(), result.Ke.item()
        angle = self.GETangle(I,J,K)
        """ r0 to radians """
        r0 = (r0*np.pi)/180
        #angle = (angle*180)/np.pi
        force_mag = k*(angle-r0)    
        """ Direction """
        rij = np.subtract(I.cartesians, J.cartesians)
        rkj = np.subtract(K.cartesians, J.cartesians)
        direction = self.normalize(np.subtract(rij, rkj))
        return np.multiply(direction, force_mag)

    def GETstrechbendForce(self, atoms:tuple)->float:
        I, J, K = atoms
        #for angle component
        result = self.paramaterQuery('angles', atoms)
        a0 = result.R0.item()
        a0 = (a0*np.pi)/180
        angle = self.GETangle(I,J,K)
        #for strech bend component
        result = self.paramaterQuery('strechbend',(J, K))
        k = result.K.item()
        force_mag = 0.5*k*(angle - a0)
        return self.TimesUnitVect(J,K, force_mag)

    def GETtorsionForce(self,atoms:tuple)->float:
        I,J,K,L = atoms
        result = self.paramaterQuery('torsions',atoms)
        V = result.V1.item(), result.V2.item(), result.V3.item()
        angle = self.GETtorsion(I,J,K,L)
        angle = (angle*180)/np.pi
        force_mag = 0.5*sum(
            v*((-1)**n*n*np.sin(n*angle)) for (v,n) in zip(V, range(1,4))
        )
        #unit vector tangent of the bond radius, perpendicular to the axis
        radial_v = np.subtract(I.cartesians,J.cartesians)
        axis_v = np.subtract(J.cartesians,K.cartesians)
        direction = np.cross(radial_v,axis_v)
        return np.multiply(self.normalize(direction), force_mag)

    def GETvanderwallForce(self, atoms:tuple)->np.array:
        #returns the vector of the force between an atom pair exurted by bonds
        I, J = atoms
        distance = self.GETdistance(I,J)
        if distance > (I.radius+J.radius):
            result_I = self.paramaterQuery('vanderwaals',I)
            result_J = self.paramaterQuery('vanderwaals',J)
            ei, ri, bi = result_I.E.item(), result_I.R.item(), result_I.B.item()
            ej, rj, bj = result_J.E.item(), result_J.R.item(), result_J.B.item()

            rij = ri + rj
            eij = np.sqrt(ei*ej)

            bij = np.sqrt(bi*bj)
            aij = 6*np.exp(bij)/(bij-6)
            cij = bij/(bij-6)

            force_mag = eij*(
                6*aij*np.power(rij,6)/np.power(distance,7)
                +
                bij*cij*np.exp(cij*distance/rij)/rij
            )
        else:
            force_mag = 0
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
            atoms[0].force -= newforce
            atoms[-1].force += newforce

    def ApplyAll(self)->None:
        #iterates throught all the lists and functions for force calculations
        graphs = [
            self.System.bonddistance_graph,
            self.System.angle_graph,
            #self.System.torsion_graph,
            self.System.strechbend_graph#,
            #self.System.non_bonddistance_graph
        ]
        formulars = [
            self.GETbondForce,
            self.GETangleForce,
            #self.GETtorsionForce,
            self.GETstrechbendForce#,
            #self.GETvanderwallForce
        ]
        for (graph, formular) in zip(graphs,formulars):
            self.SumGraph(graph, formular)