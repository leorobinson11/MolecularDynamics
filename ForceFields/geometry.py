from ForceFields.atom import Atom

import numpy as np

"""
This file contains:
    angle calculations
"""

class Geometry:
    def normalize(self,v:np.array)->np.array:
        #retrurns a normalized vector
       return v/np.linalg.norm(v)

    def GETdistance(self,I:Atom,J:Atom)->float:
        #returns the absolute cartesian distance between the coordinates of the input atoms
        return np.sqrt(
            abs(
                sum([
                    np.square(i-j) for (i,j) in zip(I.cartesians,J.cartesians)  
                ])
            )
        )
    
    def GETangle(self,I:Atom,J:Atom,K:Atom)->float:
        #returns the vector angle between the coodinates of the input atoms, J is the atom in the center of the three
        b0 = np.subtract(J.cartesians, I.cartesians)
        b1 = np.subtract(J.cartesians, K.cartesians)
        angle = abs(np.arccos(
            np.dot(
                self.normalize(b0),
                self.normalize(b1)
            )
        ))
        if angle > np.pi:angle -= np.pi
        return angle

    
    def GETtorsion(self,I:Atom,J:Atom,K:Atom,L:Atom)->float:
        #return the torsion angle, I,J,K and L are the atoms along the molecule in sequencial order
        b0 = np.subtract(J.cartesians,I.cartesians)
        b1 = self.normalize(np.subtract(K.cartesians,J.cartesians))
        b2 = np.subtract(L.cartesians,K.cartesians)
        v = np.subtract(b0,np.multiply(np.dot(b0,b1),b1))
        w = np.subtract(b2,np.multiply(np.dot(b2,b1),b1))
        angle = abs(np.arctan2(
            np.dot(v,w),
            np.dot(np.cross(b1, v),w)
        ))

        if angle > np.pi:angle -= np.pi
        return angle