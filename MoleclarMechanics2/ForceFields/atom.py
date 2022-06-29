import pandas as pd
import numpy as np
import os, sys

"""
This file contains:
    Properties of each Atom
"""

class Atom:
    def __init__(self,Symbol:str,cartesians:list)->None:
        #contains all the atoms properties, two bellow from the read in atom file
        self.symbol = Symbol

        try:
            pt = pd.read_csv(os.path.join('Data','Periodic_Table_of_Elements.csv'))
            row = pt[pt.Symbol.eq(self.symbol)]
            #gets atoms properties from csv file of periodic elements

            self.mass = row.AtomicMass.item()
            self.radius = row.AtomicRadius.item()
            self.electronegativity = row.Electronegativity.item()
            self.electrons = row.NumberofElectrons.item()
            self.protons = self.electrons

        except ValueError:
            sys.exit(f'[-] Error in retrieving data for atom {Symbol}')

        """ Electrostatic forces in development """
        self.charge = 1 

        #kinematiks paramaters
        self.cartesians = np.array(cartesians)
        self.force = np.array([0.0,0.0,0.0])

        #currently under the atoms start from rest
        self.velocity = [0,0,0]

    def GETmax_bonds(self)->int:
            #gets the max number of bonds by getting the difference between the number of electrons and the number allowed in its shells
            try:
                pt = pd.read_csv(os.path.join('Data','Periodic_Table_of_Elements.csv'))

                electrons_in_shells = {1:2,2:10,3:28,4:60}
                shell = pt[pt.Symbol.eq(self.symbol)].NumberofShells.item()
                self.max_bonds = electrons_in_shells[shell] - self.electrons
                return self.max_bonds

            except ValueError:
                sys.exit(f'[-] Error in retrieving data for atom {self.symbol}')

    def GETinital_velocity(self,Temp:float)->float:
        #gets the velocity of the atom at the start of the simulation, dependant on the temprature
        R = 8.314
        self.velocity = np.sqrt(
            (Temp*R)/self.mass
        )
        return self.velocity
