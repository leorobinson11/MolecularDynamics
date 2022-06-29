from ForceFields.atom import Atom
from ForceFields.force import Force
from ForceFields.system import System
from ForceFields.energy import Energy

from functools import reduce
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import copy

class DynamicSymulation:
    def __init__(self, name:str, stepsize:float, rec_numb:int, thresh:float, cutoff:float, find_energy:bool=True)->None:
        self.System = System()
        self.System.readXYZFile(os.path.join('Molecules',f'{name}.xyz'))
        self.System.Build_basic_Graph(thresh, cutoff)
        self.Force_calc = Force(self.System)
        self.timestep = stepsize
        self.tradj_filepath = os.path.join('Results','Trajectories',f'{name}_traj.xyz')
        self.optimize_filepath = os.path.join('Results','Optimized',f'{name}_opt.xyz')
        self.rec_numb = rec_numb
        self.find_energy = find_energy
        self.min = [float('inf'), copy(self.System)]
        self.graph_filepath = os.path.join('Results', 'Graphs', name)

    def Plot(self, x, y, name) -> None:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        #ax.plot(times, vanderwall_energies, label='VanderWalls')
        plt.ylabel('U - Potential Energy')
        plt.xlabel('T - time')
        fig.savefig(os.path.join(self.graph_filepath, name))

    def addSteptoFile(self, step:int)->None:
        #adding the coordinates the the file for a step
        with open(self.tradj_filepath, 'a') as f:
            f.write(str(self.System.atom_number)+'\n')
            f.write(f'{self.timestep*step} ps'+'\n')
            for atom in self.System.atoms:
                readable_cartesians = reduce(lambda lett, word: str(lett) + ' '*4 + str(word), atom.cartesians)
                f.write(atom.symbol + ' '*5 + readable_cartesians + '\n')

    def addBestoFile(self)->None:
        #writing a file for the best struckture
        with open(self.optimize_filepath, 'a') as f:
            f.write(str(self.System.atom_number)+'\n')
            f.write(str(self.min[0])+'\n')
            for atom in self.min[1].atoms:
                readable_cartesians = reduce(lambda lett, word: str(lett) + ' '*4 + str(word), atom.cartesians)
                f.write(atom.symbol + ' '*5 + readable_cartesians + '\n')

    def GETnewPosition(self, atom:Atom)->None:
        #recomputing the position using intergrated F = dp/dx
        a = atom.force/atom.mass
        atom.velocity += (a)*self.timestep

        dc = np.array([ -(v_comp*0.2*self.timestep)/atom.mass if v_comp > 0 
                        else (v_comp*0.2*self.timestep)/atom.mass
                        for v_comp in atom.velocity ])
                        
        atom.velocity += dc
        atom.cartesians += atom.velocity*self.timestep

    def OneSTEP(self)->None:
        #performs the presedure for one timestep
        #reseting the force to 0
        self.Force_calc.System.resetForce()
        #calculating all the forces on all the atoms
        self.Force_calc.ApplyAll()
        #recalculating all the new positions
        for atom in self.System.atoms:
            self.GETnewPosition(atom)

    def simulate(self, steps:int)->None:
        #graphing the potentials over time
        times = [t*self.timestep for t in range(0, steps, int(self.rec_numb))]
        bond_energies = []
        angle_energies = []
        torsion_energies = []
        vanderwall_energies = []
        total_energies = []

        ener = Energy(self.System)

        for step in range(steps):
            #performing the procedure for one timestep
            self.OneSTEP()

            if step % self.rec_numb == 0:
                 #printing the progress
                print(f'[+] Progress: {step}/{steps} -- ({round((step/steps)*100,2)} %)')
                #adding the current configuration to the file
                self.addSteptoFile(step)

                if self.find_energy:
                    Ub = ener.SumGraph(ener.System.bonddistance_graph, ener.GETbondPotential)
                    bond_energies.append(Ub)
                    Ua = ener.SumGraph(ener.System.angle_graph,ener.GETanglePotential)
                    angle_energies.append(Ua)
                    Ut = ener.SumGraph(ener.System.torsion_graph,ener.GETtorsionPotential)
                    torsion_energies.append(Ut)
                    Uv = ener.SumGraph(ener.System.non_bonddistance_graph,ener.GETvanderwallPotential)
                    vanderwall_energies.append(Uv)

                    U_total = Ub + Ua
                    total_energies.append(U_total)
                    if U_total < self.min[0]:
                        self.min = [U_total, copy(self.System)]

        if self.find_energy:
            try:
                os.mkdir(self.graph_filepath)
            except:
                pass
            finally:
                self.Plot(times, bond_energies, 'bonds')
                self.Plot(times, angle_energies, 'angles')
                self.Plot(times, torsion_energies, 'torsions')
                self.Plot(times, vanderwall_energies, 'Van_der_Walls')
                self.Plot(times, total_energies, 'summed energies')
                #adding the best step to the file
                self.addBestoFile()

        