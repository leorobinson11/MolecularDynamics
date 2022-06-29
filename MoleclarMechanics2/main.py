from ForceFields.system import System
from ForceFields.energy import Energy
from Simulations.dynamics import DynamicSymulation
import os

def Test_BondSetup(atomname:str)->None:
    syst = System()
    syst.readXYZFile(os.path.join('Molecules',atomname+'.xyz'))
    syst.Build_basic_Graph(thresh, cutoff)
    check = syst.check_bonded_Graph()
    print(
        f"""
        [-] There are {len(check)} atoms with too many bonds: {[I.symbol for I in check]}

        [+] Bond Graph:
        {'-'*60}
    """)
    syst.DrawGraph(syst.bonded_graph)
    print(
        f"""
        [+] Non Bonded Graph:
        {'-'*60}
    """)
    syst.DrawGraph(syst.non_bonded_graph)
    print()

def Test_EnergyCalculations(atomname:str)->None:
    syst = System()
    syst.readXYZFile(os.path.join('Molecules',atomname+'.xyz'))
    syst.Build_basic_Graph(thresh, cutoff)
    ener = Energy(syst)
    ener.GETenergyPairs()
    print(
        f"""
        [+] Energies of the current configuration:
        {'-'*60}
        {len(ener.System.bonddistance_graph)} Bonds: 
        {ener.SumGraph(ener.System.bonddistance_graph,ener.GETbondPotential)}

        {len(ener.System.angle_graph)} Angles:
        {ener.SumGraph(ener.System.angle_graph,ener.GETanglePotential)}

        {len(ener.System.torsion_graph)} Torsions:
        {ener.SumGraph(ener.System.torsion_graph,ener.GETtorsionPotential)}

        {len(ener.System.non_bonddistance_graph)} Van der Wals:
        {ener.SumGraph(ener.System.non_bonddistance_graph,ener.GETvanderwallPotential)}

        {len(ener.System.strechbend_graph)} Strech-Bend:
        {ener.SumGraph(ener.System.strechbend_graph,ener.GETstrechbendPotential)}
    """)
    print()

def Test_simulation(atomname:str, stepsize:float, stepnumber:int)->None:
    try:
        os.remove(os.path.join('Results','Trajectories',f'{atomname}_traj.xyz'))
        os.remove(os.path.join('Results','Optimized',f'{atomname}_opt.xyz'))
    except:
        pass
    finally:
        # rewrite this so that the filepath is in the dynamics module
        sim = DynamicSymulation(atomname, stepsize, rec_numb, thresh, cutoff, find_energy=True)
        sim.simulate(stepnumber)
        print('[+] Simulation compleate')


if __name__ == "__main__":
    stepsize = 5e-6 #6
    stepnumber = int(5e4) #4
    thresh, cutoff = 0.17, 10
    rec_numb = stepnumber/100
    NAME = "C2H6"
    #Test_BondSetup(NAME)
    #Test_EnergyCalculations(NAME)
    Test_simulation(NAME, stepsize, stepnumber)