Simulation
Initial Composition: C10H16
Temperature:         4000 K
Pressure:            40.5 GPa
Number of atoms:     20800
Length:              1.2 ns (coordinates saved every .012 ps)

Files:
First_Frame.npz:  Adjacency matrix at the first timestep saved as a sparse array
atom_types.npy:   Atom types as a 1D array, 1 for carbon 2 for hydrogen
bond_changes.npy: List of bond changes, column 0 is timestep, columns 1 and 2 are the atoms in the modified bond, and column 3 denotes whether the bond is added (1) or removed (2)
volume.npy:       Extra numpy variable for system volume for kinetic modeling
