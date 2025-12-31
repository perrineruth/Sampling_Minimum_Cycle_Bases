# Cycle exchange: Enhanced sampling of minimum cycle bases
The cycle exchange (`cycxchg`) is a collection of Python codes for sampling cycles from a [NetworkX](https://networkx.org/) simple graph. These codes are computationally efficient for physical networks, e.g., material networks. These codes can be used to compute:
- Polyhedron-interchangeability (`pi`) and short loop-interchangeability (`sli`) classes that partition the set of _relevant cycles_ (Vismara,1997). `pi` classes identify three-dimensional structures and `sli` classes group related large relevant cycles together.
- Uniform random sampling of the minimum cycle bases (MCBs) of a graph. This definition outputs a set of cycles that are relevant, grow linearly with system size, and are statistically well-defined.
- Pairwise intersection characteristics of MCB cycles. These are the paths that join two cycles in an MCB together. These intersections can be used to define a dual graph.

These codes are a part of the article "Theory and algorithms for clusters of cycles in graphs for material networks" listed below. If you use these codes please cite this work.

## Installation

Included in this repository is a package `cycxchg` for computing cycles from a NetworkX graph. This can be installed from PyPI as follows:
```
pip install cycxchg
```
We recommended installing this package in a new Python environment.

Alternatively, the `cycxchg` package can be installed locally, so that the files can be modified.
```
git clone https://github.com/perrineruth/Sampling_Minimum_Cycle_Bases
cd Sampling_Minimum_Cycle_Bases
pip install -e .
```
This has the same affect as including the files `cycxchg.py` (for computing cycles) and `sparseb.py` (for performing sparse binary matrix operations) in the current working directory. These files are located in the `\src` folder.

<!-- 
Depricated approach:
To run these codes, the `cycxchg.py` is imported as follows.
```
import cycxchg as cxc
```
The `sparseb.py` file must be located in the same directory as `cycxchg.py`. This file is used to implement sparse binary matrix operations (modulo 2 as in the GF(2) field). -->

## General Workflow
We assume the `cycxchg` and `networkx` packages are imported with the given syntax.
```
import networkx as nx
import cycxchg as cxc
```

The main component of these codes is the `cycle_decomposition` object. For a NetworkX graph `G` this decomposition is given by:
```
cyc_dec = cxc.cycle_decomposition(G)
```
The `cyc_dec` object breaks the relevant cycles of `G` into `pi` and `sli` classes. These are contained in `cyc_dec.pi_classes` and `cyc_dec.sli_classes`. A `sli` class "`sc`" can be visualized as a NetworkX graph by extracting its edges:
```
Gsc = nx.from_edgelist(sc.edges())
nx.draw(Gsc)
```
A `pi` class can be visualized by this method as well. More methods for the `pi` and `sli` classes are given in `cycle_decomposition.ipynb`.

A random MCB can be extracted from the `cyc_dec` object as follows:
```
rMCB = cyc_dec.random_MCB()
```

Alternatively, a random MCB can be output as a dual graph by setting the optional argument `merge_MCB` is set to `True`.
```
Dual = cyc_dec.random_MCB(merge_MCB=True)
```
`Dual` is a NetworkX graph that is structured as follows
- _Nodes_. The nodes of the dual graph are integers as indexes of each cycle in the random minimum cycle basis. Each node in the dual graph has two attributes.
  - `nodes`: the nodes in the corresponding MCB cycle.
  - `length`: the length of the corresponding MCB cycle.
- _Edges_. The edges of the dual graph represent intersection paths between pairs of MCB cycles. Each edge has two attributes
  - `length`: the number of edges in the intersection path.
  - `indices`: a pair of indices denoting the starting point of the connection path between for both of the MCB cycles. See the notebook `cycle_decomposition.ipynb` for more information on constructing the intersection paths from the dual graph.

<!-- If the optional argument `merge_MCB` is set to `True`, then the MCB is postprocessed so that every pair of cycles intersects over at most one path. These connection paths are returned as two matrices. 
1. `Weights`. `Weights[i,j]` is a signed integer whose magnitude is the length of the intersection path between MCB cycles `i` and `j`. The sign of `Weights[i,j]` gives the orientation of this path in cycle `i`.
2. `Indices`. `Indices[i,j]` points to the start of the intersection path between MCB cycles `i` and `j`. `Indices[i,j]` and `Indices[j,i]` point to the same node in cycles `i` and `j` respectively.
   
These matrices are returned as follows:
```
rMCB,Weights,Indices = cyc_dec.random_MCB(merge_MCB=True)
```
See `random_MCB_and_twistane` for more on random sampling of MCBs. -->

## Examples
This repository includes several Jupyter notebooks (located in `\examples`):
- `cycle_decomposition.ipynb`. This notebook gives an overview of the `cycle_decomposition` object including:
  - example graphs and their `pi` and `sli` classes,
  - algorithmic tests of the time-complexity to compute the `cycle_decomposition` object,
  - and an outline of the algorithmic steps to compute the `cycle_decomposition` object.
- `random_MCB_and_twistane.ipynb`. This notebook demonstrates the procedure for sampling the MCB of a graph uniformly at random including:
  - an outline of the algorithmic steps used to construct a random MCB,
  - discussion of the convergence of this approach,
  - and an example of the connection matrices `Weights` and `Indices` when `merge_MCB=True`
- `random_geometric_graphs.ipynb`. Computation of the loop length distribution for the relevant cycles and for a random MCB given by a random geometric graph model with periodic boundary conditions. These codes show the number of large relevant cycles grows much faster than the number of large MCB cycles.
- `C10H16_cycle_analysis.ipynb`. Analysis of the size distribution of the `pi` and `sli` classes in the carbon skeleton of an adamantane ($\rm C_{10}H_{16}$) simulation at 4000K. The data for this notebook are contained in the zip file in `data/C10H16_4000K_bond_data`.

## Citing this repository
If you use the codes or data in this repository please cite the following
> Theory and algorithms for clusters of cycles in graphs for material networks. Perrin E. Ruth, Maria K. Cameron, _submitted_. [arXiv:2511.09732](https://doi.org/10.48550/arXiv.2511.09732)
