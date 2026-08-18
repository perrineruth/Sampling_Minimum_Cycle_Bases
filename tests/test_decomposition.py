# test counts of the cycle decomposition
import numpy as np
import networkx as nx
import cycxchg as cxc

def test_lattice():
    max_side_len = 4 
    for m in range(1,max_side_len+1): # width, num. edges
        for n in range(1,max_side_len+1): # length
            for k in range(1,max_side_len+1): # height
                G = nx.grid_graph((m+1,n+1,k+1))
                cyc_dec = cxc.cycle_decomposition(G)
                # single pi class of m x n x k cubes
                assert len(cyc_dec.pi_classes) == cyc_dec.num_pi_classes == 1
                pc = cyc_dec.pi_classes[0]
                assert pc.length        == 4
                assert pc.dim_polyhedra == m*n*k
                assert pc.rank          == G.number_of_edges()-G.number_of_nodes()+1
                assert pc.num_cycles    == cyc_dec.num_relevant_cycles == m*n*(k+1)+m*(n+1)*k+(m+1)*n*k
                # sli class for each cycle
                assert len(cyc_dec.sli_classes) == cyc_dec.num_sli_classes == cyc_dec.num_relevant_cycles
                for sc in cyc_dec.sli_classes:
                    assert sc.length     == 4
                    assert sc.num_cycles == 1
                    edges = sc.edges()
                    assert len(edges) == 4
                    for e in edges: assert e in G.edges


def test_nested_rings():
    # do not test exact relevant cycle counts -> floating point errors may occur for large rings
    for k in range(2,101): # number of rings
        G = nx.cycle_graph(range(3*k))
        for j in range(k):
            label = -3*j-1 # -1,-4,-7,... mirror of 1,4,7,...
            G.add_edges_from([[label,3*j],[label,3*j+2]])
        cyc_dec = cxc.cycle_decomposition(G)
        assert  len(cyc_dec.pi_classes) == cyc_dec.num_pi_classes == \
            len(cyc_dec.sli_classes) == cyc_dec.num_sli_classes == \
            cyc_dec.nu == k+1 # validate dimension of cycle space and that all sli classes are pi classes
        # validate first k sli classes are a valid square
        for sc in cyc_dec.sli_classes[:k]:
            assert sc.length == 4 and sc.num_cycles == 1
            edges = sc.edges()
            for e in edges: assert e in G.edges
        # validate last large sli class
        sc = cyc_dec.sli_classes[-1]
        assert sc.length == 3*k and sc.num_cycles == 2**k
        edges = sc.random_cycle(rep='edges') # check it produces a cycle in G
        for e in edges: assert e in G.edges