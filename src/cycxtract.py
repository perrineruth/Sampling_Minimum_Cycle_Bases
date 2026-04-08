######################################################################
# Codes to extract the relevant cycles and bin them into families.
# This is separated from the main cycxchg.py file that groups the 
# relevant cycle families into pi and sli classes. These codes are 
# the primary computational bottleneck, so it is useful to separate 
# them for optimization purposes.
######################################################################

import numpy as np
import time
# optimize with numba where available
try: 
    from numba import njit
    from scipy.sparse import csr_matrix
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define a dummy decorator that does nothing
    def njit(**kwargs):
        def decorator(f):
            return f
        return decorator
# custom
import sparseb as spb # sparse binary matrix operations

# rel_cyc_matrix_constructor
class Relevant_Cycle_Matrix_Constructor:
    def __init__(self, G, FCB, nu, method='full', **kwargs):
        N = G.number_of_nodes()
        adj_list  = [list(G[u]) for u in range(len(G))]
        self.nu  = nu 
        self.FCB = FCB
        if nu == 0: # special case no cycles: handle directly
            self.Rel_Cyc_Fams = []
            self.R_Mat = spb.zeros(0,0)
        elif method == 'full':
            # CSR graph representation
            ind_ptr = np.cumsum([0]+[len(aux) for aux in adj_list])
            indices = np.concatenate(adj_list)
            # use CSR representation to lookup FCB indexes
            FCB_vals = [[FCB.edge2idx[(u,v)] if (u,v) in FCB.edge2idx else -1    for v in L]    for u,L in enumerate(adj_list)]
            FCB_vals = np.concatenate(FCB_vals)
            if HAS_NUMBA:
                self.Rel_Cyc_Fams = []
                for idx,(u,v) in enumerate(FCB.edges):
                    Fams, pind_ptr,pindices,ancestor,distance,numPaths = _Edge_Families_Full_numba(N,ind_ptr,indices,FCB_vals,idx,u,v)
                    # convert parents to a format that can be indexed like a list of lists
                    class CSR_AdjList:
                        def __init__(self,ind_ptr,indices):
                            self.ind_ptr = ind_ptr
                            self.indices = indices
                        def __getitem__(self,idx):
                            return self.indices[self.ind_ptr[idx]:self.ind_ptr[idx+1]]
                    edge_DAG_AdjList = CSR_AdjList(pind_ptr,pindices)
                    edge_DAG = {'parents':edge_DAG_AdjList, 'ancestor':ancestor, 'distance':distance, 'numPaths':numPaths}
                    # post-process cycle families
                    for row in Fams:
                        if row[0]==-1: # odd family
                            p,length,num_cycles = row[1:]
                            if num_cycles>1:    self.Rel_Cyc_Fams.append(RelCyc_Family((u,v), length, num_cycles, x=p, edge_DAG=edge_DAG, FCB=FCB, **kwargs))
                            else:               self.Rel_Cyc_Fams.append(Ess_Family((u,v), length, num_cycles, x=p, edge_DAG=edge_DAG, FCB=FCB, **kwargs))
                        else:          # even family
                            p,q,length,num_cycles = row
                            if num_cycles > 1:  self.Rel_Cyc_Fams.append(RelCyc_Family((u,v), length, num_cycles, e1=(p,q), edge_DAG=edge_DAG, FCB=FCB, **kwargs))
                            else:               self.Rel_Cyc_Fams.append(Ess_Family((u,v), length, num_cycles, e1=(p,q), edge_DAG=edge_DAG, FCB=FCB, **kwargs))
            else:
                self.Rel_Cyc_Fams = sum([list(_Edge_Families(G,i,FCB,**kwargs)) for i in range(len(FCB.edges))],start=[])
            # sort by length and extract a matrix of prototypes
            self.Rel_Cyc_Fams = sorted(self.Rel_Cyc_Fams, key=lambda x: x.length)
            Cols = [[] for i in range(self.nu)]
            for row,Fam in enumerate(self.Rel_Cyc_Fams):
                for edge in Fam.arbitrary_cycle(rep='FCB'): # edge from FCB representation
                    Cols[edge].append(row)
            self.R_Mat = spb.sparse_GF2_mat(Cols,nrows=len(self.Rel_Cyc_Fams),ncols=self.nu)
        elif method == 'iter': # for when max cycle length << diameter
            # iterators for rel cycle families which can be terminated early 
            self._edge_iterators = [_Edge_Families(adj_list, idx, FCB, **kwargs) for idx in range(nu)]
            # initialize everything else to be empty
            self.Rel_Cyc_Fams = []
            self.R_Mat = spb.zeros(0,nu)     
            self.Lmax = 2                               # no cycles length <=2
            self.buffer = [None for i in range(nu)]     # buffer for later when generating cycles
        else: raise ValueError(f'Invalid relevant cycle computation {method}')
    
    def get_dot(self, S):
        # Get the index of the first Vismara family to have dot product 1 with the input witness
        Cur_Mat = self.R_Mat
        offset = 0
        while (Cyc_idx:=next(Cur_Mat.right_vec_dot_iter(S), False)) is False: # the second argument in next is a default if the iterator empties itself
            # if fails then we need to compute more Vismara families
            # The following extends the R_Mat and outputs the block of larger cycles added so we don't need to search smaller ones again
            offset+=Cur_Mat.nrows
            Cur_Mat = self.extend_Mat()
        # The ':=' operator assigns the index of the first family with dot product 1 to Cyc_idx, return this
        Cyc_idx+=offset
        return self.Rel_Cyc_Fams[Cyc_idx].arbitrary_cycle(), Cyc_idx
        
    def extend_Mat(self):
        # add a buffer
        self.Lmax += 2  # introduce a new round of odd and even cycles
        newOddCycs  = [] # length Lmax-1 
        newEvenCycs = [] # length Lmax
        
        # pass through Vismara families at each edge
        for idx,Fam in enumerate(self.buffer):
            # start with the cycle in the buffer
            if Fam is not None:
                if Fam.length == self.Lmax-1:
                    newOddCycs.append(Fam)
                elif Fam.length == self.Lmax:
                    newEvenCycs.append(Fam)
                else: 
                    # cycle is already larger than Lmax, we can look at the next starting edge, keep the same cycle in the buffer
                    continue
                    
            # iterate through new Vismara families until we reach a cycle larger than Lmax or run out of families
            while (Fam:=next(self._edge_iterators[idx],None)) is not None and Fam.length<=self.Lmax:
                if Fam.length == self.Lmax-1:
                    newOddCycs.append(Fam)
                elif Fam.length == self.Lmax:
                    newEvenCycs.append(Fam)
                else: raise ValueError('Invalid cycle length') # it should not be possible to get here
            
            # update buffer
            self.buffer[idx] = Fam # either a larger cycle or None if out of families
            
        # update Relevant Cycle Families, sorted by length
        newFams = newOddCycs+newEvenCycs # odd cycles shorter than even cycles
        self.Rel_Cyc_Fams += newFams     # previous cycles smaller than current cycles
        # create new matrix for cycles of current length
        NewCols = [[] for i in range(self.nu)]
        for row,Fam in enumerate(newFams):
            for edge in Fam.arbitrary_cycle(rep='FCB'): # edge from FCB representation
                NewCols[edge].append(row)
        New_Mat = spb.sparse_GF2_mat(NewCols,nrows=len(newFams),ncols=self.nu)
        # append to R matrix
        self.R_Mat = spb.vstack2(self.R_Mat,New_Mat)
        # output new matrix
        return New_Mat
        
    def yield_matrix(self):
        return self.R_Mat, self.Rel_Cyc_Fams



def aux_test(G,FCB,mode,**kwargs):
    N = G.number_of_nodes()
    adj_list = [list(G[u]) for u in G]
    # CSR graph representation
    ind_ptr = np.cumsum([0]+[len(aux) for aux in adj_list])
    indices = np.concatenate(adj_list)
    # use CSR representation to lookup FCB indexes
    FCB_vals = [[FCB.edge2idx[(u,v)] if (u,v) in FCB.edge2idx else -1    for v in L]    for u,L in enumerate(adj_list)]
    FCB_vals = np.concatenate(FCB_vals)
    if mode == 'python':
        result = sum([list(_Edge_Families(G,i,FCB,**kwargs)) for i in range(len(FCB.edges))],start=[])
    elif mode == 'numpy':
        result = []
        for idx,(u,v) in enumerate(FCB.edges):
            result.append(_Edge_Families_Full_numpy(N,ind_ptr,indices,FCB_vals,idx,u,v))
    elif mode == 'numba':
        result = []
        for idx,(u,v) in enumerate(FCB.edges):
            Fams, pind_ptr,pindices,ancestor,distance,numPaths = _Edge_Families_Full_numba(N,ind_ptr,indices,FCB_vals,idx,u,v)
            # convert parents to a format that can be indexed like a list of lists
            edge_DAG_CSR = csr_matrix((np.ones_like(pindices), pindices, pind_ptr)) # data=1, neighbors, pointers to neighbor array
            class CSR_AdjList:
                def __init__(self,A):
                    self.CSR = A
                def __getitem__(self,idx):
                    return self.CSR[idx].nonzero()[1]
            edge_DAG_AdjList = CSR_AdjList(edge_DAG_CSR)
            edge_DAG = {'parents':edge_DAG_AdjList, 'ancestor':ancestor, 'distance':distance, 'numPaths':numPaths}
            result.append([u,v,Fams,edge_DAG])
    else:
        raise ValueError(f'mode: {mode}')
    return result

# Helper function for constructing relevant cycle families
# - Optimizing is important here: this is currently the highest cost
def _Edge_Families(G, e_idx, FCB, **kwargs):
    N = len(G) # number of nodes, works for list of lists and networkx
    u,v = FCB.edges[e_idx]
    # list of lists representation of directed-acyclic graph
    # with shortest paths to (u,v)
    parents = [[] for _ in range(N)] 
    # node ancestor / closest node between u and v: 
    #    1 = u,  2 = v,  3 = both
    #    0 default for unobserved,
    ancestor = [0]*N # appears to run faster with lists than arrays
    distance = [0]*N # minimum distance to u or v
    numPaths = [0]*N # number of (valid) shortest paths from to u or v
    # good to wrap these into a dictionary - Directed acyclic graph with node properties
    edge_DAG = {'parents': parents, 'ancestor': ancestor, 'distance': distance, 'numPaths': numPaths} 
    # initialize root nodes
    ancestor[u],ancestor[v] = 1,2
    distance[u],distance[v] = 0,0
    numPaths[u],numPaths[v] = 1,1

    isValid = [True]*N             # invalid if following a node with both ancestors...
    queue = [u,v]                  
    for p in queue: 
        # if p is at the tip of an odd relevant cycle family
        if ancestor[p] == 3 and isValid[p]:
            # look at parents for num. path pairs, use DAG to backtrack
            num_cycles = sum([numPaths[x] for x in parents[p] if ancestor[x]==1]) * \
                            sum([numPaths[x] for x in parents[p] if ancestor[x]==2]) # product of path counts to u and v
            # possible that there are none, this is a degenerate case and should be skipped
            if num_cycles > 0: 
                # cycles are 2 shortest paths to u and v and (u,v)
                length = 2*distance[p] + 1
                if num_cycles>1:
                    yield RelCyc_Family((u,v), length, num_cycles, x=p, edge_DAG=edge_DAG, FCB=FCB, **kwargs)
                else:
                    yield Ess_Family((u,v), length, num_cycles, x=p, edge_DAG=edge_DAG, FCB=FCB, **kwargs)
                                                        
        # look at neighbors
        for q in G[p]:
            ###### case 1: v undiscovered #######
            if ancestor[q]==0:
                # inherit distance and ancestor properties directly
                distance[q] = distance[p]+1
                ancestor[q] = ancestor[p]
                queue.append(q)
                # degenerate - if p has ancestor u & v then q is invalid
                if ancestor[p]==3:
                    isValid[q] = False # numpaths already 0,
                # "there is a valid path to p and (p,q) is valid"
                elif numPaths[p]>0 and not ((p,q) in FCB.edge2idx and e_idx<FCB.edge2idx[(p,q)]): 
                    parents[q].append(p)
                    numPaths[q] = numPaths[p]
            ###### case 2: v after u ############
            elif distance[q] > distance[p]:
                if not isValid[q]: continue
                # degenerate - if p has ancestor u & v then q is invalid
                if ancestor[p]==3:
                    isValid[q] = False
                    numPaths[q],parents[q] = 0,[]
                else:
                    # update ancestor if u adds a new one (or if v is already 3 this is no change)
                    if ancestor[p]!=ancestor[q]: ancestor[q]=3
                    # add paths and parent if valid
                    if numPaths[p]>0 and not ((p,q) in FCB.edge2idx and e_idx<FCB.edge2idx[(p,q)]): 
                        parents[q].append(p)
                        numPaths[q]+=numPaths[p]
            ###### case 3: v same distance ###### 
            #  add even family if p has ancestor u, and q ancestor v to avoid double counting
            #  need to remove corner case p=u, q=v, it may make this more efficient to parse this before the loop.
            elif ancestor[p]==1 and ancestor[q]==2 and p!=u:
                num_cycles = numPaths[p] * numPaths[q]
                if num_cycles>0 and ((p,q) not in FCB.edge2idx or e_idx>=FCB.edge2idx[(p,q)]):
                    # cycle is path p->u & q->v and edges (u,v) & (p,q)
                    length = 2*distance[p]+2 
                    if num_cycles > 1:
                        yield RelCyc_Family((u,v), length, num_cycles, e1=(p,q), edge_DAG=edge_DAG, FCB=FCB, **kwargs)
                    else:
                        yield Ess_Family((u,v), length, num_cycles, e1=(p,q), edge_DAG=edge_DAG, FCB=FCB, **kwargs)

# numba implementation for significant speed increases
@njit()
def _Edge_Families_Full_numba(N,ind_ptr,indices,FCB_vals,e_idx,u,v):
    parents = np.zeros_like(indices) # CSR representation of DAG
    degrees = np.zeros(N,dtype=np.int64)  # use the same ind_ptr format as the input adjacency matrix with a degree cutoff

    ancestor = np.zeros(N,dtype=np.int64) # node ancestor / closest node between u and v
    distance = np.zeros(N,dtype=np.int64) # minimum distance to u or v
    numPaths = np.zeros(N,dtype=np.int64) # number of (valid) shortest paths from to u or v
    # initialize root nodes
    ancestor[u] = 1
    ancestor[v] = 2
    numPaths[u] = 1
    numPaths[v] = 1

    # output families
    Fams = np.zeros((e_idx+1,4),dtype=np.int64)
    nFams = 0

    # run BFS
    isValid = np.ones(N,dtype=np.bool_)
    queue = [u,v]
    head = 0
    while head<len(queue):
        p = queue[head] 
        if ancestor[p] == 3 and isValid[p]:
            u_paths = 0
            v_paths = 0
            for idx in range(degrees[p]):
                par = parents[ind_ptr[p]+idx]
                if ancestor[par] == 1:
                    u_paths += numPaths[par]
                else:
                    v_paths += numPaths[par]
            num_cycles = u_paths*v_paths
            if num_cycles > 0: 
                length = 2*distance[p] + 1
                Fams[nFams,0] = -1
                Fams[nFams,1] = p
                Fams[nFams,2] = length
                Fams[nFams,3] = num_cycles
                nFams += 1
        
        for neigh_idx in range(ind_ptr[p],ind_ptr[p+1]):
            q = indices[neigh_idx] # q a neighbor of x
            # CASE 1: unobserved
            if ancestor[q]==0:
                distance[q] = distance[p]+1
                ancestor[q] = ancestor[p]
                queue.append(q)
                if ancestor[p]==3:
                    isValid[q] = False
                # "there is a valid path to p and the edge (p,q) is valid"
                elif numPaths[p]>0 and e_idx>=FCB_vals[neigh_idx]: 
                    parents[ind_ptr[q]] = p
                    degrees[q]  = 1 # initially 0
                    numPaths[q] = numPaths[p]
            # CASE 2: q discovered already, p on a shortest path
            elif distance[q] > distance[p]:
                if not isValid[q]: continue
                if ancestor[p]==3:
                    isValid[q]  = False
                    numPaths[q] = 0
                    degrees[q]  = 0
                else:
                    if ancestor[p]!=ancestor[q]: 
                        ancestor[q]=3
                    if numPaths[p]>0 and e_idx>=FCB_vals[neigh_idx]: 
                        parents[ind_ptr[q]+degrees[q]] = p
                        degrees[q]  += 1
                        numPaths[q] += numPaths[p]
            elif ancestor[p]==1 and ancestor[q]==2 and p!=u:
                num_cycles = numPaths[p] * numPaths[q]
                if num_cycles>0 and e_idx>=FCB_vals[neigh_idx]:
                    # cycle is path p->u & q->v and edges (u,v) & (p,q)
                    length = 2*distance[p]+2 
                    Fams[nFams,0] = p
                    Fams[nFams,1] = q
                    Fams[nFams,2] = length
                    Fams[nFams,3] = num_cycles
                    nFams += 1

        head = head+1
        
    # reduce parents DAG to CSR format for efficient return
    pind_ptr = np.zeros(N+1,dtype=np.int64)
    for i in range(N): 
        pind_ptr[i+1] = pind_ptr[i]+degrees[i]
    pindices = np.zeros(pind_ptr[N],dtype=np.int64)
    for i in range(N): 
        for j in range(degrees[i]):
            pindices[pind_ptr[i]+j] = parents[ind_ptr[i]+j]
    return Fams[:nFams,:], pind_ptr,pindices,ancestor,distance,numPaths



# class for representing Vismara's cycle families.
class RelCyc_Family:
    """Description"""
    def __init__(self, e0, length, num_cycles, x=None, e1=None, **kwargs):
        self.e0 = e0                    # base edge
        self.length = length            # length of cycles
        self.parity = self.length % 2   # even or odd cycle
        self.num_cycles = num_cycles    # number of cycles in the family
        if self.parity == 0: # even cyc., e1 is opposite edge
            assert (x is None and e1 is not None)
            self.e1 = e1
        else:
            assert (x is not None and e1 is None)
            self.x = x
        # pass additional arguments into class
        #  edge_DAG -> edges from child to parent
        #  FCB -> fundamental cycle basis
        #  node_labels / labels2idx -> convert from node index to label and vice versa
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        if self.parity == 0:
            return f"RelCyc_Family(root={(self.node_labels[self.e0[0]],self.node_labels[self.e0[1]])}, descriptor={(self.node_labels[self.e1[0]],self.node_labels[self.e1[1]])}, length={self.length}, num_cycles={self.num_cycles})"
        else:
            return f"RelCyc_Family(root={(self.node_labels[self.e0[0]],self.node_labels[self.e0[1]])}, descriptor={self.node_labels[self.x]}, length={self.length}, num_cycles={self.num_cycles})"

    def arbitrary_cycle(self, rep='FCB'):
        """Extract an arbitrary cycle from a cycle family."""
        # initialize away from the root edge
        if self.parity == 1:
            x = self.x
            u = next(u for u in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][u]==1) # arbitrary parent w/ u0 as ancestor
            v = next(v for v in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][v]==2)
            left_nodes = [x,u]
            right_nodes = [v]
        else:
            u,v = self.e1
            left_nodes,right_nodes = [u],[v]
        # backtrack cycles to first parent
        while u not in self.e0:
            u = self.edge_DAG['parents'][u][0]
            left_nodes.append(u)
        while v not in self.e0:
            v = self.edge_DAG['parents'][v][0]
            right_nodes.append(v)
        # merge both paths, move upwards through the right path
        left_nodes.extend(right_nodes[::-1])
        nodes = left_nodes
        # convert to desired format
        if rep == 'nodes': 
            return [self.node_labels[u] for u in nodes]
        edges = {tuple(sorted([nodes[i], nodes[(i+1)%len(nodes)]])) for i in range(len(nodes))}
        if rep == 'edges':
            return {(self.node_labels[u],self.node_labels[v]) for u,v in edges}
        elif rep == 'FCB':
            return [self.FCB.edge2idx[e] for e in edges if e in self.FCB.edge2idx]

    def random_cycle(self, rep='FCB'):
        """random_cycle
        choose a random cycle from Vismara's cycle family uniformly at random.
        """
        def weighted_random_node(nodes):
            if len(nodes)==1: return nodes[0] # avoid expensive random call if possible
            # draw random node from selection with probability proportional to number of cycles using node
            weights = np.array([self.edge_DAG['numPaths'][u] for u in nodes])
            return np.random.choice(nodes, p=weights/sum(weights))
        # opposite end of cycle
        if self.parity == 1:
            x = self.x
            u = weighted_random_node([u for u in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][u]==1])
            v = weighted_random_node([v for v in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][v]==2]) 
            left_nodes = [x,u]
            right_nodes = [v]
        else:
            u,v = self.e1
            left_nodes,right_nodes = [u],[v]
        # backtrack cycles at random
        while u not in self.e0:
            u = weighted_random_node(self.edge_DAG['parents'][u])
            left_nodes.append(u)
        while v not in self.e0:
            v = weighted_random_node(self.edge_DAG['parents'][v])
            right_nodes.append(v)
        # merge both paths, move upwards through the right path
        left_nodes.extend(right_nodes[::-1])
        nodes = left_nodes
        if rep == 'nodes': 
            return [self.node_labels[u] for u in nodes]
        edges = {tuple(sorted([nodes[i], nodes[(i+1)%len(nodes)]])) for i in range(len(nodes))}
        if rep == 'edges':
            return {(self.node_labels[u],self.node_labels[v]) for u,v in edges}
        elif rep == 'FCB':
            return [self.FCB.edge2idx[e] for e in edges if e in self.FCB.edge2idx]
                
    def nodes(self):
        """
        Union of nodes belonging to cycles in the cycle family. Output format is a set 
        where the order of nodes is not important.
        """
        # tree-traversal over the DAG backwards to obtain all nodes in relevant family
        # initial queue -> nodes opposite of e0
        if self.parity == 1:    queue = [self.x]
        else:                   queue = list(self.e1) # e1 is a tuple
        searched = set(queue)
        # loop through queue nodes once
        for u in queue:
            for v in self.edge_DAG['parents'][u]:
                if v not in searched:
                    searched.add(v)
                    queue.append(v)
        return {self.node_labels[u] for u in searched}
    
    def edges(self):
        """Union of edges belonging to the cycle family"""
        # tree-traversal over the DAG backwards to obtain all edges in family
        # edge indexes are sorted to make edges unique as tuples
        e_sort = lambda pair: (min(pair),max(pair))
        if self.parity == 1:
            queue = [self.x]
            edge_set = set([e_sort(self.e0)])
        else:
            queue = list(self.e1) # deque from tuple
            edge_set = set([e_sort(self.e0),e_sort(self.e1)])
        searched = set(queue)
        # loop while more nodes
        for u in queue:
            for v in self.edge_DAG['parents'][u]:
                edge_set.add(e_sort((u,v)))
                if v not in searched:
                    searched.add(v)
                    queue.append(v)
        return {(self.node_labels[u],self.node_labels[v]) for u,v in edge_set}

    def all_cycles(self,rep='nodes'):
        """
        List of cycles belonging to the cycle family.
        Warning: runtime is exponential in the worst case.
        """
        if self.parity == 0:
            u1,v1 = self.e1
            left_paths,right_paths = [[u1]],[[v1]]
        else:
            x = self.x
            left_paths = [[x,u] for u in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][u]==1]
            # skip x for right paths, only count node once in each cycle
            right_paths = [[v] for v in self.edge_DAG['parents'][x] if self.edge_DAG['ancestor'][v]==2]
        # build out paths one node at a time
        for _ in range(self.length//2 - 1):
            left_paths = [path + [u] for path in left_paths for u in self.edge_DAG['parents'][path[-1]]]
            # travel right paths in reverse direction
            right_paths = [[u]+path for path in right_paths for u in self.edge_DAG['parents'][path[0]]]
        # build out cycles as path pairs
        cycles = [l_path+r_path for l_path in left_paths for r_path in right_paths]
        # use node labels
        cycles = [[self.node_labels[u] for u in C] for C in cycles]
        # only one representation currently allowed
        if rep == 'nodes':
            return cycles
        elif rep == 'edges':
            return [{(C[-i-1],C[-i]) for i in range(len(C))} for C in cycles]
        else: raise NotImplementedError

    # test cases (1) cube and (2) bracelet w/ 4 diamonds - try reversing and shifting lists too
    def contains(self,C,rep='nodes'):
        """Returns True if cycle C belongs to the cycle family."""
        if rep == 'nodes':
            # validate length
            L = self.length
            if len(C)!=L: return False
            # convert nodes to integer labels
            C = [self.labels2node[u] for u in C]
            u0,v0 = self.e0
            # locate u0 in C
            try:                idx = C.index(u0)
            except ValueError:  return False    # not found
            C = C[idx:]+C[:idx]             # move u0 to start of list
            # locate and move v0 to end of list
            if C[1]==v0:    C[1:] = C[1:][::-1] # reverse orientation
            elif C[-1]!=v0: return False        # not found in valid location
            # validate descriptors
            if self.parity == 0:
                if self.e1 != tuple(C[L//2-1 : L//2+1]):    return False
            elif self.x != C[L//2]:                         return False
            # validate paths
            for i in range((L-1)//2):
                # left path
                if C[i] not in self.edge_DAG['parents'][C[i+1]]:
                    return False
                # right path
                if C[-i-1] not in self.edge_DAG['parents'][C[-i-2]]:
                    return False
            # passed test
            return True
        else:
            raise NotImplementedError('Only nodes representation implemented for contains method.')

    def __contains__(self,item):
        # Wrapper so that the line "C in <Family>" uses the contains function.
        # Assumes cycle uses list of nodes representation.
        return self.contains(item)
    
# special case Vismara cycle family with one cycle
class Ess_Family:
    """Description"""
    def __init__(self, e0, length, num_cycles, x=None, e1=None, edge_DAG=None, FCB=None, node_labels=None, **kwargs):
        if x is not None: # odd family
            self.des=node_labels[x]
            u,v = edge_DAG['parents'][x] # parent ordering does not matter, may not correspond to u0,v0
            left_nodes = [x,u]
            right_nodes = [v]
        elif e1 is not None: # even family
            self.des=(node_labels[e1[0]],node_labels[e1[1]])
            u,v = e1
            left_nodes,right_nodes = [u],[v]
        else: raise ValueError
        # backtrack cycles to first parent
        while u not in e0:
            u = edge_DAG['parents'][u][0]
            left_nodes.append(u)
        while v not in e0:
            v = edge_DAG['parents'][v][0]
            right_nodes.append(v)
        # merge both paths, move upwards through the right path
        left_nodes.extend(right_nodes[::-1])
        nodes = left_nodes
        self.cycle = [node_labels[u] for u in nodes]
        # save FCB representation
        edges = {tuple(sorted([nodes[i-1], nodes[i]])) for i in range(len(nodes))}
        self.FCB_vec = [FCB.edge2idx[e] for e in edges if e in FCB.edge2idx]
        # save other values
        self.e0 = (node_labels[e0[0]],node_labels[e0[1]])
        self.length = length
        self.num_cycles = num_cycles

    def __repr__(self):
        return f"RelCyc_Family(root={self.e0}, descriptor={self.des}, length={self.length}, num_cycles={self.num_cycles})"

    def arbitrary_cycle(self, rep="FCB"):
        if rep=='nodes':    return self.cycle
        elif rep=='edges':  return {tuple(sorted([self.cycle[i-1], self.cycle[i]])) for i in range(len(self.cycle))}
        elif rep=='FCB':    return self.FCB_vec

    def random_cycle(self, rep='FCB'):
        return self.arbitrary_cycle(rep=rep)
    
    def nodes(self):
        return set(self.cycle)
    
    def edges(self):
        return {tuple(sorted([self.cycle[i-1], self.cycle[i]])) for i in range(len(self.cycle))}
    
    def all_cycles(self,rep='nodes'):
        return [self.arbitrary_cycle(rep=rep)]
    
    def contains(self,C,rep='nodes'):
        if rep=='nodes': 
            try:
                idx = C.index(self.cycle[0])
                idx2 = C.index(self.cycle[1])
                if (idx2-idx)%len(C) == 1:  orientation=1
                else:                       orientation=-1
                return self.cycle == C[idx::orientation]+C[:idx:orientation]
            except ValueError:
                return False
        elif rep=='edges':  raise NotImplementedError
        elif rep=='FCB':    return self.FCB_vec==C
        else:               raise ValueError

    def __contains__(self,item):
        return self.contains(item)