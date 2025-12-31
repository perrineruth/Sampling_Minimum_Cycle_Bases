import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_left



# _______________ Special Matrices & Functions _________
def speye(n):
    """Sparse nxn identity matrix over the field of binary numbers modulo 2 (GF2)"""
    return sparse_GF2_mat([[i] for i in range(n)], nrows=n,ncols=n)

def zeros(nrows,ncols):
    """Sparse (nrows)x(ncols) zero matrix over the field of binary numbers modulo 2 (GF2)"""
    return sparse_GF2_mat([[] for i in range(ncols)],nrows=nrows,ncols=ncols)

def vstack2(Mat1,Mat2):
    """
    Stack sparse GF2 matrices Mat1 and Mat2 vertically. Mat1 and Mat2 must have the same
    number of columns.
    """
    if Mat1.ncols!=Mat2.ncols: raise ValueError('stacked matrices must have the same number of columns')
    # for each column use two lists that concatenate and shift the latter by Mat1.nrows
    aux = sparse_GF2_mat([[] for i in range(Mat1.ncols)],nrows=Mat1.nrows+Mat2.nrows, ncols=Mat1.ncols)
    for i in range(Mat1.ncols): # avoid checking inputs
        aux[:,i] = Mat1[:,i].nonzero + [row+Mat1.nrows for row in Mat2[:,i]] # cast mat1 to nonzero indexes to make
    return aux



# _______________ Sparse Matrix Class __________________
class sparse_GF2_mat:
    """
    Sparse binary matrix with input parameters
     - data = list of lists data[0],...,data[ncols-1]: data[i] is a SORTED list of indexes corresponding to the 
                nonzero entries in column i
     - nrows = number of rows.
     - ncols = number of columns, must be equal to len(data)
    """
    def __init__(self, data, nrows, ncols):
        # size of matrix m x n =  nrows x ncols
        self.nrows=nrows
        self.ncols=ncols
        if len(data)!=ncols: raise ValueError('One list must be provided per column')

        # store data as sparse GF2 vectors
        # copy data as column vectors, transform into sparse vector format if not yet formatted
        self.columnData = np.ndarray((ncols),dtype='object') # format as numpy array -> superior indexing over lists
        for idx,col in enumerate(data):
            # fill columns after array creation, otherwise numpy treats sparse vectors as lists losing their functionality
            if type(col)==sparse_GF2_vec:   # already a sparse vector, just check size
                if len(col)==nrows: self.columnData[idx] = col
                else:               raise ValueError(f'invalid vector length {len(col)} for matrix with nrows={nrows}')
            else:                           # list,iterable,etc.
                self.columnData[idx] = sparse_GF2_vec(col,nrows)

    # representation for stdout
    def __repr__(self):
        string = f"sparse_GF2_mat({self.nrows} rows, {self.ncols} cols"
        for i in range(min(self.ncols,9)):
            string+=f"\n    column {i}: {str(self.columnData[i])}"
        if self.ncols>9:
            if self.ncols>10: string+=f"\n    ..."
            string+=f"\n    column {self.ncols-1}: {str(self.columnData[self.ncols-1])}"
        string+=" )"
        return string
    
    # indexing, not all reasonable indexing is implemented! Just enough to make the code work
    def __getitem__(self,idx):
        match idx:
             # 2D input: matrix indexing, both cases lead here.
            case (x_idx,y_idx):
                # parse columns first -> rely on numpy for parsing
                aux = self.columnData[y_idx].copy() # copy array so that we do not change original matrix
                match aux:
                    # indexing gives a matrix
                    case np.ndarray():
                        if type(x_idx)==slice and x_idx==slice(None): # special case, no row indexing, use a shorthand
                            return sparse_GF2_mat(aux, nrows=self.nrows, ncols=len(aux))
                        else: 
                            # index sparse vector columns with x_idx
                            for i,col in enumerate(aux):
                                aux[i] = col[x_idx]
                            # if type(aux[0])==sparse_GF2_vec: # vector columns
                            #     return sparse_GF2_mat(aux, nrows=len(aux[0]), ncols=len(aux))
                            # elif type(aux[0])==int:          # special case single row was accessed return a binary array
                            #     return aux.astype(int)
                            if type(x_idx) == int: # special case single row was accessed, return a binary array
                                return aux.astype(int)
                            else:
                                nrows = len(np.arange(self.nrows)[x_idx]) # indexing matches numpy when available
                                return sparse_GF2_mat(aux, nrows=nrows, ncols=len(aux))
                    # y_idx is an integer -> return a single column (indexed by x_idx)
                    case sparse_GF2_vec():
                        return aux[x_idx]
            # 1D input: column indexing
            case _:
                # shorthand M[idx] = M[:,idx]
                return self[:,idx]
    
    # item assignment, only single entry M[i,j] (type(i)=type(j)=int) or single column assignment M[:,j]
    def __setitem__(self,idx,val):
        match idx:
            case (x_idx,y_idx): # x y inputs given
                # y must be an integer for column / entry assignment
                if not np.issubdtype(type(y_idx),np.integer): 
                    raise ValueError('Only single column or entry assignment implemented')
                ### column assignment
                if x_idx == slice(None):
                    if isinstance(val,list): 
                        self.columnData[y_idx] = sparse_GF2_vec(val,self.nrows)
                    elif isinstance(val,sparse_GF2_vec):
                        if len(val) == self.nrows: 
                            self.columnData[y_idx] = val
                        else: return ValueError() # wrong length
                    elif isinstance(val,sparse_GF2_mat) and val.nrows==self.nrows and val.ncols==1: 
                        self.columnData[y_idx] = val.columnData[0]
                    else: raise ValueError("Invalid column assignment")
                ### entry assignment - utilize item assignment of sparse vector class
                elif np.issubdtype(type(x_idx),np.integer):
                    self.columnData[y_idx][x_idx] = val
                else:
                    raise ValueError('Only single column or entry assignment implemented')
            # if not (x,y) input then assume it's M[y] and redo assignment as M[:,y]
            case y_idx: # column indexing
                self[:,y_idx] = val # recall function with x index = slice(None)
        
    # matrix addition A+B or column-wise vector addition (A+v)[:,i] := A[:,i]+v
    def __add__(self,B):
        match B:
            case list() | sparse_GF2_vec():     # add vector columnwise - use vector class to check validity of given operation
                return sparse_GF2_mat([col+B for col in self.columnData], nrows=self.nrows, ncols=self.ncols)
            case sparse_GF2_mat():              # matrix addition
                if B.nrows==self.nrows and B.ncols==self.ncols:
                    return sparse_GF2_mat([self.columnData[i]+B.columnData[i] for i in range(self.ncols)], nrows=self.nrows, ncols=self.ncols)
                else:
                    raise ValueError("Matrices must be of equal sizes to be added")
            case _:                  
                # error
                raise ValueError(f"Addition not implemented between type {type(self)} and {type(B)}")
        
    # matrix-matrix or matrix-vector multiplication
    def __matmul__(self,B):
        match B:
            # matrix-vector multiplication -> sparse vector
            case list() | sparse_GF2_vec():
                return sum([self.columnData[i] for i in B],start=sparse_GF2_vec([],self.nrows))
            # matrix-matrix multiplication -> sparse matrix
            case sparse_GF2_mat():
                if self.ncols!=B.nrows:
                    raise ValueError('Number of columns does not equal number of rows in multiplied matrix.')
                return sparse_GF2_mat([_iter_sum([self.columnData[i] for i in v]) for v in B.columnData], nrows=self.nrows, ncols=B.ncols)
                return sparse_GF2_mat([sum([self.columnData[i] for i in v],start=sparse_GF2_vec([],self.nrows)) for v in B.columnData], nrows=self.nrows, ncols=B.ncols)

    # specialty matrix-vector multiplication A@v output as an iterator
    def right_vec_dot_iter(self,v):
        """
        Compute a matrix-vector product as an iterator. This allows for efficient computation of the first non-zero element 
        of the product. This is used to speed up computation of a MCB by returning only an optimal cycle with given inner product.
        """
        if isinstance(v,list) or isinstance(v,sparse_GF2_vec):
            return _iter_sum([self.columnData[i] for i in v])
        elif isinstance(v,sparse_GF2_mat) and v.ncols==1 and v.nrows==self.ncols:
            return _iter_sum([self.columnData[i] for i in v.columnData[0]])
        else:
            raise ValueError("malformed input for matrix vector product")

    # vector-matrix multiplication
    def __rmatmul__(self,v):
        # this would not be called for matrix-matrix mult A@B. v has to be a vector
        if isinstance(v,list) or isinstance(v,sparse_GF2_vec):
            return [col.dot(v) for col in self.columnData]
        else:
            assert ValueError(f"cannot multiply matrix with type {type(v)}")
            
    # class function for transpose
    def transpose(self):
        rowData = [[] for i in range(self.nrows)]
        for j,col in enumerate(self.columnData):
            for i in col:
                rowData[i].append(j)
        return sparse_GF2_mat(rowData, nrows=self.ncols, ncols=self.nrows)

    # shorthand
    @property
    def T(self):
        return self.transpose()

    # as a property to do M.nonzero[i][j] with slices
    @property
    def nonzero(self):
        """
        Obtain nonzeros elements as a list of lists where the i'th interior list 
        has the coordinates for the nonzero entries of column i. This simply is the list of lists data.
        """
        return self.columnData
    
    def copy(self):
        """
        Return a copy of the sparse matrix that does not affect the original
        matrix when modified.
        """
        return sparse_GF2_mat([col.copy() for col in self.columnData], nrows=self.nrows, ncols=self.ncols)

    def to_array(self):
        aux = np.zeros((self.nrows,self.ncols),dtype=int)
        for j,col in enumerate(self.columnData):
            for i in col:
                aux[i,j]=1
        return aux
    
    # for visualization
    def plot(self,ax=None):
        if ax is None: plt.spy(self.to_array())
        else:          ax.spy(self.to_array())
    



# ______________ Sparse Vector Class ___________________
class sparse_GF2_vec:
    """
    Sparse binary vector, with operations defined modulo 2. To construct run sparse_GF2_vec(L,d) where 
    - L is a sorted integer list whose values are the indexes of the nonzero entries in the vector
    - d is the dimension of the vector.
    """
    def __init__(self,data,len):
        self.data = list(data) # data should be in list format, could be initially an iterator or np array
        self.len  = len
        # validate input format
        for i,val in enumerate(self.data):
            if not np.issubdtype(type(val),np.integer): raise ValueError('Only integer indexes permitted') # integer or numpy integer...
            if i>=1 and val<=self.data[i-1]:            raise ValueError('Input list is not sorted (strictly increasing)')
            if val>=len:                                raise ValueError('indexes can only range from 0,...,d-1 for dimension d')

    # binary vector addition modulo 2
    def __add__(self,v):
        # defer to addition of sparse vectors
        if type(v) != sparse_GF2_vec:
            try:    v = sparse_GF2_vec(v,self.len)
            except: raise ValueError(f"Addition not implemented between type sparse_GF2_vec and {type(v)}")
        elif self.len!=v.len: 
            raise ValueError(f"added vectors are not of equal length") # if it is a vector they need to be the same length
        # create empty vector for output, reduces length of time to check valid format
        out_vec = sparse_GF2_vec([],self.len)
        as_iter1,as_iter2 = iter(self),iter(v)
        # initialize search, try to obtain a single
        try: x=next(as_iter1) # first element of self
        except StopIteration: # self is empty, return v
            out_vec.data = v.data.copy()
            return out_vec
        try: y=next(as_iter2)
        except StopIteration: # v is empty, return self
            out_vec.data = self.data.copy()
            return out_vec
        # iterate through both lists
        vals = [] # this will be the final data
        while True:
            if x<y:
                # x is the next lowest element
                vals.append(x)
                try: x=next(as_iter1)
                except StopIteration:
                    vals.append(y)              # y not included in iterator
                    vals.extend(list(as_iter2)) # add rest of v to list
                    break                       # leave while loop
            if y<x:
                vals.append(y)
                try: y=next(as_iter2)
                except StopIteration:
                    vals.append(x)
                    vals.extend(list(as_iter1)) # add rest of self to list
                    break
            if y==x:
                # shared element x and y cancel, step through iteration
                try: x=next(as_iter1)
                except StopIteration:
                    # old y do not include
                    vals.extend(list(as_iter2)) # add rest of v to list
                    break                       
                try: y=next(as_iter2)
                except StopIteration:
                    vals.append(x) # new x, DO include
                    vals.extend(list(as_iter1)) # add rest of self to list
                    break
        out_vec.data=vals
        return out_vec

    # code to associate @ operator to dot product, refer to __rmatmul__ of matrix type for vector-matrix multiplication
    def __matmul__(self,v):
        if isinstance(v,list) or isinstance(v,sparse_GF2_vec):
            return self.dot(v)
        else:
            return NotImplemented # this is to implement vector-matrix multiplication as v@A where v=self

    # binary dot product, where the final sum is modulo 2
    def dot(self,v):
        # dot product between self and v modulo 2, only defined between two sparse vectors
        if type(v) != sparse_GF2_vec:
            v = sparse_GF2_vec(v,self.len)
        elif v.len != self.len:
            raise ValueError('cannot perform dot product between vectors of different lengths')
        # iterate through u and v sorted to find common value
        as_iter1,as_iter2 = iter(self),iter(v)
        val = 0 # dot product temp value
        # initial draws from iterators
        try: x,y = next(as_iter1),next(as_iter2)
        except StopIteration: return val
        # loop through lists
        while True:
            if x<y:
                try: x=next(as_iter1)
                except StopIteration: return val # multiply by 1 to cast to int
            elif y<x:
                try: y=next(as_iter2)
                except StopIteration: return val
            else:
                val = 1-val # shared index add 1 to dot product (flip value)
                try: x,y = next(as_iter1),next(as_iter2)
                except StopIteration: return val

    # return bool v[idx] for if idx is present in list data
    def __getitem__(self,idx):
        match idx:
            case int() | np.integer():
                # map to 0,...,len - 1 in the case of negative indexes
                idx = range(len(self))[idx]
                # could use bisection here, but the data list tends to be short
                return int(idx in self.data) 
            case slice():
                # use slice to determine which rows to extract. output = sparse vector
                start,stop,step = idx.indices(len(self)) # essentially range(start,stop,step)
                length = len(range(start,stop,step))     # number of rows indexed
                if step == 1: # split step logic for some optimization on if's
                    data = [i-start for i in self.data if i>=start and i<stop]
                elif step > 1:
                    data = [(i-start)//step for i in self.data if i>=start and i<stop and ((i-start)%step)==0]
                else: # 'step<0'
                    data = [(i-start)//step for i in self.data[::-1] if i<=start and i>stop and ((i-start)%step)==0]
                return sparse_GF2_vec(data,length)
            case np.ndarray():
                # only binary masks of vector
                if idx.dtype==bool and np.shape(idx)==(self.len,):
                    newRows = np.cumsum(idx)-1                          # set up so if i is a valid entry newRows[i] is its new index
                    data = [newRows[i] for i in self.data if idx[i]]    # extract valid entries
                    return sparse_GF2_vec(data, sum(idx))
                else: raise ValueError('only integer, slice, and binary mask indexing of sparse arrays implemented')
            case _:
                raise IndexError('only integer, slice, and binary mask indexing of sparse arrays implemented')

    # set index to 1 (add idx to list) or to 0 (remove idx from list)
    def __setitem__(self,idx,val):
        # assumes val is a bool, this is not formally checked. 
        # If val is, e.g., True, 1, or [3.14,2.5], then it will set the entry with the given index

        # parse index
        if not np.issubdtype(type(idx),np.integer):
            raise ValueError('only single entry assignment x[i]=val implemented for sparse vectors')
        if idx>=self.len:
            raise ValueError(f'invalid index {idx} for vector of length {self.len}')
        if idx<0:
            idx = self.len+idx
            if idx<0: # too far backwards... did not go back to 0,...,len-1
                raise ValueError(f'invalid index {idx-self.len} for vector of length {self.len}')

        # index of first entry in data >= idx
        idx2 = bisect_left(self.data, idx)
        # set to 1
        if val:
            # not present yet
            if idx2>=len(self.data) or self.data[idx2]!=idx:
                self.data.insert(idx2,idx) # format is insert(index,element): idx is the value to add and idx2 is where to add it
            # else no change
        # set to 0
        else:
            if idx2<len(self.data) and self.data[idx2]==idx:
                del self.data[idx2] 

    # as property to allow index v.nonzero[idx] = v.data[idx]
    @property
    def nonzero(self):
        """
        Indexes of nonzero entries. Cases for input index:
         - idx=None: all nonzero entries (default)
         - idx=integer i: return ith nonzero index
        """
        return self.data
    
    def __len__(self):
        # dimension of vector
        return self.len

    # if list format is needed, just return data
    def __list__(self):
        return self.data
    
    # iterator format of list
    def __iter__(self):
        return iter(self.data)
    
    # for print out to stdout
    def __repr__(self):
        return str(self.data)

    def copy(self):
        aux = sparse_GF2_vec([],self.len) # empty vector to skip format checking
        aux.data = self.data.copy()
        return aux

# _______________ Supplementary vector functions _____________________
# sum of iterable sparse binary vectors return as an iterable rather than a list
# useful when we only want the first nonzero index and the remaining are unimportant as 
def _iter_sum(iterables):
    Ni = len(iterables)
    if Ni>2: 
        return _iter_add(_iter_sum(iterables[:Ni//2]),_iter_sum(iterables[Ni//2:]))
    elif Ni==2:
        return _iter_add(iterables[0],iterables[1])
    elif Ni==1:
        return iter(iterables[0])
    else: # Ni==0
        return iter()

def _iter_add(L1,L2):
    # L1, L2 may be lists or sparse GF2 vectors
    as_iter1,as_iter2 = iter(L1),iter(L2)
    # get first elements, print if it stops
    try: x = next(as_iter1)
    except StopIteration:
        yield from as_iter2 # yield items from iter2 step by step
        return
    try: y = next(as_iter2)
    except StopIteration:
        yield x             # x already popped, then yield everything else
        yield from as_iter1
        return
    
    # loop through and take smaller, cancel if equal
    while True:
        # print next element 
        # case 1: list 1 element first
        if x<y:
            yield x
            try: x = next(as_iter1)
            except StopIteration:
                yield y
                yield from as_iter2
                return
        # case 2: list 2 element first
        elif y<x:
            yield y
            try: y = next(as_iter2)
            except StopIteration:
                yield x
                yield from as_iter1
                return
        # case 3: tie and cancel
        else:
            try: x = next(as_iter1)
            except StopIteration:
                yield from as_iter2 # don't output y which cancels
                return
            try: y = next(as_iter2)
            except StopIteration:
                yield x # this x is new
                yield from as_iter1
                return
    