"""
cycxchg
===
Construct the cycles of a networkx graph. Documentation assumes the
package is loaded as 

    >>> import cycxchg as cxc

For more information see "https://github.com/perrineruth/Sampling_Minimum_Cycle_Bases"
"""

# automatically load cycle exchange functions
from .cycxchg import (
    Fundamental_Cycle_Basis,
    cycle_decomposition,
    pair_intersect,
    merge_pair,
)

# sparseb.py is meant for backend purposes only
# These functions can be accessed through cxc.sparseb