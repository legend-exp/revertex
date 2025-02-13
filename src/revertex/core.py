from __future__ import annotations

import hist
import numpy as np


def sample_histogram(histo:hist.Hist,size: int, *,seed:int| None = None)->np.ndarray:
    """Generate samples from a 1D or 2D histogram.

    Based on approximating the histogram as a piecewise uniform,
    probability distribution.

    Parameters
    ----------
    histo
        The histogram to generate samples from.
    size
        The number of samples to generate.
    seed
        Random seed.

    Returns
    -------
    an array of the samples
    """
    
    if (not isinstance(histo,hist.Hist)):
        msg = f"sample histogram needs hist.Hist object not {type(histo)}"
        raise TypeError(msg)
    
    ndim = h.ndim

    if (ndim == 1):
        # convert to numpy
        probs, bins = hist.to_numpy()
        
        # compute the binwidth
        binwidths = np.diff(bins)

        #normalise
        probs = probs/sum(probs)
        bin_idx = rng.choice(np.arange(len(probs)), size=size, p=probs)
        start = bins[bin_idx]
        
        # the value within the bin
        delta = rng.uniform(low=0,high=1,size=size)*bin_widths[bin_idx]
        values = start + delta

    elif (ndim ==2):
        pass
    else:
        msg = f"It is only supported to sample from 1D or 2D histograms not {ndim}"
        raise ValueError(msg)

    return values

def convert_output(arr:ak.Array,*,mode:str ="pos")->Table:
    """Converts the vertices to the correct output format."""
    pass