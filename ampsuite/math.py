import numpy as np
import operator
# import bisect
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from statistics import median_low

PI = 4*np.arctan(1)
EFACT = 1/np.log(10)


def mad(x):
    """
    Compute the Median Absolute Deviation (MAD) of the input array x.
    MAD is defined as: MEDIAN(ABS(X-MEDIAN(X)))
    
    Parameters:
    x (array-like): Input array.
    
    Returns:
    mad (float): The Median Absolute Deviation of the input array.
    """
    return np.median(np.abs(x-np.median(x)))

def madn(x,k=1.4826):
    """
    Compute a normalized Median Absolute Deviation (MAD) of the input array.
    The MAD is normalized by a constant scale factor k which depends on the 
    distribution. For normally distributed data, k is typically 1.4826.
    
    The MAD may be used similarly to how one would use the standard deviation  
    of the average. In order to use the MAD as a consistent estimator for the 
    estimation of the standard deviation sigma, one multiplies
    sigma = MEDIAN(ABS(X-MEDIAN(X))) * k
    
    Parameters:
    x (array-like): Input array.
    k (float): Constant scale factor, which depends on the distribution (default=1.4826).
    
    Returns:
    madn (float): The normalized Median Absolute Deviation of the input array.
    """
    return (np.median(np.abs(x-np.median(x))))*k


def last_index(lst, value):
    """
    Find the last occurrence of a value in a list.
    
    :param lst: Input list
    :param value: Value to search for
    :return: Index of the last occurrence of the value
    """
    return len(lst) - operator.indexOf(reversed(lst), value) - 1

def lmedian(array):
    """
    Calculate the lower median of an array.
    
    :param array: Input array or pandas Series
    :return: Lower median value
    """
    if isinstance(array, pd.Series):
        array = array.values
    
    count = len(array)
    
    if count == 0:
        return np.nan
    if count == 1:
        return array[0]
    if count == 2:
        return min(array)
    
    return median_low(array)
    
def leastupperbd(array, x): 
    """
    Find the least upper bound of x in a sorted array using bisection.
    
    :param array: Sorted input array
    :param x: Value to find the least upper bound for
    :return: Index of the least upper bound
    """
    if array[-1] == x:
        return array
    if array[-1] == x:
        return np.nan()
    if array[0] >= x:
        return 0
    i = int(array/2)
    ilo, ihi = 0, last_index(array, array[-1])
    
    while array[i - 1]  <= x and x < array[i]:
        if array[i] > x:
            ihi = i 
            i = ilo +int((i - ilo)/2)
            if i == ihi:
                i -= 1
        elif array[i] < x:
            ilo = i 
            i = i + int((ihi-i)/2)
            if i == ilo:
                i += 1
        else:
            i += 1
    return i

def greatlowerbd(array, x):
    """
    Find the greatest lower bound of x in a sorted array using bisection.
    
    :param array: Sorted input array
    :param x: Value to find the greatest lower bound for
    :return: Index of the greatest lower bound
    """
    if array[-1] <= x:
        return array
    if array[0] > x:
        return np.nan()
    if array[0] == x:
        return 0 

    i = int(array/2)
    ilo, ihi = 0, last_index(array, array[-1])
    while array[i] < x and x <= array[i+1]:
        if array[i] > x:
            ihi = i 
            i = ilo + int((i -ilo)/2)
            if i == ihi:
                i -= 1
        elif array[i] < x:
            ilo = i 
            i = i + int((ihi-i)/2)
            if i == ilo:
                i -= 1
    return i

def polyfit(x, y, degree=1, rcond=2e-16, residuals=False, weights=None, covariance=False):
    """
    Perform polynomial fitting using numpy's polyfit function.
    
    :param x: x-coordinates
    :param y: y-coordinates
    :param degree: Degree of the fitting polynomial
    :param rcond: Condition number cutoff for small singular values
    :param residuals: Whether to return residuals
    :param weights: Weights for least-squares fit
    :param covariance: Whether to return the covariance matrix
    :return: Polynomial coefficients
    """
    #linefit ->numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    # least squares line fit
    # y = a + bx
    # errors, residuals?
    # https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    m, b = np.polyfit(x, y, degree, rcond=None, full=residuals, w=weights, cov=covariance)
    return m, b

def linefit(x,y):
    """
    Perform a simple linear fit using the least squares method.
    
    :param x: x-coordinates
    :param y: y-coordinates
    :return: Coefficients [a, b] for the line y = a + bx
    """
    if len(x) != len(y):
        raise Exception(f'len of x and y not equal.')
    a11, a12, a22, atd1,atd2 = 0, 0, 0, 0, 0

    for xi,yi in zip(x,y):
        a11 += 1
        a12 += xi
        a22 += xi**2
        atd1 += yi
        atd2 += xi*yi

    det = a11*a22 - a12**2;
    if not det:
        return [np.nan(), np.nan()]
    a = (a22*atd1 - a12*atd2)/det
    b = (a11*atd2 - a12*atd1)/det
    return np.array([a, b])


def rofunc(b0, ndata, x, y, aa, debug=False): 
    """
    Helper function for medfit. Calculates sum of residuals and absolute deviation.
    
    :param b0: Initial slope estimate
    :param ndata: Number of data points
    :param x: x-coordinates
    :param y: y-coordinates
    :param aa: Initial intercept estimate
    :param debug: If True, return debug information
    :return: Sum of residuals, updated intercept, and absolute deviation
    """
    #why did WSP input aa in his implementation?
    arr = [yi-b0*xi for xi,yi in zip(x,y)]
    arr = np.array(arr)
    arr = np.sort(arr) #sorting least to greatest?
    # arr = sorted([y[i] - b0 * x[i] for i in range(len(x))])
    #my @arr = sort {$a <=> $b} map {$y->[$_] - $b0*$x->[$_]} (0 .. $#$x);
    aa = np.median(arr)
    #print(type(aa))
    #$aa = median(\@arr); # slow, NR uses "select"
    sum_ref,abdev = np.array([0],dtype = 'float64'),np.array([0],dtype = 'float64')
    for i in range(0,len(x)):
        if debug:
            return x[i], y[i], b0, aa
        d = y[i] - (b0*x[i]+aa)
        #print(y[i] - (b0*x[i]+aa))
        #print(d)
        abdev = abdev+np.abs(d)
        if y[i] !=0:
            d = d/np.abs(y[i]) 
            #print('line # 350')
            #print(d)
        if d > 0:
            z = x[i]
            #print('line # 354')
            #print(d)
        else:
            z = -1*x[i]
            #print(d)

        if np.abs(d) > np.array([1e-7]):
            sum_ref += z
            #print('line # 362')
            #print(d)
    abdev = abdev/ndata
    #abdev /= ndata
    aa = np.array([aa])
    return sum_ref, aa, abdev

def sparseL1norm(A):
    """
    Calculate the L1 norm of a sparse matrix.
    
    The L1 norm of a matrix is the maximum absolute column sum.
    
    :param A: Input sparse matrix
    :return: Lambda function that computes the L1 norm when called with a matrix
    """
    return lambda A: max([numpy.abs(A).getcol(i).sum() for i in range(A.shape[1])])


def medfit(x,y,debug=False):
    """
    Perform a median-based linear fit (L1 norm minimization).
    
    This function implements an improved version of the algorithm from
    Numerical Recipes (NR) 2nd edition for L1 linear fitting.
    
    :param x: x-coordinates of the data points
    :param y: y-coordinates of the data points
    :param debug: If True, return intermediate results for debugging
    :return: Intercept, slope, and absolute deviation of the fit
    """
    # ---------------------------------------
    # NR medfit 2nd edition improved
    # L1 line fit
    # one input vector, simpler in main code{
    #ndata = zip(x,y)
    #ndata = np.column_stack((x, y))
    ndata = np.concatenate((x, y))
    ndata = ndata.size
    #$ndata = @$xy;
    #ndata is a scalar and xy is an array, 
    #so this converts an array to a scalar
    #actually gives the number of elements in the array.
    
    
    sx = np.sum(x)
    sy = np.sum(y)
    sxy = np.sum([xi*yi for xi,yi in zip(x,y)])
    sxx = np.sum([xi**2 for xi in x])
    #[x**2 for x in range(10)] == list(map(lambda x: x**2, range(10)))
    #[x for x in S if x % 2 == 0] == list(filter(lambda x: x % 2 == 0, S))
    #my $sx = sum @x;
    #my $sy = sum @y;
    #my $sxy = sum map {$x[$_]*$y[$_]} (0 .. $#x);
    #my $sxx = sum map {$x[$_]**2} (0 .. $#x);
    del_ref = ndata*sxx - sx**2
    
    
    if ndata == 0:
        raise Exception(f'medfit: del zero, ndata.')
    
    #my $del = $ndata*$sxx - $sx**2;
    #$del != 0 or print("medfit: del zero, ndata $ndata\n"), return (undef, undef, undef);
    
    aa = (sxx*sy - sx*sxy)/del_ref
    bb = (ndata*sxy - sx*sy)/del_ref
    chisq = np.sum([(yi - (aa+bb*xi))**2 for xi,yi in zip(x,y)])
    #chisq = sum map {($y[$_] - ($aa + $bb*$x[$_]))**2} (0 .. $#x);
    sigb = np.emath.sqrt(chisq/del_ref)
    b1 = bb
    if debug:
        return b1, ndata, x, y, aa
    
    
    f1, aa, abdev = rofunc(b1, ndata, x, y, aa)
    if sigb > 0:
        if f1 >= 0:
            b2 = bb + 3*sigb
        else:
            b2 = bb - 3*sigb
        f2, aa, abdev = rofunc(b2, ndata, x, y, aa)
        if b2 == b1:
            return aa,bb,abdev
        
        while f1*f2 > 0:
            bb = b2 + 1.6*(b2 - b1)
            b1 = b2
            f1 = f2
            b2 = bb
            f2, aa, abdev = rofunc(b2, ndata, x, y, aa);
        sigb *= 0.01;
        while np.abs(b2-b1) > sigb:
            bb = b1 + (b2-b1)/2
            if bb == b1 or bb == b2:
                break
            (f, aa, abdev) = rofunc(bb, ndata, x, y, aa);
            if f*f1 >= 0:
                f1 = f
                b1 = bb
            else:
                f2 = f
                b2 = bb
                
        aa = np.array([aa])
        bb = np.array([bb])
        intercept,coef,dev = aa, bb, abdev
    return intercept,coef,dev


#mov routines below
# From this post : http://stackoverflow.com/a/40085052/3293881
def strided_app(array, window_len, step_size):  
    """
    Create a 2D array view into a 1D array with overlapping windows.
    
    Parameters:
    array (array-like): Input 1D array.
    window_len (int): Length of each window.
    step_size (int): Step size between windows.
    
    Returns:
    strided (array): 2D array view with overlapping windows.
    
    """
    nrows = ((array.size-window_len)//step)+1
    n = array.strides[0]
    return np.lib.stride_tricks.as_strided(array, shape=(nrows,window_len), strides=(step*n,n))

def movpercentile(array, width, step=1, percentile=75, method='roll'):
    # pad edge values to prepend/append to array before striding/rolling
    array_padded = np.pad(array, (width//2, width-1-width//2), mode='edge')
     #compute percentile (ignoring NaNs) using np.nanpercentile()
    
    if method == 'strided':
        #note strided method already concatenates array
        result = np.nanpercentile(strided_app(array_padded, width, step), percentile, axis=-1)
    if method == 'roll':
        #roll is much faster than stride, but need to code in support for step...
        result = np.nanpercentile(np.concatenate(
            [np.roll(array_padded, shift=i, axis=0)[...,None] for i in 
             range(-n_window, n_window+1)],axis=-1),percentile,axis=-1)
        #cut away the prepended/appended values
        result = result[width//2:-(int(width-1-width//2))]
    return result


def movmean(array, width=1, step=1, method='roll'):
    """
    Compute the moving mean of an array.
    
    Parameters:
    array (array-like): Input array.
    width (int): Width of the moving window (default=1).
    step (int): Step size between windows (default=1).
    method (str): Method to use, either 'roll' or 'strided' (default='roll').
    
    Returns:
    result (array): Moving mean of the input array.
    """
    # pad edge values to prepend/append to array before striding/rolling
    array_padded = np.pad(array, (width//2, width-1-width//2), mode='edge')
     #compute percentile (ignoring NaNs) using np.nanpercentile()
    
    if method == 'strided':
        #note strided method already concatenates array
        result = np.nanmean(strided_app(array_padded, width, step), axis=-1)
    if method == 'roll':
        #roll is much faster than stride, but need to code in support for step...
        result = np.nanmean(np.concatenate(
            [np.roll(array_padded, shift=i, axis=0)[...,None] for i in 
             range(-n_window, n_window+1)],axis=-1),axis=-1)
        #cut away the prepended/appended values
        result = result[width//2:-(int(width-1-width//2))]
    return result

def movmedian(array, width=1, step=1, method='roll'):
    """
    Compute the moving median of an array.
    
    Parameters:
    array (array-like): Input array.
    width (int): Width of the moving window (default=1).
    step (int): Step size between windows (default=1).
    method (str): Method to use, either 'roll' or 'strided' (default='roll').
    
    Returns:  
    result (array): Moving median of the input array.
    """
    # pad edge values to prepend/append to array before striding/rolling
    array_padded = np.pad(array, (width//2, width-1-width//2), mode='edge')
     #compute percentile (ignoring NaNs) using np.nanpercentile()
    
    if method == 'strided':
        #note strided method already concatenates array
        result = np.nanmedian(strided_app(array_padded, width, step), axis=-1)
    if method == 'roll':
        #roll is much faster than stride, but need to code in support for step...
        result = np.nanmedian(np.concatenate(
            [np.roll(array_padded, shift=i, axis=0)[...,None] for i in 
             range(-n_window, n_window+1)],axis=-1),axis=-1)
        #cut away the prepended/appended values
        result = result[width//2:-(int(width-1-width//2))]
    return result
    
def movrms(x, window_length, run_opt,base,unit):
    """
    Compute a moving window RMS for the input signal.
    
    Parameters:
    x (array-like): Input signal.
    window_length (int): Length of the moving window.
    run_opt (str): Run option for movmean, 'same' returns same length as input, 'valid' shrinks ends.
    base (str): Base of the logarithm, either 'log10' or something else.
    unit (str): Unit of the input data, either 'meters' or 'nanometers'.
    
    Returns:
    rms_mov_amp (array): Moving window RMS of the input signal.
    """
    if base == 'log10':
        rms_mov_amp = movmean(np.square(np.ravel(x)),window_length, run_opt)
        if unit == 'meters':
            rms_mov_amp = np.log(rms_mov_amp)*EFACT - 9
        elif unit == 'nanometers':
            rms_mov_amp = np.log(rms_mov_amp)*EFACT
    else: 
        rms_mov_amp = movmean(np.square(np.ravel(x)),window_length, run_opt)
    return rms_mov_amp

# From this post : http://stackoverflow.com/a/14314054/3293881
def moving_average(a, n=3):
    """
    Compute the moving average of an array.

    Parameters:
    a (array-like): Input array.
    n (int): Size of the moving window (default=3).

    Returns:
    output (array-like): Array of the moving average.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def movmad_numpy(a, W):
    """
    Compute the moving median absolute deviation (MAD) of an array.

    Parameters:
    a (array-like): Input array.
    W (int): Size of the moving window.

    Returns:
    mad (array-like): Array of the moving MAD.
    """
    a2D = strided_app(a,W,1)
    return np.absolute(a2D - moving_average(a,W)[:,None]).mean(1)


def movstd(T, m):
    """
    Compute the moving standard deviation of an array.

    Parameters:
    T (array-like): Input array.
    m (int): Size of the moving window.

    Returns:
    output (array-like): Array of the moving standard deviation.
    """
    n = T.shape[0]
    
    cumsum = np.cumsum(T)
    cumsum_square = np.cumsum(T**2)
    
    cumsum = np.insert(cumsum, 0, 0)               # Insert a 0 at the beginning of the array
    cumsum_square = np.insert(cumsum_square, 0, 0) # Insert a 0 at the beginning of the array
    
    seg_sum = cumsum[m:] - cumsum[:-m]
    seg_sum_square = cumsum_square[m:] - cumsum_square[:-m]
    
    return np.sqrt( seg_sum_square/m - (seg_sum/m)**2 )


#taken from https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/generated/scipy.stats.median_absolute_deviation.html
def median_absolute_deviation(x, axis=0, center=np.median, scale=1.4826,
                              nan_policy='propagate'):

    """
    Compute the median absolute deviation of the data along the given axis.
    The median absolute deviation (MAD, [1]_) computes the median over the
    absolute deviations from the median. It is a measure of dispersion
    similar to the standard deviation but more robust to outliers [2]_.
    The MAD of an empty array is ``np.nan``.
    .. versionadded:: 1.3.0
    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the function
        signature ``func(arr, axis)``.
    scale : int, optional
        The scaling factor applied to the MAD. The default scale (1.4826)
        ensures consistency with the standard deviation for normally distributed
        data.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    mad : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.
    See Also
    --------
    numpy.std, numpy.var, numpy.median, scipy.stats.iqr, scipy.stats.tmean,
    scipy.stats.tstd, scipy.stats.tvar
    Notes
    -----
    The `center` argument only affects the calculation of the central value
    around which the MAD is calculated. That is, passing in ``center=np.mean``
    will calculate the MAD around the mean - it will not calculate the *mean*
    absolute deviation.
    References
    ----------
    .. [1] "Median absolute deviation" https://en.wikipedia.org/wiki/Median_absolute_deviation
    .. [2] "Robust measures of scale" https://en.wikipedia.org/wiki/Robust_measures_of_scale
    Examples
    --------
    When comparing the behavior of `median_absolute_deviation` with ``np.std``,
    the latter is affected when we change a single value of an array to have an
    outlier value while the MAD hardly changes:
    >>> from scipy import stats
    >>> x = stats.norm.rvs(size=100, scale=1, random_state=123456)
    >>> x.std()
    0.9973906394005013
    >>> stats.median_absolute_deviation(x)
    1.2280762773108278
    >>> x[0] = 345.6
    >>> x.std()
    34.42304872314415
    >>> stats.median_absolute_deviation(x)
    1.2340335571164334
    Axis handling example:
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> stats.median_absolute_deviation(x)
    array([5.1891, 3.7065, 2.2239])
    >>> stats.median_absolute_deviation(x, axis=None)
    2.9652
    """

    #from scipy.stats._stats_py import _contains_nan
    x = np.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        return np.nan

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan and nan_policy == 'propagate':
        return np.nan

    if contains_nan and nan_policy == 'omit':
        # Way faster than carrying the masks around
        arr = ma.masked_invalid(x).compressed()
    else:
        arr = x

    if axis is None:
        med = center(arr)
        mad = np.median(np.abs(arr - med))
    else:
        med = np.apply_over_axes(center, arr, axis)
        mad = np.median(np.abs(arr - med), axis=axis)

    return scale * mad

def movwin(x, window):
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _contains_nan(a, nan_policy='propagate'):
    """
    Check if an array contains NaN values.

    Parameters:
    a (array-like): Input array to check for NaN values.
    nan_policy (str): Defines how to handle when input contains NaN.
                      Options: 'propagate' (default), 'raise', 'omit'.

    Returns:
    contains_nan (bool): True if the array contains NaN values, False otherwise.
    nan_policy (str): The nan_policy used.

    Raises:
    ValueError: If nan_policy is not one of 'propagate', 'raise', or 'omit'.
    RuntimeWarning: If the input array could not be properly checked for NaN values.
    """
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)