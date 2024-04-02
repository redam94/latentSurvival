import numpy as np

def simulate_data(n_media_vars, n_controls, R2=.9, nobs=151, multiplicative=True, **kwargs):
    """Simulate data from a linear model with Gaussian noise.

    Parameters
    ----------
    n_media_vars : int
        Number of media variables
    n_controls : int
        Number of control variables
    snr : float
        Signal-to-noise ratio
    nobs : int
        Number of observations

    Returns
    -------
    X : ndarray, shape (nobs, n_media_vars + n_controls)
        Design matrix
    y : ndarray, shape (nobs,)
        Response vector
    beta : ndarray, shape (n_media_vars + n_controls,)
        True regression coefficients
    """
    M = kwargs.get('M', np.random.exponential(np.exp(np.random.normal(4, .4, size=n_media_vars)), size=(nobs, n_media_vars)))
    assert M.shape == (nobs, n_media_vars)
    X = kwargs.get('X', np.random.normal(size=(nobs, n_controls)))
    assert X.shape == (nobs, n_controls)

    beta_c = kwargs.get('beta_c', np.random.normal(loc=0, scale=.5, size=n_controls))
    assert beta_c.shape == (n_controls,)

    beta_m = kwargs.get('beta_m', np.random.lognormal(mean=.01, sigma=.05, size=n_media_vars))
    assert beta_m.shape == (n_media_vars,)
    
    
    

    half_sat = kwargs.get('half_sat', np.random.uniform(.1, 3, size=n_media_vars)*np.median(M, axis=0))
    assert half_sat.shape == (n_media_vars,)
    n = kwargs.get('n', np.random.uniform(1, 5, size=n_media_vars))
    assert n.shape == (n_media_vars,)
    M_ = M**n/(M**n + half_sat**n)
    
    deterministic = M_ @ beta_m + X @ beta_c
    sigma = (1-R2)/R2*np.var(deterministic)

    intercept = kwargs.get('intercept', np.random.normal(size=1, scale=1, loc=5))

    if (not isinstance(intercept, float|int)):
        assert (intercept.shape == (1,))
    
    random = np.random.normal(size=nobs, scale=np.sqrt(sigma))
    if multiplicative:
        y = np.exp(deterministic + random + intercept)
    else:
        y = deterministic + random + intercept

    return M, X, y, beta_m, beta_c, intercept, half_sat, n

def get_last_period(x, interval_bounds, n_intervals):
    """Get the last period for each observation.

    Parameters
    ----------
    x : ndarray, shape (nobs,)
        Vector of observations
    interval_bounds : ndarray, shape (n_intervals + 1,)
        Bounds of the intervals
    n_intervals : int
        Number of intervals

    Returns
    -------
    last_periods : ndarray, shape (nobs,)
        Last period for each observation
    """
    last_periods = np.zeros(x.shape[0], dtype=int)
    for i in range(n_intervals):
        last_periods[(x >= interval_bounds[i]) & (x <= interval_bounds[i + 1])] = i
    return last_periods

def get_last_periods(X, interval_bounds, n_intervals):
    """Get the last period for each observation.

    Parameters
    ----------
    X : ndarray, shape (nobs, nvars)
        Matrix of observations
    interval_bounds : ndarray, shape (n_intervals + 1,)
        Bounds of the intervals
    n_intervals : int
        Number of intervals

    Returns
    -------
    last_periods : ndarray, shape (nobs, nvars)
        Last period for each observation
    """
    return np.apply_along_axis(get_last_period, 1, X, interval_bounds, n_intervals)

def get_interval_length(interval_bounds):
    """Get the length of each interval.

    Parameters
    ----------
    interval_bounds : ndarray, shape (n_intervals + 1,)
        Bounds of the intervals

    Returns
    -------
    interval_length : ndarray, shape (n_intervals,)
        Length of each interval
    """
    return interval_bounds[1:] - interval_bounds[:-1]

def compute_exposure_(X, interval_bounds, last_period, n_intervals):
    """Compute the exposure for each observation.

    Parameters
    ----------
    X : ndarray, shape (nobs, nvars)
        Matrix of observations
    interval_bounds : ndarray, shape (n_intervals + 1,)
        Bounds of the intervals
    n_intervals : int
        Number of intervals

    Returns
    -------
    exposure : ndarray, shape (nobs, nvars)
        Exposure for each observation
    """
    exposure = np.zeros((X.shape[0], n_intervals), dtype=float)
    for i in range(n_intervals):
        exposure[X>interval_bounds[i], i] = get_interval_length(interval_bounds)[i]
    exposure[np.arange(X.shape[0]), last_period] = X - interval_bounds[last_period]
    return exposure

def compute_exposure(X, interval_bounds, n_intervals):
    """Compute the exposure for each observation.

    Parameters
    ----------
    X : ndarray, shape (nobs, nvars)
        Matrix of observations
    interval_bounds : ndarray, shape (n_intervals + 1,)
        Bounds of the intervals
    n_intervals : int
        Number of intervals

    Returns
    -------
    exposure : ndarray, shape (nobs, nvars)
        Exposure for each observation
    """
    last_period = get_last_periods(X, interval_bounds, n_intervals)
    return np.array([compute_exposure_(X[:, i], interval_bounds[:, i], last_period[:, i], n_intervals) for i in range(X.shape[1])]).transpose(1, 0, 2)

def cum_hazard(hazard, interval_bounds):
    return ((interval_bounds[1:]-interval_bounds[:-1]).T * hazard).cumsum(axis=-1)


def survival(hazard):
    return np.exp(-cum_hazard(hazard))


def get_mean(trace):
    return trace.mean(("chain", "draw"))

def hill(x, b, a, n):
    return b*x**n/(x**n+a**n)

def s_origin(x, coeff, alpha, beta, scale):
    return coeff*(beta**(alpha**((x/scale)*100))-beta)