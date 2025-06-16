"""Functions to sample from probabilistic models."""

import os
from cmdstanpy import CmdStanModel

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)


def sample_posterior(
        z: float,
        nsamples: int,
        mean: float = 100,
        std_dev: float = 10,
        reduction_factor: float = 0.2,
        thin_factor: int = 100,
    ):
    """Sample from posterior model defined in `posterior.stan`.

    Args:
        z (float): Measurement value to condition the posterior on.
        nsamples (int): Number of samples to draw from the posterior.
        mean (float, optional): Mean of prior dist. Defaults to 100.
        std_dev (float, optional): Std dev of prior dist. Defaults to 10.
        reduction_factor (float, optional): Std dev of likelihood model as
            fraction of prior std dev. I.e. fraction by which uncertainty is
            reduced from prior to likelihood. Defaults to 0.2.
        thin_factor (int, optional): Factor to thin samples from MCMC chain
            by to reduce correlation. Defaults to 100.

    Returns:
        np.array: Array of samples from posterior distribution.
    """

    stan_model = CmdStanModel(stan_file=os.path.join('models','posterior.stan'))

    data = {
        'mu':mean,
        'sigma':std_dev,
        'reduction_factor':reduction_factor,
        'z':z
    }
    inits = {'theta':mean}

    posterior_fit = stan_model.sample(
            data=data,
            inits=inits,
            iter_warmup=nsamples*thin_factor,
            iter_sampling=nsamples*thin_factor,
            chains=1,
            show_progress=False
        )

    return posterior_fit.stan_variable('theta')[::thin_factor]