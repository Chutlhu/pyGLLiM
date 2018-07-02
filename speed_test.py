import numpy as np

# glim speed test

theta_kdim = {
    'ct': np.array((self.Lt, self.K)),
    'cw': np.array((self.Lw, self.K)),
    'Gammat': np.array((self.Lt, self.Lt, self.K)),
    'Gammaw': np.array((self.Lw, self.Lw, self.K)),
    'pi': np.array((self.K)), # Gaussian weights of X
    'A': np.array((self.D, self.L, self.K)), # Affine transformation matrices
    'b': np.array((self.D, self.K)), # Affine transformation vectors
    'Sigma': np.array((self.D, self.D, self.K)), # Error covariances
}

theta_klist = {
    'ct': np.array((self.Lt) for k in range(self.K)),
    'cw': np.array((self.Lw) for k in range(self.K)),
    'Gammat': np.array((self.Lt, self.Lt) for k in range(self.K)),
    'Gammaw': np.array((self.Lw, self.Lw) for k in range(self.K)),
    'pi': 1./self.K for k in range(self.K)), # Gaussian weights of X
    'A': np.array((self.D, self.L, ) for k in range(self.K)), # Affine transformation matrices
    'b': np.array((self.D, ) for k in range(self.K)), # Affine transformation vectors
    'Sigma': np.array((self.D, self.D, ) for k in range(self.K)), # Error covariances
}
