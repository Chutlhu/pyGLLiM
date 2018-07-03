import numpy as np
import scipy
import progressbar

from sklearn.mixture import GaussianMixture

tmp_mat_dir = '../the_GLLiM_toolbox_matlab/tmp/' # for debugging
def my_print(theta,k=0):
    print('mean Sigma' + str(k) + '= ' , np.mean(np.diag(theta['Sigma'][:,:,k])))
    print('mean Gamma' + str(k) + '= ' , np.mean(np.diag(theta['Gamma'][:,:,k])))
    print('mean A' + str(k) + '= ' , np.mean(theta['A'][:,:,k]))
    print('mean c' + str(k) + '= ' , np.mean(theta['c'][:,k]))
    print('mean b' + str(k) + '= ' , np.mean(theta['b'][:,k]))
    print('mean pi' + str(k) + '= ' , np.mean(theta['pi'][k]))
    print('\n')
    return

def my_print_w(mu, S, k = 0):
    print('mean muw' + str(k) + '= ' , np.mean(mu[:,:,k]))
    print('mean Sw' + str(k) + '= ' , np.mean(np.diag(S[:,:,k])))
    print('\n')
    return

class GLLIM:
    '''
    Gaussian Locally-Linear Mapping
    adapted from ...

    '''

    theta_keys = ['c','Gamma','pi','A','b','Sigma']

    def __init__(self, D, Lt, Lw, K,
                 theta_init={}, cstr_init={}, r_init=[],
                 verbose=1):

        '''
        Arguments
        ---------
        init_K: int
            initial number of components
        init_D: int
            number of the high-dimensional space
        init_L: int
            total number of variables
        '''

        self.D = D # high-dimensional space dim.
        self.Lt = Lt # observed variable dimension
        self.Lw = Lw # latent variable dimension
        self.L = Lt+Lw # low-dimensional space dim.
        self.K = K # number of initial Gaussian components
        self.theta = theta_init # parameters to estimate
        self.cstr = cstr_init # contraints on parameters theta
        self.r = r_init

        self.verbose = verbose

        # DEFINITION AND INITIALIZATION
        if not self.theta:
            if self.verbose:
                print('\nNo initial value for paramters theta Defaults are used')
            self.theta = {
                'c': nans((self.L, self.K)), # Gaussian means of X
                'Gamma': nans((self.L, self.L, self.K)), # Gaussian cov. of X
                'pi': nans((self.K)), # Gaussian weights of X
                'A': nans((self.D, self.L, self.K)), # Affine transformation matrices
                'b': nans((self.D, self.K)), # Affine transformation vectors
                'Sigma': nans((self.D, self.D, self.K)), # Error covariances
            }
        else:
            # add the missing entries
            for key in self.theta_keys:
                if not key in self.theta:
                    self.theta[key] = []

        if not self.cstr:
            if self.verbose:
                print('No initial contraints. Defaults are used')
            self.cstr = {
                'ct': nans((self.Lt, self.K)),
                'cw': nans((self.Lw, self.K)),
                'Gammat': nans((self.Lt, self.Lt, self.K)),
                'Gammaw': nans((self.Lw, self.Lw, self.K)),
                'pi': nans((self.K)), # Gaussian weights of X
                'A': nans((self.D, self.L, self.K)), # Affine transformation matrices
                'b': nans((self.D, self.K)), # Affine transformation vectors
                'Sigma': 'i*', # Error covariances
            }
        else:
            # add the missing entries
            for key in ['ct','cw','Gammat','Gammaw','pi','A','b','Sigma']:
                if not key in self.theta:
                    self.theta[key] = np.NaN
    #  end def

    def fit(self, t, y, max_iter):
        '''
        fit the gllim

        Arguments
        ---------
        t: ndarray
            (Lt x N) training latent variables
        y: ndarray
            (D x N) training observed variables
        max_iter: int
            maximum number of iterations
        Returns
        -------
        '''

        self.N = t.shape[1]

        #TODO BEAUTIFULL TABLE

        cstr = self.cstr
        converged = False
        it = 0
        log_like = -np.inf*np.ones(max_iter)

        if not check_input(t, y):
            raise ValueError("Dimensions do not agree")

        # EM INITIALIZATION
        r, ec, muw, Sw = self._initialization(t, y, self.theta, cstr, self.r)

        # EM ITERATIONS
        if self.verbose:
            print('\n\n  *** EM iterations ***')

        widget = ['\nRunning: EM iteration ', progressbar.SimpleProgress(),
                    ' (', progressbar.Timer(), ') ',
                    progressbar.Bar(left_justify = False), '\n']
        bar = progressbar.ProgressBar(widgets=widget, max_value=max_iter).start()

        for it in bar(range(max_iter)):

            # MAXIMIZATION STEP
            theta = self._maximization(t, y, r, cstr, muw, Sw)

            # EXPECTATION STEP
            r, log_like[it], ec = self._expectation_z(t,y,theta)
            theta, cstr = self._remove_empty_clusters(theta,cstr,ec)
            muw, Sw = self._expectation_w(t, y, theta)

            if it > 2:
                delta_log_like_total = np.max(log_like[0:it]) - np.min(log_like[0:it])
                delta_log_like = log_like[it] - log_like[it-1]

                if delta_log_like < 0.001*delta_log_like_total:
                    break

                if delta_log_like < 0:
                    print('\nLog likelihood',log_like[np.isfinite(log_like)])
                    raise ValueError('Log likelihood is decreasing, increment: %g'%(delta_log_like))
            bar.update(it + 1)

        bar.finish()
        print('\nEM aglorithm converged')

        log_like = log_like[np.isfinite(log_like)]
        if self.verbose > 2:
            print('Log likelihood:', log_like)

        return theta, r, log_like

    def _initialization(self, t, y, theta, cstr, r):
        if self.verbose: print('\n  *** EM Initialization ***')

        if self.Lw and allnans(cstr['cw']):
            # Default value for cw
            cstr['cw'] = np.zeros((self.Lw,self.K))

        if self.Lw and allnans(cstr['Gammaw']):
            # Default value for Gammaw
            cstr['Gammaw'] = np.zeros((self.Lw,self.Lw,self.K))
            for k in range(self.K):
                cstr['Gammaw'][:,:,k] = np.eye(self.Lw)

        if not [allnans(theta[key]) for key in self.theta_keys]:
            r, log_like, empty_cluster_indeces = self._expectation_z(t, y, theta)
            theta, cstr = self._remove_empty_clusters(theta, cstr, empty_cluster_indeces)
            muw, Sw = self._expectation_w(t, y, theta)

            return r, empty_cluster_indeces, muw, Sw

        if not r:
            # Initialise posteriors with GMM on joint observed data
            print('\nRunning: Gaussian Mixture Models')
            # if True:
            gm = GaussianMixture(n_components=self.K, max_iter=100,
            verbose=self.verbose)
            X = np.concatenate((t,y)).T
            gm.fit(X)
            r = gm.predict_proba(X)
            # else:
            #     import scipy.io
            #     mat = scipy.io.loadmat('../the_GLLiM_toolbox_matlab/tmp/r.mat')
            #     r = mat['r']

        if self.Lw == 0:
            Sw = np.zeros(0)
            muw = np.zeros(0)
            empty_cluster_indeces = []
        else:
            # Start by running an M-step without hidden variables (partial
            # theta), deduce Awk by local weighted PCA on residuals (complete
            # theta), deduce r, muw and Sw from E-steps on complete theta
            # PARTIAL THETA
            theta = self._maximization(t,y,r,cstr)

            # COMPLETE THETA:
            theta['c'] = np.concatenate((theta['c'], cstr['cw'][:,0:self.K])) #LxK


            Gammaf = np.zeros((self.L,self.L,self.K))
            Gammaf[0:self.Lt,0:self.Lt,:] = theta['Gamma'] #LtxLtxK
            Gammaf[self.Lt:self.L,self.Lt:self.L,:] = cstr['Gammaw'][:,:,0:self.K] #LwxLwxK
            theta['Gamma'] = Gammaf

            # Initialize Awk with local weighted PCAs on residuals:
            Aw = np.zeros((self.D,self.Lw,self.K))
            for k in range(self.K):

                rk_bar = np.sum(r[:,k])
                bk = np.reshape(theta['b'][:,k], (self.D,1), order = 'F')
                w = y - bk #DxN
                if self.Lt > 0:
                    Ak = np.reshape(theta['A'][:,0:self.Lt,k], (self.D,self.Lt), order = 'F')
                    w -= np.matmul(Ak,t)

                w = np.multiply(w, np.sqrt(r[:,k] / rk_bar)) #DxN
                C = np.matmul(w,w.T) # Residual weighted covariance matrix
                # Eigenvalues and Eigenvectors
                # The residual variance is the discarded eigenvalues' mean
                Lambdas, Vects = scipy.sparse.linalg.eigs(C, k = self.Lw) # Vects: DxLw
                Lambdas = np.real(Lambdas) # from complex to real
                Vects = np.real(Vects)
                if np.sign(Vects[np.abs(Vects)==np.max(np.abs(Vects))]) < 0:
                    Vects = -1*Vects
                # print('\n iter %d, sum: %f'%(k,Vects.sum()))
                if self.Lw > 1:
                    sigma2k = (np.trace(C)-np.trace(Lambdas))/(self.D-self.Lw) # 1x1
                else:
                    sigma2k = (np.trace(C)-Lambdas)/(self.D-self.Lw) # 1x1
                theta['Sigma'][:,:,k] = sigma2k * np.eye(self.D)
                Aw[:,:,k] = np.matmul(Vects, np.sqrt(Lambdas - sigma2k*np.eye(self.Lw)))

            theta['A'] = np.concatenate((theta['A'], Aw), axis=1) #DxLxK

            r, log_like, empty_cluster_indeces = self._expectation_z(t,y,theta)
            theta, cstr = self._remove_empty_clusters(theta, cstr, empty_cluster_indeces)
            muw, Sw = self._expectation_w(t,y,theta)

        return r, empty_cluster_indeces, muw, Sw

    def _maximization(self, t, y, r, cstr, muw=np.zeros(0), Sw=np.zeros(0)):
        if self.verbose: print('Maximization step')

        D, N = y.shape
        Lt, N = t.shape
        K = r.shape[1]
        Lw = muw.shape[0]
        L = Lt + Lw

        theta = {
            'c': nans((L, K)),
            'Gamma': nans((L, L, K)),
            'pi': nans(K),
            'A': nans((D, L, K)),
            'b': nans((D, K)),
            'Sigma': nans((D, D, K)),
        }

        cstr_type_gamma = eval_cstr_type(cstr['Gammat'])
        cstr_type_sigma = eval_cstr_type(cstr['Sigma'])

        if Lw > 0:
            theta['c'][Lt:L] = cstr['cw']
            theta['Gamma'][Lt:L, Lt:L, :] = cstr['Gammaw']

        rk_bar = np.zeros(K)

        bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])
        for k in bar(range(K)):

            # Posterior's sums
            rk = r[:,k].T # 1xN
            rk_bar[k] = rk.sum() # 1x1

            if Lt > 0:
                if self.verbose > 1:
                    print('  --- ct')
                # compute optimal mean ctk
                if allnans(cstr['ct']):
                    theta['c'][0:Lt,k] = np.sum(rk * t, axis = 1)/rk_bar[k] # Ltx1
                else:
                    theta['c'][0:Lt,k] = cstr['ct'][:,k]

                if self.verbose > 1:
                    print('  --- Gammat')
                # Compute optimal covariance matrix Gammat
                diffGamma = np.sqrt(rk) * (t - theta['c'][0:Lt,k,None]) # LtxN

                if cstr_type_gamma == 'full':
                    theta['Gamma'][0:Lt,0:Lt,k] = np.matmul(diffGamma,diffGamma.T) / rk_bar[k] # DxD
                if cstr_type_gamma == 'fixed':
                    theta['Gamma'][0:Lt,0:Lt,k] = cstr['Gammat'][:,:,k]
                if cstr_type_gamma == 'diagonal':
                    gamma2 = np.sum(diffGamma**2, 1)/rk_bar[k] #Ltx1
                    theta['Gamma'][0:Lt,0:Lt,k] = np.diag(gamma2) # LtxLt
                if cstr_type_gamma == 'isotropic':
                    gamma2 = np.sum(diffGamma**2, 1)/rk_bar[k] #Ltx1
                    theta['Gamma'][0:Lt,0:Lt,k] = np.mean(gamma2)*np.eye(Lt) # LtxLt
                if cstr_type_gamma == 'equal_det':
                    theta['Gamma'][0:Lt,0:Lt,k] = diffGamma*diffGamma/rk_bar[k] # LtxLt

            # Compute optimal weight pik
            theta['pi'][k] = rk_bar[k]/N # 1x1

            if Lw > 0:
                x = np.concatenate(
                        (t,
                        np.reshape(muw[:,:,k], (Lw, N), order = 'F'))) # LxN
                Skx = np.zeros((L,L)) # LxL
                Skx[Lt:L,Lt:L] = Sw[:,:,k]
            else:
                x = t # LtxN
                Skx = np.zeros(Lt) #LtxLt

            # Compute optimal transformation vector Ak
            if self.verbose > 1:
                print('  --- A')

            # Compute weighted means of y and x
            if allnans(cstr['b']):
                yk_bar = np.sum(y * rk, axis=1)/rk_bar[k] # Dx1
                if L > 0:
                    xk_bar = np.sum(x * rk, axis=1)/rk_bar[k] # Lx1
                else:
                    xk_bar = np.zeros(0)
            else:
                yk_bar = cstr['b'][:,k]
                xk_bar = np.zeros((L,1))
                theta['b'][:,k] = cstr['b'][:,k]

            # Compute weighted, mean centered y and x
            weights = np.sqrt(rk)        # 1xN
            y_stark = y - yk_bar[:,None] # DxN
            y_stark = weights[None,:] * y_stark # DxN

            if L > 0:
                x_stark = x - xk_bar[:,None] # LxN
                x_stark = weights * x_stark  # LxN
            else:
                x_stark = np.zeros(0)

            # Robustly compute optimal transformation matrix Ak
            eps = 1e-8
            xx_stark = np.matmul(x_stark, x_stark.T)
            yx_stark = np.matmul(y_stark, x_stark.T)

            if not np.allclose(Skx, 0):
                Skx_xx = Skx + xx_stark
                if N >= L \
                    and np.linalg.det(Skx_xx) > eps:
                    theta['A'][:,:,k] = np.matmul(yx_stark, np.linalg.inv(Skx_xx)) # DxL
                else:
                    theta['A'][:,:,k] = np.matmul( \
                            yx_stark, np.linalg.pinv(Skx_xx)) #DxL

            elif not np.allclose(x_stark, 0):
                if N >= L \
                    and np.linalg.det(xx_stark) > eps:
                    theta['A'][:,:,k] =  np.matmul(yx_stark, np.linalg.inv(xx_stark)) # DxL
                elif N < L and np.linalg.det(x_stark.T*x_stark) > eps:
                    theta['A'][:,:,k] = y_stark \
                                    / np.matmul( np.matmul(x_stark.T, x_stark), x_stark.T) # DxL
                else:
                    if self.verbose > 1: print('  --- p')
                    theta['A'][:,:,k] = np.matmul(y_stark, np.linalg.pinv(x_stark))  # DxL

            else:
                # Correspond to null variance in cluster k or L=0:
                if self.verbose and L > 0:
                    print('  --- null variance')
                theta['A'][:,:,k] = np.zeros((D,L))  # DxL

            if self.verbose > 1:
                print('  --- b')
            # Intermediate variable wk=y-Ak*x
            if L > 0:
                Ak = np.reshape(theta['A'][:,:,k], (D, L), order = 'F')
                wk = y - np.matmul(Ak,x) # DxN
            else:
                wk = y

            # Compute optimal transformation vector bk
            if allnans(cstr['b']):
                theta['b'][:,k] = np.sum( rk * wk, axis = 1) / rk_bar[k] # Dx1

            # Compute optimal covariance matrix Sigmak
            if self.verbose > 1:
                print('  --- Sigma')
            if Lw > 0 and list(Sw):
                Awk = np.reshape(theta['A'][:,Lt:L,k], (D,Lw), order = 'F')
                Swk = np.reshape(Sw[:,:,k], (Lw,Lw), order = 'F')
                ASAwk = np.matmul(np.matmul(Awk,Swk), Awk.T)
            else:
                ASAwk = 0

            diffSigma = np.sqrt(rk) * (wk - theta['b'][:,k,None]) #DxN

            if cstr_type_sigma == 'full':
                theta['Sigma'][:,:,k] = np.matmul(diffSigma, diffSigma.T) / rk_bar[k] # DxD
                theta['Sigma'][:,:,k] = theta['Sigma'][:,:,k] + ASAwk
            elif cstr_type_sigma == 'fixed':
                theta['Sigma'] = cstr['Sigma']
            elif cstr_type_sigma == 'diagonal':
                sigma2 = np.sum(diffSigma**2, axis = 1) / rk_bar[k] #Dx1
                theta['Sigma'][:,:,k] = np.diag(sigma2) # DxD
                theta['Sigma'][:,:,k] = theta['Sigma'][:,:,k] + np.diag(np.diag(ASAwk))
            elif cstr_type_sigma == 'isotropic':
                sigma2 = np.sum(diffSigma**2, axis = 1) / rk_bar[k] #Dx1
                theta['Sigma'][:,:,k] = np.dot(np.eye(D),np.mean(sigma2)) # DxD
                if isinstance(ASAwk,int) and ASAwk == 0:
                    theta['Sigma'][:,:,k] = theta['Sigma'][:,:,k]
                else:
                    theta['Sigma'][:,:,k] = theta['Sigma'][:,:,k]  \
                                    + (np.trace(ASAwk)/D) * np.eye(D)

            # Avoid numerical problems on covariances:
            if self.verbose > 2:
                print('  --- n')
            if not np.isfinite(theta['Gamma'][0:Lt,0:Lt,k].sum()):
                theta['Gamma'][0:Lt,0:Lt,k] = 0

            theta['Gamma'][0:Lt,0:Lt,k] = theta['Gamma'][0:Lt,0:Lt,k]+eps*np.eye(Lt)

            if not np.isfinite(theta['Sigma'][:,:,k].sum()):
                theta['Sigma'][:,:,k] = 0

            theta['Sigma'][:,:,k] = theta['Sigma'][:,:,k] + eps*np.eye(D)
        #  for k

        if self.verbose > 1:
            print('end')

        # Equality constraint on Sigma
        if isinstance(cstr['Sigma'], str) and '*' in cstr['Sigma']:
            theta['Sigma'] = theta['Sigma'] * np.reshape(rk_bar, (1,1,K), order = 'F')
            theta['Sigma'] = (np.ones(theta['Sigma'].shape) * np.sum(theta['Sigma'], axis = 2)[:,:,None]) / N


        # Equal volume constraint on Gamma
        if isinstance(cstr['Gammat'], str) and cstr['Gammat'] == 'v':
            detG = np.zeros((1,K))
            for k in range(K):
                detG[k] = np.det(
                            np.reshape(theta['Gamma'][0:Lt,0:Lt,k],
                                (Lt,Lt),
                                order = 'F')) # 1x1

            theta['Gamma'][0:Lt,0:Lt,:] = theta['Gamma'][0:Lt,0:Lt,:] \
                                    / np.reshape(detG, (1,1,K), order = 'F')
            theta['Gamma'][0:Lt,0:Lt,:] = np.sum(detG**(1./Lt) * theta['pi']) \
                                          * theta['Gamma'][0:Lt,0:Lt,:]

        ### Equality constraint on Gammat
        if isinstance(cstr['Gammat'],str) \
                and not cstr['Gammat']    \
                and cstr['Gammat'][len(cstr['Gammat'])] == '*':
            theta['Gamma'][0:Lt,0:Lt,:] \
                = np.matmul(theta['Gamma'][0:Lt,0:Lt,:],
                            np.reshape(rk_bar,[1,1,K]))
            theta['Gamma'][0:Lt,0:Lt,:] \
                = np.matmul(np.ones((Lt,Lt,K)),
                            sum(theta['Gamma'][0:Lt,0:Lt,:],2))/N


        ### Constraint of pi
        if not isinstance(cstr['pi'], str) or allnans(cstr['pi']):
            if not allnans(cstr['pi']):
                theta['pi'] = cstr['pi']
        else:
            if cstr['pi'] == '*':
                theta['pi'] = np.ones((1,K))/K
            else:
                raise ValueError('  ERROR: invalid constraint on pi.')
        return theta

    def _expectation_z(self, t, y, theta):
        if self.verbose: print('Expectation Z step')

        logr = nans((self.N, self.K))

        bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])
        for k in bar(range(self.K)):

            muyk = theta['b'][:,k] # Dx1
            covyk = np.reshape(theta['Sigma'][:,:,k], (self.D, self.D), order = 'F') #DxD
            if self.Lt > 0:
                Atk = np.reshape(theta['A'][:,0:self.Lt,k], (self.D, self.Lt), order = 'F') #DxLt
                muyk = muyk[:,None] + np.matmul(Atk, t) #DxN
            if self.Lw > 0:
                Awk = np.reshape(theta['A'][:,self.Lt:self.L,k], (self.D, self.Lw), order = 'F') #DxLw
                Gammawk = np.reshape(theta['Gamma'][self.Lt:self.L, self.Lt:self.L,k], (self.Lw, self.Lw), order = 'F') #LwxLw
                cwk = theta['c'][self.Lt:self.L,k] # Lwx1
                covyk = covyk + np.matmul(np.matmul(Awk, Gammawk), Awk.T) # DxD
                muyk = muyk + np.matmul(Awk, cwk)[:,None] #DxN
            logr[:,k] = np.log(theta['pi'][k]) + (log_gauss_pdf(y, muyk, covyk)) #Nx1
            if self.Lt > 0:
                logr[:,k] += log_gauss_pdf(t, theta['c'][0:self.Lt,k], theta['Gamma'][0:self.Lt, 0:self.Lt,k])

        logr_den = log_sum_exp(logr,axis=1)
        log_like = logr_den.sum()
        r = np.exp(logr - logr_den[:,None])

        # remove empty clusters
        empty_cluster_indeces = list(range(self.K))
        indeces_to_remove = []
        for k in range(self.K):
            if r[:,k].sum() == 0 or not np.isfinite(r[:,k].sum()):
                indeces_to_remove.append(k)
                if self.verbose:
                    print('!!! class %d has been removed'%(k))
        empty_cluster_indeces = [k for k in empty_cluster_indeces if not k in indeces_to_remove]
        if not empty_cluster_indeces:
            if self.verbose: print('*** re-initialization')
            r = emgm(t, y, self.K, 2, self.verbose)
            empty_cluster_indeces = list(range(r.shape[1]))
        else:
            r = r[:,empty_cluster_indeces]

        return r, log_like, empty_cluster_indeces

    def _remove_empty_clusters(self, theta, cstr, empty_cluster_indeces):
        '''
        Remove empty components
        '''

        if len(empty_cluster_indeces) != theta['c'].shape[1]:
            if not isinstance(cstr['ct'], str) and not allnans(cstr['ct']):
                cstr['ct'] = cstr['ct'][:,empty_cluster_indeces]
            if not isinstance(cstr['cw'], str) and not allnans(cstr['cw']):
                cstr['cw'] = cstr['cw'][:,empty_cluster_indeces]
            if not isinstance(cstr['Gammat'], str) and not allnans(cstr['Gammat']):
                cstr['Gammat'] = cstr['Gammat'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['Gammaw'], str) and not allnans(cstr['Gammaw']):
                cstr['Gammaw'] = cstr['Gammaw'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['pi'], str) and not allnans(cstr['pi']):
                cstr['pi'] = cstr['pi'][empty_cluster_indeces]
            if not isinstance(cstr['A'], str) and not allnans(cstr['A']):
                cstr['A'] = cstr['A'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['b'], str) and not allnans(cstr['b']):
                cstr['b'] = cstr['b'][:,empty_cluster_indeces]
            if not isinstance(cstr['Sigma'], str) and not allnans(cstr['Sigma']):
                cstr['Sigma'] = cstr['Sigma'][:,:,empty_cluster_indeces]

            theta['c'] = theta['c'][:,empty_cluster_indeces]
            theta['Gamma'] = theta['Gamma'][:,:,empty_cluster_indeces]
            theta['pi'] = theta['pi'][empty_cluster_indeces]
            theta['A'] = theta['A'][:,:,empty_cluster_indeces]
            theta['b'] = theta['b'][:,empty_cluster_indeces]
            theta['Sigma'] = theta['Sigma'][:,:,empty_cluster_indeces]

        if self.K != len(empty_cluster_indeces):
            print('Running with %d components'%self.K)
            self.K = len(empty_cluster_indeces)
        return theta, cstr

    def _expectation_w(self, t, y, theta):
        if self.Lw == 0:
            muw = np.zeros(0)
            Sw = np.zeros(0)
            return muw, Sw

        if self.verbose: print('Expectation W step')

        muw = np.zeros((self.Lw, self.N, self.K))
        Sw = np.zeros((self.Lw, self.Lw, self.K))

        bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])

        for k in bar(range(self.K)):
            if self.verbose > 1:
                print('  - k = %d'%(k))

            # DEFINITION
            Atk = np.reshape(theta['A'][:,0:self.Lt,k], (self.D, self.Lt), order = 'F') # DxLt
            Awk = np.reshape(theta['A'][:,self.Lt:self.L,k], (self.D, self.Lw), order = 'F') # DxLw
            bk = np.reshape(theta['b'][:,k], (self.D,1), order = 'F') # Dx1
            Sigmak = np.reshape(theta['Sigma'][:,:,k], (self.D, self.D), order = 'F') # DxD
            Gammawk = np.reshape(theta['Gamma'][self.Lt:self.L, self.Lt:self.L, k], (self.Lw, self.Lw), order = 'F') # LwxLw
            cwk = theta['c'][self.Lt:self.L, k] # Lwx1

            invGammawk = np.linalg.inv(Gammawk)
            invSigmak = np.linalg.inv(Sigmak)
            invSwk = invGammawk + np.matmul(np.matmul(Awk.T, invSigmak), Awk) # LwxLw

            if not allnans(t):
                Atkt = np.dot(Atk, t) # DxLt
            else:
                Atkt = 0

            Sw[:,:,k] = np.linalg.inv(invSwk)
            muw[:,:,k] = np.dot(
                            np.linalg.inv(np.dot(invSwk, Gammawk)),
                            np.dot(np.dot(np.dot(Gammawk,Awk.T),invSigmak),
                            y - Atkt - bk) + cwk
                        )
        return muw, Sw

# END GLLiM CLASS

################################################################################
######################### U T I L S ############################################
################################################################################

def check_input(X,Y):
    return X.shape[1]==Y.shape[1]

def log_gauss_pdf(X, mu, Sigma):
    '''
    Compute the logarithm of the normal distribution applied in X given the
    mean mu and covariance Sigma.
    '''
    (D, N) = X.shape
    if len(mu.shape) < len(X.shape):
        # Restore misisng dimension
        mu1 = np.zeros((mu.shape[0],1))
        mu1[:,0] = mu
        mu = mu1
    X = X - mu # DxN
    try:
        U = np.linalg.cholesky(Sigma) # DxD
    except:
        print('!!! Sigmayk matrix is not positive definited')
        Y = -np.inf*np.ones((1,N)) # 1xN
        return Y

    Q = np.dot(np.linalg.inv(U), X) # DxN
    q = np.einsum('dn,dn->n',Q, Q) # 1xN quadratic term (M distance)
    c = D * np.log(2*np.pi) + 2*np.sum(np.log(np.diag(U))) # 1x1 normalization constant
    Y = -(c + q)/2 # 1xN
    return Y

def log_sum_exp(x, axis=0):
    '''
    Compute log(sum(exp(x), axis)) while avoiding numerical underflow
    Adapted from Micheal Caen (sth4nth@gmail.com)
    '''
    # substract the largest in each column
    y = np.max(x, axis=axis, keepdims=True)
    x = x - y
    s = y[:,0] + np.log(np.sum(np.exp(x), axis=axis))
    i = ~np.isfinite(y)
    if len(y.shape) > 1 and y.shape[-1]==1:
        y = y[:,0]
    if i.any():
        s[i] = y[i]
    return s

def eval_cstr_type(cov_cstr):
    if (not isinstance(cov_cstr, str) and allnans(cov_cstr)) or \
        (isinstance(cov_cstr,str) and (cov_cstr=='*')):
        return 'full'
    if not isinstance(cov_cstr, str):
        return 'fixed'
    if cov_cstr[0] == 'd':
        return 'diagonal'
    if cov_cstr[0] == 'i':
        return 'isotropic'
    if 'v' in cov_cstr:
        return 'equal_det_for_all_k'
    else:
        raise ValueError('  ERROR: invalid constraint')
    return

def allnans(a):
    return np.isnan(a).all()

def nans(shape):
    a = np.empty(shape)
    a.fill(np.NaN)
    return a

################################################################################
#########################    A P I   ###########################################
################################################################################

def gllim_forward_densities(x, theta, y_samples=[], verbose=1):

    ''' Forward Conditional Density from Gllim Parameters

    Calculate Gaussian mixture parameters of the forward
    conditional density p(y|x;theta) in space Y using a single observation x
    and gllim parameters theta Evaluate density at points y_samples.

    Parameters
    ----------
    x (Lx1)                Input observations to map
    theta  (struct)        Gllim model parameters
    - theta['c'] (LxK)        Gaussian means of X's prior
    - theta['Gamma'] (LxLxK)  Gaussian covariances of X's prior
    - theta['pi'] (1xK)       Gaussian weights of X's prior
    - theta['A'] (DxLxK)      Affine transformation matrices
    - theta['b'] (DxK)        Affine transformation vectors
    - theta['Sigma'] (DxDxK)  Error covariances
    y_samples (DxN)        Points where to evaluate p(y|x;theta) (def [])
    verb {0,1,2}           Verbosity (def 1)

    Returns
    -------
    - y_dens (1xN)          Density p(y|x;theta) evaluated at y_samples
    - psi (struct)          Gaussian mixture parameters of p(y|x;theta)
      - psi.S (DxDxK)       Gaussian covariance matrices
      - psi.mu (DxK)        Gaussian means
      - psi.alpha (1xK)     Gaussian weights
    '''

    L = x.shape[0]
    D,K = theta['b'].shape

    if verbose:
        print('Compute FORWARD conditional density parameters')

    # Parameters to estimate:
    mu = np.empty((D,K)) # conditional means
    log_alpha = np.zeros((1,K)) # conditional log-weights, log(p(Z=k|x;theta))
    p_y_given_xktheta = np.zeros(x, K) # Probability p(y|x,Z=k;theta)

    # Estimation
    for k in range(K):
        if verbose > 1:
            print('k = %d'%(k))

        if verbose > 1: print('    Ak, bk')
        Ak = np.reshape(theta['A'][:,:,k], (D, L), order = 'F') # DxL
        bk = np.reshape(theta['b'][:,:,k], (D, 1), order = 'F') # Dx1

        if verbose > 1: print('    mu')
        mu[:,k] = np.matmul(Ak, x) + bk # Dx1

        if verbose > 1: print('    log_alpha')
        log_alpha[k] = np.log(theta['pi'][k]) + log_gauss_pdf(x, theta['pi'][:,k], theta['Gamma'][:,:,k])

        if not y_samples.size == 0:
            if verbose > 1:
                print('    p(Y=y| X=x, Z=%d; theta)'%(k))
            p_y_given_xktheta = log_alpha[k] + log_gauss_pdf(y_samples, mu[:,k], theta['Sigma'][:,:,k]) # Nx1

        if verbose > 1:
            print('\n')

    # Normalization
    log_alpha -= log_sum_exp(log_alpha)
    alpha = np.exp(log_alpha)

    if not y_samples.size == 0:
        y_density = np.sum(p_y_given_xktheta * alpha)
    else:
        y_density = []

    psi = {}
    psi['S'] = theta['Sigma']
    psi['mu'] = mu
    psi['alpha'] = alpha
    return y_density, psi

def gllim_inverse_densities(y, theta, chi=[], x_samples=[], verbose=1):

    '''
    Inverse Conditional Density from Gllim Parameters

    Calculate Gaussian mixture parameters of the inverse
    conditional density p(x|y;theta) in space Y using gllim parameters
    theta, where y may be a set of T observations with missing components
    specified by chi. Evaluate the density at points x_samples.

    Input
    - y (DxT)                Input observations to map
    - theta  (dict)        Gllim model parameters
        - theta['c'] (LxK)        Gaussian means of X's prior
        - theta['Gamma'] (LxLxK)  Gaussian covariances of X's prior
        - theta['pi'] (1xK)       Gaussian weights of X's prior
        - theta['A'] (DxLxK)      Affine transformation matrices
        - theta['b'] (DxK)        Affine transformation vectors
        - theta['Sigma'] (DxDxK)  Error covariances
    - chi (DxT)              Indicate non-missing observations (def ones(y))
    - x_samples (LxN)        Points where to evaluate p(x|y;theta) (def [])
    - verb {0,1,2}           Verbosity (def 1)
    Output
    - x_dens (1xN)           Density p(x|y;theta) evaluated at x_samples
    - psi (dict)           Gaussian mixture parameters of p(x|y;theta)
        - psi.S (LxLxK)        Gaussian covariance matrices
        - psi.mu (LxK)         Gaussian means
        - psi.alpha (1xK)      Gaussian weights
    '''

    # Dimensions
    (D,T) = y.shape
    (L,K) = theta['c'].shape

    if verbose:
        print('Compute INVERSE conditional density parameters')

    # Pre-computation
    if not chi:
        chi = np.ones(y.shape)
    log_2piL = L*np.log(2*np.pi)
    sqrtchibar = np.sqrt(np.sum(chi, 1)) # Dx1

    # Parameters to estimate
    mu = np.empty((L,K)) # conditional means
    S = np.empty((L,L,K)) # Conditional covariance matrices
    log_alpha = np.zeros((1,K)) # conditional log-weights, log(p(Z=k|y;theta))
    p_x_given_yktheta = np.zeros(x_samples.shape[1], K) # Probability p(x|y,Z=k;theta)

    # Estimation
    for k in range(K):
        if verbose > 1:
            print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk, ck, Gammak')
        Ak = np.reshape(theta['A'][:,:,k], (D, L), order = 'F') # DxL
        bk = np.reshape(theta['b'][:,:,k], (D, 1), order = 'F') # Dx1
        Sigmak = np.reshape(theta['Sigma'][:,:,k], (D, D), order = 'F') # DxD
        invSigmak = np.linalg.inv(Sigmak)
        ck = np.reshape(theta['c'][:,k], (L, 1), order = 'F') # Lx1
        Gammak = np.reshape(theta['Gamma'][:,:,k], (L, L), order = 'F') # LxL
        invGammak = np.linalg.inv(Gammak)
        weighted_Ak = Ak * sqrtchibar

        if verbose > 1: print('    - invSk')
        invSk = np.linalg.inv(Gammak) + np.matmul( np.matmul(weighted_Ak.T, invSigmak), weighted_Ak)
        S[:,:,k] = np.linalg.inv(invSk)
        Sk = np.reshape(S[:,:,k], (L,L), order = 'F')

        if verbose > 1: print('    - muk')
        tmp_diff = chi * (y - bk) # DxT
        mu[:,k] = np.matmul(Sk,
                    (np.matmul(np.matmul(Ak.T, invSigmak), np.sum(diff, axis=1)) \
                    + np.matmul(invGammak, ck)))
        muk = np.reshape(mu[:,k], (L,1), order = 'F') # Lx1

        if verbose > 1: print('    - log(alphak)')
        log_alpha[k] = np.log(np.linalg.det(invSk)) + np.log(np.linalg.det(Gammak)) \
                    + np.matmul(np.matmul(ck.T, invGammak), ck) \
                    - np.matmul(np.matmul(muk.T, invSk), muk)
        for t in range(T):
            if verbose > 1: print('    - - t = %d'%(t))
            nonzero_indeces = [chi[:,t] == 1] # indices of the D' non-missing data in y_t
            dchi = y[nonzero_indeces, t] - bk[nonzero_indeces] # D'x1
            iSchi = invSigmak[nonzero_indeces, nonzero_indeces] # D'xD'
            log_alpha[k] = log_alpha[k]+np.matmul(np.matmul(dchi.T, iSchi), dchi.T)
        # end for t

        if not x_samples:
            if verbose > 1: print('    - p(X = x | Y = y, Z = %d; theta)\n'%(k))
            diff = x_samples - muk
            p_x_given_yktheta[:,k] = (np.log(np.linalg.det(invSk))\
                                  - log_2piL \
                                  - (np.matmul(diff.T, invSk).T * diff).sum() \
                                  )/2
    # end for k

    # Normalization
    log_alpha -= log_sum_exp(log_alpha)
    alpha = np.exp(log_alpha)

    if not x_samples.size == 0:
        x_density = np.sum(p_x_given_yktheta * alpha)
    else:
        x_density = []

    psi = {}
    psi['S'] = theta['S']
    psi['mu'] = mu
    psi['alpha'] = alpha
    return x_density, psi

def gllim_inverse_map(y, theta, verbose=1):

    ''' Inverse Mapping from Gllim Parameters
    Author: Antoine Deleforge (July 2012) - antoine.deleforge@inria.fr %

    Map N observations y using the inverse conditional
    expectation E[x|y;theta] of the gllim model with parameters theta

    Parameters
    ----------
    - y (DxN)                Input observations to map
    - theta  (struct)        Gllim model parameters
      - theta['c'] (LxK)        Gaussian means of X's prior
      - theta['Gamma'] (LxLxK)  Gaussian covariances of X's prior
      - theta['pi'] (1xK)       Gaussian weights of X's prior
      - theta['A'] (DxLxK)      Affine transformation matrices
      - theta['b'] (DxK)        Affine transformation vectors
      - theta['Sigma'] (DxDxK)  Error covariances
    - verb {0,1,2}           Verbosity (def 1)

    Returns
    -------
    x_exp (LxN)            Posterior mean estimates E[xn|yn;theta]
    alpha (NxK)            Weights of the posterior GMMs
    '''

    (D,N) = y.shape
    (L,K) = theta['c'].shape

    # parameters to estimate:
    proj = np.empty((L, N, K)) # K projection to X space
    log_alpha = np.zeros((N,K)) # conditional log-weights, log(p(Z-k|y,theta))

    # estimation:
    for k in range(K):
        if verbose > 1: print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk, Sigmak, ck, Gammak ')
        Ak = np.reshape(theta['A'][:,:,k], (D,L), order = 'F') # DxL
        bk = np.reshape(theta['b'][:,k], (D,1), order = 'F') # Dx1
        Sigmak = np.reshape(theta['Sigma'][:,:,k], (D,D), order = 'F') # DxD
        invSigmak = np.linalg.inv(Sigmak)
        ck = np.reshape(theta['c'][:,k], (L,1), order = 'F') # Lx1
        Gammak = np.reshape(theta['Gamma'][:,:,k], (L,L), order = 'F') # LxL
        invGammak = np.linalg.inv(Gammak)

        if verbose > 1: print('    - ck*')
        cks = np.matmul(Ak,ck) + bk

        if verbose > 1: print('    - Gammak*')
        Gammaks = Sigmak + np.matmul(np.matmul(Ak, Gammak), Ak.T)

        if verbose > 1: print('    - invSigmak*')
        Sigmaks = np.linalg.inv(
                invGammak + np.matmul(np.matmul(Ak.T, invSigmak), Ak))
        invSigmaks = np.linalg.inv(Sigmaks)

        if verbose > 1: print('    - Ak*')
        Aks = np.matmul(np.matmul(Sigmaks, Ak.T), invSigmak)

        if verbose > 1: print('    - bk*')
        bks = np.matmul(Sigmaks,
                (np.matmul(invGammak, ck) \
                 - np.matmul(np.matmul(Ak.T, invSigmak), bk)))

        if verbose > 1: print('    - projections')
        proj[:,:,k] = np.matmul(Aks, y) + bks   # LxN

        if verbose > 1: print('    - log(alpha)')
        log_alpha[:,k] = np.log(theta['pi'][k]) + log_gauss_pdf(y, cks, Gammaks) # Nx1S
    # Normalization
    log_alpha = log_alpha - log_sum_exp(log_alpha,1)[:,None] # NxK

    # Results
    alpha = np.exp(log_alpha) # NxK
    x_exp = np.reshape(np.sum(
        np.reshape(alpha, (1,N,K), order='F') * proj, axis = 2),
        (L,N), order='F') # LxN

    posterior_means = x_exp
    posterior_GMMs_weight = alpha

    return posterior_means, posterior_GMMs_weight

def gllim_forward_map(x, theta, verbose=1):
    '''
    Forward Mapping from Gllim Parameters
    Author: Antoine Deleforge (July 2012) - antoine.deleforge@inria.fr %

    Map N observations x using the forward conditional
    expectation E[y|x;theta] of the gllim model with parameters theta

    Input:
    ------
    - x (LxN)                Input observations to map
    - theta  (struct)        Gllim model parameters
        - theta['c'] (LxK)        Gaussian means of X's prior
        - theta['Gamma'] (LxLxK)  Gaussian covariances of X's prior
        - theta['pi'] (1xK)       Gaussian weights of X's prior
        - theta['A'] (DxLxK)      Affine transformation matrices
        - theta['b'] (DxK)        Affine transformation vectors
        - theta['Sigma'] (DxDxK)  Error covariances
    - verb {0,1,2}           Verbosity (def 1)
    Output:
    -------
    - y_exp (DxN)            Posterior mean estimates E[yn|xn;theta]
    - alpha (NxK)            Soft assigments of points to transformations
    '''

    (L,N) = x.shape
    (D,K) = theta['b'].shape

    if verbose:
        print('Compute K projection to Y space and weights')

    # Parameters to estimate
    proj = np.empty((L,N,K)) # K projection to X space
    log_alpha = np.zeros((N,K)) # Coditional log-weights , log(p(Z=k|y;theta))


    # Estimation
    for k in range(K):
        if verbose > 1: print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk')
        Ak = np.reshape(theta['A'][:,:,k], (D,L), order = 'F') # DxL
        bk = np.reshape(theta['b'][:,k], (D,1), order = 'F') # Dx1

        if verbose > 1: print('    - projections')
        proj[:,:,k] = np.matmul(Ak, x) + bk   # LxN

        if verbose > 1: print('    - log(alpha)\n')
        log_alpha[:,k] = np.log(theta['pi'][k] + log_gauss_pdf(x, theta['xk'][:,k], theta['Gammaks'][:,:,k])) # Nx1

    # Normalization
    log_alpha -= log_sum_exp(log_alpha,2) # NxK

    # Results
    alpha = np.exp(log_alpha)# NxK
    y_exp = np.reshape(np.sum(
        np.reshape(alpha, (1,N,K), order='F') * proj, axis = 2),
        (D,N), order='F') # LxN

    posterior_means = y_exp
    posterior_GMMs_weight = alpha

    return posterior_means, posterior_GMMs_weight

def gllim_train_wrapper(training_latent_var, training_obs_var, n_components,
                  dim_hidden_components=0, n_iter=100, init_parameters={},
                  init_assigment=None, contraint_params={}, verbose=1):

    # RE-NAMING
    t = training_latent_var
    y = training_obs_var
    K = n_components
    Lw = dim_hidden_components
    max_iter = n_iter
    init_theta = init_parameters
    init_r = init_assigment
    cstr = contraint_params

    Lt, N = t.shape
    D, N = y.shape

    # create the model
    model = GLLIM(D, Lt, Lw, K, verbose=1)

    # train it
    estimated_parameters, posterior_prob, costs = model.fit(t, y, max_iter)

    return estimated_parameters, posterior_prob, costs
