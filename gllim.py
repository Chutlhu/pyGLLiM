import numpy as np
import scipy
import progressbar

from sklearn.mixture import GaussianMixture

tmp_mat_dir = '../the_GLLiM_toolbox_matlab/tmp/' # for debugging
def my_print(θ,k=0):
    print('mean Σ' + str(k) + '= ' , np.mean(np.diag(θ['Σ'][:,:,k])))
    print('mean Γ' + str(k) + '= ' , np.mean(np.diag(θ['Γ'][:,:,k])))
    print('mean A' + str(k) + '= ' , np.mean(θ['A'][:,:,k]))
    print('mean c' + str(k) + '= ' , np.mean(θ['c'][:,k]))
    print('mean b' + str(k) + '= ' , np.mean(θ['b'][:,k]))
    print('mean π' + str(k) + '= ' , np.mean(θ['π'][k]))
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

    θ_keys = ['c','Γ','π','A','b','Σ']

    def __init__(self, D, Lt, Lw, K,
                 θ_init={}, cstr_init={}, r_init=[],
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
        self.θ = θ_init # parameters to estimate
        self.cstr = cstr_init # contraints on parameters θ
        self.r = r_init

        self.verbose = verbose

        # DEFINITION AND INITIALIZATION
        if not self.θ:
            if self.verbose:
                print('\nNo initial value for paramters θ Defaults are used')
            self.θ = {
                'c': nans((self.L, self.K)), # Gaussian means of X
                'Γ': nans((self.L, self.L, self.K)), # Gaussian cov. of X
                'π': nans((self.K)), # Gaussian weights of X
                'A': nans((self.D, self.L, self.K)), # Affine transformation matrices
                'b': nans((self.D, self.K)), # Affine transformation vectors
                'Σ': nans((self.D, self.D, self.K)), # Error covariances
            }
        else:
            # add the missing entries
            for key in θ_keys:
                if not key in θ:
                    self.θ[key] = []

        if not self.cstr:
            if self.verbose:
                print('No initial contraints. Defaults are used')
            self.cstr = {
                'ct': nans((self.Lt, self.K)),
                'cw': nans((self.Lw, self.K)),
                'Γt': nans((self.Lt, self.Lt, self.K)),
                'Γw': nans((self.Lw, self.Lw, self.K)),
                'π': nans((self.K)), # Gaussian weights of X
                'A': nans((self.D, self.L, self.K)), # Affine transformation matrices
                'b': nans((self.D, self.K)), # Affine transformation vectors
                'Σ': 'i*', # Error covariances
            }
        else:
            # add the missing entries
            for key in ['ct','cw','Γt','Γw','π','A','b','Σ']:
                if not key in θ:
                    self.θ[key] = np.NaN
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
        r, ec, μw, Sw = self._initialization(t, y, self.θ, cstr, self.r)

        # EM ITERATIONS
        if self.verbose:
            print('\n\n  *** EM iterations ***')

        widget = ['\nRunning: EM iteration ', progressbar.SimpleProgress(),
                    ' (', progressbar.Timer(), ') ',
                    progressbar.Bar(left_justify = False), '\n']
        bar = progressbar.ProgressBar(widgets=widget, max_value=max_iter).start()

        for it in bar(range(max_iter)):

            # MAXIMIZATION STEP
            θ = self._maximization(t, y, r, cstr, μw, Sw)

            # EXPECTATION STEP
            r, log_like[it], ec = self._expectation_z(t,y,θ)
            θ, cstr = self._remove_empty_clusters(θ,cstr,ec)
            μw, Sw = self._expectation_w(t, y, θ)

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

        return θ, r, log_like

    def _initialization(self, t, y, θ, cstr, r):
        if self.verbose: print('\n  *** EM Initialization ***')

        if self.Lw and allnans(cstr['cw']):
            # Default value for cw
            cstr['cw'] = np.zeros((self.Lw,self.K))

        if self.Lw and allnans(cstr['Γw']):
            # Default value for Γw
            cstr['Γw'] = np.zeros((self.Lw,self.Lw,self.K))
            for k in range(self.K):
                cstr['Γw'][:,:,k] = np.eye(self.Lw)

        if not [allnans(θ[key]) for key in self.θ_keys]:
            r, log_like, empty_cluster_indeces = self._expectation_z(t, y, θ)
            θ, cstr = self._remove_empty_clusters(θ, cstr, empty_cluster_indeces)
            μw, Sw = self._expectation_w(t, y, θ)

            return r, empty_cluster_indeces, μw, Sw

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
            μw = np.zeros(0)
            empty_cluster_indeces = []
        else:
            # Start by running an M-step without hidden variables (partial
            # theta), deduce Awk by local weighted PCA on residuals (complete
            # theta), deduce r, muw and Sw from E-steps on complete θ
            # PARTIAL THETA
            θ = self._maximization(t,y,r,cstr)

            # COMPLETE THETA:
            θ['c'] = np.concatenate((θ['c'], cstr['cw'][:,0:self.K])) #LxK


            Γf = np.zeros((self.L,self.L,self.K))
            Γf[0:self.Lt,0:self.Lt,:] = θ['Γ'] #LtxLtxK
            Γf[self.Lt:self.L,self.Lt:self.L,:] = cstr['Γw'][:,:,0:self.K] #LwxLwxK
            θ['Γ'] = Γf

            # Initialize Awk with local weighted PCAs on residuals:
            Aw = np.zeros((self.D,self.Lw,self.K))
            for k in range(self.K):

                rk_bar = np.sum(r[:,k])
                bk = np.reshape(θ['b'][:,k], (self.D,1), order = 'F')
                w = y - bk #DxN
                if self.Lt > 0:
                    Ak = np.reshape(θ['A'][:,0:self.Lt,k], (self.D,self.Lt), order = 'F')
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
                θ['Σ'][:,:,k] = sigma2k * np.eye(self.D)
                Aw[:,:,k] = np.matmul(Vects, np.sqrt(Lambdas - sigma2k*np.eye(self.Lw)))

            θ['A'] = np.concatenate((θ['A'], Aw), axis=1) #DxLxK

            r, log_like, empty_cluster_indeces = self._expectation_z(t,y,θ)
            θ, cstr = self._remove_empty_clusters(θ, cstr, empty_cluster_indeces)
            μw, Sw = self._expectation_w(t,y,θ)

        return r, empty_cluster_indeces, μw, Sw

    def _maximization(self, t, y, r, cstr, μw=np.zeros(0), Sw=np.zeros(0)):
        if self.verbose: print('Maximization step')

        D, N = y.shape
        Lt, N = t.shape
        K = r.shape[1]
        Lw = μw.shape[0]
        L = Lt + Lw

        θ = {
            'c': nans((L, K)),
            'Γ': nans((L, L, K)),
            'π': nans(K),
            'A': nans((D, L, K)),
            'b': nans((D, K)),
            'Σ': nans((D, D, K)),
        }

        cstr_type_gamma = eval_cstr_type(cstr['Γt'])
        cstr_type_sigma = eval_cstr_type(cstr['Σ'])

        if Lw > 0:
            θ['c'][Lt:L] = cstr['cw']
            θ['Γ'][Lt:L, Lt:L, :] = cstr['Γw']

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
                    θ['c'][0:Lt,k] = np.sum(rk * t, axis = 1)/rk_bar[k] # Ltx1
                else:
                    θ['c'][0:Lt,k] = cstr['ct'][:,k]

                if self.verbose > 1:
                    print('  --- Γt')
                # Compute optimal covariance matrix Γt
                diffΓ = np.sqrt(rk) * (t - θ['c'][0:Lt,k,None]) # LtxN

                if cstr_type_gamma == 'full':
                    θ['Γ'][0:Lt,0:Lt,k] = np.matmul(diffΓ,diffΓ.T) / rk_bar[k] # DxD
                if cstr_type_gamma == 'fixed':
                    θ['Γ'][0:Lt,0:Lt,k] = cstr['Γt'][:,:,k]
                if cstr_type_gamma == 'diagonal':
                    gamma2 = np.sum(diffΓ**2, 1)/rk_bar[k] #Ltx1
                    θ['Γ'][0:Lt,0:Lt,k] = diag(gamma2) # LtxLt
                if cstr_type_gamma == 'isotropic':
                    gamma2 = np.sum(diffΓ**2, 1)/rk_bar[k] #Ltx1
                    θ['Γ'][0:Lt,0:Lt,k] = np.mean(gamma2)*eye(Lt) # LtxLt
                if cstr_type_gamma == 'equal_det':
                    θ['Γ'][0:Lt,0:Lt,k] = diffΓ*diffΓ/rk_bar[k] # LtxLt

            # Compute optimal weight πk
            θ['π'][k] = rk_bar[k]/N # 1x1

            if Lw > 0:
                x = np.concatenate(
                        (t,
                        np.reshape(μw[:,:,k], (Lw, N), order = 'F'))) # LxN
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
                θ['b'][:,k] = cstr['b'][:,k]

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
                    θ['A'][:,:,k] = np.matmul(yx_stark, np.linalg.inv(Skx_xx)) # DxL
                else:
                    θ['A'][:,:,k] = np.matmul( \
                            yx_stark, np.linalg.pinv(Skx_xx)) #DxL

            elif not np.allclose(x_stark, 0):
                if N >= L \
                    and np.linalg.det(xx_stark) > eps:
                    θ['A'][:,:,k] =  np.matmul(yx_stark, np.linalg.inv(xx_stark)) # DxL
                elif N < L and np.linalg.det(x_stark.T*x_stark) > eps:
                    θ['A'][:,:,k] = y_stark \
                                    / np.matmul( np.matmul(x_stark.T, x_stark), x_stark.T) # DxL
                else:
                    if self.verbose > 1: print('  --- p')
                    θ['A'][:,:,k] = np.matmul(y_stark, np.linalg.pinv(x_stark))  # DxL

            else:
                # Correspond to null variance in cluster k or L=0:
                if self.verbose and L > 0:
                    print('  --- null variance')
                θ['A'][:,:,k] = np.zeros((D,L))  # DxL

            if self.verbose > 1:
                print('  --- b')
            # Intermediate variable wk=y-Ak*x
            if L > 0:
                Ak = np.reshape(θ['A'][:,:,k], (D, L), order = 'F')
                wk = y - np.matmul(Ak,x) # DxN
            else:
                wk = y

            # Compute optimal transformation vector bk
            if allnans(cstr['b']):
                θ['b'][:,k] = np.sum( rk * wk, axis = 1) / rk_bar[k] # Dx1

            # Compute optimal covariance matrix Σk
            if self.verbose > 1:
                print('  --- Σ')
            if Lw > 0 and list(Sw):
                Awk = np.reshape(θ['A'][:,Lt:L,k], (D,Lw), order = 'F')
                Swk = np.reshape(Sw[:,:,k], (Lw,Lw), order = 'F')
                ASAwk = np.matmul(np.matmul(Awk,Swk), Awk.T)
            else:
                ASAwk = 0

            diffΣ = np.sqrt(rk) * (wk - θ['b'][:,k,None]) #DxN

            if cstr_type_sigma == 'full':
                θ['Σ'][:,:,k] = np.matmul(diffΣ, diffΣ.T) / rk_bar[k] # DxD
                θ['Σ'][:,:,k] = θ['Σ'][:,:,k] + ASAwk
            elif cstr_type_sigma == 'fixed':
                θ['Σ'] = cstr['Σ']
            elif cstr_type_sigma == 'diagonal':
                sigma2 = np.sum(diffΣ**2, axis = 1) / rk_bar[k] #Dx1
                θ['Σ'][:,:,k] = np.diag(sigma2) # DxD
                θ['Σ'][:,:,k] = θ['Σ'][:,:,k] + np.diag(np.diag(ASAwk))
            elif cstr_type_sigma == 'isotropic':
                sigma2 = np.sum(diffΣ**2, axis = 1) / rk_bar[k] #Dx1
                θ['Σ'][:,:,k] = np.dot(np.eye(D),np.mean(sigma2)) # DxD
                if isinstance(ASAwk,int) and ASAwk == 0:
                    θ['Σ'][:,:,k] = θ['Σ'][:,:,k]
                else:
                    θ['Σ'][:,:,k] = θ['Σ'][:,:,k]  \
                                    + (np.trace(ASAwk)/D) * np.eye(D)

            # Avoid numerical problems on covariances:
            if self.verbose > 2:
                print('  --- n')
            if not np.isfinite(θ['Γ'][0:Lt,0:Lt,k].sum()):
                θ['Γ'][0:Lt,0:Lt,k] = 0

            θ['Γ'][0:Lt,0:Lt,k] = θ['Γ'][0:Lt,0:Lt,k]+eps*np.eye(Lt)

            if not np.isfinite(θ['Σ'][:,:,k].sum()):
                θ['Σ'][:,:,k] = 0

            θ['Σ'][:,:,k] = θ['Σ'][:,:,k] + eps*np.eye(D)
        #  for k

        if self.verbose > 1:
            print('end')

        # Equality constraint on Σ
        if isinstance(cstr['Σ'], str) and '*' in cstr['Σ']:
            θ['Σ'] = θ['Σ'] * np.reshape(rk_bar, (1,1,K), order = 'F')
            θ['Σ'] = (np.ones(θ['Σ'].shape) * np.sum(θ['Σ'], axis = 2)[:,:,None]) / N


        # Equal volume constraint on Γ
        if isinstance(cstr['Γt'], str) and cstr['Γt'] == 'v':
            detG = np.zeros((1,K))
            for k in range(K):
                detG[k] = np.det(
                            np.reshape(θ['Γ'][0:Lt,0:Lt,k],
                                (Lt,Lt),
                                order = 'F')) # 1x1

            θ['Γ'][0:Lt,0:Lt,:] = θ['Γ'][0:Lt,0:Lt,:] \
                                    / reshape(detG, (1,1,K), order = 'F')
            θ['Γ'][0:Lt,0:Lt,:] = np.sum(detG**(1./Lt) * θ['π']) \
                                          * θ['Γ'][0:Lt,0:Lt,:]

        ### Equality constraint on Γt
        if isinstance(cstr['Γt'],str) \
                and not cstr['Γt']    \
                and cstr['Γt'][len(cstr['Γt'])] == '*':
            θ['Γ'][0:Lt,0:Lt,:] \
                = np.matmul(θ['Γ'][0:Lt,0:Lt,:],
                            reshape(rk_bar,[1,1,K]))
            θ['Γ'][0:Lt,0:Lt,:] \
                = np.matmul(np.ones((Lt,Lt,K)),
                            sum(θ['Γ'][0:Lt,0:Lt,:],2))/N


        ### Constraint of π
        if not isinstance(cstr['π'], str) or allnans(cstr['π']):
            if not allnans(cstr['π']):
                θ['π'] = cstr['π']
        else:
            if cstr['π'] == '*':
                θ['π'] = np.ones((1,K))/K
            else:
                raise ValueError('  ERROR: invalid constraint on π.')
        return θ

    def _expectation_z(self, t, y, θ):
        if self.verbose: print('Expectation Z step')

        logr = nans((self.N, self.K))

        bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])
        for k in bar(range(self.K)):

            μyk = θ['b'][:,k] # Dx1
            covyk = np.reshape(θ['Σ'][:,:,k], (self.D, self.D), order = 'F') #DxD
            if self.Lt > 0:
                Atk = np.reshape(θ['A'][:,0:self.Lt,k], (self.D, self.Lt), order = 'F') #DxLt
                μyk = μyk[:,None] + np.matmul(Atk, t) #DxN
            if self.Lw > 0:
                Awk = np.reshape(θ['A'][:,self.Lt:self.L,k], (self.D, self.Lw), order = 'F') #DxLw
                Γwk = np.reshape(θ['Γ'][self.Lt:self.L, self.Lt:self.L,k], (self.Lw, self.Lw), order = 'F') #LwxLw
                cwk = θ['c'][self.Lt:self.L,k] # Lwx1
                covyk = covyk + np.matmul(np.matmul(Awk, Γwk), Awk.T) # DxD
                μyk = μyk + np.matmul(Awk, cwk)[:,None] #DxN
            logr[:,k] = np.log(θ['π'][k]) + (log_gauss_pdf(y, μyk, covyk)) #Nx1
            if self.Lt > 0:
                logr[:,k] += log_gauss_pdf(t, θ['c'][0:self.Lt,k], θ['Γ'][0:self.Lt, 0:self.Lt,k])

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
            if verbose: print('*** re-initialization')
            r = emgm(t, y, self.K, 2, self.verbose)
            empty_cluster_indeces = list(range(r.shape[1]))
        else:
            r = r[:,empty_cluster_indeces]

        return r, log_like, empty_cluster_indeces

    def _remove_empty_clusters(self, θ, cstr, empty_cluster_indeces):
        '''
        Remove empty components
        '''

        if len(empty_cluster_indeces) != θ['c'].shape[1]:
            if not isinstance(cstr['ct'], str) and not allnans(cstr['ct']):
                cstr['ct'] = cstr['ct'][:,empty_cluster_indeces]
            if not isinstance(cstr['cw'], str) and not allnans(cstr['cw']):
                cstr['cw'] = cstr['cw'][:,empty_cluster_indeces]
            if not isinstance(cstr['Γt'], str) and not allnans(cstr['Γt']):
                cstr['Γt'] = cstr['Γt'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['Γw'], str) and not allnans(cstr['Γw']):
                cstr['Γw'] = cstr['Γw'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['π'], str) and not allnans(cstr['π']):
                cstr['π'] = cstr['π'][empty_cluster_indeces]
            if not isinstance(cstr['A'], str) and not allnans(cstr['A']):
                cstr['A'] = cstr['A'][:,:,empty_cluster_indeces]
            if not isinstance(cstr['b'], str) and not allnans(cstr['b']):
                cstr['b'] = cstr['b'][:,empty_cluster_indeces]
            if not isinstance(cstr['Σ'], str) and not allnans(cstr['Σ']):
                cstr['Σ'] = cstr['Σ'][:,:,empty_cluster_indeces]

            θ['c'] = θ['c'][:,empty_cluster_indeces]
            θ['Γ'] = θ['Γ'][:,:,empty_cluster_indeces]
            θ['π'] = θ['π'][empty_cluster_indeces]
            θ['A'] = θ['A'][:,:,empty_cluster_indeces]
            θ['b'] = θ['b'][:,empty_cluster_indeces]
            θ['Σ'] = θ['Σ'][:,:,empty_cluster_indeces]

        if self.K != len(empty_cluster_indeces):
            print('Running with %d components'%self.K)
            self.K = len(empty_cluster_indeces)
        return θ, cstr

    def _expectation_w(self, t, y, θ):
        if self.Lw == 0:
            μw = np.zeros(0)
            Sw = np.zeros(0)
            return μw, Sw

        if self.verbose: print('Expectation W step')

        μw = np.zeros((self.Lw, self.N, self.K))
        Sw = np.zeros((self.Lw, self.Lw, self.K))

        bar = progressbar.ProgressBar(widgets=['k = ', progressbar.SimpleProgress(), progressbar.Bar()])

        for k in bar(range(self.K)):
            if self.verbose > 1:
                print('  - k = %d'%(k))

            # DEFINITION
            Atk = np.reshape(θ['A'][:,0:self.Lt,k], (self.D, self.Lt), order = 'F') # DxLt
            Awk = np.reshape(θ['A'][:,self.Lt:self.L,k], (self.D, self.Lw), order = 'F') # DxLw
            bk = np.reshape(θ['b'][:,k], (self.D,1), order = 'F') # Dx1
            Σk = np.reshape(θ['Σ'][:,:,k], (self.D, self.D), order = 'F') # DxD
            Γwk = np.reshape(θ['Γ'][self.Lt:self.L, self.Lt:self.L, k], (self.Lw, self.Lw), order = 'F') # LwxLw
            cwk = θ['c'][self.Lt:self.L, k] # Lwx1

            invΓwk = np.linalg.inv(Γwk)
            invΣk = np.linalg.inv(Σk)
            invSwk = invΓwk + np.matmul(np.matmul(Awk.T, invΣk), Awk) # LwxLw

            if not allnans(t):
                Atkt = np.dot(Atk, t) # DxLt
            else:
                Atkt = 0

            Sw[:,:,k] = np.linalg.inv(invSwk)
            μw[:,:,k] = np.dot(
                            np.linalg.inv(np.dot(invSwk, Γwk)),
                            np.dot(np.dot(np.dot(Γwk,Awk.T),invΣk),
                            y - Atkt - bk) + cwk
                        )
        return μw, Sw

# END GLLiM CLASS

################################################################################
######################### U T I L S ############################################
################################################################################

def check_input(X,Y):
    return X.shape[1]==Y.shape[1]

def log_gauss_pdf(X, μ, Σ):
    '''
    Compute the logarithm of the normal distribution applied in X given the
    mean μ and covariance Σ.
    '''
    (D, N) = X.shape
    if len(μ.shape) < len(X.shape):
        # Restore misisng dimension
        μ1 = np.zeros((μ.shape[0],1))
        μ1[:,0] = μ
        μ = μ1
    X = X - μ # DxN
    try:
        U = np.linalg.cholesky(Σ) # DxD
    except:
        print('!!! Σyk matrix is not positive definited')
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

def gllim_forward_densities(x, θ, y_samples=[], verbose=1):

    ''' Forward Conditional Density from Gllim Parameters

    Calculate Gaussian mixture parameters of the forward
    conditional density p(y|x;θ) in space Y using a single observation x
    and gllim parameters θ Evaluate density at points y_samples.

    Parameters
    ----------
    x (Lx1)                Input observations to map
    θ  (struct)        Gllim model parameters
    - θ['c'] (LxK)        Gaussian means of X's prior
    - θ['Γ'] (LxLxK)  Gaussian covariances of X's prior
    - θ['π'] (1xK)       Gaussian weights of X's prior
    - θ['A'] (DxLxK)      Affine transformation matrices
    - θ['b'] (DxK)        Affine transformation vectors
    - θ['Σ'] (DxDxK)  Error covariances
    y_samples (DxN)        Points where to evaluate p(y|x;θ) (def [])
    verb {0,1,2}           Verbosity (def 1)

    Returns
    -------
    - y_dens (1xN)          Density p(y|x;θ) evaluated at y_samples
    - psi (struct)          Gaussian mixture parameters of p(y|x;θ)
      - psi.S (DxDxK)       Gaussian covariance matrices
      - psi.mu (DxK)        Gaussian means
      - psi.alpha (1xK)     Gaussian weights
    '''

    L = shape(x)[0]
    (D,K) = shape(θ['b'])

    if verbose:
        print('Compute FORWARD conditional density parameters')

    # Parameters to estimate:
    μ = np.empty((D,K)) # conditional means
    log_α = np.zeros((1,K)) # conditional log-weights, log(p(Z=k|x;θ))
    p_y_given_xkθ = np.zeros(x, K) # Probability p(y|x,Z=k;θ)

    # Estimation
    for k in range(K):
        if verbose > 1:
            print('k = %d'%(k))

        if verbose > 1: print('    Ak, bk')
        Ak = np.reshape(θ['A'][:,:,k], (D, L), order = 'F') # DxL
        bk = np.reshape(θ['b'][:,:,k], (D, 1), order = 'F') # Dx1

        if verbose > 1: print('    μ')
        μ[:,k] = np.matmul(Ak, x) + bk # Dx1

        if verbose > 1: print('    log_α')
        log_α[k] = np.log(θ['π'][k]) + log_gauss_pdf(x, θ['π'][:,k], θ['Γ'][:,:,k])

        if not y.size == 0:
            if verbose > 1:
                print('    p(Y=y| X=x, Z=%d; θ)'%(k))
            p_y_given_xkθ = log_α[k] + log_gauss_pdf(y_samples, μ[:,k], θ['Σ'][:,:,k]) # Nx1

        if verbose > 1:
            print('\n')

    # Normalization
    log_α -= log_sum_exp(log_α)
    α = np.exp(log_α)

    if not y_samples.size == 0:
        y_density = np.sum(p_y_given_xkθ * α)
    else:
        y_density = []

    ψ = {}
    ψ['S'] = θ['Σ']
    ψ['μ'] = μ
    ψ['α'] = α
    return y_density, ψ

def gllim_inverse_densities(y, θ, χ=[], x_samples=[], verbose=1):

    '''
    Inverse Conditional Density from Gllim Parameters

    Calculate Gaussian mixture parameters of the inverse
    conditional density p(x|y;θ) in space Y using gllim parameters
    θ, where y may be a set of T observations with missing components
    specified by chi. Evaluate the density at points x_samples.

    Input
    - y (DxT)                Input observations to map
    - θ  (dict)        Gllim model parameters
        - θ['c'] (LxK)        Gaussian means of X's prior
        - θ['Γ'] (LxLxK)  Gaussian covariances of X's prior
        - θ['π'] (1xK)       Gaussian weights of X's prior
        - θ['A'] (DxLxK)      Affine transformation matrices
        - θ['b'] (DxK)        Affine transformation vectors
        - θ['Σ'] (DxDxK)  Error covariances
    - chi (DxT)              Indicate non-missing observations (def ones(y))
    - x_samples (LxN)        Points where to evaluate p(x|y;θ) (def [])
    - verb {0,1,2}           Verbosity (def 1)
    Output
    - x_dens (1xN)           Density p(x|y;θ) evaluated at x_samples
    - psi (dict)           Gaussian mixture parameters of p(x|y;θ)
        - psi.S (LxLxK)        Gaussian covariance matrices
        - psi.mu (LxK)         Gaussian means
        - psi.alpha (1xK)      Gaussian weights
    '''

    # Dimensions
    (D,T) = y.shape
    (L,K) = θ['c'].shape

    if verbose:
        print('Compute INVERSE conditional density parameters')

    # Pre-computation
    if not χ:
        χ = np.ones(y.shape)
    log_2πL = L*np.log(2*np.π)
    sqrtχbar = np.sqrt(np.sum(χ, 1)) # Dx1

    # Parameters to estimate
    μ = np.empty((L,K)) # conditional means
    S = np.empty((L,L,K)) # Conditional covariance matrices
    log_α = np.zeros((1,K)) # conditional log-weights, log(p(Z=k|y;θ))
    p_x_given_ykθ = np.zeros(x, K) # Probability p(x|y,Z=k;θ)

    # Estimation
    for k in range(K):
        if verbose > 1:
            print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk, ck, Γk')
        Ak = np.reshape(θ['A'][:,:,k], (D, L), order = 'F') # DxL
        bk = np.reshape(θ['b'][:,:,k], (D, 1), order = 'F') # Dx1
        Σk = np.reshape(θ['Σ'][:,:,k], (D, D), order = 'F') # DxD
        invΣk = np.linalg.inv(Σk)
        ck = np.reshape(θ['c'][:,k], (L, 1), order = 'F') # Lx1
        Γk = np.reshape(θ['Γ'][:,:,k], (L, L), order = 'F') # LxL
        invΓk = np.linalg.inv(Γk)
        weighted_Ak = Ak * sqrtχbar

        if verbose > 1: print('    - invSk')
        invSk = np.linalg.inv(Γk) + np.matmul( np.matmul(weighted_Ak.T, invΣk), weighted_Ak)
        S[:,:,k] = np.linalg.inv(invSk)
        Sk = np.reshape(S[:,:,k], (L,L), order = 'F')

        if verbose > 1: print('    - μk')
        tmp_diff = χ * (y - bk) # DxT
        μ[:,k] = np.matmul(Sk,
                    (np.matmul(np.matmul(Ak.T, invΣk), np.sum(diff, axis=1)) \
                    + np.matmul(invΓk, ck)))
        μk = np.reshape(μ[:,k], (L,1), order = 'F') # Lx1

        if verbose > 1: print('    - log(αk)')
        log_α[k] = np.log(np.linalg.det(invSk)) + np.log(np.linalg.det(Γk)) \
                    + np.matmul(np.matmul(ck.T, invΓk), ck) \
                    - np.matmul(np.matmul(μk.T, invSk), μk)
        for t in range(T):
            if verbose > 1: print('    - - t = %d'%(t))
            nonzero_indeces = [χ[:,t] == 1] # indices of the D' non-missing data in y_t
            dχ = y[nonzero_indeces, t] - bk[nonzero_indeces] # D'x1
            iSχ = invΣk[nonzero_indeces, nonzero_indeces] # D'xD'
            log_α[k] = log_α[k]+np.matmul(np.matmul(dχ.T, iSχ), dχ.T)
        # end for t

        if not x_samples:
            if verbose > 1: print('    - p(X = x | Y = y, Z = %d; θ)\n'%(k))
            diff = x_samples - μk
            p_x_given_ykθ[:,k] = (np.log(np.linalg.det(invSk))\
                                  - log_2πL \
                                  - (np.matmul(diff.T, invSk).T * diff).sum() \
                                  )/2
    # end for k

    # Normalization
    log_α -= log_sum_exp(log_α)
    α = np.exp(log_α)

    if not x_samples.size == 0:
        x_density = np.sum(p_x_given_ykθ * α)
    else:
        x_density = []

    ψ = {}
    ψ['S'] = θ['S']
    ψ['μ'] = μ
    ψ['α'] = α
    return x_density, ψ

def gllim_inverse_map(y, θ, verbose=1):

    ''' Inverse Mapping from Gllim Parameters
    Author: Antoine Deleforge (July 2012) - antoine.deleforge@inria.fr %

    Map N observations y using the inverse conditional
    expectation E[x|y;θ] of the gllim model with parameters θ

    Parameters
    ----------
    - y (DxN)                Input observations to map
    - θ  (struct)        Gllim model parameters
      - θ['c'] (LxK)        Gaussian means of X's prior
      - θ['Γ'] (LxLxK)  Gaussian covariances of X's prior
      - θ['π'] (1xK)       Gaussian weights of X's prior
      - θ['A'] (DxLxK)      Affine transformation matrices
      - θ['b'] (DxK)        Affine transformation vectors
      - θ['Σ'] (DxDxK)  Error covariances
    - verb {0,1,2}           Verbosity (def 1)

    Returns
    -------
    x_exp (LxN)            Posterior mean estimates E[xn|yn;θ]
    alpha (NxK)            Weights of the posterior GMMs
    '''

    (D,N) = y.shape
    (L,K) = θ['c'].shape

    # parameters to estimate:
    proj = np.empty((L, N, K)) # K projection to X space
    log_α = np.zeros((N,K)) # conditional log-weights, log(p(Z-k|y,θ))

    # estimation:
    for k in range(K):
        if verbose > 1: print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk, Σk, ck, Γk ')
        Ak = np.reshape(θ['A'][:,:,k], (D,L), order = 'F') # DxL
        bk = np.reshape(θ['b'][:,k], (D,1), order = 'F') # Dx1
        Σk = np.reshape(θ['Σ'][:,:,k], (D,D), order = 'F') # DxD
        invΣk = np.linalg.inv(Σk)
        ck = np.reshape(θ['c'][:,k], (L,1), order = 'F') # Lx1
        Γk = np.reshape(θ['Γ'][:,:,k], (L,L), order = 'F') # LxL
        invΓk = np.linalg.inv(Γk)

        if verbose > 1: print('    - ck*')
        cks = np.matmul(Ak,ck) + bk

        if verbose > 1: print('    - Γk*')
        Γks = Σk + np.matmul(np.matmul(Ak, Γk), Ak.T)

        if verbose > 1: print('    - invΣk*')
        Σks = np.linalg.inv(
                invΓk + np.matmul(np.matmul(Ak.T, invΣk), Ak))
        invΣks = np.linalg.inv(Σks)

        if verbose > 1: print('    - Ak*')
        Aks = np.matmul(np.matmul(Σks, Ak.T), invΣk)

        if verbose > 1: print('    - bk*')
        bks = np.matmul(Σks,
                (np.matmul(invΓk, ck) \
                 - np.matmul(np.matmul(Ak.T, invΣk), bk)))

        if verbose > 1: print('    - projections')
        proj[:,:,k] = np.matmul(Aks, y) + bks   # LxN

        if verbose > 1: print('    - log(α)')
        log_α[:,k] = np.log(θ['π'][k]) + log_gauss_pdf(y, cks, Γks) # Nx1S
    # Normalization
    log_α = log_α - log_sum_exp(log_α,1)[:,None] # NxK

    # Results
    α = np.exp(log_α) # NxK
    x_exp = np.reshape(np.sum(
        np.reshape(α, (1,N,K), order='F') * proj, axis = 2),
        (L,N), order='F') # LxN

    posterior_means = x_exp
    posterior_GMMs_weight = α

    return posterior_means, posterior_GMMs_weight

def gllim_forward_map(x, θ, verbose=1):
    '''
    Forward Mapping from Gllim Parameters
    Author: Antoine Deleforge (July 2012) - antoine.deleforge@inria.fr %

    Map N observations x using the forward conditional
    expectation E[y|x;θ] of the gllim model with parameters θ

    Input:
    ------
    - x (LxN)                Input observations to map
    - θ  (struct)        Gllim model parameters
        - θ['c'] (LxK)        Gaussian means of X's prior
        - θ['Γ'] (LxLxK)  Gaussian covariances of X's prior
        - θ['π'] (1xK)       Gaussian weights of X's prior
        - θ['A'] (DxLxK)      Affine transformation matrices
        - θ['b'] (DxK)        Affine transformation vectors
        - θ['Σ'] (DxDxK)  Error covariances
    - verb {0,1,2}           Verbosity (def 1)
    Output:
    -------
    - y_exp (DxN)            Posterior mean estimates E[yn|xn;θ]
    - alpha (NxK)            Soft assigments of points to transformations
    '''

    (L,N) = x.shape
    (D,K) = θ['b'].shape

    if verbose:
        print('Compute K projection to Y space and weights')

    # Parameters to estimate
    proj = np.empty((L,N,K)) # K projection to X space
    log_α = np.zeros((N,K)) # Coditional log-weights , log(p(Z=k|y;θ))


    # Estimation
    for k in range(K):
        if verbose > 1: print('k = %d'%(k))

        if verbose > 1: print('    - Ak, bk')
        Ak = np.reshape(θ['A'][:,:,k], (D,L), order = 'F') # DxL
        bk = np.reshape(θ['b'][:,k], (D,1), order = 'F') # Dx1

        if verbose > 1: print('    - projections')
        proj[:,:,k] = np.matmul(Ak, x) + bk   # LxN

        if verbose > 1: print('    - log(α)\n')
        log_α[:,k] = np.log(θ['π'][k] + log_gauss_pdf(x, θ['xk'][:,k], θ['Γks'][:,:,k])) # Nx1

    # Normalization
    log_α -= log_sum_exp(log_α,2) # NxK

    # Results
    α = np.exp(log_α)# NxK
    y_exp = np.reshape(np.sum(
        np.reshape(α, (1,N,K), order='F') * proj, axis = 2),
        (D,N), order='F') # LxN

    posterior_means = y_exp
    posterior_GMMs_weight = α

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
    init_θ = init_parameters
    init_r = init_assigment
    cstr = contraint_params

    Lt, N = t.shape
    D, N = y.shape

    # create the model
    model = GLLIM(D, Lt, Lw, K, verbose=1)

    # train it
    estimated_parameters, posterior_prob, costs = model.fit(t, y, max_iter)

    return estimated_parameters, posterior_prob, costs
