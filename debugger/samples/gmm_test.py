from core_tool import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from numpy.core.numeric import indices
from scipy import linalg, integrate
from scipy.stats import norm 
from sklearn import cluster, preprocessing, mixture
from gmr import GMM, plot_error_ellipses
from gmr.utils import check_random_state

def Help():
  pass

def Run(ct, *args):
  mu_list = [[0.3,0.3], [4, 4]]
  sigma_list = [[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]]
  color_list = ["blue", "red"]
  
  X = []
  C = []
  for i, (mu, sigma, c) in enumerate(zip(mu_list, sigma_list, color_list)):
    x = np.random.multivariate_normal(mu, sigma, 30).tolist()
    if X==[]:
      X = x
    else:
      X = np.concatenate([X, x])
    C += [c]*len(x)

  lowest_bic = np.Infinity
  bic_list = []
  n_components_range = range(1, 7)
  # cv_types = ['spherical', 'tied', 'diag', 'full']
  # cv_types = ['full']
  cv_types = ["diag", "full"]
  for cv_type in cv_types:
    for n_components in n_components_range:
      gmm_sl = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
      gmm_sl.fit(X)
      gmm_sl_covs = None
      if cv_type=="full":
        gmm_sl_covs = np.array([c for c in gmm_sl.covariances_])
      elif cv_type=="diag":
        gmm_sl_covs = np.array([np.diag(c) for c in gmm_sl.covariances_])
      elif cv_type=="tied":
        gmm_sl_covs = np.array([gmm_sl.covariances_]*n_components)
      elif cv_type=="spherical":
        gmm_sl_covs = np.array([c*np.eye(X.shape[1]) for c in gmm_sl.covariances_])
      gmm = GMM(n_components=n_components, priors=gmm_sl.weights_, means=gmm_sl.means_, covariances=gmm_sl_covs)
      # gmm.from_samples(X)
      bic = gmm_sl.bic(X)
      if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_type = cv_type
      bic_list.append(bic)

  if False:
    X_test = np.array([[[0.3,0.3]], [[2,4]]])
    for x_test in X_test:
      prob_density = best_gmm.to_probability_density(x_test)
      response = best_gmm.to_responsibilities(x_test)
      print(prob_density, response)

  if True:
    x1_range = np.linspace(-5,10,100)
    density_x1_outputs = []
    for x1 in x1_range:
      density_x1x2 = lambda x2: best_gmm.to_probability_density(np.array([[x1, x2]]))
      density_x1, _ = integrate.quad(density_x1x2, -np.Infinity, np.Infinity)
      density_x1_outputs.append(density_x1)
    fig = plt.figure()
    plt.plot(x1_range, density_x1_outputs)
    plt.show()

  if False:
    print(best_type)
    indices = [0]
    # X_test = np.array([[0.3],[1.55],[3.7]])
    X_test = np.array([[2.0]])
    fig = plt.figure()
    for x_test in X_test:
      conditioned = best_gmm.condition(indices, x_test)
      means = conditioned.means
      vars = conditioned.covariances
      pis = conditioned.priors
      print(means)
      print(pis)
      x = np.linspace(-5,10,1000)
      p = np.zeros(1000)
      for mean, var, pi in zip(means, vars, pis):
        # print(pi)
        # print(norm.pdf(x, loc=mean, scale=var).reshape(-1,1).shape)
        # print((pi*norm.pdf(x, loc=mean, scale=var).reshape(-1,1)).shape)
        p += pi*norm.pdf(x, loc=mean, scale=var).reshape(-1,)
      plt.plot(x,p)
    plt.show()


  if False:
    bic = np.array(bic_list)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    bars = []

    plt.figure(figsize=(8,4))
    spl = plt.subplot(1, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
      xpos = np.array(n_components_range) + .2 * (i - 2)
      bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                          width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    plt.show()

    
