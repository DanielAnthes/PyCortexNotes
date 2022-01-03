#%% 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import FastICA
from scipy import signal

#%%

def whiten(x):
    '''
    centers the data around zero and performs eigenvector decomposition
    to linearly transform the data to have zero correlations and unit variance
    '''
    x_center = x - np.repeat(np.mean(x, axis=1)[:,None], x.shape[1], axis=1)
    cov = np.cov(x_center)
    d, E = np.linalg.eig(cov) 
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x_center)))
    return x_whiten

def ica(x, niter, eta=0.01):
    '''
    x       - matrix of data, each column is a datapoint
    niter   - number of optimization steps

    performs independent component analysis by maximizing negentropy.
    Returns the source seperation matrix.
    W @ x gives the estimated source activity
    '''
    dim, ndata = x.shape
    W = np.eye(dim)
       
    for i in tqdm(range(niter)):
        W_min_T = np.linalg.inv(W.T)
        u = W @ x
        ux = np.zeros((dim,dim))
        for k in range(ndata):
            ux += np.tanh(u[:,k])[None,:] @ x[:,k][:, None]

        W += eta * (W_min_T - (2/ndata) * ux)

    return W

#%%

s = np.array([np.sin(np.linspace(0,10,1000)) + 1, signal.sawtooth(np.linspace(0,100,1000)) - 2])
# s = np.random.uniform(size=(2,1000)) 
A = np.array([[1,2],[2,1]])
x = A @ s
x_white = whiten(x)

plt.figure()
plt.subplot(311)
plt.plot(s[0])
plt.plot(s[1])
plt.title("signals")

plt.subplot(312)
plt.plot(x[0])
plt.plot(x[1])
plt.title("observations")

plt.subplot(313)
plt.plot(x_white[0])
plt.plot(x_white[1])
plt.title("data whitening")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(x[0], x[1], marker='.', label='data')
plt.scatter(x_white[0], x_white[1], marker='.', label='whitened data')
plt.legend()
plt.title("effect of data whitening")
plt.show()

# %%

W = ica(x_white, 2000, 0.1)

s_tilde = W @ x_white

plt.figure()
plt.plot(s_tilde[0])
plt.plot(s_tilde[1])
plt.title("source reconstruction")
plt.show()


W_nonwhite = ica(x, 2000, 0.1)

s_tilde_nonwhite = W_nonwhite @ x

plt.figure()
plt.plot(s_tilde_nonwhite[0])
plt.plot(s_tilde_nonwhite[1])
plt.title("source reconstruction")
plt.show()


# %%
icamodel = FastICA(whiten=False)
icamodel.fit(x_white.T)

# %%
s_tilde_scikit = icamodel.transform(x_white.T)
plt.figure()
plt.plot(s_tilde_scikit[:,0])
plt.plot(s_tilde_scikit[:,1])
plt.title("source reconstruction scikit")
plt.show()
