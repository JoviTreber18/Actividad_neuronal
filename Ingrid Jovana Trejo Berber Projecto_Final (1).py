#!/usr/bin/env python
# coding: utf-8

# # Projecto Final

# ###### Ingrid Jovana Trejo Berber

# In[2]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm


# In[148]:


def plot_raster(raster,fps,markersize=5):
    N,F=raster.shape
    plt.figure(figsize=(12,6))

    ax=plt.axes((0.05,0.35,0.75,0.6)) 
    indices = np.where(raster == 1)
    plt.plot(indices[1], indices[0] + 1,
            marker='|', linestyle='None',
            markersize=markersize, color='black')
    ax.set_xlim(0,F-1)
    ax.set_ylim(1,N)
    plt.xticks([])
    plt.ylabel("Neuron Label")

    ax=plt.axes(((0.05,0.12,0.75,0.2)))
    coactividad = np.sum(raster, axis=0)
    fpm=fps*60
    tiempo = np.arange(0,F)/fpm
    plt.plot(tiempo, coactividad, linewidth=0.5, color='black')
    ax.set_xlim(np.min(tiempo),np.max(tiempo))
    ax.set_ylim(0,np.max(coactividad)+1)
    plt.xlabel("Time (min)")
    plt.ylabel("Coactivity")

    ax=plt.axes((0.85,0.35,0.1,0.6))
    actividad=np.sum(raster,axis=1)
    plt.plot(actividad, np.arange(1, N+1), color='black', linewidth=1)
    ax.set_xlim(0,max(actividad)+1)
    ax.set_ylim(1,N)
    plt.xlabel("# Frames")
    plt.yticks([])


# In[139]:


import numpy as np

# Parámetros
num_neurons = 100
num_frames = 2000
num_stimuli = 10
stimulus_interval = 100
noise_probability = 0.1  # Probabilidad de ruido
spontaneous_activity_level = 0.05  # Nivel de actividad espontánea
response_consistency = 0.9  # Consistencia de la respuesta al estímulo

# Generar matriz de rastreo neuronal sintética con ruido y actividad espontánea
trace_matrix = np.zeros((num_neurons, num_frames), dtype=int)

for i in range(num_stimuli):
    stimulus_frames = np.random.choice(num_frames, size=stimulus_interval, replace=False)
    for frame in stimulus_frames:
        start = i * (num_neurons // num_stimuli)
        end = start + (num_neurons // num_stimuli)
        for neuron in range(start, end):
            if np.random.random() < response_consistency:
                trace_matrix[neuron, frame] = 1
            else:
                trace_matrix[neuron, frame] = 0
    for neuron in range(start, end):
        spontaneous_activity = np.random.random(num_frames) < spontaneous_activity_level
        trace_matrix[neuron] += spontaneous_activity.astype(int)


# In[149]:


plot_raster(trace_matrix,4)


# In[6]:


trace_matrix.shape


# In[7]:


plt.matshow(trace_matrix, cmap="jet")
plt.colorbar()
plt.show()


# In[8]:


g,p=trace_matrix.shape
trace_est=np.zeros((g,p))
for i in range(g):
    desv_est=np.std(trace_matrix[i,:])
    media=np.mean(trace_matrix[i,:])
    trace_est[i,:]=(trace_matrix[i,:]-media)/(desv_est)


# In[9]:


np.sum((trace_est[0,:]-trace_est[0,:].mean())*(trace_est[1,:]-trace_est[1,:].mean()))/(len(trace_est[0,:])-1)


# In[10]:


matriz_covarianza=np.cov(trace_est.T)


# In[11]:


plt.matshow(matriz_covarianza, cmap="jet")
plt.colorbar()
plt.show()


# In[12]:


matriz_covarianza[0,1]


# In[87]:


eigval,eigvec=np.linalg.eig(matriz_covarianza)


# In[88]:


eigval[0:5]


# In[89]:


np.cumsum(eigval[0:5])


# In[90]:


eigvec_seleccionados=eigvec[:,[0,1]]
b_r=trace_est@eigvec_seleccionados
b_r.shape


# In[91]:

plt.plot(b_r[:,0],b_r[:,1],"*")
plt.show()


# In[92]:


# In[93]:


ax=plt.axes(projection="3d")
ax.scatter3D(b_r[:,0],b_r[:,1],b_r[:,2])


# In[151]:


ordenar = [(np.abs(eigval[i]), eigvec[:, i]) for i in range(len(matriz_covarianza))]


# In[152]:


mk = 3
epb = np.array([ordenar[i][1] for i in range(mk)])
proyecta = np.dot(trace_matrix, epb.T)
pro=np.abs(proyecta)
pro


# In[153]:


from sklearn.metrics.pairwise import cosine_similarity
similar=cosine_similarity (pro)
sp.pprint(similar)


# In[154]:


U, Sigma, VT = np.linalg.svd(pro)

# Mostrar las matrices
print("U:\n", U, "\n")
print("Sigma:\n", Sigma, "\n")  # Nótese que Sigma es retornada como un vector 1-D
print("VT:\n", VT, "\n")


# In[155]:


# Convertir Sigma a una matriz diagonal
Sigma_matrix = np.zeros((pro.shape[0], pro.shape[1]))
for i in range(min(pro.shape[0], pro.shape[1])):
    Sigma_matrix[i, i] = Sigma[i]

# Reconstruir la matriz original
A_reconstructed = np.dot(U, np.dot(Sigma_matrix, VT))


# In[156]:


plot_raster(U,4)


# In[157]:


plot_raster(Sigma_matrix, 4)


# In[158]:


plot_raster (VT, 4)


# In[ ]:




