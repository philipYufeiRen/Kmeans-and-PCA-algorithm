
import numpy as np

def kmeans_pp(K, train_X):
    """ This function runs K-means++ algorithm to choose the centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = np.empty(shape=(K, train_X.shape[1]))

    N = train_X.shape[0]
    D = train_X.shape[1]

    centers = np.empty([K, D]) 
    distance = np.empty([N, K])
    center_index_list = np.empty([K])

    random_index = np.random.randint(0,N)
    center_index_list[0] = random_index
    
    probability = np.empty([N])
    
    for i in range(K): #loop thought each center
        index = int(center_index_list[i]) #get the center index
        for a in range(N):
            distance[:, i] = np.linalg.norm(train_X[a,:] - train_X[index,:])
        centers[i,:] = train_X[index,:]
        
        for p in range(N): #calculate weighted probability
            if p != index: 
                probability[p] = distance[p][i]**2/np.sum(distance[:, i]**2)
            else:
                probability[p] = 0
        probability = probability/np.sum(probability)
        if i != K-1: #update center index if it's not the last run
            temp_index = np.random.choice(N, p=probability)
            center_index_list[i+1] = temp_index  
    

    return centers

def random_init(K, train_X):
    """ This function randomly chooses K data points as centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = train_X[np.random.randint(train_X.shape[0], size=K)]
    return centers
