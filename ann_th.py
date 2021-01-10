import numpy as np
import torch


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    assumes: [samples x dims]
    
    TODO: Untangle all the transposes...
    
    """    
    if x.ndim == 1:
      e_x = torch.exp(x - torch.max(x))
      return e_x / e_x.sum(axis=0)
    else:
      e_x = torch.exp(x.T - torch.max(x,axis=1)[0])
      return (e_x / e_x.sum(axis=0)).T

def selectAct(action, actSelect, device='cpu'):
  if actSelect == 'hard':
    action = torch.argmax(torch.sum(action,axis=0),axis=0)
  elif actSelect == 'softmax':
    action = softmax(action)
  elif actSelect == 'prob':
    action = weightedRandom(torch.sum(action,axis=0))
  else:
    action = action.flatten()
  action = action.to(device)
  return action

# Not implemented in PyTorch yet
"""
def weightedRandom(weights):
  '''
  Takes an np array (vector) returns random index in proportion to value in each index
  '''
  minVal = np.min(weights)
  weights = weights - minVal # handle negative vals
  cumVal = np.cumsum(weights)
  pick = np.random.uniform(0, cumVal[-1])
  for i in range(len(weights)):
    if cumVal[i] >= pick:
      return i
"""
        
def applyAct(actId, x):
  '''
  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine


  '''
  if actId == 1:   # Linear
    value = x

  if actId == 2:   # Unsigned Step Function
    value = torch.heaviside(x, torch.tensor([0], dtype=torch.float))

  elif actId == 3: # Sin
    value = torch.sin(np.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = torch.exp(-torch.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = torch.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (torch.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = torch.abs(x)   
    
  elif actId == 9: # Relu
    value = torch.relu(x)   

  elif actId == 10: # Cosine
    value = torch.cos(np.pi*x)
    
  else:
    value = x

  return value

def act(weights, aVec, nInput, nOutput, inPattern, device='cpu'):
  '''  -- Activate feed-forward network once --
  If the variable weights is a vector it is turned into a square weight matrix

  This function is vectorized to allow the network to return the result of 
  several inputs at once:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)
  '''
  # Turn weight vector into weight matrix
  if np.ndim(weights) < 2:
      nNodes = int(np.sqrt(np.shape(weights)[0]))
      wMat = torch.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = weights.shape[0]
      wMat = weights

  # Vectorize input
  if np.ndim(inPattern) > 1:
      nSamples = inPattern.shape[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = torch.zeros((nSamples,nNodes)).to(device)
  nodeAct[:,0] = 1 # Bias activation
  nodeAct[:,1:nInput+1] = inPattern

  # Propagate signal through hidden to output nodes
  # For PyTorch; weights are torch tensors
  nodeAct = nodeAct[:,:nInput+1]
  for iNode in range(nInput+1,nNodes):
      rawAct = nodeAct.mv(wMat[:iNode,iNode])
      rawAct = applyAct(aVec[iNode], rawAct)  
      nodeAct = torch.cat((nodeAct, rawAct.view(-1,1)), axis=1)
  return nodeAct[:,-nOutput:]   

# Not implemented in PyTorch yet
"""
def getLayer(wMat):
  '''
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1
  '''
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while (True): # Loop until sorting is stable
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
        break
  return layer-1

def getNodeOrder(nodeG,connG):
  ''' 
  - Topological sort of nodes, returns empty if a cycle is found -
   Builds Connection Matrix from Genome

  - Disabled connections:
   Weights should be set to 0, but still counted as connections
   in the topological sort, as they could become reenabled and
   cause cycles.    

  TODO:
    setdiff1d is slow, as all numbers are positive ints is there a
    better way to do with boolean indexing tricks (ala quickINTersect)?
  '''
  conn = np.copy(connG)
  node = np.copy(nodeG)
  nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
  nOuts = len(node[0,node[1,:] == 2])
  
  # Create connection and initial weight matrices
  conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  src  = conn[1,:].astype(int)
  dest = conn[2,:].astype(int)
  
  lookup = node[0,:].astype(int)
  for i in range(len(lookup)): # Can we vectorize this?
    src[np.where(src==lookup[i])] = i
    dest[np.where(dest==lookup[i])] = i
  
  wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  wMat[src,dest] = conn[3,:]
  connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  connMat[connMat!=0] = 1

  # Topological Sort of Hidden Nodes
  edge_in = np.sum(connMat,axis=0)
  Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
  for i in range(len(connMat)):
    if (len(Q) == 0) or (i >= len(Q)):
      Q = []
      return False, False # Cycle found, can't sort
    edge_out = connMat[Q[i],:]
    edge_in  = edge_in - edge_out # Remove nodes' conns from total
    nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
    Q = np.hstack((Q,nextNodes))

    if sum(edge_in) == 0:
      break
  
  # Add In and outs back and reorder wMat according to sort
  Q += nIns+nOuts
  Q = np.r_[lookup[:nIns], Q, lookup[nIns:nIns+nOuts]]
  wMat = wMat[np.ix_(Q,Q)]
  
  return Q, wMat
"""

def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0] 

  return wVec, aVec, wKey
