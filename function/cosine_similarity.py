import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

a = np.array([1,2,3])
b = np.array([1,3,8])

dot = np.dot(a,b)
norma = np.linalg.norm(a)
normb = np.linalg.norm(b)
cos = dot /(norma * normb)
print("Python Version:")
print("norma {}, normb {}, cosine_similarity {}".format(norma, normb, cos))

print("Sklearn Version")
aa = a.reshape(1,3)
bb = b.reshape(1,3)
cos_lib = cosine_similarity(aa,bb)
print(cos_lib)

print("Pytorch Version")
pa = torch.from_numpy(a).float()
pb = torch.from_numpy(b).float()
pa = F.normalize(pa,dim=0)
pb = F.normalize(pb,dim=0)
pa = pa.view(1,-1)
pb = pb.view(-1,1)
# print(pa.shape)
# print(pb.shape)
cos = torch.mm(pa,pb)
print("pnorma {}, pnormb {}, cosine_similarity {}".format(pa, pa, cos))

pa = torch.from_numpy(a).float()
pb = torch.from_numpy(b).float()
cos = torch.nn.CosineSimilarity(dim=0)
output = cos(pa,pb)
print("pytorch CosineSimilarity: {}".format(output))
