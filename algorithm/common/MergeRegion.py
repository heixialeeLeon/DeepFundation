import numpy as np

#regions = [(1,3),(2,5),(12,20),(3,6),(7,12)]
regions = [(7,10),(1,6),(9,15),(2,5)]

max = 0
for region in regions:
    if region[1] > max:
        max = region[1]
print(max)

flag = np.zeros((max+1,))
for region in regions:
    if(flag[region[0]] == 0):
        flag[region[0]] =1
    if (flag[region[1]] == 0):
        flag[region[1]] =2
    if(flag[region[0]] == 2):
        flag[region[0]] = 0
    if(flag[region[1]] == 1):
        flag[region[1]] =0
print(flag)

st = list()
merge_region= list()
for i in range(len(flag)):
    if flag[i] == 1:
        st.append(i)
    if flag[i] == 2:
        val = st.pop()
        if len(st) == 0:
            start = val
            end = i
            merge_region.append((start,end))
print(merge_region)
# print(flag.shape)
# print(flag[0])
# print(flag[1])
# print(flag)