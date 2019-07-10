import numpy as np
import sys

test_array = np.array([1,5,3,6,4,8,9,10])
test_array2 = np.array([3,4,2,5,-1])
test_array3 = np.array([1,3,2,4,6,7,8,4,2,1,4,6,float("inf")])
test_stack = list()
target_array = np.arange(8)

def findMax(input_array, max_array):
    index = 0
    while(index < len(input_array)):
        if len(test_stack)==0 or test_array[index] <= test_array[test_stack[-1]]:
            test_stack.append(index)
            index+=1
        else:
            max_array[test_stack.pop()] = test_array[index]

    while(len(test_stack)!= 0):
        max_array[test_stack.pop()] = -1

def FindSubMaxLength(input_array):
    st = list()
    a = input_array
    ans = 0
    for i in range(len(a)):
       if len(st) == 0 or a[i] >= a[st[-1]]:
           st.append(i)
       else:
           while(len(st)!=0 and a[i]<a[st[-1]]):
               top = st[-1]
               st.pop()
               tmp = (i-top)*a[top]
               if tmp > ans:
                   ans = tmp
           st.append(top)
           a[top]=a[i]
    return ans

def FindSubMaxUpDownLength(input_array):
    st = list()
    a = input_array
    up_flag = True
    up_count = 0
    down_count = 0
    length = 0
    for i in range(len(a)):
        if up_flag:
            if len(st) == 0 or a[i] > a[st[-1]]:
                st.append(i)
            else:
                up_count = len(st)
                st.clear()
                up_flag = False
        if not up_flag:
            if len(st)==0 or a[i] < a[st[-1]]:
                st.append(i)
            else:
                down_count = len(st)
                length = max(length, up_count + down_count)
                up_flag = True
                last_val = st[-1]
                st.clear()
                st.append(last_val)
                st.append(i)
                up_count = 0
                down_count = 0
    return length

if __name__ == "__main__":
    print(test_array2)
    # findMax(test_array, target_array)
    # print(target_array)
    ans = FindSubMaxLength(test_array2)
    print(ans)
    print(test_array3)
    length = FindSubMaxUpDownLength(test_array3)
    print(length)
