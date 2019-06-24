def insert_sort(alist):
    unsort_index = 1
    while unsort_index<len(alist):
        hold = alist[unsort_index]
        i = unsort_index -1
        while i >=0 and hold < alist[i]:
            alist[i+1]=alist[i]
            i = i-1
        alist[i+1] =hold
        unsort_index +=1

if __name__ == "__main__":
   alist = [54, 26, 93, 17, 77, 31, 44, 55, 20]
   insert_sort(alist)
   print(alist)