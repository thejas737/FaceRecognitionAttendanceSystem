lst1 = [1,2,3,4,5,6,7,8,9,10]
lst2 = [2,5]
lst = []
count = 0
for i in range(1,len(lst1),2):
    if count == 0:
        lst.append(lst1[i]*2)
        count = 1
    else:
        lst.append(lst1[i]*5)
        count = 0
print(lst)