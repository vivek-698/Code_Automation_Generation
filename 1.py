s=[3, 10, 3, 11, 4, 5, 6, 7, 8, 12]
3,10,11,4 
i=0
j=i+1
m=0
while(i<len(s)):
    if s[j-1]<s[j]:
        j+=1
    else:
        m=max(m,j-i+1)
        i+=1
        j=i+1
print(m)


