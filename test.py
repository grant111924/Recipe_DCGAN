import numpy as np
layer=10

total =1
for i in range(layer):
    
    if  i%2 == 0 :
        print((" "*(layer- i))+'J'*total)
    else:
        print((" "*(layer- i))+'*'*total)

            
    
    total+=2
    

"""
    ps =False
    while  ps is False:
        array=np.random.randint(total+1, size=2)
        if np.sum(array) == total and  np.min(array)<3:
            ps=True


    print((" "*(layer- i))+('*')*np.max(array)+'J'*np.min(array))
    """