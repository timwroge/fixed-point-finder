arr = [0] *3
sizes = [3, 4, 6] 

def ripple(array):
    arr[0]  += 1
    i = 0
    while i < len(arr):
        if(arr[i]> sizes[i] ):
            arr[i] = 0
            if( i+1< len(arr) ):
                arr[i+1] += 1
        i+=1
    return array


for i in range (10) :
    print(arr)
    ripple(arr)
