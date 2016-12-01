def index_3(a, A):
    _,J,K = A
    i,j,k = a
    return ((i*J + j)*K + k)

def inverse_3(ix, A):
    _,J,K = A
    total = J*K
    i = ix // total
    ix = ix % total
    total = K
    j = ix // total
    k = ix % total
    return (i,j,k)

A,B,C = 3,4,5
key = 0
for a in range(A):
    for b in range(B):
        for c in range(C):
            print (a,b,c), '->', key
            assert inverse_3(key, (A,B,C)) == (a,b,c)
            assert index_3((a,b,c), (A,B,C)) == key            
            key += 1
