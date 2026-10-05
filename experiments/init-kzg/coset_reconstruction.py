"""Arithmetic prototype only: recover degree-<4N quotient chunks using N-point cosets."""
import random
P,N,K=97,16,4  # This field supports FFTs through 32 points, but not the required 64.
def ev(a,x):
    r=0
    for v in reversed(a):r=(r*x+v)%P
    return r
def solve(a,b):
    m=[row[:]+[v] for row,v in zip(a,b)]
    for j in range(K):
        pivot=next(i for i in range(j,K) if m[i][j])
        m[j],m[pivot]=m[pivot],m[j]
        inv=pow(m[j][j],-1,P);m[j]=[v*inv%P for v in m[j]]
        for i in range(K):
            if i!=j:
                c=m[i][j];m[i]=[(v-c*w)%P for v,w in zip(m[i],m[j])]
    return [row[-1] for row in m]
g=pow(5,(P-1)//N,P)
assert pow(g,N,P)==1 and pow(g,N//2,P)!=1
shifts=[];ts=[]
for s in range(1,P):
    t=pow(s,N,P)
    if t!=1 and t not in ts:shifts.append(s);ts.append(t)
    if len(ts)==K:break
assert len(ts)==K
vandermonde=[[pow(t,j,P) for j in range(K)] for t in ts]
rng=random.Random(0)
for trial in range(100):
    q=[rng.randrange(P) for _ in range(K*N)]
    # C(X)=(X^N-1)Q(X), mimicking the constraint numerator.
    c=[0]*((K+1)*N)
    for i,v in enumerate(q):c[i]=(c[i]-v)%P;c[i+N]=(c[i+N]+v)%P
    residues=[]
    for s,t in zip(shifts,ts):
        y=[ev(c,s*pow(g,k,P)%P)*pow(t-1,-1,P)%P for k in range(N)]
        # Inverse N-point DFT, then undo the shift on coefficient r.
        residues.append([sum(y[k]*pow(g,(-k*r)%N,P) for k in range(N))*pow(N,-1,P)*pow(s,-r,P)%P for r in range(N)])
    recovered=[0]*(K*N)
    for r in range(N):
        for j,v in enumerate(solve(vandermonde,[a[r] for a in residues])):recovered[j*N+r]=v
    assert recovered==q
print('100/100 exact coefficient reconstructions passed; 64 quotient samples using only 16-point cosets in a field whose largest radix-2 domain is 32.')
