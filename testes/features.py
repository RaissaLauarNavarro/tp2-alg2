import sys, math, glob, os, numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
INST=sys.argv[1]; FIT=sys.argv[2]
def parse(path):
    L=open(path).read().split('\n'); h={}; i=0; coords=None; disp=None; w=[]
    while i<len(L):
        s=L[i].strip()
        if ':' in s and not s.endswith('SECTION'):
            k,v=s.split(':',1); h[k.strip()]=v.strip()
        if s.startswith('NODE_COORD_SECTION') or s.startswith('DISPLAY_DATA_SECTION'):
            n=int(h['DIMENSION']); arr=[]
            for j in range(n):
                p=L[i+1+j].split(); arr.append((float(p[1]),float(p[2])))
            if s.startswith('NODE'): coords=np.array(arr)
            else: disp=np.array(arr)
            i+=n
        if s.startswith('EDGE_WEIGHT_SECTION'):
            j=i+1
            while j<len(L) and L[j].strip() and not L[j].strip()[0].isalpha():
                w+= [float(x) for x in L[j].split()]; j+=1
            i=j-1
        i+=1
    return h,coords,disp,w
def geo_rad(x):
    deg=int(x); m=x-deg; return math.pi*(deg+5*m/3)/180
def dist_matrix(h,coords,w):
    n=int(h['DIMENSION']); t=h['EDGE_WEIGHT_TYPE']; D=np.zeros((n,n))
    if t=='EXPLICIT':
        f=h['EDGE_WEIGHT_FORMAT']; it=iter(w)
        for a in range(n):
            if f=='FULL_MATRIX':
                for b in range(n): D[a,b]=next(it)
            elif f=='UPPER_ROW':
                for b in range(a+1,n): D[a,b]=D[b,a]=next(it)
            elif f=='LOWER_DIAG_ROW':
                for b in range(a+1): D[a,b]=D[b,a]=next(it)
        return D
    for a in range(n):
        for b in range(n):
            if a==b: continue
            (x1,y1),(x2,y2)=coords[a],coords[b]
            if t=='EUC_2D': D[a,b]=int(math.sqrt((x1-x2)**2+(y1-y2)**2)+0.5)
            elif t=='ATT':
                r=math.sqrt(((x1-x2)**2+(y1-y2)**2)/10); tt=int(r+0.5); D[a,b]=tt+1 if tt<r else tt
            elif t=='GEO':
                RRR=6378.388; la1,lo1,la2,lo2=geo_rad(x1),geo_rad(y1),geo_rad(x2),geo_rad(y2)
                q1=math.cos(lo1-lo2); q2=math.cos(la1-la2); q3=math.cos(la1+la2)
                D[a,b]=int(RRR*math.acos(0.5*((1+q1)*q2-(1-q1)*q3))+1)
    return D
rows=[]
for p in sorted(glob.glob(INST+'/*.tsp')):
    name=os.path.basename(p)[:-4]; h,coords,disp,w=parse(p); D=dist_matrix(h,coords,w); n=len(D)
    T=minimum_spanning_tree(D).toarray(); A=(T>0)|(T.T>0); deg=A.sum(1)
    Dn=D+np.diag([np.inf]*n); nn=Dn.min(1)
    r=dict(instancia=name,n=n,custo_mst_calc=T.sum(),grau_max_mst=deg.max(),prop_folhas_mst=(deg==1).mean(),
           cv_vizinho=nn.std()/nn.mean(),prop_impares=(deg%2==1).mean())
    P=coords if coords is not None else disp
    if P is not None:
        if h['EDGE_WEIGHT_TYPE']=='GEO':
            lat=np.array([math.degrees(geo_rad(x)) for x in P[:,0]]); lon=np.array([math.degrees(geo_rad(y)) for y in P[:,1]])
            P=np.c_[lon*math.cos(math.radians(lat.mean())),lat]
        hull=ConvexHull(P); r['prop_fecho']=len(hull.vertices)/n
        span=P.max(0)-P.min(0); r['razao_aspecto']=span.max()/span.min()
        g=math.ceil(math.sqrt(n)); ij=np.minimum(((P-P.min(0))/span*g).astype(int),g-1)
        cnt=np.bincount(ij[:,0]*g+ij[:,1],minlength=g*g); pr=cnt[cnt>0]/n
        r['entropia']=-(pr*np.log(pr)).sum()/math.log(g*g)
    rows.append(r)
F=pd.DataFrame(rows)
tw=pd.read_csv(FIT+'/tw_t.csv')[['instancia','custo_mst','gap_pct','classe_metrica']].rename(columns={'gap_pct':'gap_twice'})
ch=pd.read_csv(FIT+'/ch_t.csv')[['instancia','gap_pct','n_impares']].rename(columns={'gap_pct':'gap_christofides'})
F=F.merge(tw,on='instancia').merge(ch,on='instancia').sort_values('n')
print(F[['instancia','custo_mst','custo_mst_calc','n_impares']].assign(imp_calc=F.prop_impares*F.n).to_string())
F.drop(columns=['custo_mst_calc','custo_mst']).to_csv('/home/claude/sens/features.csv',index=False)
pd.set_option('display.width',250); print(F.drop(columns=['custo_mst_calc','custo_mst','n_impares']).round(3).to_string())
