import pandas as pd, numpy as np, sys
from scipy import stats
F=pd.read_csv('features.csv')
tw=pd.read_csv('/home/claude/fit/tw_t.csv')[['instancia','grau_max_mst','n_folhas_mst']]
ch=pd.read_csv('/home/claude/fit/ch_t.csv')[['instancia','n_impares','grau_max_mst']].rename(columns={'grau_max_mst':'gm_ch'})
F=F.drop(columns=['grau_max_mst','prop_folhas_mst','prop_impares','n_impares']).merge(tw,on='instancia').merge(ch,on='instancia')
print('grau max twice==ch', (F.grau_max_mst==F.gm_ch).all())
F['prop_folhas_mst']=F.n_folhas_mst/F.n; F['prop_impares']=F.n_impares/F.n
F['nao_metrica']=(F.classe_metrica=='nao_metrica').astype(int)
feats=['n','grau_max_mst','prop_folhas_mst','prop_impares','cv_vizinho','nao_metrica','prop_fecho','razao_aspecto','entropia']
def partial(x,y,z):
    rx,ry,rz=[stats.rankdata(v) for v in (x,y,z)]
    ex=rx-np.polyval(np.polyfit(rz,rx,1),rz); ey=ry-np.polyval(np.polyfit(rz,ry,1),rz)
    r=np.corrcoef(ex,ey)[0,1]; n=len(x); t=r*np.sqrt((n-3)/(1-r**2)); p=2*stats.t.sf(abs(t),n-3); return r,p
out=[]
for g in ['gap_twice','gap_christofides']:
    for f in feats:
        d=F[[f,g,'n']].dropna()
        rho,p=stats.spearmanr(d[f],d[g])
        pr,pp=(np.nan,np.nan) if f=='n' else partial(d[f].values,d[g].values,d['n'].values)
        out.append((g,f,len(d),rho,p,pr,pp))
R=pd.DataFrame(out,columns=['gap','feature','m','rho','p','rho_parc','p_parc'])
print(R.round(3).to_string())
F.to_csv('features_final.csv',index=False)
