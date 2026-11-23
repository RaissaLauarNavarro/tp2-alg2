import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
F=pd.read_csv('features_final.csv')
cands={'media (sem features)':[],'N':['n'],'N + grau máx. MST':['n','grau_max_mst'],'N + não métrica':['n','nao_metrica'],
       'N + CV vizinho':['n','cv_vizinho'],'N + grau máx. + não métrica':['n','grau_max_mst','nao_metrica']}
for g in ['gap_twice','gap_christofides']:
    y=F[g].values; print('==',g)
    for nm,c in cands.items():
        pred=[]
        for i in range(len(y)):
            m=np.ones(len(y),bool); m[i]=False
            if not c: pred.append(y[m].mean()); continue
            X=F[c].values; pred.append(LinearRegression().fit(X[m],y[m]).predict(X[i:i+1])[0])
        pred=np.array(pred); rmse=np.sqrt(((y-pred)**2).mean()); q2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
        if c:
            X=F[c].values; lr=LinearRegression().fit(X,y); r2=lr.score(X,y)
            Z=(F[c]-F[c].mean())/F[c].std(); beta=LinearRegression().fit(Z,(y-y.mean())/y.std()).coef_
        else: r2=0; beta=[]
        print(f"  {nm:30s} R2={r2:.3f} R2_LOO={q2:.3f} RMSE_LOO={rmse:.2f}  beta_pad={np.round(beta,2)}")
    for depth in [1,2]:
        X=F[['n','grau_max_mst','prop_folhas_mst','prop_impares','cv_vizinho','nao_metrica']].values; pred=[]
        for i in range(len(y)):
            m=np.ones(len(y),bool); m[i]=False
            pred.append(DecisionTreeRegressor(max_depth=depth,min_samples_leaf=3,random_state=0).fit(X[m],y[m]).predict(X[i:i+1])[0])
        pred=np.array(pred); q2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
        t=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=3,random_state=0).fit(X,y)
        from sklearn.tree import export_text
        print(f"  arvore d={depth} R2_LOO={q2:.3f} RMSE_LOO={np.sqrt(((y-pred)**2).mean()):.2f}")
        print(export_text(t,feature_names=['n','grau_max','folhas','impares','cv_viz','nao_metr']))
