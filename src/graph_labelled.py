import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd, scipy.sparse as sp
from data import *
def graph_feats():
    """Labelled-ness graph features, identical definition in train and test:
    a node counts as 'labelled' if it has a label (train) or a test.csv row (test)."""
    tr=load_train(); te=load_test(); e=load_edges()
    labelled=set(tr.txId[tr.label.notna()])|set(te.txId)
    nodes=pd.Index(pd.unique(pd.concat([e.txId1,e.txId2,tr.txId,te.txId])))
    a=nodes.get_indexer(e.txId1); b=nodes.get_indexer(e.txId2); N=len(nodes)
    isl=np.isin(nodes, list(labelled)).astype(float)
    D=sp.csr_matrix((np.ones(len(a)),(a,b)),shape=(N,N))  # a -> b
    U=((D+D.T)>0).astype(float)
    f=pd.DataFrame(index=nodes)
    f['g_deg']=np.asarray(U.sum(1)).ravel()
    f['g_nlab']=U@isl
    f['g_frac']=f.g_nlab/f.g_deg.clip(lower=1)
    f['g_in_lab']=D.T@isl; f['g_out_lab']=D@isl
    f['g_in']=np.asarray(D.sum(0)).ravel(); f['g_out']=np.asarray(D.sum(1)).ravel()
    U2=(U@U); U2.setdiag(0); U2=(U2>0).astype(float)
    f['g_nlab2']=U2@isl; f['g_n2']=np.asarray(U2.sum(1)).ravel()
    return f
if __name__=='__main__':
    f=graph_feats(); tr=load_train(); te=load_test()
    L=tr[tr.label.notna()]; a=f.loc[L.txId]; b=f.loc[te.txId]; post=f.loc[L.txId[L.time_step>=24]]
    from sklearn.metrics import roc_auc_score
    for c in f: print(f'{c:10s} train {a[c].mean():7.3f} post {post[c].mean():7.3f} test {b[c].mean():7.3f}  out-of-range {( (b[c]>a[c].max())|(b[c]<a[c].min())).mean():.4f} auc {roc_auc_score(L.label,a[c]):.3f}')
    # adversarial per feature
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    for c in f:
        X=np.r_[post[c].values,b[c].values][:,None]; yy=np.r_[np.zeros(len(post)),np.ones(len(b))]
        print(c,'adv AUC vs post-cliff', cross_val_score(DecisionTreeClassifier(max_depth=4),X,yy,cv=5,scoring='roc_auc').mean().round(3))
