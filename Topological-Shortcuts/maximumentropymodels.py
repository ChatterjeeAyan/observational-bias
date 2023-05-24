import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial.distance import squareform
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.model_selection import train_test_split

# traditional configuration model that fixes the expected degree sequence

def configuration_model(k, precision=10**(-5), loops=10000):
    n=len(k)
    t=np.random.uniform(size=(n,1))
    oldt=np.random.uniform(size=(n,1))

    for kk in tqdm(range(loops)):
        T=t*t.transpose()
        summat=(np.ones((n,1))*t.transpose())/(1 + T)
        summat=summat-np.diag(np.diagonal(summat))
        summat=np.sum(summat,axis=1, keepdims=True)
        t=k/(summat+(summat==0))
            
        if (max(abs((t>0)*(1-t/(oldt+(oldt==0)))))< precision):
            break

        oldt=t
            
            
    print("Loops ", kk+1)
    print('Error margin: ', max(abs((t>0)*(1-t/(oldt+(oldt==0))))))

    T=t*(t.transpose())
    Z=1+T;
    pmatrix=T/(Z+(Z==0))
    pmatrix=pmatrix-np.diag(np.diagonal(pmatrix))
    kcal=np.sum(pmatrix,axis=1,keepdims=True);  

    return (pmatrix, kcal) 


# unipartite positive and negative layers ...k01, k10 are column vectors
def multidegree_entropy_pos_neg(k01, k10, precision=10**(-5), loops=10000):
    n=len(k01)
    t01=np.random.uniform(size=(n,1))
    t10=np.random.uniform(size=(n,1))
    oldt01=np.random.uniform(size=(n,1))
    oldt10=np.random.uniform(size=(n,1))    
    
    for kk in tqdm(range(loops)):
            T01=t01*(t01.transpose())
            T10=t10*(t10.transpose())
            Z=1+ T01 + T10
            
            #p01
            summat=(np.ones((n,1))*t01.transpose())/(Z+(Z==0))
            summat=summat-np.diag(np.diagonal(summat))
            summat=np.sum(summat,axis=1, keepdims=True);
            t01=k01/(summat+(summat==0))
            T01=t01*(t01.transpose())
            
            Z=1+ T01 + T10
    
            #p10
            summat=(np.ones((n,1))*t10.transpose())/(Z+(Z==0))
            summat=summat-np.diag(np.diagonal(summat))
            summat=np.sum(summat,axis=1,keepdims=True)
            t10=k10/(summat+(summat==0))
            
            if np.logical_and((max(abs((t01>0)*(1-t01/(oldt01+(oldt01==0)))))< precision),(max(abs((t10>0)*(1-t10/(oldt10+(oldt10==0)))))<precision)):
                break

            oldt01=t01
            oldt10=t10
            
            
    print("Loops ", kk+1)
    print('Error margin: ', max((max(abs((t01>0)*(1-t01/(oldt01+(oldt01==0)))))),max(abs((t10>0)*(1-t10/(oldt10+(oldt10==0)))))))
    T01=t01*(t01.transpose());
    T10=t10*(t10.transpose());

    Z=1+ T01 + T10;
    
    
    summat01=T01/(Z+(Z==0))
    summat01=summat01-np.diag(np.diagonal(summat01))
    k01cal=np.sum(summat01,axis=1,keepdims=True);  
    
    summat10=T10/(Z+(Z==0))
    summat10=summat10-np.diag(np.diagonal(summat10))
    k10cal=np.sum(summat10,axis=1,keepdims=True)
                    
    pconditional=summat10/(summat10+summat01+(summat10==0))
    
    return (summat01, k01cal, summat10, k10cal, pconditional)  


# traditional bipartite configuration model that fixes the expected degree sequence
def bipartite_configuration_model(kr,kc, precision=10**(-5), loops=10000):
    nr=len(kr)
    nc=len(kc)
    tr=np.random.uniform(size=(nr,1))
    oldtr=np.random.uniform(size=(nr,1))
    tc=np.random.uniform(size=(nc,1))
    oldtc=np.random.uniform(size=(nc,1))

    for kk in tqdm(range(loops)):

        #rows
        T=tr*tc.transpose()
        summat=(np.ones((nr,1))*tc.transpose())/(1 + T)
        summat=np.sum(summat,axis=1, keepdims=True)
        tr=kr/(summat+(summat==0))

        #columns
        T=tr*tc.transpose()
        summat=(tr*np.ones((1, nc)))/(1 + T)
        summat=np.sum(summat,axis=0, keepdims=True).transpose()
        tc=kc/(summat+(summat==0))


        flagr=max(abs((tr>0)*(1-tr/(oldtr+(oldtr==0)))))
        flagc=max(abs((tc>0)*(1-tc/(oldtc+(oldtc==0)))))
            
        if ((flagr<precision) & (flagc<precision)):
            break

        oldtr=tr
        oldtc=tc
            
            
    print("Loops ", kk+1)
    print('Error margin: ', max(flagr, flagc))

    T=tr*(tc.transpose())
    Z=1+T;
    pmatrix=T/(Z+(Z==0))
    krcal=np.sum(pmatrix,axis=1,keepdims=True); 
    kccal=np.sum(pmatrix,axis=0,keepdims=True).transpose();  

    return (pmatrix, krcal, kccal) 

# bipartite positive and negative layers 

def bipartite_multidegree_entropy_pos_neg(k01r, k10r, k01c, k10c, precision=10**(-5), loops=10000):
    nr=len(k01r)
    nc=len(k01c)
    t01r=np.random.uniform(size=(nr,1))
    t10r=np.random.uniform(size=(nr,1))
    oldt01r=np.random.uniform(size=(nr,1))
    oldt10r=np.random.uniform(size=(nr,1)) 
    t01c=np.random.uniform(size=(nc,1))
    t10c=np.random.uniform(size=(nc,1))
    oldt01c=np.random.uniform(size=(nc,1))
    oldt10c=np.random.uniform(size=(nc,1)) 

    for kk in tqdm(range(loops)):
            T01=t01r*(t01c.transpose())
            T10=t10r*(t10c.transpose())
            Z=1+ T01 + T10
            
            #p01
            summat=(np.ones((nr,1))*t01c.transpose())/(Z+(Z==0))
            summat=np.sum(summat,axis=1, keepdims=True);
            t01r=k01r/(summat+(summat==0))
            T01=t01r*(t01c.transpose())            
            Z=1+ T01 + T10


            summat=(t01r*np.ones((1, nc)))/(Z+(Z==0))
            summat=np.sum(summat,axis=0, keepdims=True).transpose()
            t01c=k01c/(summat+(summat==0))
            T01=t01r*(t01c.transpose())            
            Z=1+ T01 + T10

            #p10

            summat=(np.ones((nr,1))*t10c.transpose())/(Z+(Z==0))
            summat=np.sum(summat,axis=1, keepdims=True);
            t10r=k10r/(summat+(summat==0))
            T10=t10r*(t10c.transpose())            
            Z=1+ T01 + T10


            summat=(t10r*np.ones((1, nc)))/(Z+(Z==0))
            summat=np.sum(summat,axis=0, keepdims=True).transpose()
            t10c=k10c/(summat+(summat==0))

            flag01r=max(abs((t01r>0)*(1-t01r/(oldt01r+(oldt01r==0)))))
            flag01c=max(abs((t01c>0)*(1-t01c/(oldt01c+(oldt01c==0)))))
            flag10r=max(abs((t10r>0)*(1-t10r/(oldt10r+(oldt10r==0)))))
            flag10c=max(abs((t10c>0)*(1-t10c/(oldt10c+(oldt10c==0)))))
            
            if ((flag01r<precision) & (flag01c<precision) & (flag10r<precision) & (flag10c<precision)):
                break
            
            oldt01r=t01r
            oldt10r=t10r  
            oldt01c=t01c
            oldt10c=t10c 

    print("Loops ", kk+1)
    print('Error margin: ', max(flag01c, flag01r, flag10c, flag10r))

    T01=t01r*(t01c.transpose());
    T10=t10r*(t10c.transpose());
    Z=1+ T01 + T10;
    
    
    summat01=T01/(Z+(Z==0))
    k01rcal=np.sum(summat01,axis=1,keepdims=True);
    k01ccal=np.sum(summat01,axis=0,keepdims=True).transpose(); 
    
    summat10=T10/(Z+(Z==0))
    k10rcal=np.sum(summat10,axis=1,keepdims=True)
    k10ccal=np.sum(summat10,axis=0,keepdims=True).transpose(); 
                    
    pconditional=summat10/(summat10+summat01+(summat10==0))
    
    return (summat01, k01rcal, k01ccal, summat10, k10rcal, k10ccal, pconditional)  





if __name__ == "__main__":
    print("testing standard configuration model")
    ba=nx.barabasi_albert_graph(500,2)

    print("500 nodes, barabasi-albert")
    degree_sequence=np.array([[ba.degree(idn) for idn in ba.nodes]]).T # degree sequence
    (pm, degree_sequence_cal)=configuration_model(degree_sequence)

    figure(figsize=(8, 8))
    ax = sns.distplot(squareform(pm), hist_kws=dict(alpha=0.1))
    ax.set(xlim = [0,1], xlabel='$p^{ij}$', ylabel='PDF')


