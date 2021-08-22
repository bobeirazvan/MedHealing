import numpy as np


# CROSSOVER VECTORI BINARI SAU INT

# crossover unipunct intre x si y, vectori cu n componente
# I :x,y,n ca mai sus
#E: c1,c2 - cei doi copii - cu n componente
def crossover_unipunct(x,y,n):
    #genereaza aleator punctul de crossover, intre 1 si n-1 pentru a avea efect
    # in pozitia n este calitata
    i = np.random.randint(0,n)
    c1=np.zeros(n)
    c2=np.zeros(n)
    # selectarea secventelor care compun primul copil si conversia catre ndarray
    c1[0:i] = x[0:i]
    c1[i:n] = y[i:n]
    # selectarea secventelor care compun cel de-al doilea copil si conversia catre ndarray
    c2[0:i] = y[0:i]
    c2[i:n] = y[i:n]
    return c1,c2


# crossover uniform intre x si y, vectori cu n componente
# I :x,y,n ca mai sus
#E: c1,c2 - cei doi copii
def crossover_uniform(x,y,n):
    #initializeaza copiii cu valorile parintilor
    c1=x.copy()
    c2=y.copy()
    #constructia copiilor
    for i in range(n):
        r = np.random.randint(0,2)
        # daca r==1 interschimba alelele din i
        if r == 1:
            c1[i] = y[i]
            c2[i] = x[i]
    return c1,c2



# CROSSOVER PERMUTARI - o permutare de dim n este cu elementele 0,1,...,n-1
#operatorul PMX
#I: permutarile x,y de dimensiune n
#E: copiii rezultati c1,c2
def crossover_PMX(x,y,n):
    #generarea secventei de crossover
    poz=np.random.randint(0,n,2)
    while poz[0]==poz[1]:
        poz=np.random.randint(0,n,2)
    p1=np.min(poz)
    p2=np.max(poz)
    c1=PMX(x,y,n,p1,p2)
    c2=PMX(y,x,n,p1,p2)
    return c1,c2

#aplica PMX pe x,y de dimensiune n, cu secventa de recombinare (p1,p2)
def PMX(x,y,n,p1,p2):
    #initializare copil - un vector cu toate elementele -1 - valori care s=sa nu fie in 0,...,n-1
    c=-np.ones(n,dtype=int)
    #copiaza secventa comuna in copilul c
    c[p1:p2+1]=x[p1:p2+1]
    # analiza secventei comune - in permutarea y
    for i in range(p1,p2+1):
        # plasarea alelei a
        a=y[i]
        if a not in c:
            curent=i
            plasat=False
            while not plasat :
                b=x[curent]
                # poz=pozitia in care se afla b in y
                [poz]=[j for j in range(n) if y[j]==b]
                if c[poz]==-1 :
                    c[poz]=a
                    plasat=True
                else:
                    curent=poz
    # z= vectorul alelelor din y inca necopiate in c
    z=[y[i] for i in range(n) if y[i] not in c]
    # poz - vectorul pozitiilor libere in c - cele cu vaori -1
    poz=[i for i in range(n) if c[i]==-1]
    #copierea alelelor inca necopiate din y in c
    m=len(poz)
    for i in range(m):
        c[poz[i]]=z[i]
    return c


#operatorul OCX
#I: permutarile x,y de dimensiune n
#E: copiii rezultati c1,c2
def crossover_OCX(x,y,n):
    #generarea secventei de crossover
    poz=np.random.randint(0,n,2)
    while poz[0]==poz[1]:
        poz=np.random.randint(0,n,2)
    p1=np.min(poz)
    p2=np.max(poz)
    c1=OCX(x,y,n,p1,p2)
    c2=OCX(y,x,n,p1,p2)
    return c1,c2

#aplica OCX pe x,y de dimensiune n, cu secventa de recombinare (p1,p2)
def OCX(x,y,n,p1,p2):
    #copiaza secventa comuna in c2
    c2=[x[i] for i in range(p1,p2+1)]
    #calculeaza z pe baza lui y:componentele incepand cu p2 pana la n si apoi de la 0 la p2-1, excluzand elementele care au fost deja copiate
    z1=[y[i] for i in range(p2,n) if y[i] not in c2]
    z2=[y[i] for i in range(p2) if y[i] not in c2]
    z=np.append(z1,z2)
    #calculeza secventa finala a individului rezultat  - din z de la 0 la n-p2
    c3=[z[i] for i in range(n-p2-1)]
    #calculeaza secventa de inceput a individului rezultat - din z de la n-p2...len(z)
    c1=[z[i] for i in range(n-p2-1,len(z))]
    #calculeaza copilul c
    c=np.append(c1,c2)
    c=np.append(c,c3)
    return c

#operatorul CX
#I: permutarile x,y de dimensiune n - completate cu fitnessul
#E: copiii rezultati c1,c2
def crossover_CX(x,y,n):
    ciclu=cicluri(x,y,n)
    c1=x.copy()
    c2=y.copy()
    for i in range(n):
        cat, rest = np.divmod(ciclu[i], 2)
        #sunt interschimbate alelele din ciclurile pare
        #primul ciclu este etichetat cu 1
        if not rest:
            c1[i]=y[i]
            c2[i]=x[i]
    return c1,c2

#calculul ciclurilor din
#I: x,y de dimensiune n
#in
#E: vectorul ciclu, unde ciclu[i]=nr. ciclului din care fac parte x[i] si y[i]
def cicluri(x,y,n):
    #numarul ciclului
    index=1
    ciclu=np.zeros(n)
    gata=0
    while not gata:
        p=np.where(ciclu==0)
        #daca exista gena inca neasignata unui ciclu
        if np.size(p):
            i=p[0][0]
            a=x[i]
            ciclu[i]=index
            b=y[i]
            while b!=a:
                r=np.where(x==b)
                j=r[0][0]
                ciclu[j]=index
                b=y[j]
            index+=1
        else:
            gata=1
    return ciclu


#EDGE CROSSOVER

#construieste tabela muchiilor pentru permutarile x si y de dimensiune n
def constr_tabel(x,y,n):
    #creaza o lista cu n elemente, toate 0
    muchii=[0]*n
    #pentru usurinta, bordeaza x cu ultimul/primul element
    x1=np.zeros(n+2, dtype='int')
    x1[1:n+1]=x[:]
    x1[0]=x[n-1]
    x1[n+1]=x[0]
    y1 = np.zeros(n + 2, dtype='int')
    y1[1:n + 1] = y[:]
    y1[0] = y[n - 1]
    y1[n + 1] = y[0]
    for i in range(1,n+1):
        a=x1[i]
        r=np.where(y==a)
        j=r[0][0]+1
        #gaseste vecinii lui a in x si y utilizand x1 si y1 si memoreaza ca multimi pentru - si intersectie
        vx={x1[i-1],x1[i+1]}
        vy={y1[j-1],y1[j+1]}
        dx=vx-vy
        dy=vy-vx
        cxy=vx & vy
        #trecem de la set la list
        lcxy=list(cxy)
        dx=list(dx)
        dy=list(dy)
        #lucram cu tip str
        for j in range(len(lcxy)):
            lcxy[j]=str(lcxy[j])+'+'
        for j in range(len(dx)):
            dx[j]=str(dx[j])
        for j in range(len(dy)):
            dy[j]=str(dy[j])
        muchii[a]=lcxy+list(dx)+list(dy)
    return muchii

# sterge un element dintr-o lista - cu chei unice, daca apare in lista
# altfel, lasa lista nemodficata
def sterge(x,a):
    #cauta aparitia - poate fi doar una
    p=[i for i in range(len(x)) if x[i]==a]
    if len(p):
        del(x[p[0]])
    return x

#alege alela de plasat, daca lp are mai mult de o valoare
def alege(lp,muchii,n):
    dim=len(lp)
    #cauta daca exista in lista muchiilor 'lp[k]+'
    #calculeaza lungimile listelor, in caz ca nu gaseste 'lp[k]'
    lliste=np.zeros(dim)
    gata=0
    k=0
    while k<dim and not gata:
        a=str(lp[k])+'+'
        i=0
        while i<n and not gata :
            l=muchii[i]
            p=[j for j in range(len(l)) if l[j]==a]
            if len(p):
                gata=1
                alela=lp[k]
            else:
                p=[j for j in range(len(l)) if l[j]==str(lp[k])]
                i=i+1
                lliste[k]=len(l)
        if not gata:
            k=k+1
    if not gata:
        #calculeaza lungimea minima si pentru ce alele se atinge
        x=[j for j in range(dim) if lliste[j]==min(lliste)]
        #alege prima alela de lungime minima
        #daca sunt mai multe, alege-o pe prima
        alela=lp[x[0]]
    return alela

# operatorul ECX - Edge crossover
def ECX(x,y,n):
    muchii=constr_tabel(x,y,n)
    #permutarea rezultata
    z=np.zeros(n, dtype='int')
    #alege initial prima alela - varianta:selecteaza aleator ap in 0...n-1
    #lp - lista alelelor posibile
    #ales - vectorul flag al alelelor alese
    ales=np.zeros(n)
    lp=[x[0]]
    for i in range(n):
        print(muchii)
        if len(lp)==0:
            #alege aleator o alela
            a=np.random.randint(n)
            while ales[a]:
                a = np.random.randint(n)
        else:
            if len(lp)>1:
                a=alege(lp,muchii,n)
            else:
                a=lp[0]
        #atribuie alela aleasa
        z[i]=a
        ales[a]=1
        print(a)
        #sterge alela din tabela de muchii
        for k in range(n):
            sterge(muchii[k],str(a))
            sterge(muchii[k],str(a)+'+')
        #alege lista posibilitatilor la urmatorul moment
        lp=[int(muchii[a][i][0]) for i in range(len(muchii[a]))]
    return z

#APEL ECX
#import numpy as np
#import FunctiiCrossoverIndivizi as c
#n=10
#x=np.random.permutation(n)
#y=np.random.permutation(n)
#z=c.ECX(y,x,10)


# CROSSOVER VECTORI NUMERE REALE

# crossover singular intre x si y, vectori cu n componente de tipul ndarray
# I :x,y,n ca mai sus
#    alpha - ponderea la mediere
#E: c1,c2 - cei doi copii - fara evaluare
def crossover_singular (x, y, n,alpha):
    #genereaza aleator gena in care este facuta recombinarea
    i = np.random.randint(0,n)
    c1 = x.copy()
    c2 = y.copy()
    c1[i]=(alpha*x[i]+(1-alpha)*y[i])
    c2[i] = (alpha * y[i] + (1 - alpha) * x[i])
    return c1, c2


# crossover simplu intre x si y, vectori cu n componente - de tipul ndarray
# I :x,y,n ca mai sus
#    alpha - ponderea la mediere
#E: c1,c2 - cei doi copii
def crossover_simplu (x, y, n,alpha):
    #genereaza aleator gena incepand cu care este facuta recombinarea
    i = np.random.randint(0,n)
    c1 = x.copy()
    c2 = y.copy()
    for j in range(i,n):
        c1[j]=(alpha*x[j]+(1-alpha)*y[j])
        c2[j] = (alpha * y[j] + (1 - alpha) * x[j])
    return c1, c2

# crossover total intre x si y, vectori cu n componente
# I :x,y,n ca mai sus
#    alpha - ponderea la mediere
#E: c1,c2 - cei doi copii - fara evaluare
def crossover_total (x, y, n,alpha):
    c1 = x.copy()
    c2 = y.copy()
    for j in range(n):
        c1[j]=(alpha*x[j]+(1-alpha)*y[j])
        c2[j] = (alpha * y[j] + (1 - alpha) * x[j])
    return c1, c2
