import numpy as np

#TOATE EVALUARILE INDIVIZILOR REZULTATI - IN APELATOR

# VECTORI BINARI
#mutatia bitflip
#I: x - valoarea care se modifica
#E: y - rezultatul mutatiei
def m_binar(x):
    y=not x
    return int(y)


#VECTORI NUMERE INTREGI
#resetare aleatoare
#I:a,b - resetarea se face pe multimea a, a+1,...,b-1
#E: y - noua valoare
def m_ra(a,b):
    y=np.random.randint(a,b)
    return y

#mutatia fluaj
#I: x - valoarea de modificat
#   a,b - limitele in care trebuie sa rezulte iesirea y, varianta a lui x modificata cu o unitate
#E:y - ca mai sus
def m_fluaj(x,a,b):
    #generare +1 sau -1
    p=np.random.randint(0,2)
    if p==0:
        sign=-1
    else:
        sign=1
    y=x+sign
    if y>b:
        y=b
    if y<a:
        y=a
    return y




#VECTORI NUMERE REALE
#mutatia uniforma
#I:a,b - intervalul in care se face resetarea
#E: y - noua valoare
def m_uniforma(a,b):
    y=np.random.uniform(a,b)
    return y

#mutatia neuniforma
#I: x - valoarea de modificat
#   sigma - pasul de fluaj
#   a,b - limitele in care trebuie sa rezulte iesirea y
#E:y - ca mai sus
def m_neuniforma(x,sigma,a,b):
    #generare zgomot
    p=np.random.normal(0,sigma)
    y=x+p
    if y>b:
        y=b
    if y<a:
        y=a
    return y




#PERMUTARI
# mutatia prin inversiune a permutarii x cu n componete
# I:x,n
# E:y - permutarea rezultat
def m_perm_inversiune(x,n):
    # generarea pozitiilor pentru inversiune
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1 = np.min(poz)
    p2 = np.max(poz)
    y=x.copy()
    y[p1:p2+1]=[x[i] for i in range(p2,p1-1,-1)]
    return y


# mutatia prin interschimbare a permutarii x cu n componete
# I:x,n
# E:y - permutarea rezultat
def m_perm_interschimbare(x,n):
    # generarea pozitiilor pentru inversiune
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1 = np.min(poz)
    p2 = np.max(poz)
    y=x.copy()
    y[p1]=x[p2]
    y[p2]=x[p1]
    return y


# mutatia prin inserare a permutarii x cu n componete
# I:x,n
# E:y - permutarea rezultat
def m_perm_inserare(x,n):
    # generarea pozitiilor pentru inversiune
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1 = np.min(poz)
    p2 = np.max(poz)
    y=x.copy()
    y[p1+1]=x[p2]
    if p1<n-2:
        y[p1+2:n]=np.array([x[i] for i in range(p1+1,n) if i != p2])
    return y