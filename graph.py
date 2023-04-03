"""
SÉANCE 1
"""

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

    def __str__(self):
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    

    # Séance 1 question 1 partie 1
    def add_edge(self, node1, node2, power_min, dist=1):
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)
        self.graph[node1].append((node2,power_min,dist))
        self.graph[node2].append((node1,power_min,dist))
        self.nb_edges +=1

    #Séance 1 Question 2
    """
        Dans la première partie de cette question, on veut trouver, pour chaque noeud, les noeuds avec lesquels il est
    connecté. Par conséquent, nous avons écrit la fonction imbriquée 'connecter' qui va permettre de trouver les 
    composantes connectées pour un noeud donné.
        Pour chaque noeud, on va procéder de la manière suivante:
        - Étape 1: on commence par récupérer les noeuds qui ont une arête avec n (ceux là font déja partis des composantes
            connectées donc peuvent aller dans 'res' qui est la liste résultat)
        - Étape 2: La liste 'a_voir' correspond à tous les noeuds qu'il faut encore explorer pour avoir leurs voisins pour 
            avoir d'autres composantes connexes. Et la liste 'vus' correspond aux noeuds déjà explorés.
        - Étape 3: Tant que cette liste est non vide, on regarde tous les noeuds qui ont une arête avec lui. Ensuite, il y a deux cas:
                1) Le voisin est déjà dans 'res', auquel cas on doit rien faire
                2) Il ne l'est pas et donc il faut le rajouter à 'res'. S'il n'a pas déjà été vu, on le met également dans
                    'a_voir'.
        - Étape 4: On enlève le noeud que l'on vient d'explorer de 'a_voir' 
    Enfin, pour obtenir les composantes connectées de chaque noeud, on renvoie la liste [[n]+ connecter(self,n) for n in self.nodes].
    """
    def connected_components(self):
        def connecter(self, n):
            #étape 1 et 2
            a_voir=[self.graph[n][i][0] for i in range (len(self.graph[n]))]
            res=a_voir.copy()
            vus=[]
            #étape 3
            while a_voir!=[]:
                noeud=a_voir[0]
                regarder=[self.graph[noeud][i][0] for i in range (len(self.graph[noeud]))]
                vus.append(noeud)
                for i in regarder:
                    if i!=n:
                        #étape 3
                        if i not in res:
                            res.append(i)
                        if (i not in a_voir) and (i not in vus):
                            a_voir.append(i)
                #Étape 4
                if len(a_voir)>1:
                    a_voir=a_voir[1:]
                else: 
                    a_voir=[]
            return (res)
        resultat= [[n]+ connecter(self,n) for n in self.nodes]
        return (resultat)
    
    """
        Dans la question précédente, on a récupéré une liste de liste de composantes connectées. Cependant, pour chaque 
    ensemble de noeuds connectés, on récupère la liste des composantes autant de fois qu'il y a de noeuds (la liste se 
    trouve juste dans un ordre différent à chaque fois).
        Par conséquent, nous souhaitons enlever les doublons. Pour cela, on créé 'vus' qui correspond aux noeuds déjà
    explorés. 
        
    Étape 1: Pour chaque liste de liste_cc=self.connected_components(), on prend le premier élément.
    Étape 2: S'il n'a pas déjà été vus, alors aucune de ses composantes n'a été vue et donc on peut rajouter cette liste 
        à la liste résultat.
    Étape 3: on rajoute tous les éléments de la liste à 'vus' (pour éviter les doublons dans 'résultat').
    Étape 4: on enlève cette liste de liste_cc afin d'explorer les autres listes
    Étape 5: On transforme cette liste resultat en frozenset.
    """
    def connected_components_set(self):
        resultat=[]
        liste_cc=self.connected_components()
        vus=[]
        #Étape 1
        while liste_cc!=[] :
            fs=liste_cc[0]
            #Étape 2
            if fs[0] not in vus:
                resultat.append(fs)
            #Étape 3
            vus+=fs
            #Étape 4
            liste_cc.remove(fs)
        #Étape 5
        res=[frozenset(l) for l in resultat]
        return (res)

  
    #Séance 1 question 3
    """
        Dans cette question, on veut savoir s'il existe un chemin entre deux noeuds (src et dest), étant donné une puissance.
    Pour cela on va procéder en plusieurs étapes:
        1) On récupère les composantes connectées du graphe et on obtient deux cas:
                - src et dest ne sont pas connectées et donc on renvoie directement 'None' car il n'existera jms de chemin.
                - elles sont connectées. Il existe donc obligatoirement un chemin (mais peut-être pas pour cette puissance)
                    Donc on va explorer les noeuds voisins.
        2) On crée 'a_voir' avec tous les voisins directs de la src (sous format de liste car les éléments de 'a_voir vont
            être des listes de chemins possibles partant de src), avec une puissance de l'arête inférieure à 'power' (car
            sinon le voisin est inaccessible avec cette puissance). (Si src et dest sont voisins directs et que l'arête 
            a un poids en dessous de 'power', alors on renvoie [src,dest] directement).
        3) Tant que 'a_voir' est non vide:
            On se place en son premier noeud. 
            On observe ses voisins qui ont une arête de puissance inférieure à 'power' et il existe plusieurs cas
                - si c'est la destination, on renvoie le chemin.
                - si ça ne l'est pas, on va rajouter à 'a_voir' les différents chemins l+[voisin] (en vérifiant toutefois
                    que le voisin n'est pas déjà dans le chemin l car sinon on aurait des boucles infinies)
        Rq: Le programme se finira dans la boucle a_voir car src et dest sont deux composantes connectées. Cependant, nous 
            avons quand même mis le 'return None' à la fin de la fonction afin d'être sûre que la fonction renvoie quelquechose
            même si ça n'arrivera jamais à là.
    """
    def get_path_with_power(self, src, dest, power):
        #Étape 1
        liste_cc=self.connected_components()
        for l in liste_cc:
            if src in l and not dest in l:
                return None
        #Étape 2
        a_voir=[]
        for i in range (len(self.graph[src])):
            if self.graph[src][i][1]<=power:
                a_voir.append([src]+[self.graph[src][i][0]])
                if self.graph[src][i][0]==dest:
                    return [src,dest]
        #Étape 3
        while a_voir!=[]:
            l=a_voir[0]
            l2=[]
            for i in range (len(self.graph[l[-1]])):
                if self.graph[l[-1]][i][1]<=power:
                    if self.graph[l[-1]][i][0]==dest:
                        return l+[dest]
                    if not self.graph[l[-1]][i][0] in l:    #empêche les cycles
                        l2.append(self.graph[l[-1]][i][0])
            a_voir_new=[]
            for i in a_voir:
                if i!=l:
                    a_voir_new.append(i)
            a_voir=a_voir_new.copy()
            for i in l2:
                a_voir.append(l+[i])
        return None
    
    #Séance 1 question 5 (bonus)
    """
    Dans cette question, on veut savoir qu'elle est la distance minimale entre src et dest, munie d'une puissance power.
    Pour cela, on procède de la même manière que pour get_path_with_power, sauf que dès que nous avons un chemin qui marche,
    on le rajoute à la liste 'finis' et à la fin de la fonction, on garde le chemin avec la distance minimale.
    Pour avoir les distances des différents chemins, on met la distance parcourue en première place dans la liste (donc 
    le premier élément des chemins est la distance et non un noeud du chemin).
    """
    def min_dist_with_power(self, src, dest, power):
        liste_cc=self.connected_components()
        if not(dest in liste_cc[src-1]):
            return None
        finis=[]
        a_voir=[]
        compconnexes=liste_cc[src-1]
        for i in range (len(self.graph[src])):
            if self.graph[src][i][1]<=power:
                a_voir.append([self.graph[src][i][2]]+[src]+[self.graph[src][i][0]])
                if self.graph[src][i][0]==dest:
                    finis.append([self.graph[src][i][2],src,dest])
        while a_voir!=[]:
            l=a_voir[0]
            l2=[]
            for i in range (len(self.graph[l[-1]])):
                if self.graph[l[-1]][i][1]<=power:
                    l2.append((self.graph[l[-1]][i][2],self.graph[l[-1]][i][0]))
                    if self.graph[l[-1]][i][0]==dest:
                        finis.append([l[0]+self.graph[l[-1]][i][2]]+l[1:]+[dest])
            a_voir_new=[]
            for i in a_voir:
                if i!=l:
                    a_voir_new.append(i)
            a_voir=a_voir_new.copy()
            for i in l2:
                l3=l+[i[1]]
                l3[0]+=i[0]
                if i[1] not in l:
                    if not l3[-1]==dest:
                        a_voir.append(l3)
        mini=finis[0][0]
        chemin=finis[0][1:]
        for i in range (len(finis)):
            if finis[i][0]<mini:
                mini=finis[i][0]
                chemin=finis[i][1:]
        return (mini,chemin)

    #Séance 1 question 6
    """
        Dans cette question, on prend deux noeuds src et dest et on veut savoir s'il existe un chemin entre les deux, et si
    oui, quelle est la puissance minimale recquise pour ce chemin. Pour cela on va avoir recours à plusieurs étapes.
        1) On regarde si src et dest sont connectées:
            - si non, on renvoie qu'il n'y a pas de chemin
            - si oui, on récupère toutes les composantes auquelles il est connecté.
        2) On récupère l'ensemble des puissances des arêtes entre les noeuds des composantes connectées. On trie la liste 
            dans le sens croissant.
        3) Dans l'ordre croissant des puissances, on regarde s'il existe un chemin entre src et dest pour cette puissance
            grâce à la fonction créée à la question précédente. S'il existe, alors on renvoit ce chemin avec la puissance
            associée. S'il n'existe pas, on voit pour la puissance suivante.
        Rq: Comme les noeuds src et dest sont connectées, alors il existe forcément un chemin possible, dans le pire cas,
        la fonction doit voir s'il existe un chemin pour toutes les puissances, jusqu'à arriver à la plus grande puissance.
    """
    def min_power(self,src,dest):
        #Étape 1
        liste_cc=self.connected_components()
        for l in liste_cc:
            if src in l and not dest in l:
                return None
            if src in l:
                compconnexes=[]
                for i in l:
                    if i!=src:
                        compconnexes.append(i)
        #Étape 2
        liste_puissance=[]
        for n in compconnexes:
            l=self.graph[n]
            for j in l:
                if not j[1] in liste_puissance:
                    liste_puissance.append(j[1])
        liste_puissance.sort()
        #Étape 3
        for puiss in liste_puissance:
            res=self.get_path_with_power(src=src, dest=dest, power=puiss)
            if res!=None:
                return (puiss,res)
        return None

# Séance 1 Question 1 partie 2 (et question 4)
def graph_from_file(filename):
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g

#TESTS UNITAIRES POUR LA SÉANCE 1
import sys 
sys.path.append("delivery_network/")

import unittest 
from graph import Graph, graph_from_file
data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"

#pour la question 1:
class Test_q1(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        self.assertEqual(g.nb_nodes, 10)
        self.assertEqual(g.nb_edges, 9)

    def test_network1(self):
        g = graph_from_file(data_path+"network.01.in")
        self.assertEqual(g.nb_nodes, 7)
        self.assertEqual(g.nb_edges, 5)
    
    def test_network4(self):
        g = graph_from_file(data_path+"network.04.in")
        self.assertEqual(g.nb_nodes, 10)
        self.assertEqual(g.nb_edges, 4)
        self.assertEqual(g.graph[1][0][2], 6)

#pour la question 2:
class Test_q2(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        cc = g.connected_components_set()
        self.assertEqual(cc,[frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})])

    def test_network1(self):
        g = graph_from_file(data_path+"network.01.in")
        cc = g.connected_components_set()
        self.assertEqual(cc, [frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})])

#pour la question 3
class Test_q3(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        self.assertEqual(g.get_path_with_power(1, 4, 11), [1, 2, 3, 4])
        self.assertEqual(g.get_path_with_power(1, 4, 10), None)

    def test_network2(self):
        g = graph_from_file(data_path+"network.02.in")
        self.assertIn(g.get_path_with_power(1, 2, 11), [[1, 2], [1, 4, 3, 2]])
        self.assertEqual(g.get_path_with_power(1, 2, 5), [1, 4, 3, 2])

#pour la question 5
class Test_q5(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file(data_path+"network.01.in")
        self.assertEqual(g.min_dist_with_power(3, 1, 11),(2, [3, 2, 1]))
        self.assertEqual(g.min_dist_with_power(2, 7, 10), None)

    def test_network4(self):
        g = graph_from_file(data_path+"network.04.in")
        self.assertEqual(g.min_dist_with_power(2, 1, 11), (11,[2, 3, 4, 1]))
        self.assertEqual(g.min_dist_with_power(2, 7, 10), None)

#pour la question 6
class Test_q6(unittest.TestCase):
    def test_network2(self):
        g = graph_from_file(data_path+"network.02.in")
        self.assertEqual(g.min_power(2,1),(4, [2,3,4,1]))
        self.assertEqual(g.min_power(2, 7), None)

    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        self.assertEqual(g.min_power(1,7), (14,[1,2,5,7]))
        self.assertEqual(g.min_power(9,4), (14,[9,8,1,2,3,4]))

"""
SÉANCE 2
"""

# Séance 2 question 1
"""
    Dans cette question, on veut estimer le temps nécessaire pour calculer la puissance minimale (et le chemin associé) 
sur l'ensemble des trajets pour chacun des fichiers routes.x.in donnés.
    Pour cela, on crée la fonction 'trajets' qui renvoie le nombre des trajets de routes.x.in, ainsi que les dix 
premiers chemins que l'on veut parcourir pour estimer le temps nécessaire.
    Dans la fonction question1_séance2(i):
    1) on commence par récupérer le graphe et les trajets que l'on veut parcourir et le nombre total de trajets
    2) on calcule le temps d'exécution pour chacun des trajets
    3) on renvoie (tps)/len(l)*n qui correspond au temps estimé d'exécution pour récupérer entièrement les trajets.
"""
def trajets(filename):
    with open(filename, "r") as file:
        n=int(file.readline())
        l=[]
        for i in range (min(n,10)):
            traj=list(map(int, file.readline().split()))
            l.append((traj[0],traj[1]))
    return n,l

def question1_séance2(i):
    #Étape 1
    import time
    data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"
    tps=0
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    n,l=trajets(data_path+file_name2)
    #étape 2
    for j in range (len(l)):
        print(j)
        a=time.time()
        g.min_power(src=l[j][0],dest=l[j][1])
        b=time.time()
        tps+=b-a
    #étape 3
    return (tps)/len(l)*n

"""
Créons maintenant une fonction qui va renvoyer routes.x.out avec pour chaque paire le chemin de puissance minimale.
    1) récupérons tous les trajets
    2) On crée une nouveau fichier
    3) Dans ce nouveau fichier, on rajoute les chemins
"""
#Étape 1
def tous_les_trajets(filemame):
    with open(filename, "r") as file:
        n=int(file.readline())
        l=[]
        for i in range (n):
            traj=list(map(int, file.readline().split()))
            l.append((traj[0],traj[1]))
    return l

def routes_x_out(i):
    #Étape 2
    data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    l=tous_les_trajets(data_path+file_name2)
    f=open(data_path+"routes."+str(i)+".out","w")
    #étape 3
    for j in range (len(l)):
        route=g.min_power(src=l[j][0],dest=l[j][1])
        f.write(route+ "\n")
    return f

#Séance 2 Question 3- Kruskal
"""
    Dans cette question, on veut écrire une fonction kruskal qui prend en entrée un graphe au format de la classe Graph (e.g., g) 
et qui retourne un autre élément de cette classe (e.g., g_mst) correspondant à un arbre couvrant de poids minimal de g.

Pour cela, on va procéder en plusieurs étapes:
    1) On récupère l'ensemble des arêtes du graph.
    2) On ordonne les arêtes du graph de la puissance minimale à la puissance maximale.
    3) find(n,classes) qui renvoie la classe dans laquelle n appartient, dans 'classes'.
    4) On relie la classe de X avec la classe de Y en rajoutant l'arête x-y et en changeant 'classes'.
"""
#Étape 1
def get_edges(g):
    edges = []
    for n1 in g.nodes:
        for n2, p, d in g.graph[n1]:
            edges.append((n1, n2, p))
    return edges

#Étape 3
def find(n,classes):
        if n!=classes[n]:
            classes[n] = find(classes[n],classes)
        return classes[n]

def kruskal(g):
    krusk=Graph(g.nodes)    #Pour avoir un nouveau graphe avec les mêmes noeuds
    tree_edges = []
    #étape 1
    edges = get_edges(g)
    classes = {n:n for n in g.nodes}
    #étape 2
    edges.sort(key=lambda x: x[2])
    for (x, y, z) in edges:
        #S'il y a autant (ou plus) d'arêtes que de noeuds, alors tous les noeuds sont déjà reliés et donc kruskal est fini.
        if krusk.nb_edges >= krusk.nb_nodes-1:
            break
        #étape 3
        reprX = find(x,classes)
        reprY = find(y,classes)
        #si les deux noeuds sont déjà reliés, on ne change rien. S'ils ne le sont pas, on doit faire l'étape 4
        if reprX != reprY:
            #étape 4
            krusk.add_edge(x, y, z)
            classes[reprX] = reprY
    return (krusk)



"""
Calculons la complexité de cette fonction.
"""

#Séance 2 question 5
"""
Dans cette question, on veut écrire la nouvelle fonction min_power qui renvoie la puissance minimale. 
Pour ce type de graph, il faut savoir que si deux noeud sont connectés, alors iles ont forcément un chemin, et celui
çi est unique et possède par conséquent la puissance minimale.
Pour cela, on procède de la même manière que dans la séance 1.
        1) On regarde si src et dest sont connectés:
            - si non, on renvoie qu'il n'y a pas de chemin
            - si oui, on continue le programme
        2) On crée 'a_voir' avec tous les voisins directs de la src (sous format de liste car les éléments de 'a_voir' vont
            être des listes de chemins possibles partant de src).
        3) Tant que 'a_voir' est non vide:
            On se place sur le premier noeud de la liste. 
            On observe ses voisins:
                - si c'est la destination, on renvoie le chemin, ainsi que la puissance
                - si ça ne l'est pas, on va rajouter à 'a_voir' les différents chemins l+[voisin] 
"""
def min_power_bis(self,src,dest):
    #Étape 1
    liste_cc=self.connected_components()
    for l in liste_cc:
        if src in l and not dest in l:
            return None
    a_voir=[]
    for i in range (len(self.graph[src])):
        a_voir.append([self.graph[src][i][1]]+[src]+[self.graph[src][i][0]])
        if self.graph[src][i][0]==dest:
            return (self.graph[src][i][1],[src,dest])
    while a_voir!=[]:
        l=a_voir[0]
        l2=[]
        for i in range (len(self.graph[l[-1]])):
            l2.append((max(l[0],self.graph[l[-1]][i][1]),self.graph[l[-1]][i][0]))
            if self.graph[l[-1]][i][0]==dest:
                return(max(l[0],self.graph[l[-1]][i][1]),l[1:]+[dest]) #Si un chemin existe, il est unique donc a la puissance minimum
        a_voir.remove(l)
        for i in l2:
            l3=l+[i[1]]
            l3[0]=i[0]
            if dest not in l[1:]and (i[1] not in l[1:]):
                if not l3[-1]==dest:
                    a_voir.append(l3)
    return None


#TESTS UNITAIRES POUR LA SÉANCE 2

#pour la question 3
class Test_kruskal(unittest.TestCase):
    def test_network00(self):
        g = graph_from_file(data_path+"network.00.in")
        g_mst = kruskal(g)
        mst_expected = {1: [(8, 0, 1), (2, 11, 1), (6, 12, 1)],
                        2: [(5, 4, 1), (3, 10, 1), (1, 11, 1)],
                        3: [(4, 4, 1), (2, 10, 1)],
                        4: [(10, 4, 1), (3, 4, 1)],
                        5: [(2, 4, 1), (7, 14, 1)],
                        6: [(1, 12, 1)],
                        7: [(5, 14, 1)],
                        8: [(1, 0, 1), (9, 14, 1)],
                        9: [(8, 14, 1)],
                        10: [(4, 4, 1)]}
        self.assertEqual(g_mst.graph, mst_expected)

    def test_network02(self):
        g = graph_from_file(data_path+"network.02.in")
        g_mst = kruskal(g)
        mst_expected = {1: [(4, 4, 1)],
                        2: [(3,4,1)],
                        3: [(4,4,1),(2,4,1)],
                        4: [(3, 4, 1),(1,4,1)],
                        5:[],
                        6:[],
                        7:[],
                        8:[],
                        9:[],
                        10:[]
                        }
        self.assertEqual(g_mst.graph, mst_expected)

#pour la question 5
class Test_s2_q5(unittest.TestCase):
    def test_network2(self):
        g = graph_from_file(data_path+"network.02.in")
        krusk=kruskal(g)
        self.assertEqual(min_power_bis(krusk,2,1),(4, [2,3,4,1]))
        self.assertEqual(min_power_bis(krusk,2, 7), None)

    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        krusk=kruskal(g)
        self.assertEqual(min_power_bis(krusk,1,7), (14,[1,2,5,7]))
        self.assertEqual(min_power_bis(krusk,9,4), (14,[9,8,1,2,3,4]))



"""
SÉANCE 3
"""
 
#récupérer les camions

def camions(filename):
    with open(filename, "r") as file:
        n=int(file.readline())
        l=[]
        for i in range (n):
            puiss,cout=list(map(int, file.readline().split()))
            l.append((puiss,cout))
    return l

"""
On récupère la liste qui possède (puissance,coût) de chaque camion, grâce à 'camions'.
On récupère la liste complète des routes qui veut parcourir, ainsi qu'une liste correspondant à leurs
puissances minimales.







"""

def puissances_minimales_routes(i):
    data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    l=tous_les_trajets(data_path+file_name2)
    résultat=[]
    for i in range (len(n)):
        résultat.append(g.min_power(src=l[j][0],dest=l[j][1]))
    return résultat


"""
SÉANCE 4
"""
début séance 4






















