"""
Début du code.
"""

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
    """
    Calculons la complexité de ces fonctions.
    Pour connected_components: 
        Pour chaque noeud, on parcourt uniquement les noeuds qui sont lui sont connectés. 
        Donc la complexité de connected_components est en O(noeud*(nb de noeuds connectés))=O(V*E) car, dans le pire des cas,
        pour chaque noeud on doit parcourir l'integralité des arêtes. 
    Pour connected_components_set, on doit calculer la liste connected_components, et enlever les doublons donc la complexité
    est au pire (si tous les noeuds sont indépendants ) en O(V^2 *E).
    """
  
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

    Complexité:
        Pour chaque path, on regarde au plus une fois chaque arête (dans le pire, les deux noeuds sont opposés dans le graphe)
        donc la complexité est en O(E).
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
    def min_dist_with_power(self, src, dest,power):
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
La fonction 'parentalité':
    1) nous créons le dictionnaire pere_dict qui recense la parenté des noeuds grâce au triplet (père, hauteur,puissance minimale)
        Pour chaque noeud, on initialise à père=noeud et hauteur=0 et puissance=0
    2) Grâce à la fonction 'recherche_fils', pour chaque triplet des voisins du père (trouvés dans la liste g.graph[pere]),
        si ce n'est pas le grand_père, on dit que le père est le père et on rajoute la hauteur. Récursivement, on en fait de même pour les fils
Complexité: On parcourt chaque noeud et chaque arête une unique fois, donc la complexité est en O(E*V).

La fonction 'min_power':
    1) on vérifie qu'ils sont dans la même composante
    2) On crée le dictionnaire de parentalité 
    3) on crée la fonction récursive 'trajet' qui renvoie (puissance minimale du trajet, le trajet).


Ces fonctions marchent uniquement quand les graphes n'ont pas de cercles. (ex: quand ils sont kurskalisés)
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

"""
La complexité de cet algorithme (dans le pire des cas) est O(V^3∗log(V)). C'est trop important pour fonctionner sur des gros graphiques.
Il faut chercher une optimisation.
"""
        
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
data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"

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
        self.assertEqual(g.min_dist_with_power(3, 1,11),(2, [3, 2, 1]))
        self.assertEqual(g.min_dist_with_power(2, 7,11), None)

    def test_network4(self):
        g = graph_from_file(data_path+"network.04.in")
        self.assertEqual(g.min_dist_with_power(2, 1,11), (11,[2, 3, 4, 1]))
        self.assertEqual(g.min_dist_with_power(2, 7,11), None)

#pour la question 6
class Test_q6(unittest.TestCase):
    def test_network2(self):
        g = graph_from_file(data_path+"network.01.in")
        self.assertEqual(min_power(g,2,1),(1, [2,1]))
        self.assertEqual(min_power(g,2, 7), None)

    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        self.assertEqual(min_power(g,1,7), (14, [1, 2, 5, 7]))
        self.assertEqual(min_power(g,9,4), (14,[9, 8, 1, 2, 3, 4]))

"""
SÉANCE 2
"""

# Séance 2 question 10
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
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"
    tps=0
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    n,l=trajets(data_path+file_name2)
    #étape 2
    for j in range (len(l)):
        print(j)
        a=time.time()
        min_power(g,src=l[j][0],dest=l[j][1])
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
def tous_les_trajets(i):
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"
    file_name2="routes."+str(i)+".in"
    filename=data_path+file_name2
    with open(filename, "r") as file:
        n=int(file.readline())
        l=[]
        for i in range (n):
            traj=list(map(int, file.readline().split()))
            l.append((traj[0],traj[1],traj[2]))
    return l

def routes_x_out(i):
    #Étape 2
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    l=tous_les_trajets(i)
    f=open(data_path+"routes."+str(i)+".out","w")
    #étape 3
    for j in range (len(l)):
        route=min_power(g,src=l[j][0],dest=l[j][1])
        f.write(route+ "\n")
    return f


"""
Malheureusement, on n'a pas de réponse sur le temps pour calculer min_power sur des gros graphes.
La complexité de min_power est trop grande
"""

#Séance 2 Question 12 - Kruskal
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
Calculons la complexité de cette fonction : c'est O(Elog(E)) (cf rapport)
"""

#Séance 2 question 14
"""
La fonction 'parentalité':
    1) nous créons le dictionnaire pere_dict qui recense la parenté des noeuds grâce au triplet (père, hauteur,puissance minimale)
        Pour chaque noeud, on initialise à père=noeud et hauteur=0 et puissance=0
    2) Grâce à la fonction 'recherche_fils', pour chaque triplet des voisins du père (trouvés dans la liste g.graph[pere]),
        si ce n'est pas le grand_père, on dit que le père est le père et on rajoute la hauteur. Récursivement, on en fait de même pour les fils
Complexité: On parcourt chaque noeud et chaque arête une unique fois, donc la complexité est en O(E*V).

La fonction 'min_power':
    1) on vérifie qu'ils sont dans la même composante
    2) On crée le dictionnaire de parentalité 
    3) on crée la fonction récursive 'trajet' qui renvoie (puissance minimale du trajet, le trajet).


Ces fonctions marchent uniquement quand les graphes n'ont pas de cercles. (ex: quand ils sont kurskalisés)
    """
    
def parentalité(g):
    pere_dict = {noeud:(noeud, 0, 0) for noeud in g.nodes} # couple (père, hauteur) 
    def recherche_fils(pere, grandpere, hauteur):
        for fils, power_min, dist in g.graph[pere]:
            if fils != grandpere:
                pere_dict[fils] = (pere, hauteur, power_min)
                recherche_fils(fils, pere, hauteur+1)
    recherche_fils(g.nodes[0], g.nodes[0], 1)
    return pere_dict # renvoie un dictionnaire contenant les informations sur les parents de chaque nœud dans l'arbre

def min_power_bis(src, dest,g): # algorithme de parcours de l'arbre de couverture minimale (MST)
    pere=parentalité(g)
    def trajet(node1, node2): # calcule le trajet le plus court entre deux nœuds node1 et node2
        if node1 == node2:
            return (0,[node1])
        pere1, h1, p1 = pere[node1]
        pere2, h2, p2 = pere[node2]
        if h1 == h2:
            if pere1==pere2: # si les deux nœuds ont le même parent
                return (max(p1, p2),[node1,pere1,node2]) # la connexion directe entre les deux nœuds.
            l=trajet(pere1, pere2)
            return (max(p1, p2, l[0]),[node1]+l[1]+[node2])
        if h1 < h2: # si le parent de node1 est plus haut dans l'arbre que celui de node2
            l=trajet(pere2, node1) # on appelle récursivement trajet avec le parent de node2 et node1
            return (max(p2, l[0]),[node2]+l[1])
        if h1 > h2:
            l=trajet(pere1, node2)
            return (max(p1, l[0]),[node1]+l[1])
    return trajet(src, dest)

#Renvoie uniquement la puissance minimale
def min_power_Trois(src, dest,g):
    pere=parentalité(g)
    def trajet(node1, node2):
        if node1 == node2:
            return 0
        pere1, h1, p1 = pere[node1]
        pere2, h2, p2 = pere[node2]
        if h1 == h2:
            return max(p1, p2, trajet(pere1, pere2))
        if h1 < h2:
            return max(p2, trajet(pere2, node1))
        if h1 > h2:
            return max(p1, trajet(pere1, node2))
    return trajet(src, dest)

#Séance 2 question 15
""" 
Analysons la compléxité de cette fonction 
En comparant avec la question 10, on voit qu'avant, la complexité de min_power était beaucoup trop élevé donc on n'avait pas de résultat.
Ici, le programme est optimisé et donc, nous avons des résultats rapides. 
La nouvelle complexité est donc beaucoup plus faible : c'est O(V^2)

"""

def question15_séance2(i):
    #Étape 1
    import time
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"
    tps=0
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    krusk=kruskal(g)
    n,l=trajets(data_path+file_name2)
    #étape 2
    for j in range (len(l)):
        a=time.time()
        min_power_bis(l[j][0],l[j][1],krusk)
        b=time.time()
        tps+=b-a
    #étape 3
    return (tps)/len(l)*n

"""
On peut exécuter pour trouver un temps (raisonnable!)
question15_séance2(9) = 190 secondes
"""

#TESTS UNITAIRES POUR LA SÉANCE 2

#Pour la question 12 (Kruskal)
class Test_kruskal(unittest.TestCase):
    def test_network00(self):
        g = graph_from_file(data_path+"network.00.in")
        g_mst = kruskal(g)
        mst_expected = {1: [(8, 0, 1), (2, 11, 1), (6, 12, 1)],
                        2: [(5, 4, 1), (3, 10, 1), (1, 11, 1)],
                        3: [(4, 4, 1), (2, 10, 1)],
                        4: [(3, 4, 1),(10, 4, 1)],
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
                        3: [(2,4,1),(4,4,1)],
                        4: [(1,4,1),(3, 4, 1)],
                        5:[],
                        6:[],
                        7:[],
                        8:[],
                        9:[],
                        10:[]
                        }
        self.assertEqual(g_mst.graph, mst_expected)

#pour la question 14
class Test_s2_q5(unittest.TestCase):
    def test_network1(self):
        g = graph_from_file(data_path+"network.01.in")
        krusk=kruskal(g)
        self.assertEqual(min_power_bis(2,1,krusk),(1,[2,1]))
        self.assertEqual(min_power_bis(3,1,krusk), (1,[3,2,1]))

    def test_network0(self):
        g = graph_from_file(data_path+"network.00.in")
        krusk=kruskal(g)
        self.assertEqual(min_power_bis(1,7,krusk), (14,[7, 5, 2, 1]))
        self.assertEqual(min_power_bis(9,4,krusk), (14,[4, 3, 2, 1, 8, 9]))



"""
SÉANCE 4
"""
# Séance 4 question 18 

#récupérer les camions
def camions(i):
    filename="/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"+"trucks."+str(i)+".in"
    with open(filename, "r") as file:
        n=int(file.readline())
        l=[]
        for i in range (n):
            puiss,cout=list(map(int, file.readline().split()))
            l.append((i,puiss,cout))
    return l

"""
On récupère la liste qui possède (puissance,coût) de chaque camion, grâce à 'camions'.
On récupère la liste complète des routes qui veut parcourir, ainsi qu'une liste correspondant à leurs
puissances minimales.
"""

def puissances_minimales_routes(i):
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/"
    file_name1 = "network."+str(i)+".in"
    file_name2="routes."+str(i)+".in"
    g = graph_from_file(data_path + file_name1)
    krusk=kruskal(g)
    l=tous_les_trajets(i)
    résultat=[]
    for j in range (len(l)):
        puiss,traj=min_power_bis(l[j][0],l[j][1],krusk)
        résultat.append((puiss,l[j][2]))   #(min_power de la route, gain rapporté par la route)
    return résultat 

# Séance 4 question 18 

"""
Dans cette fonction, on cherche à maximiser la somme des profits obtenus sur tous les trajets couverts.
Pour ce faire, on veut que la fonction retourne une collection de camions à acheter ainsi que leurs affectations sur des trajets 
en optimisant le budget et maximisant les profits qui en découlent.

Nous allons voir plusieurs approches:
- l'approche brute qui est beaucoup trop longue donc ne marche que pour de petits graphes, et petit nombre de routes
- l'approche glouton

"""

#Approche brute

def truck_choice(route, trucks):
    power = g.min_power(route[0], route[1])[0]
    if trucks[0][0]<power:    # Le camion ne peut pas prendre la route 
        return None
    i=0
    while i<len(trucks) and trucks[i][0]>=power:
        i+=1
    if i==len(trucks):
        return trucks[-1]
    return trucks[i]. # Le camion le plus adapté 

def approche_brute(i_routes,i_camion,contrainte):
    import itertools
    l_routes=tous_les_trajets(i_routes)
    l_camions=camions(i_camion)
    l_camions.sort(key=lambda x:x[1],reverse=True) # trie les camions en fonction de leur coût ordre croissant
    ens_comb_routes = []        
    ens_comb_routes_1=[]
    #on fait la liste des combinaisons de routes possibles
    for i in range(1,1+len(l_routes)):
        print(i)
        for j in itertools.combinations(l_routes, i):
            ens_comb_routes_1.append(list(j))
    #on remplace pour que 'ens_comb_routes' soit une liste de (liste de route, profit de la liste)
    for list_routes in ens_comb_routes_1:
        profit=0
        for route in list_routes:
            profit+=route[2]
        ens_comb_routes.append([list_routes, profit])
    ens_comb_routes = sorted(ens_comb_routes, key=lambda x:x[1], reverse = True)    
    # On trie l'ensemble des comb de route en fonction du profit ordre décroissant
    i=0     # variable de comptage sur ens_comb_routes
    for comb_routes in ens_comb_routes:
        #on regarde si on peut avoir des camions pour ces routes 
        choosen_trucks=[]
        cost=0
        comb_routes_1=comb_routes[0]
        for route in comb_routes_1:
            truck=truck_choice([route[0],route[1]],l_camions)
            if truck !=None:
                cost+=truck[0]
                choosen_trucks.append(truck)
            else:
                cost=contrainte+1
        #dès que le coût est en dessous du budget, on peut se permettre d'acheter les camions et donc on garde cette combinaison
        if cost<=contrainte:
            return comb_routes, choosen_trucks, cost
    return None
"""
Complexité exponentielle : impossible à appliquer pour de gros graphes
"""

#Algorithme glouton

def fonction_aux_glouton(i):
    data_path = "/home/onyxia/work/OMI1C9_rendu-Gilson_mignot/input/" # chemin
    file_name1 = "network."+str(i)+".in"                              # fichier
    g = graph_from_file(data_path + file_name1)
    krusk=kruskal(g)
    l=tous_les_trajets(i)
    puissances=[]
    efficacite=[]
    for j in range (len(l)):
        puiss=min_power_bis(l[j][0],l[j][1],krusk) # puissance minimale pour parcourir
        puissances.append(puiss[0]). # On stocke toutes les puissances 
    for i in range (len(l)):
        power=puissances[i]
        src,dest,gain=l[i]
        efficacite.append([gain/(power+0.1),power,gain,i+1]) # On calcule l'efficacité en calculant le gain divisé par la puissance minimale
    return efficacite #retourne une liste 

def glouton(i_network,i_camion,B):
    #on récupère la liste efficacité créée grâce à la fonction_aux_glouton, qu'on trie par ordre croissant en fonction de gain/(power+0.1)
    efficacite=fonction_aux_glouton(i_network)
    efficacite.sort(key= lambda x:x[0]) # On trie 
    B_dep=0
    gain_tot=0
    l_camions=camions(i_camion)
    bon_camion=[]. 
    for trajet in efficacite:
        puiss=trajet[1]
        indice_camion=-1
        cout_camion=trajet[2]
        premier=True
        for j in range(len(l_camions)):
            if l_camions[j][0]<puiss:
                continue
            else:
                if premier or l_camions[j][1]<cout_camion:
                    premier=False
                    indice_camion=j
                    cout_camion=l_camions[j][1]
        bon_camion.append([indice_camion,cout_camion]) # indice l'indice du camion qui permet de parcourir le chemin en 
                                                       # respectant la capacité maximale B et avec un coût minimal cout_camion
    i=0
    camion_et_trajet={trajet[3]: {} for trajet in efficacite} # dictionnaire  qui contient l'indice du camion qui a été utilisé pour parcourir le chemin en question
    while len(bon_camion)>i and B_dep+bon_camion[i][1]<B:
        if bon_camion[i][0] not in camion_et_trajet[efficacite[i][3]].keys():
            camion_et_trajet[efficacite[i][3]][bon_camion[i][0]]=1
        else:
            camion_et_trajet[efficacite[i][3]][bon_camion[i][0]]+=1
        gain_tot+=efficacite[i][2]
        B_dep+=bon_camion[i][1]
        i+=1
    return gain_tot,camion_et_trajet
""" 
Complexité beaucoup plus raisonnable, résultats rapides et cohérents
"""
"""
FIN
"""
