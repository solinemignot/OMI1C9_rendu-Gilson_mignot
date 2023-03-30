import sys 
sys.path.append("delivery_network/")

import unittest 
from graph import Graph, graph_from_file,Test_q1,Test_q2,Test_q3,Test_q5,Test_q6,kruskal,Test_kruskal,min_power_bis,Test_s2_q5,question1_séance2

data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"
file_name1 = "network.3.in"
file_name2="routes.3.in"
g=graph_from_file(data_path+file_name1)
krusk=kruskal(g)
print(krusk)