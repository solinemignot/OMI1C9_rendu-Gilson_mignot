import sys 
sys.path.append("delivery_network/")
import unittest 

from graph import Graph, question1_séance2, graph_from_file,min_power,parentalité
from graph import Test_q1,Test_q2,Test_q3,Test_q5,Test_q6


data_path = "/home/onyxia/work/OMI1C9_rendu-interm-diaire/input/"
g = graph_from_file(data_path+"network.00.in")
print(min_power(g,9,4))