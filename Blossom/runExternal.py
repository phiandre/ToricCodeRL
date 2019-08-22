from os import system
import subprocess

"""
#FNULL = open(os.devnull, 'w'), stdout=FNULL stderr=FNULL,
args = "C:/Users/Philip/Desktop/C++/blossom5-v2.05.src/blossom5.exe -e GRAPH1.TXT -w RES.TXT"
subprocess.call(args, shell=False)
"""

system("blossom5.exe -e TESTGRAPH.TXT -w RES.TXT")


