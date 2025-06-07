from deap import gp, base, creator, tools
from functools import partial
from enum import Enum, auto
import random
from networkx.drawing.nx_agraph import graphviz_layout


def concat(str1, str2, sep):
    return str(str1)+sep+str(str2)


def branch(left, right):
    return f"{left}({right})"


class NeuronType(Enum):
    N = auto()
    Nu = auto()
    D = auto()
    Thr = auto()
    Rnd = auto()
    Sin = auto()
    Fuzzy = auto()
    Star = "*"


def neuron(neuron_type: NeuronType, propertylist):
    return f"[{neuron_type.value},{propertylist}]"


def neuron_connection(i:int, weight:float):
    return f"{i}:{weight}"


class NeuronProperty(Enum):
    Force = "fo"
    Inertia = "in"
    Sigmoid = "si"


class ModifierUpper(Enum):
    R = auto()  # Rotation (by 45 degrees) – does NOT affect further sticks
    Q = auto()  # Twist
    C = auto()  # Curvedness

    L = auto()  # Length (Physical property)
    F = auto()  # Friction (sticks slide/stick on ground)
    W = auto()  # Weight (in water only)

    A = auto()  # Assimilation = photosynthesis (vertical stick doubles assimilation)
    S = auto()  # Stamina (increases survival chance during fights)
    M = auto()  # Muscle strength (force, speed, stress resistance, energy use)
    #I = auto()  # Ingestion (gain energy from food)

    #E = auto()  # Energy (experimental; modifies starting energy)


class ModifierLower(Enum):
    r = auto()  # Rotation (by 45 degrees) – lowercase variant
    q = auto()  # Twist
    c = auto()  # Curvedness

    l = auto()  # Length
    f = auto()  # Friction
    w = auto()  # Weight

    a = auto()  # Assimilation
    s = auto()  # Stamina
    m = auto()  # Muscle strength
    #i = auto()  # Ingestion

    #e = auto()  # Energy (experimental)

def neuron_property(prop:NeuronProperty, weight:float):
    return f"{prop.value}:{weight}"
    


pset = gp.PrimitiveSetTyped("f1", [], str)
pset.addPrimitive(neuron_connection, [int, float], str, "neuroncon")
pset.addPrimitive(neuron_property, [NeuronProperty, float], str, "neuronprop")
pset.addPrimitive(neuron, [NeuronType, str], str, "neuron")
pset.addPrimitive(partial(concat,set=","), [str, str], str,"com")
pset.addPrimitive(partial(concat,set=""), [str, str], str,"concat")
pset.addPrimitive(branch, [str, str], "branch")
pset.addEphemeralConstant("nint", lambda: random.randint(-20,20), int)
pset.addEphemeralConstant("nfloat", lambda: random.uniform(-10.0,10.0), float)
for nt in NeuronType:
    pset.addTerminal(nt, NeuronType)
for np in NeuronProperty:
    pset.addTerminal(np, NeuronProperty)
for m in ModifierUpper:
    pset.addTerminal(m.name, str)
for m in ModifierLower:
    pset.addTerminal(m.name, str)
# pset.addEphemeralConstant("npt", lambda: random.choice(list(NeuronProperty)), NeuronProperty)
# pset.addEphemeralConstant("nt", lambda: random.choice(list(NeuronType)), NeuronType)
pset.addTerminal("", str,name="empty")
pset.addTerminal("X", str,name="X")
pset.addTerminal(0, int,name="zero")

creator.create("Individual", gp.PrimitiveTree)


def generateF1Frams(pset,min_=1,max_=5,pneuron=0.2,pbranch=0.3,parm=0.4,px=0.7,pmod=0.4):
    # TODO https://www.framsticks.com/a/al_geno_f1.html
    # connections between neurons need to be fixed later as they are relative to each other
    neurons = []
    nodes = {prim.name: prim for prim in pset.primitives[object] + pset.terminals[object]}

    def genbranch():
        ...

toolbox = base.Toolbox()
toolbox.register("expr", generateF1Frams, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

fram = toolbox.individual()

print(list(fram))
print(gp.compile(fram,pset))

import matplotlib.pyplot as plt
import networkx as nx

nodes, edges, labels = gp.graph(fram)
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = graphviz_layout(g, prog="dot")

nx.draw_networkx_nodes(g, pos)
nx.draw_networkx_edges(g, pos)
nx.draw_networkx_labels(g, pos, labels)
plt.show()