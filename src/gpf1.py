from deap import gp
import re
import random
from functools import partial
from enum import Enum, auto


def simple_parser(geno: str, in_paranthesis=False):
    if len(geno) == 0:
        return ["end"]
    if in_paranthesis:
        depth = 0
        j = -1
        for i, c in enumerate(geno):
            if c == "(" or c == "[":
                depth += 1
            if c == ")" or c == "]":
                depth -= 1
            if c == "," and depth == 0:
                j = i
                break
        if j != -1:
            return ["comma"] + simple_parser(geno[j+1:],True) + simple_parser(geno[:j])
    this = geno[0]
    if this == "(":
        return ["branch"] + simple_parser(geno[1:-1],True)

    if this == "[": #handle
        #match = re.search(r'\[[^\[\]]*?\](?!\[)', geno)
        i = geno.find("]")
        type, proplist = parse_neuron(geno[1:i])
        return ["neuron"] + ["nt"+type] + proplist + simple_parser(geno[i+1:])
    return [this] + simple_parser(geno[1:])


special_neurons = {"@": "Twist", "|": "Hinge", "*": "Star"}


def parse_neuron(neuron_inside: str | list[str], is_first=True):
    if len(neuron_inside) == 0:
        return ["end"]
    if isinstance(neuron_inside, str):
        neuron_inside = neuron_inside.split(",")
    neuron_type = None
    i = 0
    if ":" not in neuron_inside[0]:
        neuron_type = neuron_inside[0]
        i = 1
    if is_first:
        neuron_type = "ntN" if not neuron_type else neuron_type
        neuron_type = special_neurons.get(neuron_type, neuron_type)
        return neuron_type, parse_neuron(neuron_inside[i:], is_first=False)

    first, second = neuron_inside[0].split(":")
    if re.match(r'^-?\d+$', first):
        thing = ["conn", int(first), float(second)]
    else:
        thing = ["prop", first, float(second)]

    return thing + parse_neuron(neuron_inside[1:], is_first=False)


# partial(lambda s: s, symbol)
# raises RuntimeWarning: Ephemeral nint function cannot be pickled
# so without lambdas:
def identity(x):
    return x

def parse(geno:str, pset):
    nodes = {prim.name: prim for prim in pset.primitives[str] + pset.terminals[str] + pset.terminals[NeuronProperty] + pset.terminals[NeuronType]}
    #print(pset.primitives)
    parsed = simple_parser(geno)
    #print(parsed)
    def map_to(symbol):
        if isinstance(symbol, int):
            return gp.MetaEphemeral("nint", partial(identity, symbol))()
            # return gp.MetaEphemeral("nint", partial(lambda s: s, symbol))()
        elif isinstance(symbol, float):
            return gp.MetaEphemeral("nfloat", partial(identity, symbol))()
            # return gp.MetaEphemeral("nfloat", partial(lambda s: s, symbol))()
        return nodes.get(symbol, None)

    #print(geno)
    mapped = [map_to(symbol) for symbol in parsed]
    mapped = [s for s in mapped if s is not None]
    return mapped


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class NeuronType(AutoName):
    N = auto()
    Nu = auto()
    D = auto()
    Thr = auto()
    Rnd = auto()
    Sin = auto()
    Fuzzy = auto()
    Star = "*"
    Twist = "@"
    Hinge = "|"
    Gpart = auto()
    S = auto()
    T = auto()
    G = auto()


class NeuronProperty(Enum):
    N_Force = "fo"
    N_Inertia = "in"
    N_Sigmoid = "si"
    Thr_Low = "lo"
    Thr_High = "hi"
    Thr_Threshold_Sin_Time = "t"
    Sin_Frequency = "f0"
    Gpart_ry = "ry"
    Gpart_rz = "rz"
    Twist_p = "p"


class ModifierUpper(AutoName):
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


class ModifierLower(AutoName):
    r = auto()  # Rotation (by 45 degrees) – lowercase variant
    q = auto()  # Twist
    c = auto()  # Curvedness

    l = auto()  # Length  # noqa: E741  Ambiguous variable name: `l`
    f = auto()  # Friction
    w = auto()  # Weight

    a = auto()  # Assimilation
    s = auto()  # Stamina
    m = auto()  # Muscle strength
    #i = auto()  # Ingestion

    #e = auto()  # Energy (experimental)


def add_neuron(type: NeuronType, properties: str, next_: str):
    return f"[{type.value}{properties}]" + next_


def add_neuron_connection(i: int, weight: float, next_: str):
    return f",{i}:{weight}{next_}"


def add_neuron_property(prop: NeuronProperty, value: float, next_: str):
    return f",{prop.value}:{value}{next_}"


def concat(right, left, sep):
    return str(left)+sep+str(right)


def branch(inside):
    return f"({inside})"


def create_f1_pset():
    pset = gp.PrimitiveSetTyped("f1", [], str)
    pset.addPrimitive(add_neuron_connection, [int, float, str], str, "conn")
    pset.addPrimitive(add_neuron_property, [NeuronProperty, float, str], str, "prop")
    pset.addPrimitive(add_neuron, [NeuronType, str, str], str, "neuron")
    pset.addPrimitive(partial(concat, sep=","), [str, str], str, "comma")
    pset.addPrimitive(branch, [str], str, "branch")

    # pset.addEphemeralConstant("nint", lambda: random.randint(-20, 20), int)
    # pset.addEphemeralConstant("nfloat", lambda: random.uniform(-10.0, 10.0), float)
    def nint_random():
        return random.randint(-20, 20)

    def nfloat_random():
        return random.uniform(-10.0, 10.0)

    pset.addEphemeralConstant("nint", nint_random, int)
    pset.addEphemeralConstant("nfloat", nfloat_random, float)
    for nt in NeuronType:
        pset.addTerminal(nt, NeuronType, name="nt"+nt.name)
    for np in NeuronProperty:
        pset.addTerminal(np, NeuronProperty, name=np.value)
    for m in list(ModifierUpper) + list(ModifierLower):
        pset.addPrimitive(partial(concat, left=m.value, sep=""), [str], str, m.name)
    # pset.addEphemeralConstant("npt", lambda: random.choice(list(NeuronProperty)), NeuronProperty)
    # pset.addEphemeralConstant("nt", lambda: random.choice(list(NeuronType)), NeuronType)
    pset.addTerminal("", str, name="end")
    pset.addPrimitive(partial(concat, left="X", sep=""), [str], str, "X")

    return pset
