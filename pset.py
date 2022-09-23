import random
import numpy as np
from helpers import protectedDiv
from deap import gp


def create_pset(num_attr):
    pset = gp.PrimitiveSet("MAIN", num_attr, "IN")
    pset.addPrimitive(np.add, 2, name="vadd")
    pset.addPrimitive(np.subtract, 2, name="vsub")
    pset.addPrimitive(np.multiply, 2, name="vmul")
    pset.addPrimitive(protectedDiv, 2)
    # pset.addPrimitive(np.negative, 1, name="vneg")
    # pset.addPrimitive(np.cos, 1, name="vcos")
    # pset.addPrimitive(np.sin, 1, name="vsin")
    pset.addEphemeralConstant("rand101", lambda: random.random())
    return pset
