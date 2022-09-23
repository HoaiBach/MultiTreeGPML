from deap import base, creator, gp
import pygraphviz as pgv


def init_draw(toolbox):
    expr = toolbox.individual()
    nodes, edges, labels = gp.graph(expr)
    ### Graphviz Section ###
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")
