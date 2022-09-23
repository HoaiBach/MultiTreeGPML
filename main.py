import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from pset import create_pset
from helpers import test_score, Sigmoid, multiCrossover, multiMutation
from skmultilearn.dataset import load_dataset
from sklearn import metrics
import multiprocessing
import eaElitism

random.seed(43)
np.random.seed(43)

toolbox = base.Toolbox()


def evaluation_pipeline(hof, x_test, y_test, f):
    bestInd = hof[0]
    f.write('Loss of the best individual %.5f\n' % bestInd.fitness.values[0])
    funcs = [toolbox.compile(expr=subindividual) for subindividual in bestInd]
    outputs = [[func(*sample) for sample in x_test] for func in funcs]
    outputs = np.array(outputs).T
    outputs[outputs > 0] = 1
    outputs[outputs <= 0] = 0
    hamming_loss, f1, acc = test_score(y_test, outputs)
    f.write('Loss: %.5f, f1-score: %.5f, acc: %.5f\n' % (hamming_loss, f1, acc))


def evalMultilabel(individual, x_train, y_train):
    # Transform the tree expression in a callable function
    funcs = [toolbox.compile(expr=subindividual) for subindividual in individual]
    outputs = [[func(*sample) for sample in x_train] for func in funcs]
    outputs = np.array(outputs).T
    outputs[outputs > 0] = 1
    outputs[outputs <= 0] = 0
    loss = metrics.hamming_loss(y_train, outputs)
    return loss,


def main():
    hof = tools.HallOfFame(5)
    pop = toolbox.population(n=512)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # pop, log = algorithms.eaSimple(
    #     pop, toolbox, 0.8, 0.2, 50, stats, halloffame=hof
    # )

    pop, log = eaElitism.eaSimple(
        pop, toolbox, 0.8, 0.2, 50, stats, halloffame=hof
    )

    evaluation_pipeline(hof, X_test, y_test, f)


if __name__ == "__main__":
    datasets = ['birds',
                'emotions',
                'enron',
                'genbase',
                'medical',
                'yeast',
                'scene',
                'rcv1subset1',
                'tmc2007_500']
    # datasets = ['emotions']

    f = open('Results.txt', 'w')
    for dataset in datasets:
        # initialise
        f.write(dataset + '\n')
        print(dataset)
        X_train, y_train, feature_names, label_names = load_dataset(dataset, "train")
        X_test, y_test, _, _ = load_dataset(dataset, "test")
        X_train = X_train.toarray()
        y_train = y_train.toarray()

        X_test = X_test.toarray()
        y_test = y_test.toarray()
        num_attr = X_train.shape[1]

        pset = create_pset(num_attr)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("SubIndividual", gp.PrimitiveTree, pset=pset)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
        toolbox.register("subIndividual", tools.initIterate, creator.SubIndividual, toolbox.expr)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.subIndividual, y_train.shape[1])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", evalMultilabel, x_train=X_train, y_train=y_train)
        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("mate", multiCrossover)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("sub_mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.register("mutate", multiMutation, toolbox=toolbox)

        # Process Pool of 4 workers
        pool = multiprocessing.Pool(processes=8)
        toolbox.register("map", pool.map)

        main()

    f.close()

