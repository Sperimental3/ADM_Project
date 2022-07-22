# containing the standard MILP model for the fully connected network, it returns the adversarial instance that wants
# to fool the network

import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
import timeit


def FC_MILP(weights, biases, sample, sample_label, gray):

    # print(weight)
    # print(bias)

    print("Preparing the MILP model...")

    sample = torch.flatten(sample)
    sample = sample.tolist()

    if gray:
        input = 1 * 100 * 100
    else:
        input = 3 * 100 * 100

    m = gp.Model("Adversarial_MILP")

    # first variables that at the end of the optimization will contain the adversarial instance
    # capable of fool the network in prediction
    x_0 = m.addVars(input, name="x_0", ub=1, lb=0, vtype=GRB.CONTINUOUS)

    x = m.addVars(2, 32, name="x", ub=100, vtype=GRB.CONTINUOUS)
    s = m.addVars(2, 32, name="s", vtype=GRB.CONTINUOUS)
    z = m.addVars(2, 32, name="z", vtype=GRB.BINARY)

    # the final variable has not s and z associated because a relu is not applied at the end of the network
    out = m.addVar(name="x_K", lb=-100, ub=100, vtype=GRB.CONTINUOUS)

    # add another set of variable for the distance between the sample and x_0
    d = m.addVars(input, name="d", vtype=GRB.CONTINUOUS)

    # minimize the distance between each pixel of the the input image that I provide and x_0 generated
    # from the optimization of the model
    m.setObjective(gp.quicksum(x[i, j] for i in range(2) for j in range(32))
                   + gp.quicksum(z[i, j] for i in range(2) for j in range(32))
                   + gp.quicksum(d[j] for j in range(input)), GRB.MINIMIZE)

    for j in range(input):
        m.addConstr(-d[j] <= (sample[j] - x_0[j]))
        m.addConstr((sample[j] - x_0[j]) <= d[j])

        # add constraints on some limitations on pixel distances
        m.addConstr(d[j] <= 0.2)

    # two layers with 32 neurons each, but each of them has a very different input dimension
    for j in range(32):
        # j is a neuron in layer k
        m.addConstr(gp.quicksum(x_0[i] * weights[0][j, i] for i in range(input)) + biases[0][j] == x[0, j] - s[0, j])
        m.addGenConstrIndicator(z[0, j], True, x[0, j] <= 0)
        m.addGenConstrIndicator(z[0, j], False, s[0, j] <= 0)

    for j in range(32):
        m.addConstr(gp.quicksum(x[0, i] * weights[1][j, i] for i in range(32)) + biases[1][j] == x[1, j] - s[1, j])
        m.addGenConstrIndicator(z[1, j], True, x[1, j] <= 0)
        m.addGenConstrIndicator(z[1, j], False, s[1, j] <= 0)

    # last layer: a single neuron with index j equal to 0
    m.addConstr(gp.quicksum(x[1, i] * weights[2][0, i] for i in range(32)) + biases[2][0] == out)

    # add constraint on the fact that the misclassification has to happen,
    # so if starting label is 1 (dog), try to output between 0 and 0.5 (cat), and vice versa
    if sample_label == 1:
        m.addConstr(out <= 0.5)
    else:
        m.addConstr(out >= 0.5001)

    print("Starting the optimization...")

    start = timeit.default_timer()

    m.optimize()

    stop = timeit.default_timer()

    print(f"Optimization finished in: {stop - start} s.")

    generated_sample = []

    for i in range(input):
        generated_sample.append(x_0[i].x)

    generated_sample = torch.tensor(generated_sample)

    if gray:
        unflatten = nn.Unflatten(0, (1, 100, 100))
    else:
        unflatten = nn.Unflatten(0, (3, 100, 100))

    return unflatten(generated_sample)
