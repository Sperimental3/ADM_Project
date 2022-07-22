import torch
import gurobipy as gp
from gurobipy import GRB
import timeit


def CNN_MILP(weights, biases, sample, sample_label):

    # print(weights[0][0, 0, 0, 2])
    # print(biases)

    print("Preparing the MILP model...")

    sample = torch.squeeze(sample)
    sample = sample.tolist()

    m = gp.Model("Adversarial_MILP_CNN")

    # standard variables 100 * 100, plus two for zero padding needed, so we are talking about the border of the image,
    # this is true also for all the other layer, zero padding is always present in each layer
    x_0 = m.addVars(102, 102, name="x_0", ub=1, lb=0, vtype=GRB.CONTINUOUS)

    for i in range(102):
        for j in range(102):
            if i == 0 or j == 0 or i == 101 or j == 101:
                m.addConstr(x_0[i, j] == 0)

    x_1 = m.addVars(2, 32, 102, 102, name="x_1", ub=100, vtype=GRB.CONTINUOUS)
    s_1 = m.addVars(2, 32, 102, 102, name="s_1", vtype=GRB.CONTINUOUS)
    z_1 = m.addVars(2, 32, 102, 102, name="z_1", vtype=GRB.BINARY)

    for c in range(2):
        for k in range(32):
            for i in range(102):
                for j in range(102):
                    if i == 0 or j == 0 or i == 101 or j == 101:
                        m.addConstr(x_1[c, k, i, j] == 0)
                        m.addConstr(s_1[c, k, i, j] == 0)
                        m.addConstr(z_1[c, k, i, j] == 0)

    x_pool_1 = m.addVars(32, 52, 52, name="x_pool_1", ub=100, vtype=GRB.CONTINUOUS)
    z_pool_1 = m.addVars(32, 102, 102, name="z_pool_1", vtype=GRB.BINARY)

    for k in range(32):
        for i in range(52):
            for j in range(52):
                if i == 0 or j == 0 or i == 51 or j == 51:
                    m.addConstr(x_pool_1[k, i, j] == 0)

    x_2 = m.addVars(2, 32, 52, 52, name="x_2", ub=100, vtype=GRB.CONTINUOUS)
    s_2 = m.addVars(2, 32, 52, 52, name="s_2", vtype=GRB.CONTINUOUS)
    z_2 = m.addVars(2, 32, 52, 52, name="z_2", vtype=GRB.BINARY)

    for c in range(2):
        for k in range(32):
            for i in range(52):
                for j in range(52):
                    if i == 0 or j == 0 or i == 51 or j == 51:
                        m.addConstr(x_2[c, k, i, j] == 0)
                        m.addConstr(s_2[c, k, i, j] == 0)
                        m.addConstr(z_2[c, k, i, j] == 0)

    x_pool_2 = m.addVars(32, 27, 27, name="x_pool_2", ub=100, vtype=GRB.CONTINUOUS)
    z_pool_2 = m.addVars(32, 52, 52, name="z_pool_2", vtype=GRB.BINARY)

    for k in range(32):
        for i in range(27):
            for j in range(27):
                if i == 0 or j == 0 or i == 26 or j == 26:
                    m.addConstr(x_pool_2[k, i, j] == 0)

    x_3 = m.addVars(2, 64, 27, 27, name="x_3", ub=100, vtype=GRB.CONTINUOUS)
    s_3 = m.addVars(2, 64, 27, 27, name="s_3", vtype=GRB.CONTINUOUS)
    z_3 = m.addVars(2, 64, 27, 27, name="z_3", vtype=GRB.BINARY)

    for c in range(2):
        for k in range(64):
            for i in range(27):
                for j in range(27):
                    if i == 0 or j == 0 or i == 26 or j == 26:
                        m.addConstr(x_3[c, k, i, j] == 0)
                        m.addConstr(s_3[c, k, i, j] == 0)
                        m.addConstr(z_3[c, k, i, j] == 0)

    x_pool_3 = m.addVars(64, 14, 14, name="x_pool_3", ub=100, vtype=GRB.CONTINUOUS)
    z_pool_3 = m.addVars(64, 27, 27, name="z_pool_3", vtype=GRB.BINARY)

    for k in range(64):
        for i in range(14):
            for j in range(14):
                if i == 0 or j == 0 or i == 13 or j == 13:
                    m.addConstr(x_pool_3[k, i, j] == 0)

    x_4 = m.addVars(2, 128, 14, 14, name="x_4", ub=100, vtype=GRB.CONTINUOUS)
    s_4 = m.addVars(2, 128, 14, 14, name="s_4", vtype=GRB.CONTINUOUS)
    z_4 = m.addVars(2, 128, 14, 14, name="z_4", vtype=GRB.BINARY)

    for c in range(2):
        for k in range(128):
            for i in range(14):
                for j in range(14):
                    if i == 0 or j == 0 or i == 13 or j == 13:
                        m.addConstr(x_4[c, k, i, j] == 0)
                        m.addConstr(s_4[c, k, i, j] == 0)
                        m.addConstr(z_4[c, k, i, j] == 0)

    # for these last variables of the CNN the padding for the convolution is not needed,
    # because there is not a next convolution
    x_pool_4 = m.addVars(128, 6, 6, name="x_pool_4", ub=100, vtype=GRB.CONTINUOUS)
    z_pool_4 = m.addVars(128, 14, 14, name="z_pool_4", vtype=GRB.BINARY)

    out = m.addVar(name="x_K", lb=-100, ub=100, vtype=GRB.CONTINUOUS)

    # add another set of variable for the distance between the sample and x_0
    d = m.addVars(100, 100, name="d", vtype=GRB.CONTINUOUS)

    # minimize the distance between each pixel of the the input image that I provide and x_0 generated
    # from the optimization of the model, minimization also of x variables and z variables
    m.setObjective(gp.quicksum(
        x_1[c, k, i, j] for c in range(2) for k in range(32) for i in range(1, 101) for j in range(1, 101))
                   + gp.quicksum(
        z_1[c, k, i, j] for c in range(2) for k in range(32) for i in range(1, 101) for j in range(1, 101))
                   + gp.quicksum(
        x_2[c, k, i, j] for c in range(2) for k in range(32) for i in range(1, 51) for j in range(1, 51))
                   + gp.quicksum(
        z_2[c, k, i, j] for c in range(2) for k in range(32) for i in range(1, 51) for j in range(1, 51))
                   + gp.quicksum(
        x_3[c, k, i, j] for c in range(2) for k in range(64) for i in range(1, 26) for j in range(1, 26))
                   + gp.quicksum(
        z_3[c, k, i, j] for c in range(2) for k in range(64) for i in range(1, 26) for j in range(1, 26))
                   + gp.quicksum(
        x_4[c, k, i, j] for c in range(2) for k in range(128) for i in range(1, 13) for j in range(1, 13))
                   + gp.quicksum(
        z_4[c, k, i, j] for c in range(2) for k in range(128) for i in range(1, 13) for j in range(1, 13))
                   + gp.quicksum(d[i, j] for i in range(100) for j in range(100)), GRB.MINIMIZE)

    # constraints for the distance between x_0 and the sample
    for i in range(100):
        for j in range(100):
            m.addConstr(-d[i, j] <= (sample[i][j] - x_0[i + 1, j + 1]))
            m.addConstr((sample[i][j] - x_0[i + 1, j + 1]) <= d[i, j])

    print("End of adding variables to the model. Start adding constraints for the convolutional neural network.")

    # _____________start of the convolutional part_____________

    for k in range(32):
        for i in range(100):
            for j in range(100):
                m.addConstr(gp.quicksum(x_0[i + 1 + x, j + 1 + y] * weights[0][k, 0, x + 1, y + 1]
                                        for x in range(-1, 2) for y in range(-1, 2))
                            + biases[0][k] == x_1[0, k, i + 1, j + 1] - s_1[0, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_1[0, k, i + 1, j + 1], True, x_1[0, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_1[0, k, i + 1, j + 1], False, s_1[0, k, i + 1, j + 1] <= 0)

                m.addConstr(gp.quicksum(x_1[0, z, i + 1 + x, j + 1 + y] * weights[1][k, z, x + 1, y + 1]
                                        for z in range(32) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[1][k] == x_1[1, k, i + 1, j + 1] - s_1[1, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_1[1, k, i + 1, j + 1], True, x_1[1, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_1[1, k, i + 1, j + 1], False, s_1[1, k, i + 1, j + 1] <= 0)

                m.addGenConstrIndicator(z_pool_1[k, i + 1, j + 1], True,
                                        x_pool_1[k, (i // 2) + 1, (j // 2) + 1] <= x_1[1, k, i + 1, j + 1])
                m.addConstr(x_pool_1[k, (i // 2) + 1, (j // 2) + 1] >= x_1[1, k, i + 1, j + 1])

    print("Debug")

    for k in range(32):
        for i in range(0, 100, 2):
            for j in range(0, 100, 2):
                m.addConstr(gp.quicksum(z_pool_1[k, i + 1 + x, j + 1 + y] for x in range(2) for y in range(2)) == 1)

    print("Debug")

    # _____________end of the first two layers plus max pooling_____________

    for k in range(32):
        for i in range(50):
            for j in range(50):
                m.addConstr(gp.quicksum(x_pool_1[z, i + 1 + x, j + 1 + y] * weights[2][k, z, x + 1, y + 1]
                                        for z in range(32) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[2][k] == x_2[0, k, i + 1, j + 1] - s_2[0, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_2[0, k, i + 1, j + 1], True, x_2[0, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_2[0, k, i + 1, j + 1], False, s_2[0, k, i + 1, j + 1] <= 0)

                m.addConstr(gp.quicksum(x_2[0, z, i + 1 + x, j + 1 + y] * weights[3][k, z, x + 1, y + 1]
                                        for z in range(32) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[3][k] == x_2[1, k, i + 1, j + 1] - s_2[1, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_2[1, k, i + 1, j + 1], True, x_2[1, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_2[1, k, i + 1, j + 1], False, s_2[1, k, i + 1, j + 1] <= 0)

                m.addGenConstrIndicator(z_pool_2[k, i + 1, j + 1], True,
                                        x_pool_2[k, (i // 2) + 1, (j // 2) + 1] <= x_2[1, k, i + 1, j + 1])
                m.addConstr(x_pool_2[k, (i // 2) + 1, (j // 2) + 1] >= x_2[1, k, i + 1, j + 1])

    for k in range(32):
        for i in range(0, 50, 2):
            for j in range(0, 50, 2):
                m.addConstr(gp.quicksum(z_pool_2[k, i + 1 + x, j + 1 + y] for x in range(2) for y in range(2)) == 1)

    # _____________end of the second two layers plus max pooling_____________

    for k in range(64):
        for i in range(25):
            for j in range(25):
                m.addConstr(gp.quicksum(x_pool_2[z, i + 1 + x, j + 1 + y] * weights[4][k, z, x + 1, y + 1]
                                        for z in range(32) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[4][k] == x_3[0, k, i + 1, j + 1] - s_3[0, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_3[0, k, i + 1, j + 1], True, x_3[0, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_3[0, k, i + 1, j + 1], False, s_3[0, k, i + 1, j + 1] <= 0)

                m.addConstr(gp.quicksum(x_3[0, z, i + 1 + x, j + 1 + y] * weights[5][k, z, x + 1, y + 1]
                                        for z in range(64) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[5][k] == x_3[1, k, i + 1, j + 1] - s_3[1, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_3[1, k, i + 1, j + 1], True, x_3[1, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_3[1, k, i + 1, j + 1], False, s_3[1, k, i + 1, j + 1] <= 0)

                # this is what pytorch does to apply the floor operator at the output dimension
                # when the pooled input is not even, he simply ignore the last element,
                # in this case the last row and the last column, that's the reason for the if condition
                if (i != 24) and (j != 24):
                    m.addGenConstrIndicator(z_pool_3[k, i + 1, j + 1], True,
                                            x_pool_3[k, (i // 2) + 1, (j // 2) + 1] <= x_3[1, k, i + 1, j + 1])
                    m.addConstr(x_pool_3[k, (i // 2) + 1, (j // 2) + 1] >= x_3[1, k, i + 1, j + 1])

    for k in range(64):
        for i in range(0, 25, 2):
            for j in range(0, 25, 2):
                if (i != 24) and (j != 24):
                    m.addConstr(gp.quicksum(z_pool_3[k, i + 1 + x, j + 1 + y] for x in range(2) for y in range(2)) == 1)

    # _____________end of the third two layers plus max pooling_____________

    for k in range(128):
        for i in range(12):
            for j in range(12):
                m.addConstr(gp.quicksum(x_pool_3[z, i + 1 + x, j + 1 + y] * weights[6][k, z, x + 1, y + 1]
                                        for z in range(64) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[6][k] == x_4[0, k, i + 1, j + 1] - s_4[0, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_4[0, k, i + 1, j + 1], True, x_4[0, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_4[0, k, i + 1, j + 1], False, s_4[0, k, i + 1, j + 1] <= 0)

                m.addConstr(gp.quicksum(x_4[0, z, i + 1 + x, j + 1 + y] * weights[7][k, z, x + 1, y + 1]
                                        for z in range(128) for x in range(-1, 2) for y in range(-1, 2))
                            + biases[7][k] == x_4[1, k, i + 1, j + 1] - s_4[1, k, i + 1, j + 1])
                m.addGenConstrIndicator(z_4[1, k, i + 1, j + 1], True, x_4[1, k, i + 1, j + 1] <= 0)
                m.addGenConstrIndicator(z_4[1, k, i + 1, j + 1], False, s_4[1, k, i + 1, j + 1] <= 0)

                # note that here we don't have the need of a plus 1 fro the padding from what concern
                # the last layer of result, x_pool_4, because as I've said in the variables definition after this
                # we don't have other convolutional layers
                m.addGenConstrIndicator(z_pool_4[k, i + 1, j + 1], True,
                                        x_pool_4[k, (i // 2), (j // 2)] <= x_4[1, k, i + 1, j + 1])
                m.addConstr(x_pool_4[k, (i // 2), (j // 2)] >= x_4[1, k, i + 1, j + 1])

    for k in range(128):
        for i in range(0, 12, 2):
            for j in range(0, 12, 2):
                m.addConstr(gp.quicksum(z_pool_4[k, i + 1 + x, j + 1 + y] for x in range(2) for y in range(2)) == 1)

    # _____________end of the fourth two layers plus max pooling_____________

    # _____________end of the convolutional part_____________

    # last layer: a single neuron, the peculiar indexing for the weights is a form of flattening
    m.addConstr(gp.quicksum(x_pool_4[z, x, y] * weights[8][0, (z * 36) + (x * 6) + y]
                            for z in range(128) for x in range(6) for y in range(6)) + biases[8][0] == out)

    # add constraint on the fact that the misclassification has to happen
    # so if starting label is 1 (dog), try to output between 0 and 0.5 (cat), and vice versa
    if sample_label == 1:
        m.addConstr(out <= 0.5)
    else:
        m.addConstr(out >= 0.5001)

    # one can add constraints on some limitations on pixel distances
    # m.addConstr(...)

    print("End of the model definition.")

    print("Starting the optimization...")

    start = timeit.default_timer()

    m.optimize()

    stop = timeit.default_timer()

    print(f"Optimization finished in: {stop - start} s.")

    generated_sample = torch.empty((1, 100, 100))

    for i in range(100):
        for j in range(100):
            generated_sample[0, i, j] = x_0[i + 1, j + 1].x

    return generated_sample
