import numpy as np
import matplotlib.pyplot as plt

"""Question-2:: Newton's Method"""

def cost_function(cf):
    cf = np.asarray(cf)
    cf_x = cf[0]
    cf_y = cf[1]
    cost = np.log(1 - cf_x - cf_y) - np.log(cf_x) - np.log(cf_y)
    return cost

def gradient_cost_function(gcf):
    gcf = np.asarray(gcf)
    gcf_x = gcf[0]
    gcf_y = gcf[1]
    gradient_x = np.divide(1, (1 - gcf_x - gcf_y)) - np.divide(1, gcf_x)
    gradient_y = np.divide(1, (1 - gcf_x - gcf_y)) - np.divide(1, gcf_y)
    new_gcf = np.asarray((gradient_x, gradient_y))
    return new_gcf

"""define fuction that calculates the hessian matrix and returns
 the pseudo inverse of the matric"""
def hessian_inverse_matrix(him):
    him = np.asarray(him)
    him_x = him[0]
    him_y = him[1]
    h_11 = np.divide(1, np.square(1 - him_x - him_y)) + np.divide(1, np.square(him_x))
    h_12 = np.divide(1, np.square(1 - him_x - him_y))
    h_21 = np.divide(1, np.square(1 - him_x - him_y))
    h_22 = np.divide(1, np.square(1 - him_x - him_y)) + np.divide(1, np.square(him_y))
    hessian_matrix = np.ndarray(shape=(2, 2))
    hessian_matrix_array = np.asarray(hessian_matrix)

    for i in range(0, 2):
        for j in range(0, 2):
            if i == 0 and j == 0:
                hessian_matrix_array[i][j] = h_11
            else:
                if i ==0 and j == 1:
                    hessian_matrix_array[i][j] = h_12
                else:
                    if i == 1 and j == 0:
                        hessian_matrix_array[i][j] = h_21
                    else:
                        if i == 1 and j == 1:
                            hessian_matrix_array[i][j] = h_22

    hessian_inverse = np.linalg.pinv(hessian_matrix_array)
    return hessian_inverse


"""weights initialization"""

w_x = np.random.uniform(0, 0.5, 1)
w_y = np.random.uniform(0, 0.4, 1)
weights = np.asarray((w_x, w_y))
print "initial weights", weights
learning_rate = 0.1
threshold = 0.0000002
iteration_number = 0
domain_miss = 0
actual_cost = [[0.0]]
actual_cost = np.asarray(actual_cost, dtype=np.float)
cost_plot = []
iteration_plot = []
w0_plot = []
w1_plot = []
w0_plot.append(weights[[0]])
w1_plot.append(weights[[0]])

while True:
    step_update = gradient_cost_function(weights)
    temp_hessian = hessian_inverse_matrix(weights)
    delta_weights_temp = np.dot(temp_hessian, step_update)
    temp_weights = weights - learning_rate * delta_weights_temp
    domain_condition_temp = np.asarray(temp_weights)
    domain_condition = domain_condition_temp[0] + domain_condition_temp[1]
    if domain_condition < 1:
        weights = temp_weights
        w0_plot.append(temp_weights[[0]])
        w1_plot.append(temp_weights[[1]])
        temp_cost = cost_function(weights)
        iteration_temp = iteration_number + 1
        iteration_plot.append(iteration_temp)
        cost_plot.append(temp_cost)

    else:
        w_x = np.random.uniform(0, 0.5, 1)
        w_y = np.random.uniform(0, 0.4, 1)
        weights = np.asarray((w_x, w_y))
        domain_miss = domain_miss +1
        print "number of domain misses:", domain_miss, "weights are:", temp_weights
        continue
    iteration_number = iteration_number + 1
    actual_cost_temp = cost_function(weights)
    actual_cost = np.append(actual_cost, actual_cost_temp)
    threshold_cal = np.subtract(actual_cost[iteration_number], actual_cost[iteration_number-1])
    print "iteration number::", iteration_number, "cost difference", threshold_cal, "cost::", actual_cost_temp
    if np.absolute(threshold_cal) < threshold:
        break

print "final weights::", weights
print "final cost::", cost_function(weights)
plt.figure(1)
plt.xlabel('iteration number')
plt.ylabel('cost')
plt.plot(iteration_plot, cost_plot, color='red')
plt.savefig('question2NM.png')
plt.figure(2)
plt.scatter(w0_plot, w1_plot, s=2)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('Q2pointPlotNM.png')


