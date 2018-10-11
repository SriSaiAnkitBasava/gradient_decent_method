import numpy as np
import matplotlib.pyplot as plt

"""Question-3:: Least Squares Fitting and Gradient Decent"""

def get_data():
    temp_x = []
    temp_y = []
    sum_x = 0.0
    sum_x_square = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    x_plot = []
    y_plot = []

    for i in range(1, 51):
        x_i = i
        x_plot.append(i)
        sum_x = sum_x + i
        sum_x_square =sum_x_square + pow(i, 2)
        temp_x.append(x_i)
        temp_point = np.random.uniform(-1, 1, 1)
        y_i = float(i + temp_point)
        y_plot.append(y_i)
        sum_y = sum_y + y_i
        sum_xy = sum_xy + (i * y_i)

        temp_y.append(y_i)
    return temp_x, temp_y, sum_x, sum_x_square, sum_y, sum_xy, x_plot, y_plot


"""write function to calculate the exact solution for weights"""
def cal_exact_weights(x, x_2, y, xy):
    w0_exact = float(((y * x_2) - (x * xy)) / ((50 * x_2) - pow(x, 2)))
    w1_exact = float(((50 * xy) - (x * y)) / ((50 * x_2) - pow(x, 2)))
    return w0_exact, w1_exact

"""write function that calculates y using calculated weights"""
def y_cal(w0, w1):
    y_out = []
    for i in range(1, 51):
        y_temp = w0 + (w1 * i)
        y_out.append(y_temp)
    return y_out

def gradient_cost_function(gcf, x1, y1):
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    gcf_x = gcf[0][0]
    gcf_y = gcf[0][1]
    temp_x = 0.0
    temp_y = 0.0
    for j in range(0,51):
        temp_x = temp_x + (y1[j - 1] - (gcf_x + (gcf_y * x1[j - 1])))
        temp_y = temp_y + x1[j - 1] * (y1[j - 1] - (gcf_x + (gcf_y * x1[j - 1])))

    gradient_x = 2 * temp_x
    gradient_y = 2 * temp_y
    new_gcf = np.asarray((gradient_x, gradient_y))
    return new_gcf

def y_cal_gradeint(w):
    w0 = w[0][0]
    w1 = w[0][1]
    y_out = []
    for i in range(1, 51):
        y_temp = w0 + (w1 * i)
        y_out.append(y_temp)
    return y_out


"""check output"""
x = []
y = []
x, y, x_sum, x_square_sum, y_sum, xy_sum, plot_x, plot_y = get_data()
print x_sum
w0_cal, w1_cal = cal_exact_weights(x_sum, x_square_sum, y_sum, xy_sum)
print "w0:", w0_cal, "w1:", w1_cal
plot_y_cal = y_cal(w0_cal, w1_cal)
plt.figure(1)
plt.scatter(plot_x, plot_y, color='black', s=4)
plt.scatter(plot_x, plot_y_cal, color='blue', s=4)
#plt.savefig('question33.png')


""""""
weights = np.random.rand(1, 2)
learning_rate = 0.000001
epoch = 0
threshold = 0.000002
total_error = 0
check_weights = weights

"""gradient decent part"""
while True:
    """calculate the total error"""
    cal_output = gradient_cost_function(weights, x, y)
    weights = weights + (learning_rate * cal_output)
    epoch = epoch + 1
    diff_w0 = -(check_weights[0][0] - weights[0][0])
    diff_w1 = -(check_weights[0][1] - weights[0][1])
    print "epoch::", epoch, "weights difference", diff_w0, " ", diff_w1
    if diff_w0 <= threshold and diff_w1 <= threshold:
        break
    else:
        check_weights = 0
        check_weights = weights

print "final weights:"
print weights
y_cal_plot = y_cal_gradeint(weights)
plt.figure(1)
plt.scatter(plot_x, y_cal_plot, color='green', s=4)
plt.savefig('question333.png')
