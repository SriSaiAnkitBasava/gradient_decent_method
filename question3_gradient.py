import numpy as np
import matplotlib.pyplot as plt

"""initialize the input vestor X and output vector Y"""


def get_data():
    temp_x = []
    temp_y = []
    for i in range(1, 51):
        x_i = i
        temp_x.append(x_i)
        temp_point = np.random.uniform(-1, 1, 1)
        y_i = float(i + temp_point)
        temp_y.append(y_i)
    return temp_x, temp_y


def gradient_cost_function(gcf, x1, y1):
    #gcf = np.asarray(gcf)
    gradient_y = 0.0
    gradient_x = 0.0
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

def y_cal(w):
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
x, y = get_data()
#x1 , y1 = x, y
# print "x values"
# print x
# print "y values"
# print y

"""initialize weights used by the algorithm"""
weights = np.random.rand(1, 2)
learning_rate = 0.000001
epoch = 0
threshold = 0.000002
total_error = 0

# print "weights"
# print weights
cal_output = np.dot(weights, x[0])
# print "calculated output"
# print cal_output
check_weights = weights

while True:
    """calculate the total error"""
    cal_output = gradient_cost_function(weights, x, y)
    weights = weights + (learning_rate * cal_output)

    # for j in range(1, 51):
    #     cal_output = np.dot(weights, x[j - 1])
    #     # temp_error = (y[j-1] - cal_output) * (y[j-1] - cal_output)
    #     temp_error = (pow((y[j - 1] - cal_output), 2))
    #     total_error = total_error + temp_error

    epoch = epoch + 1
    diff_w0 = -(check_weights[0][0] - weights[0][0])
    diff_w1 = -(check_weights[0][1] - weights[0][1])
    print "epoch::", epoch, "weights difference", diff_w0, " ", diff_w1

    if diff_w0 <= threshold and diff_w1 <= threshold:
        break
    else:
        check_weights = 0
        check_weights = weights
        # total_error = 0

print "final weights:"
print weights
y_cal_plot = y_cal(weights)
plt.figure(1)
plt.scatter(x, y, color='blue', s=4)
plt.scatter(x, y_cal_plot, color='red', s=4)
plt.savefig('question3Gradient.png')

