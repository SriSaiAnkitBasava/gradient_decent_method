import numpy as np


"""initialize the input vestor X and output vector Y"""
def get_data():
    temp_x = []
    temp_y = []
    for i in range (1, 51):
        x_i = [1, i]
        temp_x.append(x_i)
        temp_point = np.random.uniform(-1, 1, 1)
        y_i = float(i + temp_point)
        temp_y.append(y_i)
    return temp_x, temp_y

"""check output"""
x = []
y = []
x, y = get_data()
# print "x values"
# print x
# print "y values"
# print y

"""initialize weights used by the algorithm"""
weights = np.random.rand(1, 2)
learning_rate = 0.000000001
epoch = 0
threshold = 0.000002


# print "weights"
# print weights
cal_output = np.dot(weights, x[0])
# print "calculated output"
# print cal_output


while True:
    """calculate the total error"""
    total_error = 0
    for i in range(1, 51):
        total_error = 0
        for j in range(1, 51):
            cal_output = np.dot(weights, x[j-1])
            # temp_error = (y[j-1] - cal_output) * (y[j-1] - cal_output)
            temp_error = (pow((y[j-1] - cal_output), 2))
            total_error = total_error + temp_error
            #temp_error = (temp_error + cal_output) * (temp_error + cal_output)
        #weight_direction = np
        weights = weights + ((learning_rate * temp_error) * x[i-1])
    epoch = epoch + 1
    temp_see_value = total_error
    print "epoch::", epoch, "error::", total_error

    if total_error <= threshold:
        break

print "final weights :"
print weights
