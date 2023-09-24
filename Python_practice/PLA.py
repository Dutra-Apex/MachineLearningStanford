import numpy as np
import matplotlib.pyplot as plt

def perceptron_learn(data_in):

    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    x_mid = []
    y_mid = []
    for n in range (len(data_in)):
        x_mid.append(data_in[n][0])
        y_mid.append(data_in[n][1])

    x_vector = np.array(x_mid)
    y_label = y_mid
    x_t = np.transpose(x_vector)

    d_1 = len(x_t)
    w = np.zeros((1,d_1))
    w = w[0]

    loop_boolean = True
    loop_counter = 0
    loop_counter_actual = 0

    while(loop_boolean):
        loop_counter += 1
        loop_counter_actual += 1
        loop_counter = loop_counter % 100
        y_label_predictor = np.where(np.dot(w, x_t)>0, 1, -1)

        if(y_label_predictor[loop_counter] != y_label[loop_counter]):
            w = w + (-y_label_predictor[loop_counter]) * x_vector[loop_counter]
        error_counter = 0
        for i in range (len(y_label_predictor)):
            if(y_label_predictor[i] != y_label[i]):
                error_counter += 1

        if (error_counter == 0) :
            # print("SUCESS")
            loop_boolean = False

        if(loop_counter_actual > 99999):
            loop_boolean = False
            print("Fail")
            print(w)

    iterations = loop_counter_actual
    return w, iterations



def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))
    for k in range(num_exp):
        data_in = []
        rng = np.random.default_rng()
        w_star = np.random.rand(d,1)
        w_star = np.transpose(w_star)
        x_mid = []

        for i in range(N):
            rand_uni_vector = []
            for yop in range(d):
                random_uni_number = np.random.uniform(-1, 1)
                rand_uni_vector.append(random_uni_number)
            x_mid.append(rand_uni_vector)

        x_vector = np.array(x_mid)
        x_t = np.transpose(x_vector)
        y_label = np.where(np.dot(w_star, x_t)>0, 1, -1)
        y_label = y_label[0]

        x_y_pairs = []
        for i in range(len(y_label)):
            x_y_pairs.append((x_vector[i],y_label[i]))

        data_in.append(x_y_pairs)
        data_in = data_in[0]

        this_w, this_iterations = perceptron_learn(data_in)
        num_iters[k] = this_iterations


    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 10)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
