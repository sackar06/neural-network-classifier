Report: Letter Recognition using Feedforward Neural Network

1. Introduction
This project implements a feedforward neural network (FNN) from scratch using NumPy. The task is to classify binary pixel patterns of letters A, B, and C. Instead of relying on deep learning libraries such as TensorFlow or PyTorch, all key components such as weight initialization, forward propagation, activation functions, backpropagation, and gradient descent are implemented manually.

2. Problem Statement
The model should:
- Learn to classify letters A, B, and C from binary image patterns.
- Optimize the model using backpropagation and gradient descent.
- Visualize the learning process using loss and accuracy curves.
- Predict and display the class of a given letter pattern using matplotlib.

3. Methodology
3.1 Data Preparation
- Each letter (A, B, C) is defined as a 7x5 binary pixel pattern.
- Flattened into 35-dimensional vectors.
- Labels are one-hot encoded: A -> [1,0,0], B -> [0,1,0], C -> [0,0,1].

3.2 Neural Network Architecture
- Input layer: 35 neurons
- Hidden layer: 10 neurons
- Output layer: 3 neurons
- Activation Function: Sigmoid for hidden and output layers.

3.3 Forward Propagation
- Hidden layer: z1 = X*W1 + b1, a1 = sigmoid(z1)
- Output layer: z2 = a1*W2 + b2, a2 = sigmoid(z2)
- Prediction = argmax(a2)

3.4 Loss Function
- Mean Squared Error (MSE): Loss = mean((y - y_hat)^2)

3.5 Backpropagation
- Output gradient: delta_output = (y - a2) * sigmoid_derivative(a2)
- Hidden gradient: delta_hidden = (delta_output * W2.T) * sigmoid_derivative(a1)
- Weight updates: W2 += learning_rate * a1.T * delta_output, W1 += learning_rate * X.T * delta_hidden

3.6 Training
- 2000 epochs with learning rate 0.5
- Track loss and accuracy each epoch

4. Analysis Process
- Loss decreases steadily over epochs.
- Accuracy quickly reaches near 100% for training data.
- Predicted letters match input letters; visualization confirms correctness.

5. Key Findings
- One hidden layer is sufficient to classify A, B, C.
- Training converges quickly.
- Loss curve validates proper weight updates.
- Model shows some generalization to minor noise.

6. Conclusion
- Demonstrates core mechanics of neural networks: forward propagation, activation, backpropagation, gradient descent.
- Provides a foundation for scaling to more complex tasks.

7. Future Work
- Extend to A-Z letters.
- Test noise robustness.
- Try different activation functions (ReLU, tanh).
- Explore deeper architectures and advanced optimizers.