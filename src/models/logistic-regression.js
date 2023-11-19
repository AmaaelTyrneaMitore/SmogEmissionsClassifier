import { tensor, ones, zeros, moments, sigmoid, Tensor } from '@tensorflow/tfjs-node';

/**
 * Logistic Regression class that performs binary classification using TensorFlow.js.
 */
export default class LogisticRegression {
  /**
   * Constructs a Logistic Regression model with tensors for features, labels, and weights.
   * @param {number[][]} features - Array of feature values for training.
   * @param {number[][]} labels - Array of label values for training.
   * @param {Object} [options={}] - Options for the logistic regression model.
   * @param {number} [options.learningRate=0.1] - Learning rate for gradient descent.
   * @param {number} [options.iterations=1000] - Maximum number of iterations for gradient descent.
   * @param {number} [options.batchSize=1] - Size of each batch for batch gradient descent.
   * @param {number} [options.decisionBoundary=0.5] - Threshold for classification decision.
   */
  constructor(
    features,
    labels,
    options = { learningRate: 0.1, iterations: 1000, batchSize: 1, decisionBoundary: 0.5 }
  ) {
    // Convert input arrays to TensorFlow tensors
    this.features = this.preprocessFeatures(features);
    this.labels = tensor(labels);

    // Set model options and initialize weights
    this.options = options;
    this.costHistory = []; // Array to store cost (cross-entropy) history
    this.weights = zeros([this.features.shape[1], 1]); // Initialize weights as zeros
  }

  /**
   * Performs batch gradient descent to update weights based on the given features and labels.
   * @param {Tensor} features - Tensor of feature values.
   * @param {Tensor} labels - Tensor of label values.
   */
  gradientDescent(features, labels) {
    // Calculate predictions using sigmoid function
    const currentPredictions = sigmoid(features.matMul(this.weights));

    // Compute differences between predicted labels and actual labels
    const differences = currentPredictions.sub(labels);

    // Calculate gradients using matrix operations
    const gradients = features.transpose().matMul(differences).div(features.shape[0]);

    // Update weights using the gradients and learning rate
    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  /**
   * Trains the logistic regression model using batch gradient descent.
   * Optimizes weights for the model through iterations and batch updates.
   */
  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    // Iterate over the specified number of iterations
    for (let i = 0; i < this.options.iterations; i++) {
      // Loop through each batch for gradient descent
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options;
        const startIndex = j * batchSize;

        // Extract the current batch of features and labels for gradient descent
        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        // Run gradient descent for the current batch
        this.gradientDescent(featureSlice, labelSlice);
      }

      // Record cross-entropy cost and optimize learning rate after each epoch
      this.recordCost();
      this.optimizeLearningRate();
    }
  }

  /**
   * Test the trained logistic regression model's accuracy using test data.
   * Calculates the accuracy rate based on the specified decision boundary.
   * @param {number[][]} testFeatures - Array of feature values for testing.
   * @param {number[][]} testLabels - Array of label values for testing.
   * @returns {number} - Accuracy rate of the model.
   */
  test(testFeatures, testLabels) {
    testLabels = tensor(testLabels);

    // Calculate predictions using the model
    const predictions = this.predict(testFeatures);

    // Calculate the number of incorrect predictions
    const incorrectPredictions = predictions.sub(testLabels).abs().sum().arraySync();

    // Calculate and return the accuracy rate
    const accuracy = (predictions.shape[0] - incorrectPredictions) / predictions.shape[0];
    return accuracy;
  }

  /**
   * Preprocesses the features by standardizing and adding a column of ones for intercept calculation.
   * @param {number[][]} features - Array of feature values.
   * @returns {Tensor} - Processed features tensor.
   */
  preprocessFeatures(features) {
    features = tensor(features);

    // Standardize features if mean and variance are available, else perform standardization
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    // Add a column of ones to features for intercept calculation
    const onesColumn = ones([features.shape[0], 1]);
    features = onesColumn.concat(features, 1);

    return features;
  }

  /**
   * Helper function to standardize the features.
   * @param {Tensor} features - TensorFlow tensor of feature values.
   * @returns {Tensor} - Standardized features tensor.
   */
  standardize(features) {
    const { mean, variance } = moments(features, 0);

    // Save mean and variance for later use
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  /**
   * Records the current value of cross-entropy cost.
   * Calculates the cross-entropy and adds it to the history.
   */
  recordCost() {
    const predictions = sigmoid(this.features.matMul(this.weights));

    // Calculate cross-entropy cost
    const termOne = this.labels.transpose().matMul(predictions.log());
    const termTwo = this.labels.mul(-1).add(1).transpose().matMul(predictions.mul(-1).add(1).log());
    const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).arraySync()[0][0];

    // Store the cost in the history
    this.costHistory.unshift(cost);
  }

  /**
   * Updates the learning rate based on the cross-entropy cost history.
   * Adjusts the learning rate for optimization based on the cost trend.
   */
  optimizeLearningRate() {
    // Ensure enough cost values are available for comparison
    if (this.costHistory.length < 2) return;

    // If cost increased, decrease learning rate; else, increase it
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2; // Reduce learning rate by 50% if cost increased
    } else {
      this.options.learningRate *= 1.05; // Increase learning rate by 5% if cost decreased
    }
  }

  /**
   * Predicts the label values for a new set of observations.
   * @param {number[][]} observations - Array of feature values for prediction.
   * @returns {Tensor} - Tensor containing predicted label values.
   */
  predict(observations) {
    // Process the provided observations to generate predictions
    const processedObservations = this.preprocessFeatures(observations);

    // Calculate predicted values using current weights and decision boundary
    const predictedLabels = processedObservations
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32');

    return predictedLabels;
  }
}
