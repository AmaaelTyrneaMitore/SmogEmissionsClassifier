import { join } from 'path';
import appRootPath from 'app-root-path';
import plot from 'node-remote-plot';

import loadCSV from './utils/csv-loader.js';
import LogisticRegression from './models/logistic-regression.js';

// File path for CSV data
const csvFilePath = join(appRootPath.path, 'data', 'vintage_cars_data.csv');

// Define options for loading CSV data
const csvLoadingOptions = {
  // Define columns for data and labels
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],

  // Configure data processing
  shuffle: true,
  splitTest: 50,

  // Convert 'passedemissions' to binary (1 or 0)
  converters: {
    passedemissions: (ifPassed) => (ifPassed === 'TRUE' ? 1 : 0),
  },
};

// Load CSV data based on the provided options
const { features, labels, testFeatures, testLabels } = loadCSV(csvFilePath, csvLoadingOptions);

// Initialize and train Logistic Regression model
const logisticRegressionModel = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5,
});

logisticRegressionModel.train();

// Test the trained model and calculate accuracy
const modelAccuracy = logisticRegressionModel.test(testFeatures, testLabels);

console.log(`\n\n[+] Model Accuracy: ${modelAccuracy}%\n\n`);

// Plot cost history for visualization
plot({
  x: logisticRegressionModel.costHistory.reverse(),
  name: 'data/cost_history',
  title: 'Cost History',
  xLabel: 'No of Iterations #',
  yLabel: 'Cost',
});
