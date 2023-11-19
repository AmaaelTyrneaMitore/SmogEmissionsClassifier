import { join } from 'path';
import appRootPath from 'app-root-path';

import loadCSV from './utils/csv-loader.js';
import LogisticRegression from './models/logistic-regression.js';

// Load the CSV file path
const csvFilePath = join(appRootPath.path, 'data', 'vintage_cars_data.csv');

// Define options for loading CSV data
const csvLoadingOptions = {
  // Columns to be considered as data and labels
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],

  // Options for data processing
  shuffle: true,
  splitTest: 50,

  // Define converters to encode 'passedemissions' as 1 or 0
  converters: {
    passedemissions: (ifPassed) => (ifPassed === 'TRUE' ? 1 : 0),
  },
};

// Load the CSV data using the provided options
const { features, labels, testFeatures, testLabels } = loadCSV(csvFilePath, csvLoadingOptions);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5,
});

regression.train();

const modelAccuracy = regression.test(testFeatures, testLabels);

console.log(`\n\n[+] Model Accuracy: ${modelAccuracy}%\n\n`);
