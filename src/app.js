import { join } from 'path';
import appRootPath from 'app-root-path';

import loadCSV from './utils/csv-loader.js';

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
const { labels } = loadCSV(csvFilePath, csvLoadingOptions);

// Log the loaded labels for verification
console.log(labels);
