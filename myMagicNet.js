var traceback = require("traceback");
var convnetjs = require("convnetjs");
var fs = require('fs');
var $ = jQuery = require('jquery');
require('./lib/jquery.csv-0.71.min.js');

var myMagicNet = new MyMagicNet();
myMagicNet.init();

function MyMagicNet() {

	var _ = this;
	
	var myMagicNetOpts;
	var folds_evaluated;
	var batches_evaluated;
	var magicNet;
	var magicNetTesting;
	var opts;
	var iter;
	var bestModelAccuracy;	
	var train_dataset;
	var train_import_data;
	
	_.init = function () {
	
		_.folds_evaluated = 0;
		_.batches_evaluated = 0;
		_.opts = {};
		_.iter = 0;
		_.bestModelAccuracy = 0;
		_.magicNet = null;
		_.magicNetTesting = null;
		
		_.myMagicNetOpts = {}
		// Used to find training sets and save models.
		_.myMagicNetOpts.dataTitle = 'horse';
		// Set this to true to train, false to predict.
		_.myMagicNetOpts.runTraining = false;
		// Resume training from an existing trained model. TODO: implement
		_.myMagicNetOpts.resumeTraining = true;
		// Set this to true for faster code path testing
		_.myMagicNetOpts.simpleTraining = false;

		// Set this to a file containing data to train with.
		_.myMagicNetOpts.trainingData = '' + _.myMagicNetOpts.dataTitle + '.training.data.csv';
		// Set this to a file containing data to evaluate.
		_.myMagicNetOpts.evaluateData = '' + _.myMagicNetOpts.dataTitle + '.testing.data.csv';
		// Set this to the trained model to load.
		_.myMagicNetOpts.trainedModel = '' + _.myMagicNetOpts.dataTitle + '.magicNetModel.json';

		debugObject('_.myMagicNetOpts: ', _.myMagicNetOpts);

		if ( _.myMagicNetOpts.runTraining ) {
			_.importTrainData();
		} else {
			_.importMagicNet();
		}	
		
	}
	
	// Import Magic Net
	_.importMagicNet = function () {

		fs.readFile(__dirname + '/trained_models/' + _.myMagicNetOpts.trainedModel + '', function (err, jsonData) {
			if (err) { throw err; }

			debug('Import Trained Model ' + _.myMagicNetOpts.trainedModel);

			var json = JSON.parse(jsonData.toString());
			_.magicNetTesting = new convnetjs.MagicNet();
			_.magicNetTesting.fromJSON(json);
			
			debug('MagicNet loaded from JSON with ' +  + _.magicNetTesting.evaluated_candidates.length + ' evaluated candidates.');
						
			// set options for magic net
			_.magicNetTesting.ensemble_size = 10;

			_.importTrainData(); // For column defs.	
		});

	}	
	
	// Import training data.
	_.importTrainData = function () {

		fs.readFile(__dirname + '/data/' + _.myMagicNetOpts.trainingData + '', function (err, csvData) {
			if (err) {throw err;}

			debug('Import Train Data ' + _.myMagicNetOpts.trainingData);

			var csv_txt = csvData.toString();
			//debug(csvData.toString());

			var arr = $.csv.toArrays(csv_txt);
			var arr_train = arr;
			
			if ( _.myMagicNetOpts.runTraining ) {
				debug('Set random test data.');
				var arr_test = [];

				var test_ratio = Math.floor(20); // send 20% of imported data randomly into test set below
				if (test_ratio !== 0) {
					// send some lines to test set
					var test_lines_num = Math.floor(arr.length * test_ratio / 100.0);
					var rp = randperm(arr.length);
					arr_train = [];
					for (var i = 0; i < arr.length; i++) {
						if (i < test_lines_num) {
							arr_test.push(arr[rp[i]]);
						} else {
							arr_train.push(arr[rp[i]]);
						}
					}
					// enter test lines to test box
					var t = "";
					for (var i = 0; i < arr_test.length; i++) {
						t += arr_test[i].join(",") + "\n";
					}
					//debug(t);

				}
			}

			//debug('importTrainData: Training data length ' + arr_train.length);
			_.train_import_data = _.importData(arr_train);
			_.train_dataset = _.makeDataset(_.train_import_data.arr, _.train_import_data.colstats);

			if ( _.myMagicNetOpts.runTraining ) {
				_.startCV(); // CV = ConVnetJS
			} else {
				// read in the data in the text field
				_.importTestData();
			}
		});

	}	
	

	_.importTestData = function () {

		fs.readFile(__dirname + '/data/' + _.myMagicNetOpts.evaluateData + '', function (err, csvData) {
			if (err) { throw err; }
			
			debug("Import Test Data " + _.myMagicNetOpts.evaluateData);
			
			var csv_txt = csvData.toString();
			var arr_test = $.csv.toArrays(csv_txt);
			
			var test_import_data = _.importData(arr_test);
			//debugObject('test_import_data', test_import_data);
			
			// note important that we use colstats of train data!
			var test_dataset = _.makeDataset(test_import_data.arr, _.train_import_data.colstats);
			//debugObject('_.train_import_data.colstats', _.train_import_data.colstats);
			//debugObject('test_import_data.colstats', test_import_data.colstats);
			
			// use magic net to predict
			var n = test_dataset.data.length;
			var acc = 0.0;
						
			var predictions = {};
			predictions.outcome = {};
						
			for (var i = 0; i < n; i++) {
				
				//debugObject('test_dataset.data[i]: ', test_dataset.data[i]);
				
				var predictedCategoryIndex = _.magicNetTesting.predict(test_dataset.data[i]);
				
				if (predictedCategoryIndex === -1) {
					debug("The MagicNet is not yet ready! It must process at least one batch of candidates across all folds first. Wait a bit.");
					return;
				}
				
				var actualCategoryIndex = test_dataset.labels[i];
				
				//debug('n: ' + n);
				//debugObject('train_import_data.colstats', train_import_data.colstats[train_import_data.colstats.length-1]);
				// SH: ?? Is this the problem?
				var predictedCategory = _.train_import_data.colstats[_.train_import_data.colstats.length-1].uniques[predictedCategoryIndex];
				var actualCategory =    _.train_import_data.colstats[_.train_import_data.colstats.length-1].uniques[actualCategoryIndex];
				
				acc += (predictedCategoryIndex === actualCategoryIndex ? 1 : 0); // 0-1 loss
				
				var predictionOutcome = 'P';
				if ( actualCategoryIndex !== predictedCategoryIndex ) {
					var predictionOutcome = '-';
				}
				//debug(predictionOutcome + '  ' + i + ' ' + predictedCategoryIndex + ' ' + actualCategoryIndex + ' ' + predictedCategory + ' ' + actualCategory + '');
				
				predictions.outcome[i] = {};
				predictions.outcome[i].csvrow = i;
				predictions.outcome[i].predictionOutcome = predictionOutcome;
				predictions.outcome[i].predictedCategoryIndex = predictedCategoryIndex;
				predictions.outcome[i].actualCategoryIndex = actualCategoryIndex;
				predictions.outcome[i].predictedCategory = predictedCategory;
				predictions.outcome[i].actualCategory = actualCategory;
				
				//var predictions[i].data = {};
				//predictions[i].data = '' + i + ' ' + predictedCategoryIndex + ' ' + actualCategoryIndex + ' ' + predictedCategory + ' ' + actualCategory + '';
			}
			acc /= n;

			var accuracyRounded = Math.round(acc * 100) / 100;
			//debugObject('Prediction accuracy ' + accuracyRounded, predictions);
			debug('Prediction accuracy ' + (accuracyRounded*100) + '%');

			// report accuracy
			//debug("Test set accuracy: " + acc);		
			
			//debugObject('test_dataset', test_dataset);
			//debugObject('train_import_data.colstats', train_import_data.colstats);
			
			//debug(csvData.toString());
		});

	}
	
	// returns arr (csv parse)
	// and colstats, which contains statistics about the columns of the input
	// parsing results will be appended to a div with id outdivid
	_.importData = function (arr) {

		// find number of datapoints
		var N = arr.length;
		var t = [];
		debug('ImportData: found ' + N + ' data points');
		if (N === 0) {
			debug('ImportData: no data points found?');
			return;
		}

		// find dimensionality and enforce consistency
		var D = arr[0].length;
		for (var i = 0; i < N; i++) {
			var d = arr[i].length;
			if (d !== D) {
				debug('ImportData: data dimension not constant: line ' + i + ' has ' + d + ' entries.');
				return;
			}
		}
		debug('ImportData: data dimensionality is ' + (D - 1));

		// go through columns of data and figure out what they are
		var colstats = [];
		for (var i = 0; i < D; i++) {
			var res = _.guessColumn(arr, i);
			colstats.push(res);
			if (D > 20 && i > 3 && i < D - 3) {
				if (i == 4) {
					debug('ImportData: ...'); // suppress output for too many columns
				}
			} else {
				debug('ImportData: column ' + i + ' looks ' + (res.numeric ? "numeric" : "NOT numeric") + " and has " + res.num + " unique elements");
			}
		}

		return {
			arr : arr,
			colstats : colstats
		};
	}

	// process input mess into vols and labels
	// SH: Is this the problem? Labels are rebuilt differently???
	_.makeDataset = function (arr, colstats) {

		var D = arr[0].length;
		var labelix = -1;
		if (labelix < 0) {
			labelix = D + labelix; // -1 should turn to D-1
		}
		
		var data = [];
		var labels = [];
		var N = arr.length;
		for (var i = 0; i < N; i++) {
			var arri = arr[i];

			// create the input datapoint Vol()
			var p = arri.slice(0, D - 1);
			var xarr = [];
			for (var j = 0; j < D; j++) {
				if (j === labelix)
					continue; // skip!

				if (colstats[j].numeric) {
					xarr.push(parseFloat(arri[j]));
				} else {
					var u = colstats[j].uniques;
					var ix = u.indexOf(arri[j]); // turn into 1ofk encoding
					for (var q = 0; q < u.length; q++) {
						if (q === ix) {
							xarr.push(1.0);
						} else {
							xarr.push(0.0);
						}
					}
				}
			}
			var x = new convnetjs.Vol(xarr);

			// process the label (last column)
			if (colstats[labelix].numeric) {
				var L = parseFloat(arri[labelix]); // regression
			} else {
				var L = colstats[labelix].uniques.indexOf(arri[labelix]); // classification
				if (L == -1) {
					debug('whoa label not found! CRITICAL ERROR, very fishy.');
				}
			}
			data.push(x);
			labels.push(L);
		}

		var dataset = {};
		dataset.data = data;
		dataset.labels = labels;
		return dataset;
	}


	_.finishedFold = function () {
		_.folds_evaluated++;
		debug("finishedFold: So far evaluated a total of " + _.folds_evaluated + "/" + _.magicNet.num_folds + " folds in current batch");
	}

	_.finishedBatch = function () {
		_.batches_evaluated++;
		debug("finishedBatch: So far evaluated a total of " + _.batches_evaluated + " batches of candidates");
		_.folds_evaluated = 0;
	}


	// Start Training ConVnetJS
	_.startCV = function () {  
		
		debug('startCV');
		if ( _.myMagicNetOpts.simpleTraining ) {
			_.opts.train_ratio = 70 / 100.0; // default 70/100
			_.opts.num_folds = 1; // default 1
			_.opts.num_candidates = 5; // default 50
			_.opts.num_epochs = 2; // default 20
			_.opts.neurons_min = 5; // default 5
			_.opts.neurons_max = 10; // default 30
		} else {
			_.opts.train_ratio = 70 / 100.0; // default 70/100
			_.opts.num_folds = 1; // default 1
			_.opts.num_candidates = 50; // default 50
			_.opts.num_epochs = 20; // default 20
			_.opts.neurons_min = 5; // default 5
			_.opts.neurons_max = 30; // default 30
		}
		
		debugObject('MagicNet Options: ', _.opts);
		_.magicNet = new convnetjs.MagicNet(_.train_dataset.data, _.train_dataset.labels, _.opts);
		_.magicNet.onFinishFold(_.finishedFold);
		_.magicNet.onFinishBatch(_.finishedBatch);

		_.folds_evaluated = 0;
		_.batches_evaluated = 0;
		debug("Evaluated a total of " + _.batches_evaluated + " batches of candidates");
		debug("Evaluated a total of " + _.folds_evaluated + "/" + _.magicNet.num_folds + " folds in current batch");

		setInterval(_.step, 0);
	}


	// Iterate training
	_.step = function () {
		_.iter++;
		_.magicNet.step();
		
		if (_.iter % 300 == 0) {

			// var vals = _.magicNet.evalValErrors();
			//debugObject('vals', vals);
			
			//debug('magicNet.evaluated_candidates.length: ' + _.magicNet.evaluated_candidates.length);
			if (_.magicNet.evaluated_candidates.length > 0) {
				var cm = _.magicNet.evaluated_candidates[0];
				var currentModelAccuracy = (cm.accv / cm.acc.length);
				
				// Only save new bestModels.
				if ( currentModelAccuracy > _.bestModelAccuracy ) {

					debug('Iteration: ' + _.iter + ' currentModelAccuracy: ' + currentModelAccuracy + ' bestModelAccuracy: ' + _.bestModelAccuracy);
					_.bestModelAccuracy = currentModelAccuracy;
					
					// Save the best model so far
					_.exportMagicNet();
				}				
			} 
			
			debugNoNewline(_.iter + ' ');
		}
	}

	// Export Magic Net
	_.exportMagicNet = function () {

		if (_.magicNet.evaluated_candidates.length > 0) {
			var cm = _.magicNet.evaluated_candidates[0];
			
			var accuracyRounded = Math.round(cm.accv / cm.acc.length * 100) / 100;

			var t = '';
			t = '\n\n==================================================================================\n';
			t += 'Iteration: ' + _.iter  + '\n\n';
			t += 'Model accuracy: ' + cm.accv / cm.acc.length + '\n\n';
			t += 'Training options set: ' + JSON.stringify(_.opts) + '\n\n';
			t += 'Layer definitions: ' + JSON.stringify(cm.layer_defs) + '\n\n';
			t += 'Trainer definition: ' + JSON.stringify(cm.trainer_def) + '\n\n';
			t += '==================================================================================';
			debug(t);

			// Write the model and some definition info to file.
			var fileNameWithoutExtension = './trained_models/' + _.myMagicNetOpts.dataTitle + '.magicNetModel-accuracy-' + accuracyRounded + '-iter-' + _.iter;
			fs.writeFile(fileNameWithoutExtension + '.json', JSON.stringify(_.magicNet.toJSON()), function (err) {
				if (err) { debug(err); } else {	debug('exportMagicNet: file saved to ' + fileNameWithoutExtension + '.json'); }
			});
			fs.writeFile(fileNameWithoutExtension + '.def', t, function (err) {
				if (err) { debug(err); } else { debug('exportMagicNet: file saved to ' + fileNameWithoutExtension + '.def'); }
			});
					
		}

		/*
		// for debugging
		var j = JSON.parse($("#taexport").val());
		var m = new convnetjs.MagicNet();
		m.fromJSON(j);
		testEval(m);
		 */
	}

	// looks at a column i of data and guesses what's in it
	// returns results of analysis: is column numeric? How many unique entries and what are they?
	_.guessColumn = function (data, c) {	
		var numeric = true;
		var vs = [];
		for (var i = 0, n = data.length; i < n; i++) {
			var v = data[i][c];
			vs.push(v);
			if (isNaN(v))
				numeric = false;
		}
		var u = vs.unique();
		if (!numeric) {
			// if we have a non-numeric we will map it through uniques to an index
			return {
				numeric : numeric,
				num : u.length,
				uniques : u
			};
		} else {
			return {
				numeric : numeric,
				num : u.length
			};
		}
	}
	
}


// utility functions
Array.prototype.contains = function (v) {
	for (var i = 0; i < this.length; i++) {
		if (this[i] === v)
			return true;
	}
	return false;
};

Array.prototype.unique = function () {
	var arr = [];
	for (var i = 0; i < this.length; i++) {
		if (!arr.contains(this[i])) {
			arr.push(this[i]);
		}
	}
	return arr.sort();
}

// TODO: MOVE TO CONVNETJS UTILS
function randperm(n) {
	var i = n,
	j = 0,
	temp;
	var array = [];
	for (var q = 0; q < n; q++)
		array[q] = q;
	while (i--) {
		j = Math.floor(Math.random() * (i + 1));
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	return array;
}

var lastMessageNoNewLine = false;

function debug(msg) {
	if ( lastMessageNoNewLine ) {
		console.log('\n');
		lastMessageNoNewLine = false;
	}

	var d = new Date();
	var tb = traceback()[1]; // 1 because 0 should be your enterLog-Function itself
	console.log(d.toJSON() + ' ' + tb.file + ':' + tb.line + ':\t' + msg);
}

function debugObject(msg, obj) {
	debug(msg + '\n====================\n' + JSON.stringify(obj, null, 4) + '\n====================\n');
}

function debugNoNewline(msg) {
	lastMessageNoNewLine = true;
	process.stdout.write(msg);
}
