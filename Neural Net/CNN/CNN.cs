using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using CNNF.Neural_Net.Feedback;

namespace CNNF.Neural_Net.CNN
{
	public class CNNOutput
	{
		public DataPoint2D input;
		double[] output;
	}

	/// <summary> Single layer artificial neural network </summary>
	public class CNN
	{
		public const double learningRate = 0.0015f;
		public int hiddenNodes { get; private set; } //Number of neurodes that should be on the hidden layer
		public int convolutionalLayerDepth { get; private set; } //Number of conv layers per time step. For normal CNN, just one time step
		public int convolutionalLayerWidth { get; private set; } //Number of sub conv layers per conv layer
		protected int inputWidth, outputWidth;
		public const int subimageSize = 3; //Dimension of subimage to feature maps

		//Layers of the CNN
		ConvolutionalLayer[,] convolutionalLayers;
		Layer feedForwardLayer;

		public LossFunction lossFunction;
		public LossFunction lossDerivative;

		#region ***************** Propogation Methods *****************

		/// <summary> Feeds the input vector through the neural net and returns an array of size 3
		/// of all of the Layers that were used </summary>
		/// <param name="input_vector"></param>
		/// <returns></returns>
		public CNNOutput FeedForward(MNISTChar input)
		{
			//Create the output object
			CNNOutput output = new CNNOutput(mapCount);
			output.input = input;
			int i; //For iteration

			output.hiddenLayer = Layer.MakeZeroLayer(hiddenNodes);

			//Do the feature maps then max pool. Multiply each by their corresponding weight matrix, then generate the hidden layer
			for (i = 0; i < mapCount; i++)
			{
				output.maps[i] = featureWeights[i] * input;
				output.pools[i] = output.maps[i].GetMaxPool();
				output.hiddenLayer += poolWeights[i] * output.pools[i];
			}

			//Do the output layer now that the hidden layer is found
			output.hiddenLayer.ApplyActivationFunction(); //Apply the activation function now that all values are found
			output.outputLayer = secondWeights * output.hiddenLayer; //Calculate the output layer

			//Adjust for softmax
			double[] values = output.outputLayer.dendriteValues; //Get the pre-activation values to use for softmax
			output.outputLayer.OverwriteAxonValues(SoftMax(values)); //Replace the output layer with one that uses softmax

			return output; //Return all of the Layers
		}

		/// <summary>
		/// Takes the error signals output by a FeedForward run of an ANN and backpropagates it, updating all weights using
		/// calculated gradients. </summary>
		/// <param name="errors"></param>
		public void BackPropagate(CNNOutput output)
		{
			int i, j, k;
			int featureIndex;

			//Calculate the deltas for the output layer
			double[] softmaxDerivValues = SoftMaxDerivOut(output.outputLayer.axonValues); //Get the softmax derivative values for the whole output
			for (k = 0; k < outputWidth; k++)
				output_deltas[k] = lossDerivative(output.outputLayer.axonValues[k], output.input.data.output[k]) * softmaxDerivValues[k];

			//Calculate the deltas for the hidden layer
			for (j = 0; j < hiddenNodes; j++)
			{
				hidden_deltas[j] = 0.0; //Initialize to 0 before we take the sum from all of the nodes
				for (k = 0; k < outputWidth; k++) //Take delta from last layer times weight used to get to node j in hidden layer
					hidden_deltas[j] += output_deltas[k] * secondWeights.weights[k, j];
				hidden_deltas[j] *= output.hiddenLayer.activationDerivative(output.hiddenLayer.axonValues[j]); //Multiply the derivative of the activation function for this node
			}

			//Calculate deltas for the max pools/ feature maps. Will avoid calculating all feature map deltas to save on time/space
			for (i = 0; i < mapCount; i++) //Go through each pool/ map
			{
				for (k = 0; k < output.pools[i].values.Length; k++) //Go through all of the elements in the max pool vector
				{
					feature_deltas[i, k] = 0.0; //Initialize to 0
					for (j = 0; j < hiddenNodes; j++) //Take the sum over all hidden nodes' error signals
						feature_deltas[i, k] += poolWeights[i].weights[j, k] * hidden_deltas[j];
					feature_deltas[i, k] *= output.maps[i].activationDerivative(output.pools[i].values[k]);
				}
			}

			//########################## Adjust the weights #######################

			double nodeAcc; //Used to store values for a node so we don't calculate twice

			//Adjust the weights between the input and the feature maps
			for (k = 0; k < mapCount; k++) //Go through each map
			{
				for (i = 0; i < output.pools[k].values.Length; i++) //Go through the elements of each map (just the maxes)
				{
					featureIndex = output.pools[k].source[i]; //Get the index that was used for the current delta. Also top left of MNIST subimage

					for (j = 0; j < subimageSize * subimageSize; j++) //Update the weights connected to the subimage that generate this feature element
						featureWeights[k].weights[featureIndex, j] -= learningRate * feature_deltas[k, i] * output.input.data.input[featureIndex + (j / subimageSize) * 28 + (j % subimageSize)];

					featureWeights[k].biases[featureIndex] -= learningRate * feature_deltas[k, i]; //Update the bias
				}
			}

			//Adjust the weights and biases between the feature maps and the hidden layer
			for (k = 0; k < mapCount; k++)
			{
				for (j = 0; j < hiddenNodes; j++)
				{
					//Weights and biases share delta of node * learning rate * activationDeriv to get to input side of node
					nodeAcc = learningRate * hidden_deltas[j];

					for (i = 0; i < poolDimension * poolDimension; i++) //Apply eta * delta_j * dyj/dsj * dsj/dwij = eta * delta_j * activationDeriv(sj) * Xi
						poolWeights[k].weights[j, i] -= nodeAcc * output.pools[k].values[i];

					//Update the bias similarly
					poolWeights[k].biases[j] -= nodeAcc; //Deriv of input side of node with respect to bias is 1, so no extra multiplication
				}
			}

			//Now adjust the weights and biases between the hidden and output layers
			for (k = 0; k < outputWidth; k++)
			{
				for (j = 0; j < hiddenNodes; j++) //Apply eta * delta_k * y_j
					secondWeights.weights[k, j] -= learningRate * output_deltas[k] * output.hiddenLayer.axonValues[j];

				secondWeights.biases[k] -= learningRate * output_deltas[k]; //Apply deltas to the biases, again because deriv is 1
			}
		}

		/// <summary> Returns a vector of errors between generated output layer and expected values using the loss function for this neural net. </summary>
		/// <param name="outputs"></param>
		/// <param name="expected_values"></param>
		/// <returns></returns>
		public double[] CalculateErrors(double[] outputs, double[] expected_values)
		{
			double[] errors = new double[outputs.Length];

			for (int i = 0; i < errors.Length; i++)
				errors[i] = lossFunction(outputs[i], expected_values[i]);

			return errors;
		}

		#endregion

		#region ***************** String Methods *****************
		/// <summary> ToString override for printing </summary>
		/// <returns></returns>
		public override string ToString()
		{
			return "";
		}

		public string GetWeightsString()
		{
			/*int i, j, k;
			string result = "";

			//Get the weights between the input and the feature maps
			for (i = 0; i < mapCount; i++)
				result += featureWeights[i].GetWeightString();

			//Get the weights between the max pools and hidden layer
			for (i = 0; i < mapCount; i++)
				result += poolWeights[i].GetWeightString();

			result += secondWeights.GetWeightString();

			return result;*/
			return "";
		}
		#endregion

		#region ***************** SoftMax Methods *****************
		/// <summary> Applies softmax to the vector and returns all of the softmaxes as a vector
		/// </summary>
		/// <param name="vector"></param>
		public static double[] SoftMax(double[] vector)
		{
			double sum = 0;
			double[] exps = new double[vector.Length];
			int i;

			for (i = 0; i < vector.Length; i++)
			{
				//Take exp of vector and try to prevent overflow that would create Infinity or NaN
				exps[i] = Math.Exp(vector[i]);
				if (double.IsInfinity(exps[i]))
					exps[i] = 5.0;
				else if (double.IsNaN(exps[i]))
					exps[i] = -5.0;

				sum += exps[i];
			}

			for (i = 0; i < vector.Length; i++)
				exps[i] = exps[i] / sum;

			return exps;
		}

		/// <summary> 
		/// Applies the softmax derivative based on all of the values in the vector and returns a new vector with the results
		/// </summary>
		/// <param name="vector"></param>
		/// <returns></returns>
		public static double[] SoftMaxDeriv(double[] vector)
		{
			double[] z = SoftMax(vector); //Find the zs
			double[] values = new double[z.Length];
			int i, j;

			for (i = 0; i < z.Length; i++)
			{
				values[i] = z[i] * (1.0f - z[i]);
				/*
				values[i] = 0.0f;
				for (j = 0; j < z.Length; j++)
				{ 
					double temp = (i == j) ? z[i] * (1.0f - z[i]) : -z[i] * z[j];
					values[i] += temp;
				} 
				*/

				if (double.IsNaN(values[i]))
					Console.WriteLine("Got NaN!");
			}

			return values;
		}

		public static double[] SoftMaxDerivOut(double[] z)
		{
			double[] values = new double[z.Length];
			int i, j;

			for (i = 0; i < z.Length; i++)
			{
				values[i] = z[i] * (1.0f - z[i]);

				if (double.IsNaN(values[i]))
					Console.WriteLine("Got NaN!");
			}

			return values;
		}
		#endregion

		#region ***************** Constructors *****************
		public CNN(int hidden_nodes, int input_width, int output_width, int feature_maps, int number_of_conv_layers)
		{
			//Set all of the variables passed
			this.hiddenNodes = hidden_nodes;
			this.inputWidth = input_width;
			this.outputWidth = output_width;
			this.convolutionalLayerWidth = feature_maps;
			this.convolutionalLayerDepth = number_of_conv_layers;

			//Allocate convolutional layers

			//Allocate last fully connected layer
		}

		public CNN(int hidden_nodes, int maps, StreamReader reader)
		{
			//TODO
		}
		#endregion
	}

	public class CNNF : CNN
	{
		public int timeSteps { get; private set; } //Number of total time steps
		FeedbackLayer[] feedbackLayers;
	}
}
