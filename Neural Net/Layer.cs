using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNNF.Neural_Net.CNN;

namespace CNNF.Neural_Net
{
	public class InnerDimensionException : Exception { }

	#region Data Point Classes
	/// <summary> DataPoint with matching input and actual output </summary>
	public abstract class DataPointBase
	{
		public double[] output { get; protected set; }
		public int outputSize { get; protected set; }

		protected DataPointBase(int output_size) 
		{
			outputSize = output_size;
			output = new double[output_size]; 
		}
	}

	/// <summary> DataPoint with matching input and actual output </summary>
	public class DataPoint : DataPointBase
	{
		public double[] input { get; protected set; }
		public int inputSize { get; protected set; }

		public DataPoint(int input_size, int output_size) : base(output_size)
		{
			inputSize = input_size;
			input = new double[input_size];
		}
	}

	/// <summary> Class for 2D data points. Used primarly for CNNs </summary>
	public class DataPoint2D : DataPointBase
	{
		public double[,] input { get; protected set; }
		public int width { get; protected set; }
		public int height { get; protected set; }

		public DataPoint2D(int input_width, int input_height, int output_size) : base(output_size)
		{
			input = new double[input_height, input_width];
		}
	}

	#endregion

	public class Layer : LayerObject
	{

		//Member variables
		public int size { get; protected set; } //Number of neurodes in this layer, minus the constant
		public double[] dendriteValues { get; protected set; } //Values received prior to activation function
		public double[] axonValues { get; protected set; } //Values with activation function applied
		public double[] deltaValues{ get; protected set; } //Backprop delta values

		#region Normal Methods
		public override void ApplyActivationFunction()
		{
			//axonValues = new double[dendriteValues.Length];

			for (int i = 0; i < dendriteValues.Length; i++)
				axonValues[i] = activationFunction(dendriteValues[i]);
		}

		/// <summary> Makes the layer not use any activation function and adjusts the output side values accordingly </summary>
		public void MakeConstant()
		{
			activationFunction = activationDerivative = ActivationFunctions.Constant;
			axonValues = dendriteValues;
		}

		public void OverwriteAxonValues(double[] new_values)
		{
			axonValues = new_values;
		}

		/// <summary> Constructs a Layer with all values as zero </summary>
		/// <param name="size"></param>
		/// <returns></returns>
		public static Layer MakeZeroLayer(int size)
		{
			Layer returnLayer = new Layer(size, defaultActivationFunction);

			for (int i = 0; i < size; i++)
				returnLayer.dendriteValues[i] = 0.0;

			return returnLayer;
		}

		public override string ToString()
		{
			string output = "[";
			for (int i = 0; i < size; i++)
				output += $" {axonValues[i]}";

			output += " ]";

			return output;
		}

		/// <summary> Zeroes out the dendrite and delta arrays</summary>
		public void ZeroVectors()
		{
			for (int i = 0; i<size; i++)
			{
				dendriteValues[i] = 0.0;
				deltaValues[i] = 0.0;
			}
		}
		#endregion

		#region Methods that Populate Values

		/// <summary> Multiplies the layer by all of the weights to generate the next Layer. Throws an InnerDimensionException on error </summary>
		/// <param name="layer"></param>
		/// <param name="matrix"></param>
		/// <returns></returns>
		public void PopulateLayer(WeightMatrix matrix, Layer inputLayer)
		{
			if (inputLayer.size != matrix.width) //If the dimensions don't match, throw an exception. Subtract 1 to account for constants
				throw new InnerDimensionException();

			int i, j;

			//Initialize all output values to 0
			for (i = 0; i < size; i++)
				dendriteValues[i] = 0.0f;

			//Multiply the input by the matrix
			for (i = 0; i < matrix.height; i++) //Output layer loops
			{
				for (j = 0; j < matrix.width; j++) //Input layer loops
					dendriteValues[i] += matrix.weights[i, j] * inputLayer.axonValues[j];

				dendriteValues[i] += matrix.biases[i]; //Make sure to add the bias for this node
			}

			ApplyActivationFunction(); //Apply the activation function to all outputs
		}


		public void PopulateLayer(WeightMatrix matrix, MaxPool pool)
		{
			int i, j;

			//Initialize all values to 0
			for (i = 0; i < size; i++)
				dendriteValues[i] = 0.0f;

			for (i = 0; i < matrix.height; i++)
			{
				for (j = 0; j < matrix.width; j++)
				{
					dendriteValues[i] += matrix.weights[i, j] * pool[i, j];
				}
					

				dendriteValues[i] += matrix.biases[i]; //Make sure to add the bias for this node
			}

			Layer returnLayer = new Layer(values);

			return returnLayer; //Create a layer out of all of the values that we calculated and apply the activation function
		}
		#endregion

		#region Operators

		/// <summary>
		/// Adds the lhs Layer to the rhs Layer and stores the result in the passed result Layer, overwriting any values
		/// </summary>
		/// <param name="lhs"></param>
		/// <param name="rhs"></param>
		/// <param name="result"></param>
		public static void Add(Layer lhs, Layer rhs, Layer result)
		{
			if (lhs.size != rhs.size)
				throw new InnerDimensionException();

			//Layer returnLayer = new Layer(lhs.size, defaultActivationFunction);

			for (int i = 0; i < lhs.size; i++)
				result.dendriteValues[i] = lhs.dendriteValues[i] + rhs.dendriteValues[i];
		}

		#endregion

		#region Constructors ------------
		public Layer(int size, ActivationFunctions.FunctionType activation_function_type)
		{
			this.size = size;

			//Allocate all of the arrays
			dendriteValues = new double[size];
			axonValues = new double[size];
			deltaValues = new double[size];

			SetActivationFunction(activation_function_type);
		}

		public Layer(int size) : this(size, defaultActivationFunction) { }

		public Layer(double[] values, ActivationFunctions.FunctionType activation_function_type)
		{
			this.size = values.Length;
			this.dendriteValues = values;
			this.axonValues = values;

			SetActivationFunction(activation_function_type);
		}

		public Layer(double[] values) : this(values, defaultActivationFunction) { }

		#endregion
	}
	//IEnumerable<double>

}
