using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using CNNF.Neural_Net;
using CNNF.Neural_Net.Feedback;

namespace CNNF.Neural_Net
{
	public class Position
	{
		public int x;
		public int y;

		public void SetPosition(int x, int y) { this.x = x; this.y = y; }
	}

	public class WeightMatrix
	{
		public int width { get; protected set; }
		public int height { get; protected set; }
		public double[,] weights; //2D array for holding weights
		public double[] biases;
		protected static Random random = new Random();

		//Operator overloads

		//ToString override
		public override string ToString()
		{
			string output = "";
			int i, j;

			for (i = 0; i < height; i++)
			{
				output += "| ";

				for (j = 0; j < width; j++)
					output += $"\t{weights[i, j]}";

				output += $"|\t|{ biases[i]}|\n";
			}

			return output;
		}

		//Gets a string for writing to a file
		public string GetWeightString()
		{
			int j, k;
			string result = $"{width} {height} {biases.Length}\n"; //Write some header info for this matrix

			for (j = 0; j < height; j++)
			{
				for (k = 0; k < width; k++)
					result += $"{weights[j, k]} ";
				result += $"{biases[j]}\n";
			}

			return result;
		}

		/// <summary> Reads a weight matrix from a file </summary>
		/// <returns></returns>
		public WeightMatrix(StreamReader reader)
		{
			string[] line = reader.ReadLine().Split(' '); //Read the header line
			width = Int32.Parse(line[0]);
			height = Int32.Parse(line[1]);
			int biasCount = Int32.Parse(line[2]);

			//Allocate the arrays
			weights = new double[height, width];
			biases = new double[biasCount];

			//Go through an set all of the values
			int j, k;

			for (j = 0; j < height; j++)
			{
				line = reader.ReadLine().Split(' ');
				for (k = 0; k < width; k++)
					weights[j, k] = double.Parse(line[k]);
				biases[j] = double.Parse(line[k]);
			}
		}

		#region Constructors
		public WeightMatrix(int input_layer_size, int output_layer_size)
		{
			this.height = output_layer_size;
			this.width = input_layer_size; //Create a width for the input layer size and also a constant

			this.weights = new double[height, width]; //Create a new weight array
			this.biases = new double[height];

			//Initialize all of the weights
			int i, j;

			for (i = 0; i < height; i++)
			{
				biases[i] = (double)random.NextDouble() - 0.5f; //Generate random bias for the output node
				for (j = 0; j < width; j++)
					weights[i, j] = (double)random.NextDouble() - 0.5f; //Generate a random value on [0.5, 0.5]
			}
		}

		protected WeightMatrix() { }

		#endregion
	}

	public class WeightedConvolutionMatrix : WeightMatrix
	{
		public WeightedConvolutionMatrix(int size)
		{
			this.height = size;
			this.width = size; //Create a width for the input layer size and also a constant

			this.weights = new double[size, size]; //Create a new weight array
			this.biases = new double[1]; //Only need 1 bias

			//Initialize all of the weights
			int i, j;
			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++)
					weights[i, j] = (double)random.NextDouble() - 0.5f; //Generate a random value on [0.5, 0.5]

			biases[0] = (double)random.NextDouble() - 0.5f; //Generate random bias for the output node
		}
	}
}
