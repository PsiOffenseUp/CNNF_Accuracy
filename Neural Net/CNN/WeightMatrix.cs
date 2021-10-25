using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using CNNF.Neural_Net;
using CNNF.Neural_Net.Feedback;

namespace CNNF.Neural_Net.CNN
{
	public class Position
	{
		public int x;
		public int y;

		public void SetPosition(int x, int y) { this.x = x; this.y = y; }
	}

	public class WeightMatrix
	{
		public int width { get; private set; }
		public int height { get; private set; }
		public double[,] weights; //2D array for holding weights
		public double[] biases;
		static Random random = new Random();

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

		//Constructors
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
				biases[i] = (double)random.NextDouble() - 0.5f; //Generate random bias for the outpu node
				for (j = 0; j < width; j++)
					weights[i, j] = (double)random.NextDouble() - 0.5f; //Generate a random value on [0.5, 0.5]
			}
		}
	}


	/// <summary> Class for Feature Maps generated from DataPoint2Ds being given a WeightMatrix </summary>
	public class FeatureMap : LayerObject
	{
		public int width { get; private set; } //Length of a single dimension
		public int height { get; private set; } //Length of a single dimension		
		public double[,] axonValues;
		public double[,] dendriteValues;
		MaxPool maxPool;

		//----------------- Methods ----------------- 
		#region Max Pool Methods
		public void GetMaxPool()
		{
			int x, y;
			int hCount = width / MaxPool.maxPoolSize; //How many max pool spaces there will be horizontally
			int wCount = width / MaxPool.maxPoolSize;

			for (x = 0; x < hCount; x++)//Find the max of each square
				for (y = 0; y < wCount; y++)
					FindMax(x, y, maxPool.source); //Find where the max of this square is
		}

		public void GetAdaMaxPool(FeedbackLayer feedback)
		{
			int x, y;
			int hCount = width / MaxPool.maxPoolSize; //How many max pool spaces there will be horizontally
			int wCount = width / MaxPool.maxPoolSize;

			for (x = 0; x < hCount; x++)//Find the max of each square
				for (y = 0; y < wCount; y++)
				{
					//TODO Go through feedback and find min or max where relevant
					FindMax(x, y, maxPool.source); //Find where the max of this square is
				}
					
		}

		//Finds the max for a max pool at the given x and y position
		void FindMax(int x, int y, Position[,] locationArr)
		{
			int topLeftX = x * MaxPool.maxPoolSize, topLeftY = y * MaxPool.maxPoolSize;
			locationArr[y, x].SetPosition(topLeftX, topLeftY);
			double max = axonValues[topLeftY, topLeftX];

			int i, j; //Offset from the top left
			for(i = 0; i < MaxPool.maxPoolSize; i++)
				for(j = 0; j < MaxPool.maxPoolSize; j++)
				{
					//If we find a new max, record it
					if(axonValues[topLeftY + i, topLeftX + j] > max)
					{
						max = axonValues[topLeftY + i, topLeftX + j];
						locationArr[y, x].SetPosition(topLeftX + j, topLeftY + i);
					}
				}
		}

		//Finds the min for a min pool at the given x and y position
		void FindMin(int x, int y, Position[,] locationArr)
		{
			int topLeftX = x * MaxPool.maxPoolSize, topLeftY = y * MaxPool.maxPoolSize;
			locationArr[y, x].SetPosition(topLeftX, topLeftY);
			double min = axonValues[topLeftY, topLeftX];

			int i, j; //Offset from the top left
			for (i = 0; i < MaxPool.maxPoolSize; i++)
				for (j = 0; j < MaxPool.maxPoolSize; j++)
				{
					//If we find a new max, record it
					if (axonValues[topLeftY + i, topLeftX + j] < min)
					{
						min = axonValues[topLeftY + i, topLeftX + j];
						locationArr[y, x].SetPosition(topLeftX + j, topLeftY + i);
					}
				}
		}

		#endregion

		#region Other Methods
		public override string ToString()
		{
			string result = "";
			int i, j;

			for (i = 0; i < height; i++)
			{
				result += "| ";
				for (j = 0; j < width; j++)
					result += $"{Math.Round(axonValues[i, j], 4)} ";

				result += "|\n";
			}

			return result;
		}

		/// <summary> Applies the activation function to all values </summary>
		public override void ApplyActivationFunction()
		{
			//TODO Parallelize
			int i, j;
			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++)
					axonValues[i, j] = activationFunction(dendriteValues[i, j]);
		}

		public void PopulateFeatureMap(WeightMatrix matrix, DataPoint2D input)
		{
			int i, j, k, l;

			for (i = 0; i < height; i++) //Rows of the feature map
				for (j = 0; j < width; j++) //Columns of input
				{
					//Go through the current sub image with (j, i) as the coordinate of the top left
					dendriteValues[i, j] = 0.0f; //Initialize value to 0
					for (k = 0; k < CNN.subimageSize; k++) //Subimage row
						for (l = 0; l < CNN.subimageSize; l++) //Subimage column
							dendriteValues[i, j] += input.input[i + k, j + l] * matrix.weights[l, k];

					dendriteValues[i, j] += matrix.biases[0]; //Add in the bias (should be just one number)
				}

			ApplyActivationFunction();
		}

		#endregion

		//Constructors
		public FeatureMap(int height, int width)
		{
			this.width = width;
			this.height = height;

			//Allocate relevant fields
			this.axonValues = new double[height, width];
			this.dendriteValues = new double[height, width];
			this.maxPool = new MaxPool(height / MaxPool.maxPoolSize, width / MaxPool.maxPoolSize, this);
		}
	}

	/// <summary> Extension of feature map that uses Ada </summary>
	public class AdaFeatureMap : FeatureMap
	{
		public AdaFeatureMap(int height, int width) : base(height, width)
		{

		}
	}

	/// <summary>
	/// Class for all Max Pools that will be part of convolutional layers
	/// </summary>
	public class MaxPool
	{
		public const int maxPoolSize = 2; //Size of single dimension of Max Pool
		FeatureMap sourceMap; //Reference to the FeatureMap that this MaxPool uses
		public Position[,] source { get; private set; } //Index into the FeatureMap that generated this MaxPool. Used to quickly identify source value
		public int height { get; private set; } //Length as a square
		public int width { get; private set; } //Length as a square
		public int count { get; private set; } //Total number of things in the MaxPool
		public double this[int y, int x] { get { return sourceMap.axonValues[source[y, x].y, source[y, x].x]; } }

		public override string ToString()
		{
			string result = "";
			int i, j;

			for (i = 0; i < height; i++)
			{
				result += "| ";
				for (j = 0; j < width; j++)
					result += $"{Math.Round(sourceMap.axonValues[source[i, j].y, source[i,j].x], 4)} ";

				result += "|\n";
			}

			return result;
		}

		public MaxPool(int height, int width, FeatureMap source)
		{
			this.width = width;
			this.height = height;
			this.count = width * height;

			//Allocate arrays
			sourceMap = source;
			this.source = new Position[height, width]; //Where the max value came from

			//Create all of the Position objects
			int i, j;
			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++)
					this.source[i, j] = new Position();
		}

	}
}
