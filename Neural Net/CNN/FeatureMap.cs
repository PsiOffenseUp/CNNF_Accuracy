using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNNF.Neural_Net;
using CNNF.Neural_Net.Feedback;

namespace CNNF.Neural_Net.CNN
{

	/// <summary> Class for Feature Maps generated from DataPoint2Ds being given a WeightMatrix </summary>
	public class FeatureMap : LayerObject
	{
		public int width { get; private set; } //Length of a single dimension
		public int height { get; private set; } //Length of a single dimension		
		public double[,] axonValues;
		public double[,] dendriteValues;

		//----------------- Methods ----------------- 
		#region Max Pool Methods
		public virtual void FillMaxPool(MaxPool pool, FeedbackLayer feedback = null)
		{
			int x, y;
			int hCount = width / MaxPool.maxPoolSize; //How many max pool spaces there will be horizontally
			int wCount = width / MaxPool.maxPoolSize;

			for (x = 0; x < hCount; x++)//Find the max of each square
				for (y = 0; y < wCount; y++)
					FindMax(x, y, pool.source); //Find where the max of this square is
		}

		//Finds the max for a max pool at the given x and y position
		protected void FindMax(int x, int y, Position[,] locationArr)
		{
			int topLeftX = x * MaxPool.maxPoolSize, topLeftY = y * MaxPool.maxPoolSize;
			locationArr[y, x].SetPosition(topLeftX, topLeftY);
			double max = axonValues[topLeftY, topLeftX];

			int i, j; //Offset from the top left
			for (i = 0; i < MaxPool.maxPoolSize; i++)
				for (j = 0; j < MaxPool.maxPoolSize; j++)
				{
					//If we find a new max, record it
					if (axonValues[topLeftY + i, topLeftX + j] > max)
					{
						max = axonValues[topLeftY + i, topLeftX + j];
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

		/// <summary> Populates the values in this feature map using a given weight matrix and an input </summary>
		/// <param name="matrix"></param>
		/// <param name="input"></param>
		public void PopulateFeatureMap(WeightMatrix matrix, DataPoint2D input)
		{
			int i, j, k, l;

			for (i = 0; i < height; i++) //Rows of the feature map
				for (j = 0; j < width; j++) //Columns of input
				{
					//Go through the current sub image with (j, i) as the coordinate of the top left
					dendriteValues[i, j] = 0.0; //Initialize value to 0
					for (k = 0; k < CNN.subimageSize; k++) //Subimage row
						for (l = 0; l < CNN.subimageSize; l++) //Subimage column
							dendriteValues[i, j] += input.input[i + k, j + l] * matrix.weights[l, k];

					dendriteValues[i, j] += matrix.biases[0]; //Add in the bias (should be just one number)
				}

			ApplyActivationFunction();
		}

		public void PopulateFeatureMap(WeightMatrix matrix, MaxPool pool)
		{
			int i, j, k, l;

			for (i = 0; i < height; i++) //Rows of the feature map
				for (j = 0; j < width; j++) //Columns of input
				{
					//Go through the current sub image with (j, i) as the coordinate of the top left
					dendriteValues[i, j] = 0.0; //Initialize value to 0
					for (k = 0; k < CNN.subimageSize; k++) //Subimage row
						for (l = 0; l < CNN.subimageSize; l++) //Subimage column
							dendriteValues[i, j] += pool[i + k, j + l] * matrix.weights[l, k];

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
		}
	}

	/// <summary> Extension of feature map that uses Ada </summary>
	public class AdaFeatureMap : FeatureMap
	{
		FeedbackLayer feedback; //Reference to the feedback layer need
		public AdaFeatureMap(int height, int width, FeedbackLayer feedback) : base(height, width)
		{
			this.feedback = feedback; //Store the reference
		}

		#region Methods
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

		public override void FillMaxPool(MaxPool pool, FeedbackLayer feedback)
		{
			int x, y;
			int hCount = width / MaxPool.maxPoolSize; //How many max pool spaces there will be horizontally
			int wCount = width / MaxPool.maxPoolSize;

			for (x = 0; x < hCount; x++)//Find the max of each square
				for (y = 0; y < wCount; y++)
				{
					//TODO Go through feedback and find min or max where relevant
					FindMax(x, y, pool.source); //Find where the max of this square is
				}
		}

		public override void ApplyActivationFunction()
		{
			//TODO Parallelize
			int i, j;
			for (i = 0; i < height; i++)
				for (j = 0; j < width; j++)
					axonValues[i, j] = ActivationFunctions.AdaRELU(dendriteValues[i, j], feedback); //TODO Access feedback layer
		}

		#endregion
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
					result += $"{Math.Round(sourceMap.axonValues[source[i, j].y, source[i, j].x], 4)} ";

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
