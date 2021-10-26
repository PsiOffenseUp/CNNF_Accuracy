using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNF.Neural_Net.CNN
{
	public class ConvolutionalLayer
	{
		public FeatureMap featureMap { get; protected set; }
		protected WeightedConvolutionMatrix weights;
		public MaxPool pool { get; protected set; }

		//------------------- Methods ------------------
		/// <summary> Feeds input forward given a previous convolutional layer </summary>
		/// <param name="previousLayer"></param>
		public virtual void FeedForward(ConvolutionalLayer previousLayer)
		{
			featureMap.PopulateFeatureMap(weights, previousLayer.pool); //Generate the feature map
			featureMap.FillMaxPool(pool);
		}

		/// <summary> Feeds input forward given an input matrix or a latent feature representation of the same
		/// size as the input matrix </summary>
		public virtual void FeedForward(DataPoint2D input)
		{
			//TODO
		}

		//------------------- Constructors ------------------
		protected ConvolutionalLayer() { } //Default constructor for derived class to use
		public ConvolutionalLayer(int input_height, int input_width)
		{
			featureMap = new FeatureMap(input_height, input_width);
			pool = new MaxPool(input_height / MaxPool.maxPoolSize, input_width / MaxPool.maxPoolSize, featureMap);
			weights = new WeightedConvolutionMatrix(MaxPool.maxPoolSize);
		}
	}

	/// <summary>
	/// Convolutional Layer thaat uses 
	/// </summary>
	public class AdaConvolutionalLayer : ConvolutionalLayer
	{
		Feedback.FeedbackLayer feedbackLayer; //Corresponding feedback layer for this convolutional layer

		#region Methods
		/// <summary> Feeds input forward given a previous convolutional layer </summary>
		/// <param name="previousLayer"></param>
		public override void FeedForward(ConvolutionalLayer previousLayer)
		{
			featureMap.PopulateFeatureMap(weights, previousLayer.pool); //Generate the feature map
			featureMap.FillMaxPool(pool, feedbackLayer); //TODO Add Feedback Layer here
		}

		/// <summary> Feeds input forward given an input matrix or a latent feature representation of the same
		/// size as the input matrix </summary>
		public override void FeedForward(DataPoint2D input)
		{
			featureMap.PopulateFeatureMap(weights, input); //Generate the feature map
			featureMap.FillMaxPool(pool, feedbackLayer); //TODO Add Feedback Layer here
		}
		#endregion

		public AdaConvolutionalLayer(int input_height, int input_width, Feedback.FeedbackLayer feedback)
		{
			featureMap = new AdaFeatureMap(input_height, input_width, feedback);
			pool = new MaxPool(input_height / MaxPool.maxPoolSize, input_width / MaxPool.maxPoolSize, featureMap);
			this.feedbackLayer = feedback;
		}
	}
}
