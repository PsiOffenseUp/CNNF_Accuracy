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
		public MaxPool pool { get; protected set; }

		//------------------- Methods ------------------

		/// <summary> Feeds input forward given a previous convolutional layer </summary>
		/// <param name="previousLayer"></param>
		public void FeedForward(ConvolutionalLayer previousLayer)
		{
			//TODO
		}

		/// <summary> Feeds input forward given an input matrix or a latent feature representation of the same
		/// size as the input matrix </summary>
		public void FeedForward(DataPoint2D input)
		{
			//TODO
		}

		//------------------- Constructors ------------------
		public ConvolutionalLayer(int input_height, int input_width)
		{
			featureMap = new FeatureMap(input_height, input_width);
			pool = new MaxPool(input_height / MaxPool.maxPoolSize, input_width / MaxPool.maxPoolSize, featureMap);
		}
	}
}
