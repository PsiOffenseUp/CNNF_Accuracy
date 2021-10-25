using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNF.Neural_Net
{
	public delegate double ActivationFunction(double input); //Delegate type for activation functions. Can also be used for the partial derivatives of activation functions.
	public delegate double LossFunction(double output, double actual); //Delegate type for loss functions
	public delegate double MultiLossFunction(double[] output, double[] actual); //Delegate type for loss functions that take in multiple errors
	public delegate double AdaActivationFunction(double input, double feedback); //Delegate type for Ada functions
	public delegate double[,] MultiAdaActivationFunction(double[,] input, double[,] feedback); //Delegate type for Ada functions

	/// <summary> Static collection of activation functions and their corresponding partial derivatives </summary>
	public static class ActivationFunctions
	{
		public enum FunctionType { TANH, SIGMOID, DOUBLE_SIGMOID, DISCRETE_SIGNMOID, RELU, ADARELU, NONE } //Used for convenience in Layer class

		#region Hyperbolic Tangent methods
		public static double TanH(double input) { return (double)Math.Tanh(input); }
		public static double TanHDeriv(double x)
		{
			double hyperTan = (double)Math.Tanh(x); //Get the hyperbolic tangent so we don't call it twice
			return 1.0f - hyperTan * hyperTan;
		}

		public static double TanHDerivOut(double x)
		{
			return 1.0f - x * x;
		}
		#endregion

		#region  Sigmoid methods
		public static double Sigmoid(double x) { return 1.0f / (1.0f + (double)Math.Exp(-x)); }

		public static double SigmoidDeriv(double x)
		{
			double sigmoid = Sigmoid(x);
			return sigmoid * (1.0f - sigmoid);
		}

		public static double SigmoidDerivOut(double x)
		{
			return x * (1.0f - x);
		}
		#endregion

		#region Double sigmoid methods
		public static double DoubleSigmoid(double x) { return (2.0f / (1.0f + (double)Math.Exp(-x))) - 1.0f; }

		public static double DoubleSigmoidDeriv(double x)
		{
			double dSigmoid = DoubleSigmoid(x);
			return 0.5f * (1 - dSigmoid * dSigmoid);
		}
		#endregion

		#region  RELU methods

		public static double RELU(double x) { if (double.IsNaN(x) || x > 100.0) x = 100.0; return x > 0.0f ? x : 0.0f; }
		public static double RELUDeriv(double x) { return x > 0.0f ? 1.0f : 0.0f; }

		#endregion

		#region AdaRELU Methods

		public static double AdaRELU(double input, double feedback)
		{
			return feedback >= 0.0 ? RELU(input) : RELU(-input);
		}

		public static double AdaRELUDeriv(double input, double feedback)
		{
			return feedback >= 0.0 ? RELUDeriv(input) : RELUDeriv(-input);
		}
		#endregion

		#region Constant method (for input layer as safeguard)
		public static double Constant(double input) { return input; }
		#endregion
	}

}
