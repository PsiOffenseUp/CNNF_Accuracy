using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNF.Neural_Net
{
	//Base class for layer-like objects
	public abstract class LayerObject
	{
		public ActivationFunction activationFunction { get; protected set; } //Activation function to be applied to all 
		public ActivationFunction activationDerivative { get; protected set; }

		protected static ActivationFunctions.FunctionType defaultActivationFunction = ActivationFunctions.FunctionType.RELU;
		public static void SetDefaultActivationFunction(ActivationFunctions.FunctionType function_type) { defaultActivationFunction = function_type; }

		/// <summary> Sets the acitvation function and its corresponding partial derivative for this layer </summary>
		/// <param name="function_type"></param>
		public void SetActivationFunction(ActivationFunctions.FunctionType function_type)
		{
			switch (function_type)
			{
				case ActivationFunctions.FunctionType.TANH:
					activationFunction = ActivationFunctions.TanH;
					//activationDerivative = ActivationFunctions.TanHDeriv;
					activationDerivative = ActivationFunctions.TanHDerivOut;
					break;
				case ActivationFunctions.FunctionType.SIGMOID:
					activationFunction = ActivationFunctions.Sigmoid;
					//activationDerivative = ActivationFunctions.SigmoidDeriv;
					activationDerivative = ActivationFunctions.SigmoidDerivOut;
					break;
				case ActivationFunctions.FunctionType.DOUBLE_SIGMOID:
					activationFunction = ActivationFunctions.DoubleSigmoid;
					activationDerivative = ActivationFunctions.DoubleSigmoidDeriv;
					break;
				case ActivationFunctions.FunctionType.DISCRETE_SIGNMOID:
					break;
				case ActivationFunctions.FunctionType.RELU:
					activationFunction = ActivationFunctions.RELU;
					activationDerivative = ActivationFunctions.RELUDeriv;
					break;
				case ActivationFunctions.FunctionType.NONE:
					activationFunction = activationDerivative = ActivationFunctions.Constant;
					break;
				default:
					break;
			}
		}

		/// <summary> Applies the activation function to all dendrite values, setting the axon values </summary>
		/// <returns></returns>
		public abstract void ApplyActivationFunction();
	}
}
