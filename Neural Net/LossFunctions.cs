using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNNF.Neural_Net
{
	/// <summary> Contains static loss functions for use with ML models </summary>
	public static class LossFunctions
	{
		#region Mean Square Error methods
		public static double SquareError(double output, double actual)
		{
			double diff = output - actual; //Take y_hat - y
			return 0.5f * diff * diff;
		}

		public static double SquareErrorDeriv(double output, double actual)
		{
			return output - actual;
		}

		#endregion
	}
}
