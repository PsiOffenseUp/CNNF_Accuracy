using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CNNF_Accuracy
{
	public partial class Form1 : Form
	{
		public Form1()
		{
			InitializeComponent();
		}

		#region Console Text Methods

		static readonly Color consoleMessageColor = Color.FromArgb(70, 240, 80);
		static readonly Color consoleErrorColor = Color.FromArgb(250, 30, 40);
		static readonly Font consoleFont = new Font(FontFamily.GenericMonospace, 10.0f, FontStyle.Regular);

		/// <summary>  Puts text into the console panel </summary>
		/// <param name="message"></param>
		void ConsolePrint(string message)
		{
			Label label = new Label()
			{
				Text = message,
				BackColor = consoleMessageColor,
				Font = consoleFont
			};

			consolePanel.Controls.Add(label); //Add the newly created label to the console
		}

		/// <summary> Puts text into the console panel as an error</summary>
		void ConsoleErrorPrint(string message)
		{
			Label label = new Label()
			{
				Text = message,
				BackColor = consoleErrorColor,
				Font = consoleFont
			};

			consolePanel.Controls.Add(label); //Add the newly created label to the console
		}

		#endregion
	}


}
