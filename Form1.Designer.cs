namespace CNNF_Accuracy
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.panel1 = new System.Windows.Forms.Panel();
			this.panel2 = new System.Windows.Forms.Panel();
			this.label1 = new System.Windows.Forms.Label();
			this.panel3 = new System.Windows.Forms.Panel();
			this.panel5 = new System.Windows.Forms.Panel();
			this.consolePanel = new System.Windows.Forms.FlowLayoutPanel();
			this.panel2.SuspendLayout();
			this.SuspendLayout();
			// 
			// panel1
			// 
			this.panel1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(223)))), ((int)(((byte)(240)))), ((int)(((byte)(253)))));
			this.panel1.Location = new System.Drawing.Point(3, 51);
			this.panel1.Name = "panel1";
			this.panel1.Size = new System.Drawing.Size(458, 552);
			this.panel1.TabIndex = 0;
			// 
			// panel2
			// 
			this.panel2.Controls.Add(this.label1);
			this.panel2.Controls.Add(this.panel1);
			this.panel2.Location = new System.Drawing.Point(12, 12);
			this.panel2.Name = "panel2";
			this.panel2.Size = new System.Drawing.Size(464, 606);
			this.panel2.TabIndex = 1;
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Font = new System.Drawing.Font("Doppio One", 15.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
			this.label1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(253)))), ((int)(((byte)(253)))), ((int)(((byte)(253)))));
			this.label1.Location = new System.Drawing.Point(3, 13);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(196, 26);
			this.label1.TabIndex = 1;
			this.label1.Text = "Hyperparameters";
			// 
			// panel3
			// 
			this.panel3.Location = new System.Drawing.Point(482, 12);
			this.panel3.Name = "panel3";
			this.panel3.Size = new System.Drawing.Size(562, 55);
			this.panel3.TabIndex = 2;
			// 
			// panel5
			// 
			this.panel5.Location = new System.Drawing.Point(482, 74);
			this.panel5.Name = "panel5";
			this.panel5.Size = new System.Drawing.Size(562, 369);
			this.panel5.TabIndex = 4;
			// 
			// consolePanel
			// 
			this.consolePanel.AutoScroll = true;
			this.consolePanel.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(10)))), ((int)(((byte)(3)))), ((int)(((byte)(12)))));
			this.consolePanel.Location = new System.Drawing.Point(482, 449);
			this.consolePanel.Name = "consolePanel";
			this.consolePanel.Size = new System.Drawing.Size(562, 169);
			this.consolePanel.TabIndex = 1;
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(18)))), ((int)(((byte)(20)))), ((int)(((byte)(26)))));
			this.ClientSize = new System.Drawing.Size(1056, 630);
			this.Controls.Add(this.consolePanel);
			this.Controls.Add(this.panel5);
			this.Controls.Add(this.panel3);
			this.Controls.Add(this.panel2);
			this.Name = "Form1";
			this.Text = "CNNF Accuracy";
			this.panel2.ResumeLayout(false);
			this.panel2.PerformLayout();
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.Panel panel1;
		private System.Windows.Forms.Panel panel2;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Panel panel3;
		private System.Windows.Forms.Panel panel5;
		private System.Windows.Forms.FlowLayoutPanel consolePanel;
	}
}

