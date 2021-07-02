using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Drawing.Drawing2D;

namespace GaussBlur
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            beeps.Initialize();
        }
        private Bitmap curBitmap = null;
        private Bitmap srcBitmap = null;
        private string curFileName;

        BEEPS beeps = new BEEPS();
 
        //打开图像函数
        public void OpenFile()
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "所有图像文件 | *.bmp; *.pcx; *.png; *.jpg; *.gif;" +
                   "*.tif; *.ico; *.dxf; *.cgm; *.cdr; *.wmf; *.eps; *.emf|" +
                   "位图( *.bmp; *.jpg; *.png;...) | *.bmp; *.pcx; *.png; *.jpg; *.gif; *.tif; *.ico|" +
                   "矢量图( *.wmf; *.eps; *.emf;...) | *.dxf; *.cgm; *.cdr; *.wmf; *.eps; *.emf";
            ofd.ShowHelp = true;
            ofd.Title = "打开图像文件";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                curFileName = ofd.FileName;
                try
                {
                    srcBitmap = (Bitmap)System.Drawing.Image.FromFile(curFileName);
                    curBitmap = new Bitmap(srcBitmap);
                }
                catch (Exception exp)
                { MessageBox.Show(exp.Message); }
            }
        }
        //保存图像函数
        public void SaveFile()
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "位图文件(*.bmp)|*.bmp|Jpeg文件(*.jpg)|*.jpg|GIF文件(*.gif)|*.gif";
            if (sfd.ShowDialog() == DialogResult.OK)
                pictureBox1.Image.Save(sfd.FileName);
        }
      
        private void button1_Click(object sender, EventArgs e)
        {
            OpenFile();
            if(curBitmap!=null)
            {
                pictureBox1.Image = (Image)curBitmap;         
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            SaveFile();
        }
       

        private void button3_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                double v = Convert.ToDouble(textBox1.Text);
                int v1 = Convert.ToInt32(textBox2.Text);
                DateTime start = DateTime.Now;
                curBitmap = BEEPSProcess(srcBitmap, v, 8);
                DateTime end = DateTime.Now;
                label2.Text = "Time:" + (end - start).ToString();
                pictureBox1.Image = (Image)curBitmap;
            }
        }
        private Bitmap BEEPSProcess(Bitmap a, double sigma,int c)
        {
            Bitmap src = new Bitmap(a);
            System.Drawing.Imaging.BitmapData srcData = src.LockBits(new Rectangle(0, 0, src.Width, src.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            IntPtr ptr = srcData.Scan0;
            beeps.BEEPSProcess(ptr, srcData.Stride, src.Width, src.Height, sigma,c);
            src.UnlockBits(srcData);
            return src;
        }



        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            if (srcBitmap != null)
                pictureBox1.Image = srcBitmap;
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            if (curBitmap != null)
                pictureBox1.Image = curBitmap;
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            System.Diagnostics.Process.Start("https://blog.csdn.net/trent1985");
        }
        


    }
}
