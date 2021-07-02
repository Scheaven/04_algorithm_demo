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
                    curBitmap = (Bitmap)System.Drawing.Image.FromFile(curFileName);

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
        public Bitmap PCluster(Bitmap a)
        {
            try
            {
                Rectangle rect = new Rectangle(0, 0, a.Width, a.Height);
                System.Drawing.Imaging.BitmapData bmpData = a.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                int stride = bmpData.Stride;
                unsafe
                {
                    byte* pIn = (byte*)bmpData.Scan0.ToPointer();
                    byte* P;
                    int R, G, B;
                    for (int y = 0; y < a.Height; y++)
                    {
                        for (int x = 0; x < a.Width; x++)
                        {
                            P = pIn;
                            B = P[0];
                            G = P[1];
                            R = P[2];
                            P[0] = (byte)(B & 248);  //屏蔽末6位
                            P[1] = (byte)(G & 248);
                            P[2] = (byte)(R & 248);
                            pIn += 3;

                        }
                        pIn += stride - a.Width * 3;
                    }


                }
                a.UnlockBits(bmpData);
                return a;
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message.ToString());
                return null;
            }
        }
        private void button1_Click(object sender, EventArgs e)
        {
            OpenFile();
            if(curBitmap!=null)
            {
                pictureBox1.Image = (Image)curBitmap;
                pictureBox1.Width = curBitmap.Width;
                pictureBox1.Height = curBitmap.Height;           
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
                DateTime start = DateTime.Now;
                Bitmap t = SkinGrindingProcess(curBitmap, v, 8);
                //Bitmap t = BEEPSProcess(curBitmap, v, 8);
                DateTime end = DateTime.Now;
                label2.Text = "Time:" + (end - start).ToString();
                pictureBox1.Image = (Image)t;
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
        private Bitmap SkinGrindingProcess(Bitmap a, double sigma, int c)
        {
            Bitmap src = new Bitmap(a);
            System.Drawing.Imaging.BitmapData srcData = src.LockBits(new Rectangle(0, 0, src.Width, src.Height), System.Drawing.Imaging.ImageLockMode.ReadWrite, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            IntPtr ptr = srcData.Scan0;
            beeps.SkinGrindProcess(ptr, srcData.Stride, src.Width, src.Height, sigma, c);
            //beeps.SkinDetectProcessAB(ptr, srcData.Stride, src.Width, src.Height);
            src.UnlockBits(srcData);
            return src;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                pictureBox1.Image = (Image)PCluster(curBitmap);
            }
        }
        


    }
}
