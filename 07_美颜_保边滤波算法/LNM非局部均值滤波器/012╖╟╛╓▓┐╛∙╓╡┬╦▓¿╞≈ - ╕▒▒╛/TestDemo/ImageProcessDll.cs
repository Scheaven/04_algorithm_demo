using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TestDemo
{
    unsafe class ImageProcessDll
    {
        [DllImport("TestDemo_C.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode, ExactSpelling = true)]
        private static extern void f_NLMFilter(byte* srcData, int nWidth, int nHeight, int nStride, int dRadius, int sRadius, int h);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        public static Bitmap NLMFilter(Bitmap src, int dRadius, int sRadius, int h)
        {
            Bitmap dst = new Bitmap(src);
            BitmapData srcData = dst.LockBits(new Rectangle(0, 0, dst.Width, dst.Height), ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            f_NLMFilter((byte*)srcData.Scan0, dst.Width, dst.Height, srcData.Stride, dRadius, sRadius, h);
            dst.UnlockBits(srcData);
            return dst;
        }
       
    }
}
