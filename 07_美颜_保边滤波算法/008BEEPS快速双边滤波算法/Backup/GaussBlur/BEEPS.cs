using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GaussBlur
{
    class BEEPS
    {
        private double[] gaussMap = new double[65025];
        private static int mSkinMin = 256;
        private static int mSkinMax = 0;
        private static int mSkinAverage = 0;
        public void Initialize()
        {
            for (int i = 0; i < gaussMap.Length; i++)
            {
                gaussMap[i]=Math.Exp(-i);
            }
        }
        public unsafe void BEEPSProcess(IntPtr srcPtr, Int32 stride, Int32 width, Int32 height, double sigma,int c)
        {
            sigma = sigma > 50 ? 50 : sigma;
            byte* pSrc = (byte*)srcPtr.ToPointer();
            byte[] dstValue = new byte[height * stride];
            byte[] hValue = BEEPSHorizontal(pSrc, stride, width, height, sigma, c);
            byte[] vValue;
            fixed (byte* p = hValue)
            {
            vValue = BEEPSVertical(p, stride, width, height, sigma, c);
            }
            hValue = BEEPSVertical(pSrc, stride, width, height, sigma, c);
            fixed (byte* p = hValue)
            {
                dstValue = BEEPSHorizontal(p, stride, width, height, sigma, c);
            }
            for (int i = 0; i < dstValue.Length; i++)
            {
                *pSrc = (byte)((vValue[i] + dstValue[i]) / 2);
                pSrc++;
            }
        }
        private unsafe byte[] BEEPSHorizontal(byte* srcPtr, Int32 stride, Int32 width, Int32 height, double sigma, int c)
        {
            byte[] F = new byte[height * stride];
            byte[] D = new byte[height * stride];
            int[] s = new int[width * 3];
            int[] v = new int[width * 3];
            int pos = 0, X = 0, Y = 0;
            int p = 0;
            byte* data;
            sigma = sigma * sigma * 2.0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 1; x < width; x++)
                {
                    X = width - 1 - x;
                    Y = height - 1 - y;
                    if (x == 0)
                    {
                        pos = x * 3 + y * stride;
                        data = srcPtr + pos;
                        F[pos] = data[0];
                        s[0] = data[0];
                        ++pos;
                        F[pos] = data[1];
                        s[1] = data[1];
                        ++pos;
                        F[pos] = data[2];
                        s[2] = data[2];

                        p = X * 3;
                        pos = p + Y * stride;
                        data = srcPtr + pos;
                        v[p] = data[0];
                        D[pos] = data[0];
                        ++pos;
                        ++p;
                        v[p] = data[1];
                        D[pos] = data[1];
                        ++pos;
                        ++p;
                        v[p] = data[2];
                        D[pos] = data[2];
                    }
                    else
                    {
                        p = x * 3;
                        pos = p + y * stride;
                        data = srcPtr + pos;
                        s[p] = (int)(10.0 * Gaussian(data[0], F[pos - 3], sigma));
                        F[pos] = (byte)(((100 - s[p] * c) * data[0] + s[p] * c * F[pos - 3]) / 100);
                        ++pos;
                        ++p;
                        s[p] = (int)(10.0 * Gaussian(data[1], F[pos - 3], sigma));
                        F[pos] = (byte)(((100 - s[p] * c) * data[1] + s[p] * c * F[pos - 3]) / 100);
                        ++pos;
                        ++p;
                        s[p] = (int)(10.0 * Gaussian(data[2], F[pos - 3], sigma));
                        F[pos] = (byte)(((100 - s[p] * c) * data[2] + s[p] * c * F[pos - 3]) / 100);

                        p = X * 3;
                        pos = p + Y * stride;
                        data = srcPtr + pos;
                        v[p] = (int)(10.0 * Gaussian(data[0], D[pos + 3], sigma));
                        D[pos] = (byte)(((100 - v[p] * c) * data[0] + v[p] * c * D[pos + 3]) / 100);
                        ++pos;
                        ++p;
                        v[p] = (int)(10.0 * Gaussian(data[1], D[pos + 3], sigma));
                        D[pos] = (byte)(((100 - v[p] * c) * data[1] + v[p] * c * D[pos + 3]) / 100);
                        ++pos;
                        ++p;
                        v[p] = (int)(10.0 * Gaussian(data[2], D[pos + 3], sigma));
                        D[pos] = (byte)(((100 - v[p] * c) * data[2] + v[p] * c * D[pos + 3]) / 100);
                    }

                }
            }
            data = srcPtr;
            for (int i = 0; i < height * stride; i++)
            {
                D[i] = (byte)((10 * F[i] - (10 - c) * (*data) + 10 * D[i]) / (10 + c));
                data++;
            }
            return D;
        }
        private unsafe byte[] BEEPSVertical(byte* srcPtr, Int32 stride, Int32 width, Int32 height, double sigma, int c)
        {
            byte[] F = new byte[height*stride];
            byte[] D = new byte[height * stride];
            int[] sR = new int[height];
            int[] sG = new int[height];
            int[] sB = new int[height];
            int[] vR = new int[height];
            int[] vG = new int[height];
            int[] vB = new int[height];
            int pos = 0, X = 0, Y = 0;
            sigma = sigma * sigma * 2.0;
            byte* data;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    X = width - 1 - x;
                    Y = height - 1 - y;
                    if (y == 0)
                    {
                        pos = x * 3 + y * stride;
                        data = srcPtr + pos;
                        F[pos] = data[0];
                        sB[y] = data[0];
                        sG[y] = data[1];
                        sR[y] = data[2];

                        pos = X * 3 + Y * stride;
                        data = srcPtr + pos;
                        D[pos] = data[0];
                        vB[Y] = data[0];
                        vG[Y] = data[1];
                        vR[Y] = data[2];
                    }
                    else
                    {
                        pos = x * 3 + y * stride;
                        data = srcPtr + pos;
                        sB[y] = (int)(10.0 * Gaussian(data[0], F[pos - stride], sigma));
                        F[pos] = (byte)(((100 - sB[y] * c) * data[0] + sB[y] * c * F[pos - stride]) / 100);
                        sG[y] = (int)(10.0 * Gaussian(data[1], F[pos - stride + 1], sigma));
                        F[pos + 1] = (byte)(((100 - sG[y] * c) * data[1] + sG[y] * c * F[pos - stride + 1]) / 100);
                        sR[y] = (int)(10.0 * Gaussian(data[2], F[pos - stride + 2], sigma));
                        F[pos + 2] = (byte)(((100 - sR[y] * c) * data[2] + sR[y] * c * F[pos - stride + 2]) / 100);

                        pos = X * 3 + Y * stride;
                        data = srcPtr + pos;
                        vB[Y] = (int)(10.0 * Gaussian(data[0], D[pos + stride], sigma));
                        D[pos] = (byte)(((100 - vB[Y] * c) * data[0] + vB[Y] * c * D[pos + stride]) / 100);
                        vG[Y] = (int)(10.0 * Gaussian(data[1], D[pos + stride + 1], sigma));
                        D[pos + 1] = (byte)(((100 - vG[Y] * c) * data[1] + vG[Y] * c * D[pos + stride + 1]) / 100);
                        vR[Y] = (int)(10.0 * Gaussian(data[2], D[pos + stride + 2], sigma));
                        D[pos + 2] = (byte)(((100 - vR[Y] * c) * data[2] + vR[Y] * c * D[pos + stride + 2]) / 100);

                    }

                }
            }
            data = srcPtr;
            for (int i = 0; i < height*stride; i++)
            {
                D[i] = (byte)((10 * F[i] - (10 - c) * (*data) + 10 * D[i]) / (10 + c));
                data++;
            }
            return D;
        }
        //private byte[] BEEPSHorizontal(byte[] data, Int32 stride, Int32 width, Int32 height, double sigma,int c)
        //{
        //    byte[] F = new byte[data.Length];
        //    byte[] D = new byte[data.Length];
        //    int[] s = new int[width * 3];
        //    int[] v = new int[width * 3];
        //    int pos = 0, X = 0, Y = 0;
        //    int p = 0;
        //    sigma = sigma * sigma * 2.0;
        //    for (int y = 0; y < height; y++)
        //    {
        //        for (int x = 1; x < width; x++)
        //        {
        //            X = width - 1 - x;
        //            Y = height - 1 - y;
        //            if (x == 0)
        //            {
        //                pos = x * 3 + y * stride;
        //                F[pos] = data[pos];
        //                s[0] = data[pos];
        //                ++pos;
        //                F[pos] = data[pos];
        //                s[1] = data[pos];
        //                ++pos;
        //                F[pos] = data[pos];
        //                s[2] = data[pos];

        //                p = X * 3;
        //                pos = p + Y * stride;
        //                v[p] = data[pos];
        //                D[pos] = data[pos];
        //                ++pos;
        //                ++p;
        //                v[p] = data[pos];
        //                D[pos] = data[pos];
        //                ++pos;
        //                ++p;
        //                v[p] = data[pos];
        //                D[pos] = data[pos];
        //            }
        //            else
        //            {
        //                p = x * 3;
        //                pos = p + y * stride;
        //                s[p] = (int)(10.0*Gaussian(data[pos], F[pos - 3], sigma));
        //                F[pos] = (byte)(((100 - s[p] * c) * data[pos] + s[p] * c * F[pos - 3])/100);
        //                ++pos;
        //                ++p;
        //                s[p] = (int)(10.0*Gaussian(data[pos], F[pos - 3], sigma));
        //                F[pos] = (byte)(((100 - s[p] * c) * data[pos] + s[p] * c * F[pos - 3])/100);
        //                ++pos;
        //                ++p;
        //                s[p] = (int)(10.0*Gaussian(data[pos], F[pos - 3], sigma));
        //                F[pos] = (byte)(((100 - s[p] * c) * data[pos] + s[p] * c * F[pos - 3])/100);

        //                p = X * 3;
        //                pos = p + Y * stride;
        //                v[p] = (int)(10.0*Gaussian(data[pos], D[pos + 3], sigma));
        //                D[pos] = (byte)(((100 - v[p] * c) * data[pos] + v[p] * c * D[pos + 3])/100);
        //                ++pos;
        //                ++p;
        //                v[p] = (int)(10.0*Gaussian(data[pos], D[pos + 3], sigma));
        //                D[pos] = (byte)(((100 - v[p] * c) * data[pos] + v[p] * c * D[pos + 3])/100);
        //                ++pos;
        //                ++p;
        //                v[p] = (int)(10.0*Gaussian(data[pos], D[pos + 3], sigma));
        //                D[pos] = (byte)(((100 - v[p] * c) * data[pos] + v[p] * c * D[pos + 3])/100);
        //            }

        //        }
        //    }
        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        D[i] = (byte)((10*F[i] - (10 - c) * data[i] + 10*D[i]) / (10 + c));
        //    }
        //    return D;
        //}
        //private byte[] BEEPSVertical(byte[] data, Int32 stride, Int32 width, Int32 height, double sigma, int c)
        //{
        //    byte[] F = new byte[data.Length];
        //    byte[] D = new byte[data.Length];
        //    int[] sR = new int[height];
        //    int[] sG = new int[height];
        //    int[] sB = new int[height];
        //    int[] vR = new int[height];
        //    int[] vG = new int[height];
        //    int[] vB = new int[height];
        //    int pos = 0, X = 0, Y = 0;
        //    sigma = sigma * sigma * 2.0;
        //    for (int x = 0; x < width; x++)
        //    {
        //        for (int y = 0; y < height; y++)
        //        {
        //            X = width - 1 - x;
        //            Y = height - 1 - y;
        //            if (y == 0)
        //            {
        //                pos = x * 3 + y * stride;
        //                F[pos] = data[pos];
        //                sB[y] = data[pos];
        //                sG[y] = data[pos+1];
        //                sR[y] = data[pos+2];

        //                pos = X * 3 + Y * stride;
        //                D[pos] = data[pos];
        //                vB[Y] = data[pos];
        //                vG[Y] = data[pos + 1];
        //                vR[Y] = data[pos + 2];
        //            }
        //            else
        //            {
        //                pos = x * 3 + y * stride;
        //                sB[y] = (int)(10.0*Gaussian(data[pos], F[pos - stride], sigma));
        //                F[pos] = (byte)(((100 - sB[y] * c) * data[pos] + sB[y] * c * F[pos - stride])/100);
        //                sG[y] = (int)(10.0*Gaussian(data[pos+1], F[pos - stride+1], sigma));
        //                F[pos + 1] = (byte)(((100 - sG[y] * c) * data[pos + 1] + sG[y] * c * F[pos - stride + 1])/100);
        //                sR[y] = (int)(10.0*Gaussian(data[pos+2], F[pos - stride+2], sigma));
        //                F[pos + 2] = (byte)(((100 - sR[y] * c) * data[pos + 2] + sR[y] * c * F[pos - stride + 2])/100);

        //                pos = X * 3 + Y * stride;
        //                vB[Y] = (int)(10.0*Gaussian(data[pos], D[pos + stride], sigma));
        //                D[pos] = (byte)(((100 - vB[Y] * c) * data[pos] + vB[Y] * c * D[pos + stride])/100);
        //                vG[Y] = (int)(10.0*Gaussian(data[pos + 1], D[pos + stride + 1], sigma));
        //                D[pos + 1] = (byte)(((100 - vG[Y] * c) * data[pos + 1] + vG[Y] * c * D[pos + stride + 1])/100);
        //                vR[Y] = (int)(10.0*Gaussian(data[pos + 2], D[pos + stride + 2], sigma));
        //                D[pos + 2] = (byte)(((100 - vR[Y] * c) * data[pos + 2] + vR[Y] * c * D[pos + stride + 2])/100);

        //            }
                   
        //        }
        //    }

        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        D[i] = (byte)((10 * F[i] - (10 - c) * data[i] + 10 * D[i]) / (10 + c));
        //    }
        //    return D;
        //}
        private double Gaussian(int u, int v, double sigma)
        {
            //int t = -(u - v) * (u - v);
            //return Math.Exp((double)t / sigma);
            int t = (u - v) * (u - v)/(int)sigma;
            return gaussMap[t];
        }
        public unsafe void SkinGrindProcess(IntPtr srcPtr, Int32 stride, Int32 width, Int32 height, double sigma, int c)
        {
            int[] skin = SkinDetectProcessA(srcPtr, stride, width, height);
            //sigma = (double)((double)(Math.Min(mSkinMax-mSkinAverage, mSkinAverage-mSkinMin)) * 50.0/((double)(mSkinMax-mSkinMin)));
            sigma = sigma > 50 ? 50 : sigma;
            byte* pSrc = (byte*)srcPtr.ToPointer();
            byte[] dstValue = new byte[height * stride];
            byte[] hValue = BEEPSHorizontal(pSrc, stride, width, height, sigma, c);
            byte[] vValue;
            fixed (byte* p = hValue)
            {
                vValue = BEEPSVertical(p, stride, width, height, sigma, c);
            }
            hValue = BEEPSVertical(pSrc, stride, width, height, sigma, c);
            fixed (byte* p = hValue)
            {
                dstValue = BEEPSHorizontal(p, stride, width, height, sigma, c);
            }
            
            for (int i = 0; i < dstValue.Length; i++)
            {
                if (skin[i] == 256)
                {
                    *pSrc = (byte)((vValue[i] + dstValue[i]) / 2);
                }
                pSrc++;
            }
            pSrc = (byte*)srcPtr.ToPointer();
            byte* ptr;
            for (int y = 1; y < height-1; y++)
            {
                for (int x = 1; x < width-1; x++)
                {
                    ptr = pSrc + x * 3 + y * stride;
                    if (!((skin[(x - 1) * 3 + (y - 1) * stride] == 256) && (skin[x * 3 + (y - 1) * stride] == 256) && (skin[(x + 1) * 3 + (y - 1) * stride] == 256) && (skin[(x - 1) * 3 + y * stride] == 256) &&
                        (skin[(x + 1) * 3 + y * stride] == 256) && (skin[(x - 1) * 3 + (y + 1) * stride] == 256) && (skin[x * 3 + (y + 1) * stride] == 256) && (skin[(x + 1) * 3 + (y + 1) * stride] == 256))&&((skin[x*3+y*stride]==256))||
                        (!((skin[(x - 1) * 3 + (y - 1) * stride] != 256) && (skin[x * 3 + (y - 1) * stride] != 256) && (skin[(x + 1) * 3 + (y - 1) * stride] != 256) && (skin[(x - 1) * 3 + y * stride] != 256) &&
                        (skin[(x + 1) * 3 + y * stride] != 256) && (skin[(x - 1) * 3 + (y + 1) * stride] != 256) && (skin[x * 3 + (y + 1) * stride] != 256) && (skin[(x + 1) * 3 + (y + 1) * stride] != 256))&&((skin[x*3+y*stride]!=256))))
                    {
                        ptr[0] = (byte)(((ptr - 3 - stride)[0] + (ptr + 3 - stride)[0] + (ptr - stride)[0] + (ptr - 3)[0] + (ptr)[0] +
                            (ptr + 3)[0] + (ptr - 3 + stride)[0] + (ptr + stride)[0] + (ptr + 3 + stride)[0]) / 9);
                        ptr[1] = (byte)(((ptr - 3 - stride)[1] + (ptr + 3 - stride)[1] + (ptr - stride)[1] + (ptr - 3)[1] + (ptr)[1] +
                            (ptr + 3)[1] + (ptr - 3 + stride)[1] + (ptr + stride)[1] + (ptr + 3 + stride)[1]) / 9);
                        ptr[2] = (byte)(((ptr - 3 - stride)[2] + (ptr + 3 - stride)[2] + (ptr - stride)[2] + (ptr - 3)[2] + (ptr)[2] +
                            (ptr + 3)[2] + (ptr - 3 + stride)[2] + (ptr + stride)[2] + (ptr + 3 + stride)[2]) / 9);
                        //ptr[0] = (byte)255;
                        //ptr[1] = 0;
                        //ptr[2] = 0;
                    }

                }
            }
        }

        public unsafe int[] SkinDetectProcessA(IntPtr srcPtr, Int32 stride, Int32 width, Int32 height)
        {
            byte* ptr = (byte*)srcPtr.ToPointer();
            int R = 0, G = 0, B = 0;
            int[] res = new int[stride * height];
            int count = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    B = ptr[0];
                    G = ptr[1];
                    R = ptr[2];
                   // if (R > 95 && G > 40 && B > 20 && Math.Max(R, Math.Max(G, B)) - Math.Min(R, Math.Min(G, B)) > 15 && Math.Abs(R - G) > 15 && R > G && R > B)
                        if (R > 50 && G > 35 && B > 5 && Math.Max(R, Math.Max(G, B)) - Math.Min(R, Math.Min(G, B)) > 10 && Math.Abs(R - G) > 1 && R > G && R > B)
                    {
                        res[x * 3 + y * stride] = 256;
                        res[x * 3 + 1 + y * stride] = 256;
                        res[x * 3 + 2 + y * stride] = 256;
                        int t = (B + G + R) / 3;
                        mSkinMax = mSkinMax > t ? mSkinMax : t;
                        mSkinMin = mSkinMin < t ? mSkinMin : t;
                        mSkinAverage += t;
                        count++;
                    }
                    else
                    {        
                        res[x * 3 + y * stride] = B;
                        res[x * 3 + 1 + y * stride] = G;
                        res[x * 3 + 2 + y * stride] = R;
                    }
                    ptr += 3;
                }
                ptr += stride - width * 3;
            }
            if (count != 0)
                mSkinAverage = mSkinAverage / count;
            else
                mSkinAverage = 0;
            return res;
        }
        public unsafe void SkinDetectProcessAB(IntPtr srcPtr, Int32 stride, Int32 width, Int32 height)
        {
            byte* ptr = (byte*)srcPtr.ToPointer();
            int R = 0, G = 0, B = 0;
            int[] res = new int[stride * height];
            double Cr, Cg;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    B = ptr[0];
                    G = ptr[1];
                    R = ptr[2];
                    //if (R > 95 && G > 40 && B > 20 && Math.Max(R, Math.Max(G, B)) - Math.Min(R, Math.Min(G, B)) > 15 && Math.Abs(R - G) > 15 && R > G && R > B)
                    if (R > 50 && G > 35 && B > 5 && Math.Max(R, Math.Max(G, B)) - Math.Min(R, Math.Min(G, B)) > 15 && Math.Abs(R - G) > 5 && R > G && R > B)
                    {
                        //ptr[0] = (byte)B;
                        //ptr[1] = (byte)G;
                        //ptr[2] = (byte)R;
                        ptr[0] = ptr[1] = ptr[2] = 0;
                    }
                    else
                    {
                        //res[x * 3 + y * stride] = B;
                        //res[x * 3 + 1 + y * stride] = G;
                        //res[x * 3 + 2 + y * stride] = R;
                        ptr[0] = (byte)B;
                        ptr[1] = (byte)G;
                        ptr[2] = (byte)R;
                        //ptr[0] = ptr[1] = ptr[2] = 0;
                    }
                    //Cg = 128.0 - 0.318 * (double)R + 0.4392 * (double)G - 0.1212 * (double)B;
                    //Cr = 128.0 + 0.4392 * (double)R - 0.3677 * (double)G - 0.0714 * (double)B;

                    //if ((Cg >= 85) && (Cg <= 140) && ((Cr <= (280 - Cg)) && (Cr >= (260 - Cg))))
                    //{
                    //        ptr[0] = (byte)B;
                    //        ptr[1] = (byte)G;
                    //        ptr[2] = (byte)R;

                    //}
                    //else
                    //{
                    //    //res[x * 3 + y * stride] = B;
                    //    //res[x * 3 + 1 + y * stride] = G;
                    //    //res[x * 3 + 2 + y * stride] = R;
                    //    ptr[0] = ptr[1] = ptr[2] = 0;
                    //}
                    ptr += 3;
                }
                ptr += stride - width * 3;
            }
            //return res;
        }
        public unsafe void GaussianBlur(IntPtr srcPtr, Int32 width, Int32 height, Int32 stride, Int32 radius, double sigma)
        {
            byte* ptr = (byte*)srcPtr.ToPointer();
            double[] kernel = GaussKernel1D(radius, sigma);
            double tempR = 0.0, tempG = 0.0, tempB = 0.0;
            int v = 0;
            double K = 0.0;
            int rem = 0;
            int t = 0;
            byte[] tempValues = new byte[height * stride];
            byte* p;
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    tempR = 0.0; tempG = 0.0; tempB = 0.0;
                    for (int k = -radius; k <= radius; k++)
                    {
                        rem = (Math.Abs(i + k) % width);
                        t = rem * 3 + j * stride;
                        p = ptr + t;
                        K = kernel[k + radius];
                        tempB += p[0] * K;
                        tempG += p[1] * K;
                        tempR += p[2] * K;
                    }
                    v = i * 3 + j * stride;
                    tempValues[v] = (byte)tempB;
                    tempValues[v + 1] = (byte)tempG;
                    tempValues[v + 2] = (byte)tempR;

                }
            }
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    tempR = 0.0; tempG = 0.0; tempB = 0.0;
                    for (int k = -radius; k <= radius; k++)
                    {
                        rem = (Math.Abs(j + k) % height);
                        t = rem * stride + i * 3;
                        K = kernel[k + radius];
                        tempB += tempValues[t] * K;
                        tempG += tempValues[t + 1] * K;
                        tempR += tempValues[t + 2] * K;
                    }
                    v = i * 3 + j * stride;
                    p = ptr + v;
                    p[0] = (byte)tempB;
                    p[1] = (byte)tempG;
                    p[2] = (byte)tempR;
                }
            }
        }
        private static double[] GaussKernel1D(int r, double sigma)
        {
            double[] filter = new double[2 * r + 1];
            double sum = 0.0;
            for (int i = 0; i < filter.Length; i++)
            {
                filter[i] = Math.Exp((double)(-(i - r) * (i - r)) / (2.0 * sigma * sigma));
                sum += filter[i];
            }
            for (int i = 0; i < filter.Length; i++)
            {
                filter[i] = filter[i] / sum;
            }
            return filter;
        }
       
    }
   
}
