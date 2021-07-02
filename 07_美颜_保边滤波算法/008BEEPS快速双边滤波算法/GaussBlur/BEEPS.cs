using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GaussBlur
{
    class BEEPS
    {
        private double[] gaussMap = new double[65025];
        public void Initialize()
        {
            string t = "";
            for (int i = 0; i < gaussMap.Length; i++)
            {
                gaussMap[i]=Math.Exp(-i);
                t += gaussMap[i].ToString() + ",";
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
        private double Gaussian(int u, int v, double sigma)
        {
            //int t = -(u - v) * (u - v);
            //return Math.Exp((double)t / sigma);
            int t = (u - v) * (u - v)/(int)sigma;
            return gaussMap[t];
        }
       
       
    }
   
}
