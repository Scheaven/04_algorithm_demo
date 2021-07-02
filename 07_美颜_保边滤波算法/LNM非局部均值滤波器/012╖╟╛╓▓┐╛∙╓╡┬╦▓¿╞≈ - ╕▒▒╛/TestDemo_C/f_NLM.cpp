#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include"f_NLM.h"

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define CLIP3(x, a, b) MIN2(MAX2(a,x), b)


void NLM(unsigned char* srcData, int width, int height, int D, int d, float h)
{
	unsigned char* tempData = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
	memcpy(tempData, srcData, sizeof(unsigned char) * height * width);
	float sw = 0;
	float sum = 0;
	int px, py, cx, cy;
	float zx;
	float vxsy = 0;
	float DD = d * d;
	float HH = h * h;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			sw = 0;
			zx = 0; 
			sum = 0;
			for(int n = -D; n <= D; n++)
			{
				for(int m = -D; m <= D; m++)
				{			
					vxsy = 0;
					for(int kn = -d; kn <= d; kn++)
					{
						for(int km = -d; km <= d; km++)
						{
							cx = CLIP3(i - d + km, 0, width - 1);
							cy = CLIP3(j - d + kn, 0, height - 1);
							px = CLIP3(i + m + km, 0, width - 1);
							py = CLIP3(j + n + kn, 0, height - 1);
							vxsy += (tempData[px + py * width] - tempData[cx + cy * width]) * (tempData[px + py * width] - tempData[cx + cy * width]);
						}
					}
					vxsy = vxsy / DD;
					sw = exp(-vxsy / HH);
					zx += sw;
					int ox = CLIP3(i + m, 0, width - 1);
					int oy = CLIP3(j + n, 0, height - 1);
					sum += sw * tempData[ox + oy * width];
				}
			}
			srcData[i + j * width] = zx == 0 ? srcData[i + j * width] : CLIP3(sum / zx, 0, 255);
		}
	}
	free(tempData);
};

void f_NLMFilter(unsigned char* srcData, int nWidth, int nHeight, int nStride, int dRadius, int sRadius, int h)
{
	if (srcData == NULL)
	{
		return;
	}
	if(dRadius == 0 || sRadius == 0 || h == 0 || dRadius <= sRadius)
		return;
	unsigned char* rData = (unsigned char*)malloc(sizeof(unsigned char) * nWidth * nHeight);
	unsigned char* gData = (unsigned char*)malloc(sizeof(unsigned char) * nWidth * nHeight);
	unsigned char* bData = (unsigned char*)malloc(sizeof(unsigned char) * nWidth * nHeight);
	unsigned char* pSrc = srcData;
	unsigned char* pR = rData;
	unsigned char* pG = gData;
	unsigned char* pB = bData;
	for(int j = 0; j < nHeight; j++)
	{
		for(int i = 0; i < nWidth; i++)
		{
			*pR = pSrc[2];
			*pG = pSrc[1];
			*pB = pSrc[0];
			pR++;
			pG++;
			pB++;
			pSrc += 4;
		}
	}
	NLM(rData, nWidth, nHeight, dRadius, sRadius, h);
	NLM(gData, nWidth, nHeight, dRadius, sRadius, h);
	NLM(bData, nWidth, nHeight, dRadius, sRadius, h);
	pSrc = srcData;
	pR = rData;
	pG = gData;
	pB = bData;
	for(int j = 0; j < nHeight; j++)
	{
		for(int i = 0; i < nWidth; i++)
		{
            pSrc[2] = * pR;
			pSrc[1] = * pG;
			pSrc[0] = * pB;
			pR++;
			pG++;
			pB++;
			pSrc += 4;
		}
	}
	free(rData);
	free(gData);
	free(bData);
}



