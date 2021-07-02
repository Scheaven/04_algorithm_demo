
/*************************************************************************
Copyright:   HZ.
Author:		 Hu Yaowu
Date:		 2015-4-23
Mail:        dongtingyueh@163.com
Description: Selective Filter .
Refference: None
*************************************************************************/
#ifndef __T_SELECTIVE_BLUR__
#define __T_SELECTIVE_BLUR__


#ifdef _MSC_VER

#ifdef __cplusplus
#define EXPORT extern "C" _declspec(dllexport)
#else
#define EXPORT __declspec(dllexport)
#endif

EXPORT void f_NLMFilter(unsigned char* srcData, int nWidth, int nHeight, int nStride, int dRadius, int sRadius, int h);


#else

#ifdef __cplusplus
extern "C" {
#endif    

    void f_NLMFilter(unsigned char* srcData, int nWidth, int nHeight, int nStride, int dRadius, int sRadius, int h);

#ifdef __cplusplus
}
#endif


#endif



#endif
