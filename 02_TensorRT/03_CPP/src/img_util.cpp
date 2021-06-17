#include "img_util.hpp"
#include "log4plus_util.h"

// static const float kMean[3] = { 0.485f, 0.456f, 0.406f};
// static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f};
// static const int map_[7][3] = { {0,0,0} ,
//                 {128,0,0},
//                 {0,128,0},
//                 {0,0,128},
//                 {128,128,0},
//                 {128,0,128},
//                 {0,128,0}};

float* img_normal(cv::Mat img, int BATCHSIZE, char* model_type)
{
    //cv::Mat image(img.rows, img.cols, CV_32FC3);
    float * data;
    int a = 0;
    int index_img=0;
    data = (float*)calloc(img.rows*img.cols * 3 * BATCHSIZE, sizeof(float));

    for (int h = 0; h < BATCHSIZE; ++h)
    {
        for (int i = 0; i < img.rows; ++i)
        {   //获取第i行首像素指针
            cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
            //cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
            for (int j = 0; j < img.cols; ++j)
            {
                for (int c = 0; c < 3; ++c)
                {
                    // data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = (p1[j][c] / 255.0f - kMean[c]) / kStdDev[c];
                    // float tmp_val = data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j];
                    float tmp_val = p1[j][c];
                    if (strcmp(model_type, "pytorch")==0)
                    {
                        data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = (tmp_val/255.0f - kMean[c])/kStdDev[c];
                    }else if(strcmp(model_type, "keras")==0)
                    {
                        data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = (tmp_val/255.0f - kMean[c])/kStdDev[c];
                    }else if(strcmp(model_type, "keras_else")==0)
                    {
                        // data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = (tmp_val/1.0f - keras_EMean[c]);
                        data[index_img++] = (tmp_val/1.0f - keras_EMean[c]);
                        // DEBUG(to_string(data[index_img-1]));
                    }else if(strcmp(model_type, "tf")==0)
                    {
                        data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = (tmp_val+1.0f)*127.5;
                    }else
                    {
                        data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j] = tmp_val;
                    }

                    if(1)
                    {
                        // DEBUG(to_string(data[h * c * img.cols * img.rows + c * img.cols * img.rows + i * img.cols + j]) + " : " + to_string(tmp_val));
                    }
                }
            }
        }
    }

    // for (int i = 1 * img.cols * img.rows * 3; i > 0; i--)
    // {
    //     if(i%1000 == 0)
    //         std::cout << "::"<< data[i] << std::endl;
    //     // std::cout << "::"<< (float *)buffers[inputIndex] << std::endl;
    //     /* code */
    // }
    return data;
}



