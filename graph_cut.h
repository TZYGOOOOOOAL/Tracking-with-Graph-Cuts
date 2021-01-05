#ifndef __GRAPH_CUT_H__
#define __GRAPH_CUT_H__

#include <opencv.hpp>

using namespace std;
using namespace cv;

Mat my_graph_cut(Rect& ROI_fgd_rect, Mat &frame, int iter_cnt);

Mat make_visual_mask(Mat &frame, Mat& mask, Rect predict_bbox, const Scalar& color, double mask_weight, bool inplace);
#endif