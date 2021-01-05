#include "graph_cut.h"

Mat my_graph_cut(Rect& ROI_fgd_rect, Mat &frame, int iter_cnt)
{
	Mat result_mask(frame.size(), CV_8UC1, Scalar::all(0));
	Mat result_mask_ROI;
	Mat bgModel, fgModel;

	// 背景区域不在原图上进行
	Rect ROI_bgd_rect;
	//ROI_bgd_rect.x = std::max(0, ROI_fgd_rect.x - 2 * ROI_fgd_rect.width + 1);
	//ROI_bgd_rect.y = std::max(0, ROI_fgd_rect.y - 2 * ROI_fgd_rect.height + 1);
	//ROI_bgd_rect.width = std::min(frame.cols - ROI_bgd_rect.x, 5 * ROI_fgd_rect.width);
	//ROI_bgd_rect.height = std::min(frame.rows - ROI_bgd_rect.y, 5 * ROI_fgd_rect.height);
	ROI_bgd_rect.x = std::max(0, ROI_fgd_rect.x - 1 * ROI_fgd_rect.width + 1);
	ROI_bgd_rect.y = std::max(0, ROI_fgd_rect.y - 1 * ROI_fgd_rect.height + 1);
	ROI_bgd_rect.width = std::min(frame.cols - ROI_bgd_rect.x, 3 * ROI_fgd_rect.width);
	ROI_bgd_rect.height = std::min(frame.rows - ROI_bgd_rect.y, 3 * ROI_fgd_rect.height);

	// 前景相对位置
	Rect fgd_rect;
	fgd_rect.x = ROI_fgd_rect.x - ROI_bgd_rect.x;
	fgd_rect.y = ROI_fgd_rect.y - ROI_bgd_rect.y;
	fgd_rect.width = ROI_fgd_rect.width;
	fgd_rect.height = ROI_fgd_rect.height;

	//Mat t1 = frame(ROI_bgd_rect).clone();
	//Mat t2 = t1(fgd_rect).clone();

	// Cut
	grabCut(frame(ROI_bgd_rect), result_mask_ROI, fgd_rect, bgModel, fgModel, iter_cnt, cv::GC_INIT_WITH_RECT);

	// 与"可能是前景区域"相符的置为255
	cv::compare(result_mask_ROI, cv::GC_PR_FGD, result_mask_ROI, cv::CMP_EQ);
	result_mask_ROI.copyTo(result_mask(ROI_bgd_rect));

	return result_mask;
}

// 可视化 分割结果
Mat make_visual_mask(Mat &frame, Mat& mask, Rect predict_bbox, const Scalar& color, double mask_weight, bool inplace)
{
	vector<Mat> v_mat(3);
	Mat mask_bgr;

	for (int i = 0; i < 3; i++)
	{
		v_mat[i] = mask.clone();
		v_mat[i] = v_mat[i] / 255 * color[i];
	}

	cv::merge(v_mat, mask_bgr);
	Mat fg = frame & mask_bgr;
	Mat bg = frame & (~mask_bgr);
 	if (inplace)
	{
		cv::addWeighted(fg, 1 - mask_weight, mask_bgr, mask_weight, 0, fg);
		frame = fg + bg;
		return frame;
	}
	else
	{
		Mat result;
		cv::addWeighted(fg, 1 - mask_weight, mask_bgr, mask_weight, 0, fg);
		result = fg + bg;
		return result;
	}

}