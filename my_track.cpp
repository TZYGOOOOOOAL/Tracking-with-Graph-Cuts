#include "my_track.h"
#include "Config.h"
#include "particles.h"
#include "graph_cut.h"
#include "frame_difference.h"
#include "utils.h"

vector<Rect> track(string video_path, Rect init_bbox, const Config& config, const vector<Rect>& annotation_bboxes /*= vector<Rect>()*/)
{
	// 视频相关
	VideoCapture capture;			// 视频
	Mat frame;						// 当前帧
	Mat f_copy;
	int particle_idx, frame_idx;	// 帧数

	// 粒子群相关
	Mat ROI_hsv;			// ROI 转换到HSV空间
	Mat ROI;				// ROI
	Rect ROI_rect;			// ROI区域
	Rect predict_bbox;		// 预测bbox

	MatND hist_object;		// 目标特征（直方图）
	MatND hist_roi;			// 粒子区域直方图

	vector<PARTICLE> particles(config.NUM_PARTICLE);
	vector<PARTICLE> new_particles(config.NUM_PARTICLE);

	cv::RNG rng(12);		// 随机数生成器
	
	// 帧差
	FrameDiff fd(config.FRAME_DIFF_TH, config.FRAME_DIFF_SEARCH_RATIO);
	Mat moving_obj_mask;

	// 结果
	vector<Rect> predict_bboxes;

	// 判断视频是否打开
	capture.open(video_path);
	if (!capture.isOpened())
	{
		cout << "Cannot Open " << video_path << " !!!" << endl;
		return predict_bboxes;
	}

	// 保存相关
	VideoWriter writer;
	string video_save_path;
	string frames_save_dir;
	string frame_save_path;
	fstream fs_save;

	if (config.SAVE_RESULT)
	{
		// 视频保存路径
		Size video_size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		string video_save_dir = path_join(config.SAVE_DIR_PATH, get_filename(video_path));
		make_dir(video_save_dir);
		video_save_path = path_join(video_save_dir, get_basename(video_path));
		writer.open(video_save_path, CV_FOURCC('X', 'V', 'I', 'D'), config.VIDEO_FPS, video_size);

		// 视频每帧保存路径
		frames_save_dir = path_join(video_save_dir, "frames");
		make_dir(frames_save_dir);
		
		// 结果输出txt
		string predicts_save_txt_path = path_join(video_save_dir, "predicts.txt");
		fs_save.open(predicts_save_txt_path, ios::out);
	}


	// 读取视频
	frame_idx = 0;
	while (true)
	{
		capture >> frame;

		if (frame.empty()){
			cout << "Read " << video_path << " Finish" << endl;
			break;
		}

		/****** 第一帧读取目标框 并 初始化  ******/
		if (frame_idx == 0){
			// 起始框
			frame(init_bbox).copyTo(ROI);
			
			// 目标HS特征
			hist_object = get_norm_HShist(ROI, ROI_hsv, config.HIST_CHANNELS, config.HIST_SIZE, config.HIST_RANGES);

			//粒子初始化
			particle_init(particles, hist_object, init_bbox);

			// 帧差初始化
			fd.get_object_mask(frame, init_bbox);
			predict_bbox = init_bbox;
		}


		/****** 开始跟踪  ******/
		else
		{
			f_copy = frame.clone();
			// 先得帧差结果
			if ((frame_idx - 1) % config.FRAME_DIFF_INTERVAL == 0){
				moving_obj_mask = fd.get_object_mask(frame, predict_bbox);
			}
			
			// 对每个粒子的操作：
			for (particle_idx = 0; particle_idx < particles.size(); particle_idx++){

				// 这里利用高斯分布的随机数来生成每个粒子下一次的位置以及范围
				transition(particles[particle_idx], frame.cols, frame.rows, rng, config.A1, config.A2, config.B0,
					config.TRANS_X_STD, config.TRANS_Y_STD, config.TRANS_S_STD);

				// 根据新生成的粒子信息截取对应frame上的区域
				ROI_rect = get_ROI_rect_from_frame(particles[particle_idx].x, particles[particle_idx].y, particles[particle_idx].width,
					particles[particle_idx].height, frame.cols, frame.rows, particles[particle_idx].scale);
				frame(ROI_rect).copyTo(ROI);
			
				// 新框特征
				hist_roi = get_norm_HShist(ROI, ROI_hsv, config.HIST_CHANNELS, config.HIST_SIZE, config.HIST_RANGES);

				// 画出蓝色的粒子框
				////if (config.VISUAL_TEMP_RESULT)
				//{
				//	// 不要在原frame上画
				//	//rectangle(f_copy, ROI_rect, Scalar(255, 0, 0), 1);
				//	//imshow("particle", frame);
				//}
			
				//比较目标的直方图和上述计算的区域直方图,更新particle权重
				// https://blog.csdn.net/shuiyixin/article/details/80257822
				particles[particle_idx].weight = exp(-100 * (compareHist(hist_object, hist_roi, cv::HISTCMP_BHATTACHARYYA)));
			}

			// 归一化权重 
			normalize_weights(particles);

			// 加入运动目标帧差权重
			if ((frame_idx - 1) % config.FRAME_DIFF_INTERVAL == 0){
				merge_particles_frameDiff_weights(particles, moving_obj_mask, config.LAMBDA_FRAME_DIFF);
			}

			// 将粒子按权重从高到低排序
			des_sort(particles);

			// 重采样
			resample(particles, new_particles, rng);
		
			// 再次排序
			des_sort(particles);

			// 计算目标位置	
			predict_bbox = get_predict_bbox(particles, frame.rows, frame.cols, config.PREDICT_WITH_WEIGHT, config.PREDICT_USE_RATIO);
		}

		// **** 分割目标 **** 
		if (frame_idx % config.GRAPH_CUT_INTERVAL == 0)
		{
			Mat mask = my_graph_cut(predict_bbox, frame, config.GRAPH_CUT_ITERS_COUNT);
			make_visual_mask(frame, mask, predict_bbox, config.VISUAL_PREDICT_MASK_COLOR, config.VISUAL_MASK_WEIGHT, true);
		}

		// 跟踪框显示
		if (config.TEST_WITH_LABEL)
		{
			rectangle(frame, annotation_bboxes[frame_idx], config.VISUAL_TARGET_BBOX_COLOR, 1);
		}
		rectangle(frame, predict_bbox, config.VISUAL_PREDICT_BBOX_COLOR, 1);

		// 保存
		predict_bboxes.push_back(predict_bbox);
		if (config.SAVE_RESULT)
		{
			writer << frame;
			frame_save_path = path_join(frames_save_dir, std::to_string(frame_idx) + ".jpg");
			cv::imwrite(frame_save_path, frame);

			fs_save << predict_bbox.x << ' ' << predict_bbox.y << ' ' << predict_bbox.width \
				<< ' ' << predict_bbox.height << '\n';
		}

		imshow("frame", frame);
		cvWaitKey(40);

		frame_idx++;
	}

	writer.release();
	cv::destroyAllWindows();

	return predict_bboxes;
}

// 粒子框中 运动目标像素比重
void merge_particles_frameDiff_weights(vector<PARTICLE>& particles, Mat& obj_mask, double frame_diff_lambda)
{
	if (obj_mask.empty())
		return;

	double nozero_sum = 1e-12;
	vector<double> frame_diff_weights(particles.size());

	// 统计帧差法产生权重
	for (int i = 0; i < particles.size(); i++)
	{
		Rect ROI_rect = get_ROI_rect_from_frame(particles[i].x, particles[i].y, particles[i].width,
			particles[i].height, obj_mask.cols, obj_mask.rows, particles[i].scale);

		frame_diff_weights[i] = (double)(cv::countNonZero(obj_mask(ROI_rect)));

		nozero_sum += frame_diff_weights[i];
	}

	// 归一化帧差法权重
	for (int i = 0; i < frame_diff_weights.size(); i++)
	{
		particles[i].weight += frame_diff_lambda * frame_diff_weights[i] / nozero_sum;
	}

	// 归一化总权重
	normalize_weights(particles);
	return;
}