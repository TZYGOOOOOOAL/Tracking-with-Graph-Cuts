#include "my_track.h"
#include "Config.h"
#include "particles.h"
#include "graph_cut.h"
#include "frame_difference.h"
#include "utils.h"

vector<Rect> track(string video_path, Rect init_bbox, const Config& config, const vector<Rect>& annotation_bboxes /*= vector<Rect>()*/)
{
	// ��Ƶ���
	VideoCapture capture;			// ��Ƶ
	Mat frame;						// ��ǰ֡
	Mat f_copy;
	int particle_idx, frame_idx;	// ֡��

	// ����Ⱥ���
	Mat ROI_hsv;			// ROI ת����HSV�ռ�
	Mat ROI;				// ROI
	Rect ROI_rect;			// ROI����
	Rect predict_bbox;		// Ԥ��bbox

	MatND hist_object;		// Ŀ��������ֱ��ͼ��
	MatND hist_roi;			// ��������ֱ��ͼ

	vector<PARTICLE> particles(config.NUM_PARTICLE);
	vector<PARTICLE> new_particles(config.NUM_PARTICLE);

	cv::RNG rng(12);		// �����������
	
	// ֡��
	FrameDiff fd(config.FRAME_DIFF_TH, config.FRAME_DIFF_SEARCH_RATIO);
	Mat moving_obj_mask;

	// ���
	vector<Rect> predict_bboxes;

	// �ж���Ƶ�Ƿ��
	capture.open(video_path);
	if (!capture.isOpened())
	{
		cout << "Cannot Open " << video_path << " !!!" << endl;
		return predict_bboxes;
	}

	// �������
	VideoWriter writer;
	string video_save_path;
	string frames_save_dir;
	string frame_save_path;
	fstream fs_save;

	if (config.SAVE_RESULT)
	{
		// ��Ƶ����·��
		Size video_size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		string video_save_dir = path_join(config.SAVE_DIR_PATH, get_filename(video_path));
		make_dir(video_save_dir);
		video_save_path = path_join(video_save_dir, get_basename(video_path));
		writer.open(video_save_path, CV_FOURCC('X', 'V', 'I', 'D'), config.VIDEO_FPS, video_size);

		// ��Ƶÿ֡����·��
		frames_save_dir = path_join(video_save_dir, "frames");
		make_dir(frames_save_dir);
		
		// ������txt
		string predicts_save_txt_path = path_join(video_save_dir, "predicts.txt");
		fs_save.open(predicts_save_txt_path, ios::out);
	}


	// ��ȡ��Ƶ
	frame_idx = 0;
	while (true)
	{
		capture >> frame;

		if (frame.empty()){
			cout << "Read " << video_path << " Finish" << endl;
			break;
		}

		/****** ��һ֡��ȡĿ��� �� ��ʼ��  ******/
		if (frame_idx == 0){
			// ��ʼ��
			frame(init_bbox).copyTo(ROI);
			
			// Ŀ��HS����
			hist_object = get_norm_HShist(ROI, ROI_hsv, config.HIST_CHANNELS, config.HIST_SIZE, config.HIST_RANGES);

			//���ӳ�ʼ��
			particle_init(particles, hist_object, init_bbox);

			// ֡���ʼ��
			fd.get_object_mask(frame, init_bbox);
			predict_bbox = init_bbox;
		}


		/****** ��ʼ����  ******/
		else
		{
			f_copy = frame.clone();
			// �ȵ�֡����
			if ((frame_idx - 1) % config.FRAME_DIFF_INTERVAL == 0){
				moving_obj_mask = fd.get_object_mask(frame, predict_bbox);
			}
			
			// ��ÿ�����ӵĲ�����
			for (particle_idx = 0; particle_idx < particles.size(); particle_idx++){

				// �������ø�˹�ֲ��������������ÿ��������һ�ε�λ���Լ���Χ
				transition(particles[particle_idx], frame.cols, frame.rows, rng, config.A1, config.A2, config.B0,
					config.TRANS_X_STD, config.TRANS_Y_STD, config.TRANS_S_STD);

				// ���������ɵ�������Ϣ��ȡ��Ӧframe�ϵ�����
				ROI_rect = get_ROI_rect_from_frame(particles[particle_idx].x, particles[particle_idx].y, particles[particle_idx].width,
					particles[particle_idx].height, frame.cols, frame.rows, particles[particle_idx].scale);
				frame(ROI_rect).copyTo(ROI);
			
				// �¿�����
				hist_roi = get_norm_HShist(ROI, ROI_hsv, config.HIST_CHANNELS, config.HIST_SIZE, config.HIST_RANGES);

				// ������ɫ�����ӿ�
				////if (config.VISUAL_TEMP_RESULT)
				//{
				//	// ��Ҫ��ԭframe�ϻ�
				//	//rectangle(f_copy, ROI_rect, Scalar(255, 0, 0), 1);
				//	//imshow("particle", frame);
				//}
			
				//�Ƚ�Ŀ���ֱ��ͼ���������������ֱ��ͼ,����particleȨ��
				// https://blog.csdn.net/shuiyixin/article/details/80257822
				particles[particle_idx].weight = exp(-100 * (compareHist(hist_object, hist_roi, cv::HISTCMP_BHATTACHARYYA)));
			}

			// ��һ��Ȩ�� 
			normalize_weights(particles);

			// �����˶�Ŀ��֡��Ȩ��
			if ((frame_idx - 1) % config.FRAME_DIFF_INTERVAL == 0){
				merge_particles_frameDiff_weights(particles, moving_obj_mask, config.LAMBDA_FRAME_DIFF);
			}

			// �����Ӱ�Ȩ�شӸߵ�������
			des_sort(particles);

			// �ز���
			resample(particles, new_particles, rng);
		
			// �ٴ�����
			des_sort(particles);

			// ����Ŀ��λ��	
			predict_bbox = get_predict_bbox(particles, frame.rows, frame.cols, config.PREDICT_WITH_WEIGHT, config.PREDICT_USE_RATIO);
		}

		// **** �ָ�Ŀ�� **** 
		if (frame_idx % config.GRAPH_CUT_INTERVAL == 0)
		{
			Mat mask = my_graph_cut(predict_bbox, frame, config.GRAPH_CUT_ITERS_COUNT);
			make_visual_mask(frame, mask, predict_bbox, config.VISUAL_PREDICT_MASK_COLOR, config.VISUAL_MASK_WEIGHT, true);
		}

		// ���ٿ���ʾ
		if (config.TEST_WITH_LABEL)
		{
			rectangle(frame, annotation_bboxes[frame_idx], config.VISUAL_TARGET_BBOX_COLOR, 1);
		}
		rectangle(frame, predict_bbox, config.VISUAL_PREDICT_BBOX_COLOR, 1);

		// ����
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

// ���ӿ��� �˶�Ŀ�����ر���
void merge_particles_frameDiff_weights(vector<PARTICLE>& particles, Mat& obj_mask, double frame_diff_lambda)
{
	if (obj_mask.empty())
		return;

	double nozero_sum = 1e-12;
	vector<double> frame_diff_weights(particles.size());

	// ͳ��֡�����Ȩ��
	for (int i = 0; i < particles.size(); i++)
	{
		Rect ROI_rect = get_ROI_rect_from_frame(particles[i].x, particles[i].y, particles[i].width,
			particles[i].height, obj_mask.cols, obj_mask.rows, particles[i].scale);

		frame_diff_weights[i] = (double)(cv::countNonZero(obj_mask(ROI_rect)));

		nozero_sum += frame_diff_weights[i];
	}

	// ��һ��֡�Ȩ��
	for (int i = 0; i < frame_diff_weights.size(); i++)
	{
		particles[i].weight += frame_diff_lambda * frame_diff_weights[i] / nozero_sum;
	}

	// ��һ����Ȩ��
	normalize_weights(particles);
	return;
}