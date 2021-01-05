#include "make_data.h"
#include "utils.h"
#include "Config.h"

void imgs2video(const vector<string> &img_paths, string video_save_path, double fps)
{
	assert(!img_paths.empty());

	Size video_size = imread(img_paths[0]).size();
	VideoWriter writer(video_save_path, CV_FOURCC('X', 'V', 'I', 'D'), fps, video_size);

	// ��ȡ
	Mat frame;
	for (int i = 0; i < img_paths.size(); i++)
	{
		frame = imread(img_paths[i]);
		if (frame.size() != video_size)
			resize(frame, frame, video_size);
		writer << frame;
	}

	writer.release();
	return;
}


// һ��ͼ����ϳ���Ƶ
void make_video(const Config &config)
{
	// ����ͼƬ�ļ���
	vector<string> data_dirs = get_child_dirs(config.DATA_DIR);

	// ����ÿ���ļ���
	for (int i = 0; i < data_dirs.size(); i++)
	{
		// ����·��
		string dir_path = data_dirs[i];
		vector<string> img_paths = get_all_files(dir_path, vector<string>({ config.IMG_FMT }));
		string video_path = get_filename(dir_path) + config.VIDEO_FMT;
		video_path = path_join(dir_path, video_path);

		if (is_exist(video_path))
		{
			cout << video_path << " is exist !!!" << endl;
			continue;
		}

		// ������Ƶ
		imgs2video(img_paths, video_path, config.VIDEO_FPS);
		cout << "Make Video " << video_path << endl;
	}
}


// ��ʾ��ע
void show_annotations(const Config &config)
{
	vector<string> data_dirs = get_child_dirs(config.DATA_DIR);
	for (int i = 0; i < data_dirs.size(); i++)
	{
		string annotation_path = get_child_files(data_dirs[i], vector<string>({".txt"}))[0];
		vector<Rect> gt_bboxes = parse_one_annotation(annotation_path);

		// ����Ƶ
		string video_path = get_child_files(data_dirs[i], vector<string>({config.VIDEO_FMT}))[0];
		VideoCapture capture(video_path);
		Mat frame;
		int frame_idx = 0;
		while (true){
			capture >> frame;
			if (frame.empty()){
				cout << "Read " << video_path << " Finish" << endl;
				break;
			}
			vector<Rect> gt_bbox;
			gt_bbox.emplace_back(gt_bboxes[frame_idx]);
			visual_bboxes(frame, gt_bbox, Scalar(0, 0, 255), 1);
			imshow("", frame);
			waitKey(60);
			frame_idx++;
		}
	}

}