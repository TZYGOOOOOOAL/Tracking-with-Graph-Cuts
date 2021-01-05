#include "eval.h"
#include "Config.h"
#include "utils.h"


// ��������ָ�� https://www.cnblogs.com/P3nguin/p/10570053.html
// https://blog.csdn.net/weixin_36761349/article/details/90370537
// ׼ȷ�ʣ�����ֵ�����ľ��� <= ��ֵ ����  & ƽ��������� Average Pixel Error��APE��
// �ɹ��ʣ�����ֵ��IoU >= ��ֵ     ����   & ƽ���ص��� Average Overlap Rate��AOR��

void eval(const vector<Rect>& predicts, const vector<Rect>& targets, const Config& config)
{
	int size = predicts.size();
	assert(size == targets.size());

	double pixel_diff_sum = 0.0;	// �������������
	double iou_sum = 0.0;			// ��iou

	int tp_pixel_diff_num = 0;		// ��ֵ�����ľ��� < ��ֵ ����
	int tp_iou_num = 0;				// ��ֵ��IoU >= ��ֵ ����

	double predict_center_x, predict_center_y;
	double target_center_x, target_center_y;

	double iou, pixel_diff;

	for (size_t i = 0; i < size; i++)
	{
		// �������
		predict_center_x = predicts[i].x + predicts[i].width / 2.0;
		predict_center_y = predicts[i].y + predicts[i].height / 2.0;
		target_center_x = targets[i].x + targets[i].width / 2.0;
		target_center_y = targets[i].y + targets[i].height / 2.0;

		pixel_diff = std::sqrt((predict_center_x - target_center_x) * (predict_center_x - target_center_x) + \
			(predict_center_y - target_center_y) * (predict_center_y - target_center_y));

		pixel_diff_sum += pixel_diff;
		if (pixel_diff <= config.EVAL_PIXEL_DIFF_TH){
			tp_pixel_diff_num++;
		}

		// iou���
		iou = bbox_iou(predicts[i], targets[i]);
		iou_sum += iou;
		if (iou >= config.EVAL_IOU_TH){
			tp_iou_num++;
		}
	}

	// ��ʾ���
	cout << "\n   ***   Eval Result   ***" << endl;
	cout << "׼ȷ�� Precision = " << 1.0 * tp_pixel_diff_num / size << endl;
	cout << "�ɹ��� Success   = " << 1.0 * tp_iou_num / size << endl;
	cout << "ƽ��������� APE = " << pixel_diff_sum / size << endl;
	cout << "ƽ���ص���   AOR = " << iou_sum / size << endl;

	return;
}

void eval_by_txts(Config &config)
{
	vector<string> data_dirs = get_child_dirs(config.DATA_DIR);

	for (int pixel_diff_th = 0; pixel_diff_th <= 50; pixel_diff_th += 5)
	{
		config.EVAL_PIXEL_DIFF_TH = pixel_diff_th;

		for (double iou_th = 1.0; iou_th > 0.0; iou_th -= 0.1)
		{
			config.EVAL_IOU_TH = iou_th;
			cout << "\n\npixel_diff_th = " << pixel_diff_th << endl;
			cout << "iou_th = " << iou_th << endl;

			for (int i = 0; i < data_dirs.size(); i++)
			{
				// ����·��
				string dir_path = data_dirs[i];
				string ann_path = get_child_files(dir_path, vector<string>({ ".txt" }))[0];

				string video_save_dir = path_join(config.SAVE_DIR_PATH, get_filename(dir_path));
				string predicts_save_txt_path = path_join(video_save_dir, "predicts.txt");

				cout << "\n ###### " << get_filename(dir_path) << " ######" <<endl;
				assert(is_file(predicts_save_txt_path) && is_file(ann_path));

				// ��ע
				vector<Rect> ann_bboxes = parse_one_annotation(ann_path);

				// Ԥ��
				vector<Rect> predicts = parse_one_annotation(predicts_save_txt_path);

				eval(predicts, ann_bboxes, config);
			}
		}
	}
}