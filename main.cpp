#include <opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "particles.h"
#include "utils.h"
#include "make_data.h"
#include "Config.h"
#include "my_track.h"
#include "eval.h"

using namespace std;
using namespace cv;


///*****************************************************************/
int main()
{
	Config config;
	
	make_video(config);
	//show_annotations(config);

	// 创建保存路径
	if (config.SAVE_RESULT){
		make_dir(config.SAVE_DIR_PATH);
	}

	//eval_by_txts(config);

	// 对所有文件遍
	vector<string> data_dirs = get_child_dirs(config.DATA_DIR);

	//for (int i = 0; i < data_dirs.size(); i++)
	for (int i = 3; i < data_dirs.size(); i++)
	{
		// 处理路径
		string dir_path = data_dirs[i];
		string video_path = get_filename(dir_path) + config.VIDEO_FMT;
		video_path = path_join(dir_path, video_path);
		string ann_path = get_child_files(dir_path, vector<string>({ ".txt" }))[0];

		assert(is_file(video_path) && is_file(ann_path));

		// 起始框
		Rect init_bbox = get_init_bbox(ann_path);

		// 标注
		vector<Rect> ann_bboxes = parse_one_annotation(ann_path);

		cout << "TRACK in " << video_path << endl;
		vector<Rect> predicts;
		predicts = track(video_path, init_bbox, config, ann_bboxes);

		eval(predicts, ann_bboxes, config);
	}

	system("pause");
}
