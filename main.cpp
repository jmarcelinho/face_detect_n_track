#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "VideoFaceDetector.h"

using namespace std;
using namespace cv;

const String WINDOW_NAME("Camera video");
const String CASCADE_FILE("haarcascade_frontalface_alt.xml");

int main(int argc, char **argv)
{
	// Try opening camera
	VideoCapture camera(0);
	//VideoCapture camera("D:\\video.mp4");
	if (!camera.isOpened())
	{
		fprintf(stderr, "Error getting camera...\n");
		exit(1);
	}
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0); // 0 to High Quality
	namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

	VideoFaceDetector detector(CASCADE_FILE, camera);
	Mat frame;
	int id = 0;
	double fps = 0, time_per_frame;
	while (true)
	{
		auto start = getCPUTickCount();
		detector >> frame;
		auto end = getCPUTickCount();

		time_per_frame = (end - start) / getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f\n", time_per_frame, fps);

		if (detector.isFaceFound())
		{
			String path = "image/captureFace_" + to_string(id++) + ".png";
			imwrite(path, frame(detector.face()), compression_params);
			rectangle(frame, detector.face(), Scalar(255, 0, 0));
			circle(frame, detector.facePosition(), 30, Scalar(0, 255, 0));
		}

		imshow(WINDOW_NAME, frame);
		if (waitKey(25) == 27)
			break;
	}

	return 0;
}
