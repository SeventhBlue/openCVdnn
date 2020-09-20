#include <opencv2/highgui.hpp>

#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"

#include "Yolo.h"
#include "OpenPose.h"
#include "TextDetection.h"

using namespace cv;

void runningYoloV3();
void runningYoloV4();
void runningOpenPose();
void runningTextDetection();


int main(int argc, char** argv)
{
	//runningYoloV3();
	//runningYoloV4();
	//runningOpenPose();
	runningTextDetection();

	return 0;
}

void runningTextDetection() {

	String eastModelPath = "./cfg/east.pb";
	String crnnModelPath = "./cfg/crnn.onnx";
	String wordDict = "";
	TextDetection textDetection = TextDetection(eastModelPath, crnnModelPath, wordDict);
	textDetection.loadModel();

	std::vector<TextDetSt> textDetRet;
	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0)
	{
		cap >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}
		double start_time = (double)cv::getTickCount();

		std::vector<TextDetSt> textDetRet;
		textDetection.runningTextDetection(frame, textDetRet);
		textDetection.showDetectRet(frame, textDetRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("Text Detection results", frame);

		//textDetection.saveVider(frame, textDetRet);
	}
}

void runningOpenPose() {
	String modelPath = "./cfg/openpose_coco.caffemodel";
	String configPath = "./cfg/openpose_coco.prototxt";
	String dataType = "COCO";
	OpenPose openPose = OpenPose(modelPath, configPath, dataType);
	openPose.loadModel();


	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<cv::Point> openPoseRet(22);
		openPose.runningOpenPose(frame, openPoseRet);
		openPose.showDetectRet(frame, openPoseRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV4 detect results", frame);

		//openPose.saveVider(frame, yoloRet);
	}
}

void runningYoloV4() {
	String modelPath = "./cfg/yolov4_coco.weights";
	String configPath = "./cfg/yolov4_coco.cfg";
	String classesFile = "./data/coco.names";
	Yolo yolov4 = Yolo(modelPath, configPath, classesFile, true);
	yolov4.loadModel();


	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<YoloDetSt> yoloRet;
		yolov4.runningYolo(frame, yoloRet);
		yolov4.drowBoxes(frame, yoloRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV4 detect results", frame);

		//yolov4.saveVider(frame, yoloRet);
	}
}

void runningYoloV3() {
	String modelPath = "./cfg/yolov3_coco.weights";
	String configPath = "./cfg/yolov3_coco.cfg";
	String classesFile = "./data/coco.names";
	Yolo yolov3 = Yolo(modelPath, configPath, classesFile, true);
	yolov3.loadModel();


	VideoCapture cap;
	cap.open(0);
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<YoloDetSt> yoloRet;
		yolov3.runningYolo(frame, yoloRet);
		yolov3.drowBoxes(frame, yoloRet);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV3 detect results", frame);

		//yolov3.saveVider(frame, yoloRet);
	}
}