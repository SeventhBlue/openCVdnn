#include "TextDetection.h"

TextDetection::TextDetection(cv::String& EASTModelPath, cv::String& CRNNModelPath, cv::String& wordDict) {
	m_EASTModelPath = EASTModelPath;
	m_CRNNModelPath = CRNNModelPath;
	if (wordDict == "") {
		m_wordDict = "0123456789abcdefghijklmnopqrstuvwxyz";
	}
	else {
		std::cout << "The m_wordDict is null!" << std::endl;
	}
	m_outNames.push_back("feature_fusion/Conv_7/Sigmoid");
	m_outNames.push_back("feature_fusion/concat_3");
}

TextDetection::~TextDetection() {
	m_viderWriter.release();
}

int TextDetection::loadModel() {
	CV_Assert(!m_EASTModelPath.empty());
	// Load networks.
	m_east = cv::dnn::readNet(m_EASTModelPath);
	m_east.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	m_east.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	if (!m_CRNNModelPath.empty()) {
		m_crnn = cv::dnn::readNet(m_CRNNModelPath);
		m_crnn.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		m_crnn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else {
		std::cout << "The CRNNs model file not read!" << std::endl;
	}
}

int TextDetection::runningTextDetection(cv::Mat& img, std::vector<TextDetSt>& textDetRet) {
	cv::dnn::blobFromImage(img, m_blob, 1.0, cv::Size(m_inpWidth, m_inpHeight), cv::Scalar(123.68, 116.78, 103.94), true, false);
	m_east.setInput(m_blob);
	m_east.forward(m_outs, m_outNames);

	cv::Mat scores = m_outs[0];
	cv::Mat geometry = m_outs[1];

	// Decode predicted bounding boxes.
	std::vector<cv::RotatedRect> boxes;
	std::vector<float> confidences;
	decodeBoundingBoxes(scores, geometry, m_confThreshold, boxes, confidences);

	// Apply non-maximum suppression procedure.
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);
	cv::Point2f ratio((float)img.cols / m_inpWidth, (float)img.rows / m_inpHeight);

	// Render text.
	for (size_t i = 0; i < indices.size(); ++i)
	{
		TextDetSt td;
		cv::RotatedRect& box = boxes[indices[i]];
		cv::Point2f vertices[4];
		box.points(vertices);
		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}
		if (!m_CRNNModelPath.empty())
		{
			cv::Mat cropped;
			fourPointsTransform(img, vertices, cropped);
			cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);
			cv::Mat blobCrop = cv::dnn::blobFromImage(cropped, 1.0 / 127.5, cv::Size(), cv::Scalar::all(127.5));
			m_crnn.setInput(blobCrop);
			
			cv::Mat result = m_crnn.forward();

			decodeText(result, td.text);
		}
		for (int j = 0; j < 4; ++j) {
			td.vertices[j] = vertices[j];
		}
		textDetRet.push_back(td);
	}
}

void TextDetection::showDetectRet(cv::Mat& img, std::vector<TextDetSt>& textDetRet) {
	for (int i = 0; i < textDetRet.size(); ++i) {
		putText(img, textDetRet[i].text, textDetRet[i].vertices[1], cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255));
		for (int j = 0; j < 4; ++j) {
			line(img, textDetRet[i].vertices[j], textDetRet[i].vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
		}
	}
}


void TextDetection::decodeText(const cv::Mat& scores, std::string& text)
{
	cv::Mat scoresMat = scores.reshape(1, scores.size[0]);
	std::vector<char> elements;
	elements.reserve(scores.size[0]);
	for (int rowIndex = 0; rowIndex < scoresMat.rows; ++rowIndex)
	{
		cv::Point p;
		cv::minMaxLoc(scoresMat.row(rowIndex), 0, 0, 0, &p);
		if (p.x > 0 && static_cast<size_t>(p.x) <= m_wordDict.size())
		{
			elements.push_back(m_wordDict[p.x - 1]);
		}
		else
		{
			elements.push_back('-');
		}
	}
	if (elements.size() > 0 && elements[0] != '-')
		text += elements[0];
	for (size_t elementIndex = 1; elementIndex < elements.size(); ++elementIndex)
	{
		if (elementIndex > 0 && elements[elementIndex] != '-' &&
			elements[elementIndex - 1] != elements[elementIndex])
		{
			text += elements[elementIndex];
		}
	}
}

void TextDetection::fourPointsTransform(const cv::Mat& frame, cv::Point2f vertices[4], cv::Mat& result)
{
	const cv::Size outputSize = cv::Size(100, 32);
	cv::Point2f targetVertices[4] = { cv::Point(0, outputSize.height - 1),
								  cv::Point(0, 0), cv::Point(outputSize.width - 1, 0),
								  cv::Point(outputSize.width - 1, outputSize.height - 1),
	};
	cv::Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	warpPerspective(frame, result, rotationMatrix, outputSize);
}

void TextDetection::decodeBoundingBoxes(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
	std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
	detections.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);
	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < scoreThresh)
				continue;
			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];
			cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
				offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
			cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
			cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			detections.push_back(r);
			confidences.push_back(score);
		}
	}
}

// 返回格式化时间：20200426_150925
std::string TextDetection::getLocNameTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_name = cv::format("%d", t.tm_year + 1900) + cv::format("%.2d", t.tm_mon + 1) + cv::format("%.2d", t.tm_mday) + "_" +
		cv::format("%.2d", t.tm_hour) + cv::format("%.2d", t.tm_min) + cv::format("%.2d", t.tm_sec);
	return time_name;
}

void TextDetection::setViderWriterPara(const cv::Mat& img) {
	m_saveH = img.size().height;
	m_saveW = img.size().width;
	m_viderName = "./data/" + getLocNameTime() + ".avi";
	m_viderWriter = cv::VideoWriter(m_viderName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(m_saveW, m_saveH));
	m_frames = 0;
}

void TextDetection::saveVider(cv::Mat img, std::vector<TextDetSt>& textDetRet) {
	showDetectRet(img, textDetRet);
	if ((m_saveH == 0) && (m_saveW == 0)) {
		setViderWriterPara(img);
		m_viderWriter << img;
	}
	else {
		if ((m_saveH != img.size().height) || (m_saveW != img.size().width)) {
			cv::resize(img, img, cv::Size(m_saveW, m_saveH));
			m_viderWriter << img;
		}
		else {
			m_viderWriter << img;
		}
	}

	++m_frames;
	if (m_frames == 25 * 60 * 10) {   // 每十分钟从新录制新视频
		m_saveH = 0;
		m_saveW = 0;
		m_viderWriter.release();
	}
}