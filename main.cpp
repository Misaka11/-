//
//	主力输出：赵晨余
//	环境配置：石一泽
//	辅助输出：董伟良，赵佳
//	资料查询：黄菩臣
//  日期：2017.7.14
//	测试环境：vs2013 opencv3.0
//	分类器使用: haarcascade_frontalface_alt2.xml， haarcascade_eye.xml
#include "opencv2\opencv.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <ctime>
using namespace std;
using namespace cv;

template<typename T>
string ToString(const T &t, const int &precision = 6)
{
	stringstream oss;
	oss.precision(precision);
	oss.setf(ios_base::fixed);
	oss << t;
	return oss.str();
}

bool cmp1(double x, double y)
{
	return x < y;
}

bool input(CascadeClassifier &cascade, CascadeClassifier &cascade2,Mat &srcImage)
{
	string name;
	cout << "Input file name: ";
	cin >> name;
	if (!cascade.load("haarcascade_frontalface_alt2.xml")
	|| !cascade2.load("haarcascade_eye.xml"))
	{
		cout << "Classifier error!";
		return false;
	}
	srcImage = imread(name);
	return true;
}

void check(vector<Rect>&rect, vector<Rect>&rect2, vector<int>&Rlvl, vector<int>&Rlvl2, vector<double>&lvlW, vector<double>&lvlW2)
{
	for (int i = 0; i < rect.size(); ++i)
	{
		int target = -1;
		double W;
		for (int j = 0; j < rect2.size(); ++j)
		{
			Point center = Point(rect2[j].x + (double)rect2[j].width / 2, rect2[j].y + (double)rect2[j].height / 2);
			if (center.x >= rect[i].x&&center.x <= rect[i].x + rect[i].width&&center.y >= rect[i].y&&center.y <= rect[i].y + rect[i].height
				&&lvlW2[j]>0 && lvlW2[j]<2 && (target == -1 || lvlW2[j]>lvlW2[target]))
				target = j;
		}
		W = ((lvlW[i]-50.5) / 4.75)*((lvlW[i]-50.5) / 4.75);
		if (target != -1)
			W += (lvlW2[target] / 1)*(lvlW2[target] / 1);
		if (lvlW[i]<50.5 || lvlW[i]>110 || W < 1)
		{
			rect.erase(rect.begin() + i), Rlvl.erase(Rlvl.begin() + i), lvlW.erase(lvlW.begin() + i);
			--i;
		}
	}
	return;
}

void output(Mat dstImage, vector<Rect>&rect, vector<double>&lvlW,Scalar* color,int tim)
{
	cout<<"Number of faces："<<rect.size()<<endl<<"Time: "<<(unsigned)time(NULL)-tim<<"s"<<endl;
	double scaling = 1;
	if (dstImage.size().width>1024 || dstImage.size().height > 768)
		scaling = min(1024.0 / dstImage.size().width, 768.0 / dstImage.size().height);
	for (int i = 0; i < rect.size(); ++i)
		rectangle(dstImage, rect[i], color[i%7], 1 / scaling, 8, 0);
	resize(dstImage, dstImage, Size(dstImage.size().width *scaling, dstImage.size().height *scaling));
	for (int i = 0; i < rect.size(); ++i)
		putText(dstImage, ToString(lvlW[i],2), Point(rect[i].x*scaling, rect[i].y*scaling), 0, 0.5, color[i%7]);
	imshow("dstImage", dstImage);
	return;
}

int main()
{
	CascadeClassifier cascade,cascade2;
	Mat srcImage, grayImage, dstImage;
	double scaling = 1;
	vector<Rect>rect, rect2;
	vector<int>Rlvl, Rlvl2;
	vector<double>lvlW, lvlW2;
	int tim;
	Scalar colors[] =
	{
		CV_RGB(255, 0, 0),
		CV_RGB(255, 97, 0),
		CV_RGB(255, 255, 0),
		CV_RGB(0, 255, 0),
		CV_RGB(0, 255, 255),
		CV_RGB(0, 0, 255),
		CV_RGB(160, 32, 240)
	};

	if (!input(cascade, cascade2, srcImage))
		return 0;

	tim = (unsigned)time(NULL);
	if (srcImage.size().width>2048 || srcImage.size().height > 1536)
		scaling = min(2048.0 / srcImage.size().width, 1536.0 / srcImage.size().height);
	resize(srcImage, srcImage, Size(srcImage.size().width *scaling, srcImage.size().height *scaling));
	dstImage = srcImage.clone();
	grayImage.create(srcImage.size(), srcImage.type());
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	equalizeHist(grayImage, grayImage);
	
	cascade.detectMultiScale(grayImage, rect, Rlvl, lvlW, 1.1, 1, 0, Size(), Size(), true);
	cascade2.detectMultiScale(grayImage, rect2, Rlvl2, lvlW2, 1.1, 1, 0, Size(), Size(), true);

	check(rect, rect2, Rlvl, Rlvl2, lvlW, lvlW2);
	output(dstImage, rect, lvlW, colors, tim);
	waitKey(0);

	return 0;
}