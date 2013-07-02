// SIF01.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"


#include <stdio.h>
#include <iostream>
#include <core/core.hpp>
#include <features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "WIR01.h"
#include "WIR_OCR.h"
#include "dirent.h"
#define _CRT_SECURE_NO_WARNINGS
#define _ATL_SECURE_NO_WARNINGS

using namespace std;
using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }
  //WIR_OCR tmpOCR;
  //tmpOCR.m();
  //tmpOCR.saveTrainingDB("OCR.XML");
  //additional data
  cout<<"Please select mode:\n 1 standart\n2 fast\n3 DFast"<<endl;
  int workingMode;
  cin>>workingMode;

  WIRParam param;
  switch (workingMode)
  {
  case 1:
	  param = WIRParam(standartWIR);
	  break;
  case 2:
	  param = WIRParam(fastWIR);
	  break;
  case 3:
	  param = WIRParam(doubleFastWIR);
	  break;
  default:
	  param = WIRParam();
  };
  param.labelExtraction = WIR_EL_SOFT;
  char dirSpec[2048];
  WIR01 classificator(param);
  WIRTrainSample tmpTrainSample;
  vector<WIRTrainSample> trainSamples;
  dirSpec[0]=0;
  strcpy_s(dirSpec,argv[2]);
  DIR *dir;
  struct dirent *ent;
  //получаеми путь ко всем файлам в папке
  if ((dir = opendir (dirSpec)) != NULL) {
	  // print all the files and directories within directory
	  while ((ent = readdir (dir)) != NULL) {
		  if (ent->d_namlen<4) 
			  continue;
		  if (ent->d_name[0] == '.')
			  continue;
		  if (ent->d_type == DT_REG)
		  {
				tmpTrainSample.classLabel = 0; //not used here;
				tmpTrainSample.imagePath[0] = 0;
				tmpTrainSample.imageName[0] = 0;
				strcpy(tmpTrainSample.imagePath, argv[2]);
				strcat(tmpTrainSample.imagePath,"\\");
				strcat(tmpTrainSample.imagePath,ent->d_name);
				strcpy(tmpTrainSample.imageName,ent->d_name);

				//cout << tmpTrainSample.imagePath << endl;
				trainSamples.push_back(tmpTrainSample);
		  }
	  }
	  closedir (dir);

	} 
  else 
  {
	 cout << "No pictures have been found" <<endl;
	 readme();
	return -1;
	}

	vector<WIRResult> results;
	 //для проведения эксмперементов и исследования устанавливает опциональное 
	//сравнение гистограммы фрагментов представляющих интерес в каждом из моментов.
	//сейчас флаг должен быть установлен на FALSE при работе на сервере
	//classificator.setHistogramUse(true);
	//обучаемся на созданных файлах
	classificator.addTrainSamples(trainSamples);
	//Загрузка данных в бинарном формате из директории. Основоне предназначение: мобильные устройства.
	//classificator.LoadBinary("C:\\LGP500\\1");

	//заместо преведущей строчки можно загрузуить уже обученные данные классификаторов
	//classificator.loadTrainingDB("test.xml");
	cout<<"LOADED"<<endl;
	if(classificator.Recognize(argv[1],results,3))
	{
		Mat img_object = imread(argv[1], IMREAD_GRAYSCALE );
		Mat img_match = imread(results[0].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[0].fileName << endl;
		cout <<"Detected Year: "<< results[0].year<<endl;
		cout <<"Hist: "<< results[0].hist<<endl;
		if(!img_object.empty() && !img_match.empty())
		{
			imshow("Match",img_match);
			imshow("Object", img_object);
		}
		//Histogramm
		cout<<"Histograms data"<<endl;
		unsigned int minId = 0;
		for(size_t i = 0; i<results.size(); i++)
		{
			if(results[i].hist<results[minId].hist)
				minId = i;
			cout<<"#"<<i<<" : " <<results[i].hist<<endl;
		}
		cout << "Best Hist feeting"<<endl;
		cout << "-------------------------------------------------------"<<endl;
		img_match = imread(results[minId].filePath);
		cout << "File "<< argv[1] <<" Matchs "<< results[minId].fileName << endl;
		cout <<"Detected Year: "<< results[minId].year<<endl;
		cout <<"Hist: "<< results[minId].hist<<endl;
		if(!img_match.empty())
			imshow("Match HIst",img_match);
	}
	else
		cout<<"No match has been found"<<endl;
	//сохраняем настройки классификатора
	//classificator.saveTrainingDB("test.xml");

	//ГЕнерируем обновление на основе переданных данных. 
	//vector<const char*> inputNames; inputNames.push_back("C:\\LGP500");
	//WIR01::GenerateUpdates(param,trainSamples,inputNames,10,10,"Test");

	//Сохраняем данные в бинарном формате предназначен в основном для мобильных устройств.
	//classificator.SaveBinary("C:\\LGP500\\1");
	cv::waitKey(0);
  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIF01 <image> <lib_path>" << std::endl; }




