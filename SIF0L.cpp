// SIF01.cpp: определяет точку входа для консольного приложения.
//

#include <stdio.h>
#include <iostream>
#include <core/core.hpp>
#include <features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "WIR01.h"
#include <dirent.h>

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
  //additional data
  char dirSpec[2048];
  WIR01 classificator;
  WIRTrainSample tmpTrainSample;
  vector<WIRTrainSample> trainSamples;
  dirSpec[0]=0;
  strcpy_s(dirSpec,argv[2]);
  DIR *dir;
  struct dirent *ent;

  //получаеми путь ко всем файлам в папке
  if ((dir = opendir (dirSpec)) != NULL) {
	  /* print all the files and directories within directory */
	  while ((ent = readdir (dir)) != NULL) {
		  if (ent->d_namlen<4) continue;
		  if (ent->d_type == DT_REG)
		  {
				tmpTrainSample.classLabel = 0; //not used here;
				tmpTrainSample.imagePath[0] = 0;
				tmpTrainSample.imageName[0] = 0;
				strcpy_s(tmpTrainSample.imagePath, argv[2]);
				strcat_s(tmpTrainSample.imagePath,"\\");
				strcat_s(tmpTrainSample.imagePath,ent->d_name);
				strcpy_s(tmpTrainSample.imageName,ent->d_name);

				cout << tmpTrainSample.imagePath << endl;
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
	
	WIRResult result;
	//обучаемся на созданных файлах
	classificator.addTrainSamples(trainSamples);
	//заместо преведущей строчки можно загрузуить уже обученные данные классификаторов
	//classificator.loadTrainingDB("text.xml");
	if(classificator.Recognize(argv[1],result))
	{
		Mat img_object = imread(argv[1], IMREAD_GRAYSCALE );
		Mat img_match = imread(result.filePath);
		cout << "File "<< argv[1] <<" Matchs "<< result.fileName << endl; 
		imshow("Match",img_match);
		imshow("Object", img_object);
		waitKey(0);
	}
	else
		cout<<"No match has been found"<<endl;
	//сохраняем настройки классификатора
	//classificator.saveTrainingDB("text.xml");
  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SIF01 <image> <lib_path>" << std::endl; }




