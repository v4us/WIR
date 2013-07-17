#include "WIR_clustering.h"

bool WIR_clustering::getCentroidsBRIEF(const cv::Mat& descriptors, cv::Mat& centroids, unsigned int countCentroids)
{
	if (!getMostValuebleDescriptors(descriptors, centroids, countCentroids))
		return false;
	if ((unsigned int) centroids.rows <= countCentroids)
		return true;

	//compute clusters
	Mat dist;
	unsigned int** tmpArray = NULL;
	tmpArray = new unsigned int*[countCentroids];
	if(tmpArray == NULL)
		return false;
	unsigned int* selectCCount = NULL;
	selectCCount =  new unsigned int[countCentroids];
	if(selectCCount == NULL)
	{
		delete[] tmpArray;
		return false;
	};
	for (size_t i = 0; i<countCentroids; i++)
	{
		selectCCount[i] = 0;
		tmpArray[i] = NULL;
		tmpArray[i] = new unsigned int[8*BRIEF_DECTRIPTOR_SIZE];
		if(tmpArray[i] == NULL)
		{
			for(size_t j = 0; j<i; j++)
				delete[] tmpArray[j];
			delete[] tmpArray;
			delete[] selectCCount;
			return false;
		};
		for(size_t j = 0; j<8*BRIEF_DECTRIPTOR_SIZE; j++)
			tmpArray[i][j] = 0;
	};
	int iteration = maxIteration;
	while(iteration > 0)
	{
		//clearnig tmpArray
		for(size_t i = 0; i<countCentroids; i++)
		{
			for(size_t j = 0; j<8*BRIEF_DECTRIPTOR_SIZE; j++)
				tmpArray[i][j] = 0;
			selectCCount[i] = 0;
		};
		//calculating distance
		cv::batchDistance(descriptors,centroids,dist,CV_32S,noArray(),NORM_HAMMING);
		for(size_t i = 0; i<(unsigned int)descriptors.rows; i++)
		{
			unsigned int selectedComponent = 0;
			for(size_t j = 0; j<countCentroids; j++)
				if(dist.at<int>(i,j)<dist.at<int>(i,selectedComponent))
					selectedComponent = j;
			selectCCount[selectedComponent]++;
			//passing bits
			for(size_t j = 0; j<BRIEF_DECTRIPTOR_SIZE;j++)
			{
				for(int k = 0; k<8; k++)
					if(getBit(descriptors.at<unsigned char>(i,j),k)>0)
						tmpArray[selectedComponent][j*8+k]++;
			};
		};
		//Normolizing
		Mat tmpCentroids;
		centroids.copyTo(tmpCentroids);
		for(size_t i = 0; i<countCentroids; i++)
		{
			if(selectCCount[i]>0)
				for(size_t j = 0; j<BRIEF_DECTRIPTOR_SIZE;j++)
				{
					tmpCentroids.at<unsigned char>(i,j) = 0;
					for(int k = 0; k<8; k++)
						if(tmpArray[i][j*8+k]/(double)selectCCount[i]>=0.5)
						{
							unsigned char tmpchar = tmpCentroids.at<unsigned char>(i,j);
							tmpchar = tmpchar | (unsigned int)(1<<k);
							tmpCentroids.at<unsigned char>(i,j) = tmpchar;
						}
						else
						{
							unsigned char tmpchar = tmpCentroids.at<unsigned char>(i,j);
							tmpchar = tmpchar & (unsigned int)~(1<<k);
							tmpCentroids.at<unsigned char>(i,j) = tmpchar;
						};
				};
		};
		cv::batchDistance(tmpCentroids,centroids,dist,CV_32S,noArray(),NORM_HAMMING);
		tmpCentroids.copyTo(centroids);
		//epsilon calculation
		bool breakCondition = true;
		for(size_t i = 0; i<countCentroids; i++)
		{
			breakCondition = (dist.at<int>(i,i)<epsilon) && breakCondition;
				
		};
		if(breakCondition)
			break;
		else
		{
				iteration--;
		};
	};

	//
	for(size_t j = 0; j<countCentroids; j++)
		delete[] tmpArray[j];
	delete[] tmpArray;
	delete[] selectCCount;
	return true;
};

bool WIR_clustering::getMostValuebleDescriptors(const cv::Mat& descriptors, cv::Mat& centroids, unsigned int countCentroids)
{
	if(descriptors.cols!=BRIEF_DECTRIPTOR_SIZE || descriptors.type() != CV_8U || countCentroids == 0)
		return false;
	if ((unsigned int)descriptors.rows<=countCentroids)
	{
		descriptors.copyTo(centroids);
		return true;
	}
	//initiating centroid matrix
	centroids = Mat::zeros(countCentroids,BRIEF_DECTRIPTOR_SIZE, CV_8U);
	//Getting centroids
	//!!! REALLY IMPORTANT THING
	Mat dist;
	stack<DistRecord> distances;
	distances.push(DistRecord());
	cv::batchDistance(descriptors,descriptors,dist,CV_32S,noArray(),NORM_HAMMING);
	DistRecord tmpRecord;
	for (size_t i = 0; i<(unsigned int)dist.rows; i++)
		for (size_t j = i; j<(unsigned int)dist.cols; j++)
		{
			tmpRecord = distances.top();
			if(dist.at<int>(i,j)>=tmpRecord.distance)
				distances.push(DistRecord(i,j,dist.at<int>(i,j)));
		};
	//Coping data
	int tmpCount = countCentroids;
	set<unsigned int> inserted_id;
	while(tmpCount>0)
	{
		if(distances.empty())
		{
			int k = rand() % descriptors.rows;
			if (inserted_id.count(k)==0)
			{
				inserted_id.insert(k);
				for(size_t i = 0; i<BRIEF_DECTRIPTOR_SIZE;i++)
					centroids.at<unsigned char>(tmpCount-1,i) = descriptors.at<unsigned char>(k,i); ///!!!
				tmpCount--;
			}
			continue;
		}
		tmpRecord = distances.top();
		if (inserted_id.count(tmpRecord.i)==0)
		{
			inserted_id.insert(tmpRecord.i);
			for(size_t i = 0; i<BRIEF_DECTRIPTOR_SIZE;i++)
				centroids.at<unsigned char>(tmpCount-1,i) = descriptors.at<unsigned char>(tmpRecord.i,i); ///!!!
			tmpCount--;
		}
		if(tmpCount>0)
			if (inserted_id.count(tmpRecord.j)==0)
			{
				inserted_id.insert(tmpRecord.j);
				for(size_t i = 0; i<BRIEF_DECTRIPTOR_SIZE;i++)
					centroids.at<unsigned char>(tmpCount-1,i) = descriptors.at<unsigned char>(tmpRecord.j,i); ///!!!
				tmpCount--;
			}
		distances.pop();
	};
	return true;
}


