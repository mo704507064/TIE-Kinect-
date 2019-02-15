#include <time.h>
#include "kinect.h"
#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <thread>
#include <mutex>
//#define GL_DISPLAY
//#define SAVE_IMG
using namespace cv;
using namespace std;

// Release Pointer
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

void
// Two-layer pixelFilterter
PixelFilter(unsigned short* depthArray, unsigned short* smoothDepthArray, int innerBandThreshold = 3, int outerBandThreshold = 7)
{
	// The bound of the depthmap
	int widthBound = 512 - 1;
	int heightBound = 424 - 1;

	// Each row
	for (int depthArrayRowIndex = 0; depthArrayRowIndex<424; depthArrayRowIndex++)
	{
		// Each colum
		for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
		{
			int depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * 512);

			// select candidate pixel with zero value
			if (depthArray[depthIndex] == 0)
			{
				// calculate the position of the selected pixel
				int x = depthIndex % 512;
				int y = (depthIndex - x) / 512;

				// collection the pixel value in the filter 
				unsigned short filterCollection[24][2] = { 0 };

				// the number of the non-zero pixel to determine whether it is flitered
				int innerBandCount = 0;
				int outerBandCount = 0;
				//The following loop will traverse the 5 X 5 pixel array centered on the candidate pixels.
				//If the pixels in the array are non-zero, then we will record the depth value and add the 
				//counter at its boundary to one. If the counter is higher than the set threshold, then we will 
				//apply the mode of the depth value (the highest frequency depth value) of the filter to the candidate pixels.
				for (int yi = -2; yi < 3; yi++)
				{
					for (int xi = -2; xi < 3; xi++)
					{
						if (xi != 0 || yi != 0)
						{
							// decide the position of the pixel
							int xSearch = x + xi;
							int ySearch = y + yi;
							//check whether out of boundry of depth map.
							if (xSearch >= 0 && xSearch <= widthBound &&
								ySearch >= 0 && ySearch <= heightBound)
							{
								int index = xSearch + (ySearch * 512);
								if (depthArray[index] != 0)
								{
									// calculate the number of the pixel
									for (int i = 0; i < 24; i++)
									{
										if (filterCollection[i][0] == depthArray[index])
										{
											filterCollection[i][1]++;
											break;
										}
										else if (filterCollection[i][0] == 0)
										{
											filterCollection[i][0] = depthArray[index];
											filterCollection[i][1]++;
											break;
										}
									}
									//  decide which filter is non-zero.
									if (yi != 2 && yi != -2 && xi != 2 && xi != -2)
										innerBandCount++;
									else
										outerBandCount++;
								}
							}
						}
					}
				}

				// if the number of non-zero pixel in each filter is out of the threshold,set the candidate pixel value to the mode.
				if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold)
				{
					short frequency = 0;
					short depth = 0;
					for (int i = 0; i < 24; i++)
					{
						if (filterCollection[i][0] == 0)
							break;
						if (filterCollection[i][1] > frequency)
						{
							depth = filterCollection[i][0];
							frequency = filterCollection[i][1];
						}
					}

					smoothDepthArray[depthIndex] = depth;
				}
				else
				{
					smoothDepthArray[depthIndex] = 0;
				}
			}
			else
			{
				// if the candidate pixel is non-zero,the keep the own value.
				smoothDepthArray[depthIndex] = depthArray[depthIndex];
			}
		}
	}
}

Mat
// visualize the depthmap
ShowDepthImage(unsigned short* depthData)
{
	Mat result(424, 512, CV_8UC4);
	for (int i = 0; i < 512 * 424; i++)
	{
		UINT16 depthValue = depthData[i];
		if (depthValue == 0)
		{
			result.data[i * 4] = 255;
			result.data[i * 4 + 1] = 0;
			result.data[i * 4 + 2] = 0;
			result.data[i * 4 + 3] = depthValue % 256;
		}
		else
		{
			result.data[i * 4] = depthValue % 256;
			result.data[i * 4 + 1] = depthValue % 256;
			result.data[i * 4 + 2] = depthValue % 256;
			result.data[i * 4 + 3] = depthValue % 256;
		}
	}
	return result;
}

int main()
{
#pragma region 
	// Acquire Kinect
	IKinectSensor* m_pKinectSensor;
	ICoordinateMapper*      m_pCoordinateMapper;
	CameraIntrinsics* m_pCameraIntrinsics = new CameraIntrinsics();
	HRESULT hr;
	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	IMultiSourceFrameReader* m_pMultiFrameReader;
	IBodyFrameSource* m_pBodyFrameSource;
	IBodyFrameReader* m_pBodyFrameReader;
	if (m_pKinectSensor)
	{
		hr = m_pKinectSensor->Open();
		if (SUCCEEDED(hr))
		{
			m_pKinectSensor->get_BodyFrameSource(&m_pBodyFrameSource); 
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Color |
				FrameSourceTypes::FrameSourceTypes_Infrared |
				FrameSourceTypes::FrameSourceTypes_Depth,
				&m_pMultiFrameReader);
		}
	}
	if (SUCCEEDED(hr))
	{
		hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	}
	if (!m_pKinectSensor || FAILED(hr))
	{
		return E_FAIL;
	}
	IDepthFrameReference* m_pDepthFrameReference = NULL;
	IColorFrameReference* m_pColorFrameReference = NULL;
	IDepthFrame* m_pDepthFrame = NULL;
	IColorFrame* m_pColorFrame = NULL;
	Mat i_rgb(1080, 1920, CV_8UC4);      //attention£ºthe pic format of kinect is RGBA. 


	IMultiSourceFrame* m_pMultiFrame = NULL;

	DepthSpacePoint*        m_pDepthCoordinates = NULL;
	ColorSpacePoint*        m_pColorCoordinates = NULL;
	CameraSpacePoint*        m_pCameraCoordinates = NULL;

#pragma endregion
	m_pColorCoordinates = new ColorSpacePoint[512 * 424];
	m_pCameraCoordinates = new CameraSpacePoint[512 * 424];
	UINT16 *pixelFilterData = new UINT16[424 * 512];
	UINT16 *averagedDepthData = new UINT16[424 * 512];
	
	BYTE *bgraData = new BYTE[1080 * 1920 * 4];
	UINT16 *depthData = new UINT16[424 * 512];
	new UINT16[424 * 512];
	Mat i_depth(424, 512, CV_16UC1);
	Mat i_average(424, 512, CV_16UC1);
	Mat i_before(424, 512, CV_8UC4);
	Mat i_pixFilter(424, 512, CV_8UC4);
	// store the N frames of the depthmap
	std::vector<UINT16*> queDepthArrays;
#pragma region 
	std::mutex mutex1;
	std::thread th1 = thread([&]{
		while (1)
		{
			mutex1.lock();
			vector<float> cloud;
			HRESULT hr = 0;
			// acquire a new frame
			hr = m_pMultiFrameReader->AcquireLatestFrame(&m_pMultiFrame);
			if (m_pMultiFrame == NULL)
			{
				mutex1.unlock();
				continue;
			}
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_ColorFrameReference(&m_pColorFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pColorFrameReference->AcquireFrame(&m_pColorFrame);
			if (SUCCEEDED(hr))
				hr = m_pMultiFrame->get_DepthFrameReference(&m_pDepthFrameReference);
			if (SUCCEEDED(hr))
				hr = m_pDepthFrameReference->AcquireFrame(&m_pDepthFrame);

			// color map
			UINT nColorBufferSize = 1920 * 1080 * 4;
			if (SUCCEEDED(hr))
				hr = m_pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, bgraData, ColorImageFormat::ColorImageFormat_Bgra);

			// depth map
			UINT nDepthBufferSize = 424 * 512;
			if (SUCCEEDED(hr))
			{
				hr = m_pDepthFrame->CopyFrameDataToArray(nDepthBufferSize, depthData); 
			}

			// two-layer pixelfilter
			PixelFilter(depthData, pixelFilterData, 3, 5);

			// weighted mean
			// num of frame N
			int N = 5;
			UINT16 sumDepthData[424 * 512] = { 0 };
			if (queDepthArrays.size() < N)
			{
				UINT16 *temp = new UINT16[424 * 512];
				memcpy(temp, depthData, 424 * 512 * 2);
				queDepthArrays.push_back(temp);
			}
			else
			{
				if (queDepthArrays.size() == N)
				{
					// add current frame to the queue
					UINT16 *temp = new UINT16[424 * 512];
					memcpy(temp, depthData, 424 * 512 * 2);
					queDepthArrays.push_back(temp);
				}
				else
				{
					memcpy(queDepthArrays.back(), depthData, 424 * 512 * 2);
				}
				int Denominator = 0;
				int Count = 1;

				// create an empty depth map, which stores the weighted sum of the previous N frames and the current frame depth values at each pixel position. 
				// Finally, divide each pixel by the sum of weights
				for each (auto item in queDepthArrays)
				{
					// Each row
					for (int depthArrayRowIndex = 0; depthArrayRowIndex < 424; depthArrayRowIndex++)
					{
						// Each colum
						for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++)
						{
							int index = depthArrayColumnIndex + (depthArrayRowIndex * 512);
							sumDepthData[index] += item[index] * Count;
						}
					}
					Denominator += Count;
					Count++;
				}

				// divide each pixel by the sum of weights
				for (int i = 0; i<512 * 424;i++) 
				{
					queDepthArrays.back()[i] = depthData[i];
					averagedDepthData[i] = (short)(sumDepthData[i] / Denominator);
				}
				// queue.pop()
				auto temp = queDepthArrays.begin();
				for (auto iter = queDepthArrays.begin(); iter !=queDepthArrays.end()-1; iter++)
				{
					*iter = *(iter + 1);
				}
				queDepthArrays.back() = *temp;
			}
			i_average.data = (unsigned char *)averagedDepthData;
#pragma region showImage
			if (SUCCEEDED(hr))
			{
				i_depth.data = (BYTE*)depthData;
				imshow("depth", i_depth);
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("before", ShowDepthImage(depthData));
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("pixFilter", ShowDepthImage(pixelFilterData));
				if (waitKey(1) == VK_ESCAPE)
					break;
				imshow("average", ShowDepthImage(averagedDepthData));
				if (waitKey(1) == VK_ESCAPE)
					break;

			}
#pragma endregion

#pragma region save pic
#ifdef SAVE_IMG
			imwrite("depth.png", i_depth_raw);
			imwrite("color.jpg", i_rgb);
			imwrite("depth2rgb.jpg", i_depthToRgb);
#endif
#pragma endregion
			SafeRelease(m_pColorFrame);
			SafeRelease(m_pDepthFrame);
			SafeRelease(m_pColorFrameReference);
			SafeRelease(m_pDepthFrameReference);
			SafeRelease(m_pMultiFrame);
			mutex1.unlock();
		}
	});
#pragma endregion
	th1.join();
	cv::destroyAllWindows();
	delete[] depthData;
	delete[] bgraData;
	delete[] pixelFilterData;

	SafeRelease(m_pCoordinateMapper);
	m_pKinectSensor->Close();
	std::system("pause");
	return 0;
}
