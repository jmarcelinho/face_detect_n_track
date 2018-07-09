#include "VideoFaceDetector.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const double VideoFaceDetector::TICK_FREQUENCY = getTickFrequency();

VideoFaceDetector::VideoFaceDetector(const string cascadeFilePath, VideoCapture &videoCapture)
{
    setFaceCascade(cascadeFilePath);
    setVideoCapture(videoCapture);
}

void VideoFaceDetector::setVideoCapture(VideoCapture &videoCapture)
{
    m_videoCapture = &videoCapture;
}

VideoCapture *VideoFaceDetector::videoCapture() const
{
    return m_videoCapture;
}

void VideoFaceDetector::setFaceCascade(const string cascadeFilePath)
{
    if (m_faceCascade == NULL)
    {
        m_faceCascade = new CascadeClassifier(cascadeFilePath);
    }
    else
    {
        m_faceCascade->load(cascadeFilePath);
    }

    if (m_faceCascade->empty())
    {
        cerr << "Error creating cascade classifier. Make sure the file" << endl
             << cascadeFilePath << " exists." << endl;
    }
}

CascadeClassifier *VideoFaceDetector::faceCascade() const
{
    return m_faceCascade;
}

void VideoFaceDetector::setResizedWidth(const int width)
{
    m_resizedWidth = max(width, 1);
}

int VideoFaceDetector::resizedWidth() const
{
    return m_resizedWidth;
}

bool VideoFaceDetector::isFaceFound() const
{
    return m_foundFace;
}

Rect VideoFaceDetector::face() const
{
    Rect faceRect = m_trackedFace;
    faceRect.x = (int)(faceRect.x / m_scale);
    faceRect.y = (int)(faceRect.y / m_scale);
    faceRect.width = (int)(faceRect.width / m_scale);
    faceRect.height = (int)(faceRect.height / m_scale);
    return faceRect;
}

Point VideoFaceDetector::facePosition() const
{
    Point facePos;
    facePos.x = (int)(m_facePosition.x / m_scale);
    facePos.y = (int)(m_facePosition.y / m_scale);
    return facePos;
}

void VideoFaceDetector::setTemplateMatchingMaxDuration(const double s)
{
    m_templateMatchingMaxDuration = s;
}

double VideoFaceDetector::templateMatchingMaxDuration() const
{
    return m_templateMatchingMaxDuration;
}

VideoFaceDetector::~VideoFaceDetector()
{
    if (m_faceCascade != NULL)
    {
        delete m_faceCascade;
    }
}

Rect VideoFaceDetector::doubleRectSize(const Rect &inputRect, const Rect &frameSize) const
{
    Rect outputRect;
    // Double rect size
    outputRect.width = inputRect.width * 2;
    outputRect.height = inputRect.height * 2;

    // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2;
    outputRect.y = inputRect.y - inputRect.height / 2;

    // Handle edge cases
    if (outputRect.x < frameSize.x)
    {
        outputRect.width += outputRect.x;
        outputRect.x = frameSize.x;
    }
    if (outputRect.y < frameSize.y)
    {
        outputRect.height += outputRect.y;
        outputRect.y = frameSize.y;
    }

    if (outputRect.x + outputRect.width > frameSize.width)
    {
        outputRect.width = frameSize.width - outputRect.x;
    }
    if (outputRect.y + outputRect.height > frameSize.height)
    {
        outputRect.height = frameSize.height - outputRect.y;
    }

    return outputRect;
}

Point VideoFaceDetector::centerOfRect(const Rect &rect) const
{
    return Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

Rect VideoFaceDetector::biggestFace(vector<Rect> &faces) const
{
    assert(!faces.empty());

    Rect *biggest = &faces[0];
    for (auto &face : faces)
    {
        if (face.area() < biggest->area())
            biggest = &face;
    }
    return *biggest;
}

/*
* Face template is small patch in the middle of detected face.
*/
Mat VideoFaceDetector::getFaceTemplate(const Mat &frame, Rect face)
{
    face.x += face.width / 4;
    face.y += face.height / 4;
    face.width /= 2;
    face.height /= 2;

    Mat faceTemplate = frame(face).clone();
    return faceTemplate;
}

void VideoFaceDetector::detectFaceAllSizes(const Mat &frame)
{
    // Minimum face size is 1/5th of screen height
    // Maximum face size is 2/3rds of screen height
    m_faceCascade->detectMultiScale(frame, m_allFaces, 1.1, 3, 0,
                                    Size(frame.rows / 5, frame.rows / 5),
                                    Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

    if (m_allFaces.empty())
        return;

    m_foundFace = true;

    // Locate biggest face
    m_trackedFace = biggestFace(m_allFaces);

    // Copy face template
    m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

    // Calculate roi
    m_faceRoi = doubleRectSize(m_trackedFace, Rect(0, 0, frame.cols, frame.rows));

    // Update face position
    m_facePosition = centerOfRect(m_trackedFace);
}

void VideoFaceDetector::detectFaceAroundRoi(const Mat &frame)
{
    // Detect faces sized +/-20% off biggest face in previous search
    m_faceCascade->detectMultiScale(frame(m_faceRoi), m_allFaces, 1.1, 3, 0,
                                    Size(m_trackedFace.width * 8 / 10, m_trackedFace.height * 8 / 10),
                                    Size(m_trackedFace.width * 12 / 10, m_trackedFace.width * 12 / 10));

    if (m_allFaces.empty())
    {
        // Activate template matching if not already started and start timer
        m_templateMatchingRunning = true;
        if (m_templateMatchingStartTime == 0)
            m_templateMatchingStartTime = getTickCount();
        return;
    }

    // Turn off template matching if running and reset timer
    m_templateMatchingRunning = false;
    m_templateMatchingCurrentTime = m_templateMatchingStartTime = 0;

    // Get detected face
    m_trackedFace = biggestFace(m_allFaces);

    // Add roi offset to face
    m_trackedFace.x += m_faceRoi.x;
    m_trackedFace.y += m_faceRoi.y;

    // Get face template
    m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

    // Calculate roi
    m_faceRoi = doubleRectSize(m_trackedFace, Rect(0, 0, frame.cols, frame.rows));

    // Update face position
    m_facePosition = centerOfRect(m_trackedFace);
}

void VideoFaceDetector::detectFacesTemplateMatching(const Mat &frame)
{
    // Calculate duration of template matching
    m_templateMatchingCurrentTime = getTickCount();
    double duration = (double)(m_templateMatchingCurrentTime - m_templateMatchingStartTime) / TICK_FREQUENCY;

    // If template matching lasts for more than 2 seconds face is possibly lost
    // so disable it and redetect using cascades
    if (duration > m_templateMatchingMaxDuration)
    {
        m_foundFace = false;
        m_templateMatchingRunning = false;
        m_templateMatchingStartTime = m_templateMatchingCurrentTime = 0;
        m_facePosition.x = m_facePosition.y = 0;
        m_trackedFace.x = m_trackedFace.y = m_trackedFace.width = m_trackedFace.height = 0;
        return;
    }

    // Edge case when face exits frame while
    if (m_faceTemplate.rows * m_faceTemplate.cols == 0 || m_faceTemplate.rows <= 1 || m_faceTemplate.cols <= 1)
    {
        m_foundFace = false;
        m_templateMatchingRunning = false;
        m_templateMatchingStartTime = m_templateMatchingCurrentTime = 0;
        m_facePosition.x = m_facePosition.y = 0;
        m_trackedFace.x = m_trackedFace.y = m_trackedFace.width = m_trackedFace.height = 0;
        return;
    }

    // Template matching with last known face
    //matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_CCOEFF);
    matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_SQDIFF_NORMED);
    normalize(m_matchingResult, m_matchingResult, 0, 1, NORM_MINMAX, -1, Mat());
    double min, max;
    Point minLoc, maxLoc;
    minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

    // Add roi offset to face position
    minLoc.x += m_faceRoi.x;
    minLoc.y += m_faceRoi.y;

    // Get detected face
    //m_trackedFace = Rect(maxLoc.x, maxLoc.y, m_trackedFace.width, m_trackedFace.height);
    m_trackedFace = Rect(minLoc.x, minLoc.y, m_faceTemplate.cols, m_faceTemplate.rows);
    m_trackedFace = doubleRectSize(m_trackedFace, Rect(0, 0, frame.cols, frame.rows));

    // Get new face template
    m_faceTemplate = getFaceTemplate(frame, m_trackedFace);

    // Calculate face roi
    m_faceRoi = doubleRectSize(m_trackedFace, Rect(0, 0, frame.cols, frame.rows));

    // Update face position
    m_facePosition = centerOfRect(m_trackedFace);
}

Point VideoFaceDetector::getFrameAndDetect(Mat &frame)
{
    *m_videoCapture >> frame;

    // Downscale frame to m_resizedWidth width - keep aspect ratio
    m_scale = (double)min(m_resizedWidth, frame.cols) / frame.cols;
    Size resizedFrameSize = Size((int)(m_scale * frame.cols), (int)(m_scale * frame.rows));

    Mat resizedFrame;
    resize(frame, resizedFrame, resizedFrameSize);

    if (!m_foundFace)
        detectFaceAllSizes(resizedFrame); // Detect using cascades over whole image
    else
    {
        detectFaceAroundRoi(resizedFrame); // Detect using cascades only in ROI
        if (m_templateMatchingRunning)
        {
            detectFacesTemplateMatching(resizedFrame); // Detect using template matching
        }
    }

    return m_facePosition;
}

Point VideoFaceDetector::operator>>(Mat &frame)
{
    return this->getFrameAndDetect(frame);
}