/* Copyright 2016 Michael Sladoje and Mike Schälchli. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package ch.zhaw.facerecognitionlibrary.PreProcessor;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.media.FaceDetector;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.Eyes;
import ch.zhaw.facerecognitionlibrary.Helpers.FaceDetection;
import ch.zhaw.facerecognitionlibrary.Helpers.MatOperation;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

public class PreProcessor {
    private Context context;
    private int angle;
    private Mat img;
    private List<Mat> images;
    private Rect[] faces;
    private Eyes[] eyes;
    private FaceDetection faceDetection;

    public Context getContext(){
        return context;
    }

    public PreProcessor(FaceDetection faceDetection, List<Mat> images, Context context){
        this.faceDetection = faceDetection;
        this.images = images;
        this.context = context;
    }

    public void setFaces(PreProcessorFactory.PreprocessingMode preprocessingMode) {
        List<Mat> images = getImages();

        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        if (preferencesHelper.getDetectionMethod()){
            faces = faceDetection.getFaces(images.get(0));
            angle = faceDetection.getAngle();
        } else {
            Mat img = images.get(0);
            FaceDetector faceDetector = new FaceDetector(img.cols(), img.rows(), 1);
            Bitmap bmp = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(img, bmp);
            FaceDetector.Face[] facesAndroid = new FaceDetector.Face[1];
            if (faceDetector.findFaces(bmp, facesAndroid) > 0){
                faces = new Rect[facesAndroid.length];
                for (int i=0; i<facesAndroid.length; i++){
                    PointF pointF = new PointF();
                    facesAndroid[i].getMidPoint(pointF);
                    int xWidth = (int) (1.34 * facesAndroid[i].eyesDistance());
                    int yWidth = (int) (1.12 * facesAndroid[i].eyesDistance());
                    int dist = (int) (2.77 * facesAndroid[i].eyesDistance());
                    Rect face = new Rect((int) pointF.x - xWidth, (int) pointF.y - yWidth, dist, dist);
                    faces[i] = face;
                }
            }
        }

        if (preprocessingMode == PreProcessorFactory.PreprocessingMode.RECOGNITION && preferencesHelper.getDetectionMethod()){
            // Change the image rotation to the angle where the face was detected
            images.remove(0);
            images.add(faceDetection.getImg());
            setImages(images);
        }
    }

    public void setFaces(Rect[] faces){
        this.faces = faces;
    }

    public Eyes[] setEyes() {
        List<Mat> images = getImages();
        eyes = new Eyes[images.size()];
        for (int i=0; i<images.size(); i++){
            Mat img = images.get(i);
            normalize0255(img);
            eyes[i] = faceDetection.getEyes(img);
        }
        return eyes;
    }

    public Eyes[] getEyes() {
        return eyes;
    }

    public Rect[] getFaces() {
        return faces;
    }

    public int getAngle() { return angle; }

    public void setAngle(int angle) {
        this.angle = angle;
        for (Mat img : images){
            MatOperation.rotate_90n(img, angle);
        }
    }

    public Mat getImg() {
        return img;
    }

    public void setImages(List<Mat> images) {
        this.images = images;
    }

    public List<Mat> getImages() {
        return images;
    }

    public void setImg(Mat img) {
        this.img = img;
    }

    public void normalize0255(Mat norm){
        Core.normalize(norm, norm, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
    }
}
