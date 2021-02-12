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

package ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPreprocessing;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.Eyes;
import ch.zhaw.facerecognitionlibrary.PreProcessor.Command;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessor;

public class EyeAlignment implements Command {
    private static final double DESIRED_RIGHT_EYE_X = 0.24;
    private static final double DESIRED_RIGHT_EYE_Y = 0.30;
    private static final double DESIRED_LEFT_EYE_X = (1.0 - DESIRED_RIGHT_EYE_X);

    public PreProcessor preprocessImage(PreProcessor preProcessor) {
        List<Mat> images = preProcessor.getImages();
        List<Mat> processed = new ArrayList<Mat>();
        Eyes[] eyes = preProcessor.setEyes();
        if (eyes == null || eyes[0] == null){
            return null;
        }
        for (int i=0; i<images.size(); i++){
            Mat img = images.get(i);
            Eyes eye = eyes[i];
            double desiredLen = (DESIRED_LEFT_EYE_X - DESIRED_RIGHT_EYE_X) * img.cols();
            double scale = 0.9 * desiredLen / eye.getDist();
            MatOfFloat leftCenter = eye.getLeftCenter();
            MatOfFloat rightCenter = eye.getRightCenter();
            double centerX = ((leftCenter.get(0,0)[0] + rightCenter.get(0,0)[0]) / 2);
            double centerY = ((leftCenter.get(1,0)[0] + rightCenter.get(1,0)[0]) / 2);
            Mat rotMat = Imgproc.getRotationMatrix2D(new Point(centerX,centerY), eye.getAngle(), scale);
            rotMat.put(2, 0, img.cols() * 0.5 - centerX);
            rotMat.put(2, 1, img.rows() * DESIRED_RIGHT_EYE_Y - centerY);
            Imgproc.warpAffine(img, img, rotMat, new Size(img.cols(),img.rows()));
            processed.add(img);
        }
        preProcessor.setImages(processed);
        return preProcessor;
    }
}
