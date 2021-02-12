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

package ch.zhaw.facerecognitionlibrary.PreProcessor.Contours;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.PreProcessor.Command;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessor;

public class Masking implements Command {

    public PreProcessor preprocessImage(PreProcessor preProcessor) {
        List<Mat> images = preProcessor.getImages();
        List<Mat> processed = new ArrayList<Mat>();
        for (Mat img : images){
            preProcessor.normalize0255(img);

            /***************************************************************************************
             *    Title: Automatic calculation of low and high thresholds for the Canny operation in opencv
             *    Author: VP
             *    Date: 16.04.2013
             *    Code version: -
             *    Availability: http://stackoverflow.com
             *
             ***************************************************************************************/

            double otsu_thresh_val = Imgproc.threshold(img, img, 0, 255, Imgproc.THRESH_OTSU);
            Imgproc.Canny(img, img, otsu_thresh_val * 0.5, otsu_thresh_val);
            processed.add(img);
        }
        preProcessor.setImages(processed);
        return preProcessor;
    }
}
