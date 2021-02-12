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

package ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPostprocessing;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;
import ch.zhaw.facerecognitionlibrary.PreProcessor.Command;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessor;

public class Resize implements Command {

    public PreProcessor preprocessImage(PreProcessor preProcessor) {
        List<Mat> images = preProcessor.getImages();
        PreferencesHelper preferencesHelper = new PreferencesHelper(preProcessor.getContext());
        Size size = new Size(preferencesHelper.getN(), preferencesHelper.getN());
        preProcessor.setImages(preprocessImages(images, size));
        return preProcessor;
    }

    public static List<Mat> preprocessImage(List<Mat> images, int n){
        Size size = new Size(n, n);
        return preprocessImages(images,size);
    }

    private static List<Mat> preprocessImages(List<Mat> images, Size size){
        List<Mat> processed = new ArrayList<Mat>();
        for (Mat img : images){
            Imgproc.resize(img, img, size);
            processed.add(img);
        }
        return processed;
    }
}
