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
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.PreProcessor.Command;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessor;

public class Crop implements Command {

    public PreProcessor preprocessImage(PreProcessor preProcessor) {
        Mat img = preProcessor.getImages().get(0);
        List<Mat> processed = new ArrayList<Mat>();
        if (preProcessor.getFaces().length == 0){
            return null;
        } else {
            for (Rect rect : preProcessor.getFaces()){
                Mat subImg = img.submat(rect);
                processed.add(subImg);
            }
        }
        preProcessor.setImages(processed);
        return preProcessor;
    }

    public Mat preprocessImage(Mat img, Rect rect){
        return img.submat(rect);
    }
}
