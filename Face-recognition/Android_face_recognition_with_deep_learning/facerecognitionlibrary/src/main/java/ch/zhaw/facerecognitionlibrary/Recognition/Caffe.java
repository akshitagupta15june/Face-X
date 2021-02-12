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

package ch.zhaw.facerecognitionlibrary.Recognition;


import android.content.Context;
import android.os.ParcelFormatException;

import ch.zhaw.facerecognitionlibrary.Helpers.CaffeMobile;

import org.opencv.core.Mat;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.MatName;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;


/***************************************************************************************
 *    Title: caffe-android-demo
 *    Author: sh1r0
 *    Date: 21.06.2016
 *    Code version: -
 *    Availability: https://github.com
 *
 ***************************************************************************************/

public class Caffe implements Recognition {
    private CaffeMobile caffe;
    private Recognition rec;
    private FileHelper fh;
    String layer;

    static {
        System.loadLibrary("caffe");
        System.loadLibrary("caffe_jni");
    }

    public Caffe(Context context, int method) {
        fh = new FileHelper();
        String dataPath = fh.CAFFE_PATH;
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        String modelFile = preferencesHelper.getCaffeModelFile();
        String weightsFile = preferencesHelper.getCaffeWeightsFile();
        layer = preferencesHelper.getCaffeOutputLayer();
        float[] meanValues = preferencesHelper.getCaffeMeanValues();

        Boolean classificationMethod = preferencesHelper.getClassificationMethodTFCaffe();

        caffe = new CaffeMobile();
        caffe.setNumThreads(4);
        caffe.loadModel(dataPath + modelFile, dataPath + weightsFile);
        caffe.setMean(meanValues);
        if(classificationMethod){
            rec = new SupportVectorMachine(context, method);
        } else {
            rec = new KNearestNeighbor(context, method);
        }

    }

    @Override
    public boolean train() {
        return rec.train();
    }

    @Override
    public String recognize(Mat img, String expectedLabel) {
        return rec.recognize(getFeatureVector(img), expectedLabel);
    }

    @Override
    public void saveToFile() {

    }

    @Override
    public void saveTestData() {
        rec.saveTestData();
    }

    @Override
    public void loadFromFile() {

    }

    @Override
    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        if (featuresAlreadyExtracted){
            rec.addImage(img, label, true);
        } else {
            rec.addImage(getFeatureVector(img), label, true);
        }
    }

    public Mat getFeatureVector(Mat img){
        float[][] vector = caffe.getRepresentationLayer(saveMatToImage(img), layer);

        List<Float> fVector = new ArrayList<>();
        for(float f : vector[0]){
            fVector.add(f);
        }

        return Converters.vector_float_to_Mat(fVector);
    }

    private String saveMatToImage(Mat img){
        MatName m = new MatName("caffe_vector", img);
        return fh.saveMatToImage(m, fh.CAFFE_PATH);
    }
}
