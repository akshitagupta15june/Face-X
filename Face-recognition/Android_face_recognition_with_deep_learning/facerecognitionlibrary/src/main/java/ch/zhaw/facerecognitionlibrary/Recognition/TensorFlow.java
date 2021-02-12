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
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

/***************************************************************************************
 *    Title: TensorFlowAndroidDemo
 *    Author: miyosuda
 *    Date: 23.04.2016
 *    Code version: -
 *    Availability: https://github.com
 *
 ***************************************************************************************/

public class TensorFlow implements Recognition {
    private String inputLayer;
    private String outputLayer;

    private int inputSize;
    private int channels;
    private int imageMean;
    private int imageStd;
    private int outputSize;

    private Recognition rec;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean logStats = false;

    public TensorFlow(Context context, int method) {
        String dataPath = FileHelper.TENSORFLOW_PATH;
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        inputSize = preferencesHelper.getTensorFlowInputSize();
        channels = preferencesHelper.getTensorFlowInputChannels();
        imageMean = preferencesHelper.getTensorFlowImageMean();
        imageStd = preferencesHelper.getTensorFlowImageStd();
        outputSize = preferencesHelper.getTensorFlowOutputSize();
        inputLayer = preferencesHelper.getTensorFlowInputLayer();
        outputLayer = preferencesHelper.getTensorFlowOutputLayer();

        String modelFile = preferencesHelper.getTensorFlowModelFile();
        Boolean classificationMethod = preferencesHelper.getClassificationMethodTFCaffe();

        // Use internal assets file as fallback, if no model file is provided
        File file = new File(dataPath + modelFile);
        if(file.exists()){
            inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), dataPath + modelFile);
        } else {
            inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), modelFile);
        }

        if(classificationMethod){
            rec = new SupportVectorMachine(context, method);
        }
        else {
            rec = new KNearestNeighbor(context, method);
        }
    }

    public TensorFlow(Context context, int inputSize, int outputSize, String inputLayer, String outputLayer, String modelFile){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;

        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), modelFile);
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
    public void loadFromFile() {

    }

    @Override
    public void saveTestData() {
        rec.saveTestData();
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
        Imgproc.resize(img, img, new Size(inputSize, inputSize));

        inferenceInterface.feed(inputLayer, getPixels(img), 1, inputSize, inputSize, channels);
        inferenceInterface.run(new String[]{outputLayer}, logStats);
        float[] outputs = new float[outputSize];
        inferenceInterface.fetch(outputLayer, outputs);

        List<Float> fVector = new ArrayList<>();
        for(float o : outputs){
            fVector.add(o);
        }

        return Converters.vector_float_to_Mat(fVector);
    }

    private float[] getPixels(Mat img){
        Bitmap bmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp);
        int[] intValues = new int[inputSize * inputSize];
        bmp.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        float[] floatValues = new float[inputSize * inputSize * channels];
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((float)((val >> 16) & 0xFF)) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((float)((val >> 8) & 0xFF)) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = (((float)(val & 0xFF)) - imageMean) / imageStd;
        }

        return floatValues;
    }
}
