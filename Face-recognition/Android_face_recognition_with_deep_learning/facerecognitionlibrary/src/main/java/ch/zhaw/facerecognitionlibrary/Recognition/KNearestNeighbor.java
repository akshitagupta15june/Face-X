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

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.ml.KNearest;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.MatName;
import ch.zhaw.facerecognitionlibrary.Helpers.OneToOneMap;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

public class KNearestNeighbor implements Recognition{
    private Context context;
    private FileHelper fh;
    private Mat trainingList;
    private Mat testList;
    private Mat labels;
    private int k;
    private KNearest knn;
    private List<Integer> labelList;
    private List<Integer> labelListTest;
    private OneToOneMap<String,Integer> labelMap;
    private OneToOneMap<String,Integer> labelMapTest;
    private String trainingFile;
    private String testFile;
    private int method;


   public KNearestNeighbor(Context context, int method)  {
       this.context = context;
       fh = new FileHelper();
       k = 20;
       trainingList = new Mat();
       testList = new Mat();
       this.labelList = new ArrayList<>();
       this.labelListTest = new ArrayList<>();
       this.labelMap = new OneToOneMap<String, Integer>();
       this.labelMapTest = new OneToOneMap<String, Integer>();
       trainingFile = "knn_traininglist.xml";
       testFile = "knn_testlist.xml";
       this.method = method;
       if(method == RECOGNITION){
           loadFromFile();
       }

   }

    @Override
    public boolean train() {
        // Return if no images
        if (trainingList.empty()){
            return false;
        }
        saveToFile();
        return true;
    }

    @Override
    public String recognize(Mat img, String expectedLabel) {
        Mat result = new Mat();
        float nearest;

        img = getFeatureVector(img);
        addImage(img, expectedLabel, true);
        nearest = knn.findNearest(img,k,result);

        return labelMap.getKey((int) nearest);
    }

    @Override
    public void saveToFile() {

        fh.saveIntegerList(labelList, fh.createLabelFile(fh.KNN_PATH, "train"));
        fh.saveLabelMapToFile(fh.KNN_PATH, labelMap, "train");

        MatName mtrainingList = new MatName("TrainingList", trainingList);

        List<MatName> listMat = new ArrayList<MatName>();
        listMat.add(mtrainingList);

        fh.saveMatListToXml(listMat, fh.KNN_PATH, trainingFile);
    }

    @Override
    public void saveTestData() {
        MatName mTestList = new MatName("TestList", testList);
        List<MatName> listMat = new ArrayList<>();
        listMat.add(mTestList);
        fh.saveMatListToXml(listMat, fh.KNN_PATH, testFile);
        fh.saveIntegerList(labelListTest, fh.createLabelFile(fh.KNN_PATH, "test"));
        fh.saveLabelMapToFile(fh.KNN_PATH, labelMapTest, "test");
    }

    @Override
    public void loadFromFile() {
        MatName mtrainingList = new MatName("TrainingList", trainingList);

        List<MatName> listMat = new ArrayList<MatName>();
        listMat.add(mtrainingList);

        labelList = fh.loadIntegerList(fh.createLabelFile(fh.KNN_PATH, "train"));
        labelMap = fh.getLabelMapFromFile(fh.KNN_PATH);
        trainingList = fh.getMatListFromXml(listMat, fh.KNN_PATH, trainingFile).get(0).getMat();

        labels = new Mat(labelList.size(), 1, CvType.CV_8UC1);
        for (int i=0; i<labelList.size(); i++) {
            Integer label = labelList.get(i);
            // Fill shorter labels with 0s
            labels.put(i, 0, label);
        }

        labels.convertTo(labels, CvType.CV_32F);
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        k = preferencesHelper.getK();

        knn = KNearest.create();
        knn.setIsClassifier(true);
        knn.train(trainingList, 0,labels);

    }

    @Override
    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        // Ignore featuresAlreadyExtracted because either KNN get the features from TensorFlow or Caffe
        int iLabel = 0;
        if(method == TRAINING){
            // Reshape image to have only 1 row, then add it to GammaList
            trainingList.push_back(img.reshape(1,1));
            if (labelMap.containsKey(label)) {
                iLabel = labelMap.getValue(label);
            } else {
                iLabel = labelMap.size() + 1;
                labelMap.put(label, iLabel);
            }
            labelList.add(iLabel);
        } else {
            testList.push_back(img);
            if (labelMapTest.containsKey(label)){
                iLabel = labelMapTest.getValue(label);
            } else {
                iLabel = labelMapTest.size() + 1;
                labelMapTest.put(label, iLabel);
            }
            labelListTest.add(iLabel);
        }
    }


    @Override
    public Mat getFeatureVector(Mat img) {
        return img.reshape(1,1);
    }

    private byte[] stringToByteArray(String s){
        return s.getBytes();
    }
}
