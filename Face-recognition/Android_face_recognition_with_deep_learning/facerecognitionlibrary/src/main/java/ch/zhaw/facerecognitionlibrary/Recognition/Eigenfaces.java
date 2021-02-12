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

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.MatName;
import ch.zhaw.facerecognitionlibrary.Helpers.OneToOneMap;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

public class Eigenfaces implements Recognition {
    private Context context;
    private Mat Gamma = new Mat();
    private Mat Psi = new Mat();
    private Mat Phi = new Mat();
    private Mat eigVectors = new Mat();
    private Mat Omega = new Mat();
    private Mat testList = new Mat();
    private List<Integer> labelList;
    private List<Integer> labelListTest;
    private OneToOneMap<String,Integer> labelMap;
    private OneToOneMap<String,Integer> labelMapTest;
    private String filename = "eigenfaces.xml";
    private int method;

    public Eigenfaces(Context context, int method) {
        this.context = context;
        this.labelList = new ArrayList<>();
        this.labelListTest = new ArrayList<>();
        this.labelMap = new OneToOneMap<String, Integer>();
        this.labelMapTest = new OneToOneMap<String, Integer>();
        this.method = method;
        if(method == RECOGNITION){
            loadFromFile();
        }
    }

    public boolean train(){
        // Return if no images
        if (Gamma.empty()){
            return false;
        }
        computePsi();
        computePhi();
        computeEigVectors();
        Omega = getFeatureVector(Phi);
        saveToFile();
        return true;
    }

    public String recognize(Mat img, String expectedLabel){
        // Ignore
        img = img.reshape(1,1);
        // Subtract mean
        img.convertTo(img, CvType.CV_32F);
        Core.subtract(img, Psi, img);
        // Project to subspace
        Mat projected = getFeatureVector(img);
        // Save all points of image for tSNE
        img.convertTo(img, CvType.CV_8U);
        addImage(projected, expectedLabel, true);
        //addImage(projected, expectedLabel);
        Mat distance = new Mat(Omega.rows(), 1, CvType.CV_64FC1);
        for (int i=0; i<Omega.rows(); i++){
            double dist = Core.norm(projected.row(0), Omega.row(i), Core.NORM_L2);
            distance.put(i, 0, dist);
        }
        Mat sortedDist = new Mat(Omega.rows(), 1, CvType.CV_8UC1);
        Core.sortIdx(distance, sortedDist, Core.SORT_EVERY_COLUMN + Core.SORT_ASCENDING);
        // Give back the name of the found person
        int index = (int)(sortedDist.get(0,0)[0]);
        return labelMap.getKey(labelList.get(index));
    }

    private void computePsi(){
        Core.reduce(Gamma, Psi, 0, Core.REDUCE_AVG);
    }

    private void computePhi(){
        Mat Psi_repeated = new Mat();
        Core.repeat(Psi, Gamma.rows(), 1, Psi_repeated);
        Core.subtract(Gamma, Psi_repeated, Phi);
    }

    private void computeEigVectors(){
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        float pca_threshold = preferencesHelper.getPCAThreshold();
        Core.PCACompute(Phi, Psi, eigVectors, pca_threshold);
    }

    public Mat getFeatureVector(Mat original){
        Mat projected = new Mat();
        Core.PCAProject(original, Psi, eigVectors, projected);
        return projected;
    }

    public void saveToFile(){
        FileHelper fh = new FileHelper();
        fh.saveIntegerList(labelList, fh.createLabelFile(fh.EIGENFACES_PATH, "train"));
        fh.saveLabelMapToFile(fh.EIGENFACES_PATH, labelMap, "train");
        MatName mOmega = new MatName("Omega", Omega);
        MatName mPsi = new MatName("Psi", Psi);
        MatName mEigVectors = new MatName("eigVectors", eigVectors);
        // Save Phi for tSNE
        MatName mPhi = new MatName("Phi", Phi);
        List<MatName> listMat = new ArrayList<MatName>();
        listMat.add(mOmega);
        listMat.add(mPsi);
        listMat.add(mEigVectors);
        listMat.add(mPhi);
        fh.saveMatListToXml(listMat, fh.EIGENFACES_PATH, filename);
    }

    @Override
    public void saveTestData() {
        FileHelper fh = new FileHelper();
        fh.saveIntegerList(labelListTest, fh.createLabelFile(fh.EIGENFACES_PATH, "test"));
        fh.saveLabelMapToFile(fh.EIGENFACES_PATH, labelMapTest, "test");
        MatName mTestList = new MatName("TestList", testList);
        List<MatName> listMat = new ArrayList<>();
        listMat.add(mTestList);
        fh.saveMatListToXml(listMat, fh.EIGENFACES_PATH, "testlist.xml");
    }

    public void loadFromFile(){
        FileHelper fh = new FileHelper();
        MatName mOmega = new MatName("Omega", Omega);
        MatName mPsi = new MatName("Psi", Psi);
        MatName mEigVectors = new MatName("eigVectors", eigVectors);
        List<MatName> listMat = new ArrayList<MatName>();
        listMat.add(mOmega);
        listMat.add(mPsi);
        listMat.add(mEigVectors);
        listMat = fh.getMatListFromXml(listMat, fh.EIGENFACES_PATH, filename);
        for (MatName mat : listMat){
            switch (mat.getName()){
                case "Omega":
                    Omega = mat.getMat();
                    break;
                case "Psi":
                    Psi = mat.getMat();
                    break;
                case "eigVectors":
                    eigVectors = mat.getMat();
                    break;
            }
        }
        labelList = fh.loadIntegerList(fh.createLabelFile(fh.EIGENFACES_PATH, "train"));
        labelMap = fh.getLabelMapFromFile(fh.EIGENFACES_PATH);
    }

    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        // Ignore featuresAlreadyExtracted because with Eigenfaces all the original images are needed to extract the eigenfaces (feature vector)
        int iLabel = 0;
        if(method == TRAINING){
            // Reshape image to have only 1 row, then add it to GammaList
            Gamma.push_back(img.reshape(1,1));
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
}
