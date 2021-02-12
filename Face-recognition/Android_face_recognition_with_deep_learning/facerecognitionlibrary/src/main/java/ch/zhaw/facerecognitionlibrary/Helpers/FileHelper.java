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

package ch.zhaw.facerecognitionlibrary.Helpers;

import android.graphics.Bitmap;
import android.os.Environment;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessorFactory;

/**
 * This class is used to access the file system where the training and test data is stored
 */

public class FileHelper {
    public static String getFolderPath() {
        return FOLDER_PATH;
    }

    private static final String FOLDER_PATH = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/facerecognition";
    public static final String TRAINING_PATH = FOLDER_PATH + "/training/";
    public static final String TEST_PATH = FOLDER_PATH + "/test/";
    public static final String DETECTION_TEST_PATH = FOLDER_PATH + "/detection_test/";
    public static final String DATA_PATH = FOLDER_PATH + "/data/";
    public static final String RESULTS_PATH = FOLDER_PATH + "/results/";
    public static final String EIGENFACES_PATH = DATA_PATH + "Eigenfaces/";
    public static final String SVM_PATH = DATA_PATH + "SVM/";
    public static final String KNN_PATH = DATA_PATH + "KNN/";
    public static final String CAFFE_PATH = DATA_PATH + "Caffe/";
    public static final String TENSORFLOW_PATH = DATA_PATH + "TensorFlow/";
    private static final String SEPARATOR = ";";
    /**
     * Name of the person (subdirectory)
     */
    private String name = "";

    public FileHelper(String name) {
        this.name = name;
    }

    public FileHelper(){}

    public void createDataFolderIfNotExsiting(){
        File folder = new File(DATA_PATH);
        folder.mkdir();
    }

    private void createFolderIfNotExisting(String path){
        File folder = new File(path);
        folder.mkdir();
    }

    public static boolean isFileAnImage(File file){
        if (file.toString().endsWith(".jpg") || file.toString().endsWith(".jpeg") || file.toString().endsWith(".gif") || file.toString().endsWith(".png")){
            return true;
        } else {
            return false;
        }
    }

    /**
     * Returns an array of all files in the specified directory
     * @param path of the directory
     * @return
     */
    private File[] getListOfFiles(String path){
        File directory = new File(path + name);
        if(directory.exists()){
            return directory.listFiles();
        } else {
            return new File[]{};
        }
    }

    /**
     * Returns an array of all training files in the specified person directory
     * @return
     */
    public File[] getTrainingList(){
        return getListOfFiles(TRAINING_PATH);
    }

    /**
     * Returns an array of all test files in the specified person directory
     * @return
     */
    public File[] getTestList(){
        return getListOfFiles(TEST_PATH);
    }

    public File[] getDetectionTestList() {
        return getListOfFiles(DETECTION_TEST_PATH);
    }

    public void saveMatListToXml(List<MatName> matList, String path, String filename){
        createFolderIfNotExisting(path);
        MatXml matXml = new MatXml();
        matXml.create(path + filename);
        // Write Mats to file
        for(MatName mat : matList){
            matXml.writeMat(mat.getName(), mat.getMat());
        }
        matXml.release();
    }

    public List<MatName> getMatListFromXml(List<MatName> matList, String path, String filename){
        String filepath = path + filename;
        MatXml matXml = new MatXml();
        File file = new File(filepath);
        if (file.exists()){
            matXml.open(filepath);
            for (MatName mat : matList){
                mat.setMat(matXml.readMat(mat.getName()));
            }
        }
        return matList;
    }

    public String saveMatToImage(MatName m, String path){
        // Create folder if not already existing
        new File(path).mkdirs();
        String fullpath = path + m.getName() + ".png";
        Mat mat = m.getMat();
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        File file = new File(fullpath);
        try {
            FileOutputStream os = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
            os.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return fullpath;
    }

    public void saveBitmapToImage(Bitmap bmp){
        File file = new File(DATA_PATH + "bitmap.png");
        try {
            FileOutputStream os = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.PNG, 100, os);
            os.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public File createSvmTrainingFile(){
        createFolderIfNotExisting(SVM_PATH);
        String filepath = SVM_PATH + "svm_train";
        File trainingFile = new File(filepath);
        return trainingFile;
    }

    public File createSvmPredictionFile(){
        String filepath = SVM_PATH + "svm_predict";
        File predictionFile = new File(filepath);
        return predictionFile;
    }

    public File createSvmTestFile(){
        String filepath = SVM_PATH + "svm_test";
        File testFile = new File(filepath);
        return testFile;
    }

    public File createLabelFile(String path, String name){
        createFolderIfNotExisting(path);
        String filepath = path + "label_" + name;
        File trainingFile = new File(filepath);
        return trainingFile;
    }

    public void saveLabelMapToFile(String path, OneToOneMap<String, Integer> labelMap, String name){
        createFolderIfNotExisting(path);
        String filepath = path + "labelMap_" + name;
        try {
            FileWriter fw = new FileWriter(filepath);
            for (String s : (Set<String>)labelMap.getKeyToValMap().keySet()){
                fw.append(s + SEPARATOR + labelMap.getValue(s) + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveResultsToFile(Map<String, ?> map, double accuracy, double accuracy_reference, double accuracy_deviation, double robustness, int duration, List<String> results){
        String timestamp = new SimpleDateFormat("ddMMyyyyHHmm").format(new java.util.Date());
        createFolderIfNotExisting(RESULTS_PATH);
        String filepath = RESULTS_PATH + "Accuracy_" + String.format("%.2f", accuracy * 100) + "_" + timestamp + ".txt";
        try {
            FileWriter fw = new FileWriter(filepath);
            for (Map.Entry entry : map.entrySet()){
                fw.append(entry.getKey() + ": " + entry.getValue() + "\n");
            }
            fw.append("Accuracy: " + accuracy * 100 + "%\n");
            fw.append("Accuracy reference: " + accuracy_reference * 100 + "%\n");
            fw.append("Accuracy deviation: " + accuracy_deviation * 100 + "%\n");
            fw.append("Robustness: " + robustness * 100 + "%\n");
            fw.append("Duration per image: " + duration + "ms\n");
            for (String result : results){
                fw.append(result + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveResultsToFile(Map<String, ?> map, double accuracy, int duration, List<String> results){
        String timestamp = new SimpleDateFormat("ddMMyyyyHHmm").format(new java.util.Date());
        createFolderIfNotExisting(RESULTS_PATH);
        String filepath = RESULTS_PATH + "Accuracy_" + String.format("%.2f", accuracy * 100) + "_" + timestamp + ".txt";
        try {
            FileWriter fw = new FileWriter(filepath);
            for (Map.Entry entry : map.entrySet()){
                fw.append(entry.getKey() + ": " + entry.getValue() + "\n");
            }
            fw.append("Accuracy: " + accuracy * 100 + "%\n");
            fw.append("Duration per image: " + duration + "ms\n");
            for (String result : results){
                fw.append(result + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public OneToOneMap<String, Integer> getLabelMapFromFile(String path){
        String filepath = path + "labelMap_train";
        OneToOneMap<String, Integer> labelMap = new OneToOneMap<>();
        try {
            BufferedReader buf = new BufferedReader(new FileReader(filepath));
            String line = buf.readLine();
            while (line != null){
                String[] split = line.split(SEPARATOR);
                labelMap.put(split[0], Integer.valueOf(split[1]));
                line = buf.readLine();
            }
            buf.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return labelMap;
    }

    public void saveStringList(List<String> list, File file){
        try {
            FileWriter fw = new FileWriter(file, false);
            for (String line : list){
                fw.append(line + "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void saveIntegerList(List<Integer> list, File file){
        try {
            FileWriter fw = new FileWriter(file, false);
            for (int line : list){
                fw.append(Integer.toString(line)+ "\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public List<String> loadStringList(File file){
        List<String> list = new ArrayList<>();
        try {
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            String line;
            while ((line = br.readLine()) != null){
                list.add(line);
            }
            br.close();
            fr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }

    public List<Integer> loadIntegerList(File file){
        List<Integer> list = new ArrayList<>();
        try {
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            Integer line = 0;
            String sLine;
            while ((sLine = br.readLine()) != null){
                line = Integer.parseInt(sLine);
                list.add(line);
            }
            br.close();
            fr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }
}
