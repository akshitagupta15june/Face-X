import requests





def get_dataset():
    pretrained_dataset_url= "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download="

    file_name = "face_detection_yunet_2023mar.onnx"


    response = requests.get(url=pretrained_dataset_url)

    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print("File downloaded successfully.")
        return 1
    else:
        print("Failed to download file. Status code:", response.status_code)
        return 0