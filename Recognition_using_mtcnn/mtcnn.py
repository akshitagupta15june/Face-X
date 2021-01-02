import mtcnn
import matplotlib.pyplot as plt
filename = "1.jpg"
pixels = plt.imread(filename)
# print("Shape of image/array:",pixels.shape)
imgplot = plt.imshow(pixels)
# plt.show()

detector = mtcnn.MTCNN()
faces = detector.detect_faces(pixels)
# for face in faces:
    # print(face)
def draw_facebox(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='orange')
        ax.add_patch(rect)
    
    plt.show()
 
faces = detector.detect_faces(pixels)
draw_facebox(filename, faces)
def draw_facedot(filename, result_list):
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()
    
    for result in result_list:
        x, y, width, height = result['box']
        rect = plt.Rectangle((x, y), width, height,fill=False, color='orange')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = plt.Circle(value, radius=10, color='red')
            ax.add_patch(dot)
        plt.show()
 
faces = detector.detect_faces(pixels)
draw_facedot(filename, faces)