import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import Data


test_data = ['test/1.jpg', 'test/2.jpg', 'test/3.jpg', 'test/4.jpg']

def test_model():
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    batch_size = 4

    test_datagen = Data(test_data, input_shape[:2], local_shape[:2])
    completion_model = load_model("model/completion.h5", compile=False)
    #completion_model.summary()

    cnt = 0
    for inputs, points, masks in test_datagen.flow(batch_size):
        completion_image = completion_model.predict([inputs, masks])
        num_img = min(5, batch_size)
        fig, axs = plt.subplots(num_img, 3)

        for i in range(num_img):
            axs[i, 0].imshow(inputs[i] * (1 - masks[i]))
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')
            axs[i, 1].imshow(completion_image[i])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Output')
            axs[i, 2].imshow(inputs[i])
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Ground Truth')
        fig.savefig(os.path.join("result", "result_test_%d.png" % cnt))
        plt.close()
        cnt += 1


def main():
    test_model()


if __name__ == "__main__":
    main()