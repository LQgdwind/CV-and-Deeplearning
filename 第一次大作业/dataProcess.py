import os
import random
import cv2
import dataEnhance as de
import dataLoading as dl

def display_test_pic(img):
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def data_json_haddling():
    if not os.path.exists(".//train"):
        dl.data_to_train()
        # 根据json文件生成训练集并且改名
    if not os.path.exists(".//valid"):
        dl.data_to_valid()
        # 根据json文件生成验证集并且改名


def augment_image_generate_and_saving(type = 'light'):
    for k in range(10):
        for i in range(16):
            # cv2.imwrite方法不会创建新的目录!!!
            if not os.path.exists(".//tiny_data_after_light_augment//train" + str(k) + "//" + str(i)):
                os.makedirs(".//data_after_light_augment//train" + str(k) + "//" + str(i))
            path = ".//train//" + str(i)
            num_image = len(os.listdir(path))
            # 查看每个类别初始有多少图片
            j = 0
            while True:
                j = j + 1
                if j == 65:
                    break
                if j % 16 == 0:
                    print("第" + str(k) + "轮,第" + str(i) + "类,第" + str(j) + "张")
                target_path = ".//data_after_light_augment//train" + str(k) + "//" + str(i) + "//" + str(i) + "_" + str(j) + ".jpg"
                # print(target_path)
                id = str(random.randint(1, num_image))
                selected_picture = cv2.imread(path + "//" + str(i) + "_" + id + ".jpg", cv2.IMREAD_UNCHANGED)
                if type == 'light':
                    selected_picture = de.image_light_augment(selected_picture, size=224)
                    cv2.imwrite(target_path, selected_picture)
                # display_test_pic(selected_picture)
                else :
                    selected_picture = de.image_augment(selected_picture, size=224)
                    cv2.imwrite(target_path, selected_picture*255)
                    # 这里要做反归一化!


# augment_image_generate_and_saving()
