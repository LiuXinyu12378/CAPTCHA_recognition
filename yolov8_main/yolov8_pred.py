from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train/weights/best.pt')

# Define path to directory containing images and videos for inference
source = 'data/bground'

# Run inference on the source
results = model(source, stream=True,conf=0.1)  # generator of Results objects

for result in results:
    print(result)
    print(result.boxes)

    # # 读入图片
    img = cv2.imread(result.path)

    for (x1,y1,x2,y2),cls in zip(result.boxes.xyxy,result.boxes.cls):

        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        # 在图片上绘制红色矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        # 在矩形框上加上文字
        cv2.putText(img, str(int(cls)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    # # 显示结果图片
    cv2.imshow('image', img)

    # # 等待按下任意按键退出程序
    cv2.waitKey(0)

