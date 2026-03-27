from ultralytics import YOLO

model = YOLO(r'C:/Users/sarun/OneDrive/Desktop/flask2-main/flask2-main/train9/weights/best.pt')
res = model(r'C:/Users/sarun/OneDrive/Desktop/flask2-main/flask2-main/test_upload.jpg')
print('len', len(res))
if res:
    boxes = res[0].boxes
    print('boxes', len(boxes), type(boxes))
    for i, box in enumerate(boxes):
        print('box', i)
        print('  xyxy raw:', box.xyxy)
        print('  xyxy tolist:', box.xyxy.tolist())
        print('  conf raw:', box.conf)
        print('  conf tolist:', box.conf.tolist())
        print('  cls raw:', box.cls)
        print('  cls tolist:', box.cls.tolist())
