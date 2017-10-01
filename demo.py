import cv2
import torch
from model import CNN
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def start_cam(cap):
    cnn = CNN()
    ckpt = torch.load('cnn.pkl')

    cnn.load_state_dict(ckpt['model'])
    classes = ckpt['classes']

    cnn.cuda()
    cnn.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
    ])

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = transform(frame)
            pred = cnn(Variable(img.unsqueeze(0)).cuda())
            cv2.imshow('frame', frame)

            pred_prob = F.softmax(pred).data.cpu().numpy()[0]
            pred = np.argmax(pred_prob)
            print("{}: {}".format(classes[pred], pred_prob[pred]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not cap.isOpened():
        print("camera is not opened")

def main():
    cap = cv2.VideoCapture(0)

    try:
        start_cam(cap)
    except KeyboardInterrupt:
        print("quitting")

    print("releasing camera")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()