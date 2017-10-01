import cv2
import torch
from model import CNN, DoE
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def start_cam(cap):
    cnn = DoE()
    ckpt = torch.load('cnn.pkl')

    cnn.load_state_dict(ckpt['model'])
    classes = ["damage", "no line", "no damage"]

    cnn.cuda()
    cnn.eval()

    transform = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
    ])

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_gray = rgb2gray(frame).astype(np.uint8)
            frame_pil = Image.fromarray(frame_gray)
            img = transform(frame_pil)
            pred = cnn(Variable(img.unsqueeze(0)).cuda())
            cv2.imshow('frame', frame)

            pred_prob = F.softmax(pred).data.cpu().numpy()
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
