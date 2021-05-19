import cv2
import os
# Opens the Video file

cap = cv2.VideoCapture('C:/Projects/redesNeuronales/DectectarMultiplesRostros/video.avi')
nombre = input("Enter a name:")
personPath = 'C:/Projects/redesNeuronales/DectectarMultiplesRostros/Data/' + nombre

i=0

if not os.path.exists(personPath):
    print("Carpeta creada:", personPath)
    os.makedirs(personPath)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(personPath+'/martin'+str(i)+'.jpg',frame)
    i+=1
cap.release()
cv2.destroyAllWindows()
