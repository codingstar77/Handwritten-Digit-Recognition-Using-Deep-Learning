import numpy as np,cv2,imutils
from sklearn.externals import joblib


img = cv2.imread('sample_image.jpg')

img = imutils.resize(img,width=300)
cv2.imshow("Original",img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",gray)


kernel = np.ones((20,20),np.uint8)


blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)


ret,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

ret,cnts,hie = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


model = joblib.load('model.pkl')
for c in cnts:
    try:
        mask = np.zeros(gray.shape,dtype="uint8")   

        (x,y,w,h) = cv2.boundingRect(c)
        
        hull = cv2.convexHull(c)
        cv2.drawContours(mask,[hull],-1,255,-1)    
        mask = cv2.bitwise_and(thresh,thresh,mask=mask) 

        roi = mask[y-9:y+h+9,x-9:x+w+9]       
        roi = cv2.resize(roi,(28,28))
        roi = np.array(roi)

        roi = roi.reshape(1,784)
        prediction = model.predict(roi)
        probability = model.predict_proba(roi)
        print(probability)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.putText(img,str(int(prediction)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1)
        cv2.putText(img,str((max(probability[0])*100))+"%",(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    except Exception as e:
        print(e)
        pass
cv2.imshow('Detection',img)
               
