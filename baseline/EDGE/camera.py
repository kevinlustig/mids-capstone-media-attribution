from datetime import datetime
import numpy as np
import cv2
import paho.mqtt.client as mqtt

###Will stay on screen for 5 seconds
ttl = 5
####
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
text_ori = "Press Space to Predict!"
global text
text = text_ori

# LOCAL_MQTT_HOST = "mosquitto-service.default.svc.cluster.local"
LOCAL_MQTT_HOST ="10.43.152.40"
# LOCAL_MQTT_PORT = 30001
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "image"
LOCAL_MQTT_TOPIC_Listen ="predict"

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC_Listen)

def on_message(client,userdata, msg):
  try:
    # get payload
    global text
    text = str(msg.payload.decode("utf-8"))
    text = "Carrying Capacity: " + text
    print("Message received.")
    print("topic: ",msg.topic)
    print(text)
  except:
    print("Unexpected error")

local_mqttclient = mqtt.Client("image")
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 1200)
# local_mqttclient.loop_forever()
print("connected")

local_mqttclient.on_message = on_message

font = cv2.FONT_HERSHEY_SIMPLEX
loc = (10,30)
fontScale = 1
fontColor= (0,255,255)
thickness= 2
lineType= cv2.LINE_4

while(True):
    
    if text != text_ori:
        if (datetime.now() - t).seconds > ttl:
            text = text_ori

    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.putText(frame,text, loc, font, fontScale,fontColor,thickness,lineType)
    k = cv2.waitKey(1) 
    
    if k & 0xFF == ord('q'):
        break
    
    if k %256 == 32:
        t = datetime.now()
        # text = t.strftime("%m/%d/%Y %H:%M:%S")
        try:
            rc,img = cv2.imencode('.png', frame)
            msg = img.tobytes()
            local_mqttclient.publish(LOCAL_MQTT_TOPIC, msg)
            print("Publish Sucsess.")
        except:
            print("Publish Failure")
        print((datetime.now() - t).seconds)
        print(text)
    
    cv2.imshow('frame',frame)

    local_mqttclient.loop()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
