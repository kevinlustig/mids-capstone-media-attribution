NAME_SPACE=w2511
dev=true
if ($dev);
# git pull
# git add .
# git commit -m "edge"
# TAG_HASH=$(git log -1 --pretty=format:%h)
# echo ${TAG_HASH}
then
echo DEV MODE
fi

echo -1.Stop all docker containers
docker stop $(docker ps -aq)
# docker rm $(docker ps -aq)

echo 1.Start up k3
sudo systemctl start k3s
sudo k3s server --docker

echo 2. Clean up
# ####!!!!<----->!!!! docker rmi $(docker images -q)
kubectl delete all --all -n ${NAME_SPACE}
# kubectl delete --all pods
# kubectl delete --all deployments
# kubectl delete --all services
kubectl delete namespaces ${NAME_SPACE}
echo 3. Apply your Deployments and Services
kubectl get namespaces
kubectl apply -f kube/namespace.yaml
kubectl get namespaces
echo Config default namespace to w251
sudo kubectl config set-context --current --namespace=w2511
echo 3.1 Apply MQTT
echo "Kube get nodes"
kubectl get pods
echo "Docker Mqtt"
docker build -t wolu0901/w251-project-mqtt -f kube/Dockerfile.mqtt kube/.
if ($dev)
then
docker push wolu0901/w251-project-mqtt
fi
echo "Get Mosquitto Kuber pods"
kubectl apply -f kube/mosquitto.yaml
kubectl get pods -l app=mosquitto
echo "Get service"
kubectl apply -f kube/mosquittoService.yaml
kubectl get service mosquitto-service
echo "Take note of the NodePort Kubernetes assigns."
echo "Sleep 5"
sleep 10

MOSIP="\"$(kubectl get service/mosquitto-service -o jsonpath='{.spec.clusterIP}')\""
sed "s/MQTT_HOST_IP/${MOSIP}/" ./kube/prediction/main_copy.py > ./kube/prediction/main.py
sed "s/MQTT_HOST_IP/${MOSIP}/" camera_copy.py > camera.py
echo 3.2 Apply Prediction
echo "Kube get nodes"
kubectl get nodes
echo "Docker Predict"
docker build -t wolu0901/w251-project-predict -f kube/prediction/Dockerfile.predict kube/prediction/.
# docker push wolu0901/w251-project-predict
# docker run --rm -ti wolu0901/w251-project-predict
echo "Get Mosquitto Kuber pods"
kubectl apply -f kube/prediction/predict.yaml
kubectl get pods -l app=predict
echo "Sleep 5"
sleep 5

docker build -t cam -f Dockerfile .
xhost +

CAM=0
docker run --privileged --rm -v /data:/data -e DISPLAY -v /tmp:/tmp -ti cam python3 camera.py --device /dev/video0 
