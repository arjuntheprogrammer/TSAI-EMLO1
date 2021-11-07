# TSAI - Session 5

KUBERNETES HANDS ON

## COMMANDS

INSTALLATION

- brew install minikube
- brew link minikube

- minikube start --driver=docker
- minikube config set driver docker

LIST RUNNING SERVICES

- kubectl status
- kubectl get pods
- kubectl get deployments
- kubectl get replicasets
- kubectl get services

- kubectl logs mongodb-deployment-8f6675bc5-6fbc8
- kubectl describe pod mongodb-deployment-8f6675bc5-6fbc8
- kubectl get all

CREATE DEPLOYMENTS USING YAML FILES

- kubectl apply -f mongo-secret.yaml
- kubectl apply -f mongo-configmap.yaml
- kubectl apply -f mongo-deployment.yaml
- kubectl apply -f mongo-express-deployment.yaml

RUN SERVICE AND VIEW DASHBOARD

- minikube service mongodb-express-service
- kubectl cluster-info
- minikube dashboard --url

---

### TERMINAL OUTPUT

![image](https://user-images.githubusercontent.com/15984084/140642925-b09bd932-5a21-4879-9307-89123e7cc165.png)

![image](https://user-images.githubusercontent.com/15984084/140643308-f1d05375-a149-45aa-9ff6-5b26f58f24d5.png)

---

### DASHBOARD VIEW

![image](https://user-images.githubusercontent.com/15984084/140643002-2b31a8d2-917e-429a-ade7-29f68e9b8e43.png)

![image](https://user-images.githubusercontent.com/15984084/140643062-d39e8c01-5a88-49bc-96f4-0e13630c4ab2.png)

![image](https://user-images.githubusercontent.com/15984084/140643020-e604cc76-1b91-4157-9880-67d3e160ebf7.png)

![image](https://user-images.githubusercontent.com/15984084/140643031-4c259ad1-6a5f-4719-b69c-d0f2aec36b6a.png)

---

## REFERENCE

1. <https://minikube.sigs.k8s.io/docs/start/>
2. <https://kubernetes.io/docs/tasks/run-application/run-stateless-application-deployment/>
3. <https://hub.docker.com/_/mongo>
4. <https://hub.docker.com/_/mongo-express>

---
