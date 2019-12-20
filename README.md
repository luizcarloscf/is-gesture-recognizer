# is-gesture-recognizier

## Docker

Firts you need to log in the docker. Substitute the "luizcarloscf" for your docker username.

```bash
docker login --username=luizcarloscf
```
Now you can push image to repositories on dockerhub. To do so,
```bash
docker build -f etc/docker/Dockerfile --tag=luizcarloscf/is-gesture-recognizier:0.0.1 .
```
If you doesnt have the repository create on dockerhub, there is no problem.When you push the repository will be created.

```bash
docker push luizcarloscf/is-gesture-recognizier:0.0.1
```

## Kubernetes

Before you apply the deployment.yaml, remember to update the name of the image Docker.

```bash
kubectl apply -f etc/k8s/deployment.yaml 
```

If something went wrong and you wanna to debug inside the container, just edit the deployment to:

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: is-gesture-recognizier
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: is-gesture-recognizier
    spec:
      containers:
        - name: is-gesture-recognizier
          image: luizcarloscf/is-gesture-recognizier:0.0.1
          command: ["sleep"]
          args: ["300000"]
          #command: ["python3"]
          #args: ["src/is_gesture_recognizier/main.py", "/conf/options.json"]
          imagePullPolicy: Always
          ports:
            - name: web
              containerPort: 8000
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 30
          resources:
            limits:
              cpu: "1"
              memory: 2048Mi
              nvidia.com/gpu: 1
          volumeMounts:
            - name: options
              mountPath: /conf/
      volumes:
        - name: options
          configMap:
            name: is-gesture-recognizier
            items:
              - key: recognizier
                path: options.json

```

Now the container is running but with a process of sleep. To access the container, take the pod name and run the command just like the example below.

```bash
kubectl exec -it is-gesture-recognizier-66585449f4-rx8xk -- /bin/bash
```