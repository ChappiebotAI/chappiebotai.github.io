---
layout: post
title: Kubernetes Tutorial
author: hoangbm
---

# How to create and train a machine learning task on Kubernetes autoscaler cluster

To run a machine learning job on kubernetes autoscaler cluster. You have to define three main components. The first component is task template, the second is task definition, and the remaining component is filled values which defined in task definition - they is used to fill to task template to generate a completed [yaml](http://yaml.org/start.html) file.


## Task template

**(if you aren't familiar with kubernetes, you can skip this part)**
Task template is based on [jinja](http://jinja.pocoo.org/) template format.

After filling values to task template, it includes a list of kubernetes resources will be created on the kubernetes cluster. When you call Restful API (APIs have been managed by web service on kubernetes cluster) to create a training job, the kubernetes autoscaler cluster will create all necessary resources for this job. The following example a task template for non distributed task run on a single CPU node:

```yaml
{% if configs is defined %}
apiVersion: v1
kind: ConfigMap
metadata:
  name: "{{ global.release }}"
  labels:
    mltask: mltask
data:
  globalCfg: '{{ configs.get("global") }}'
  datasetCfg: '{{ configs.get("dataset") }}'
  processCfg: '{{ configs.get("process") }}'
---
{% endif %}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: "{{ global.release }}"
spec:
  storageClassName: aws-efs
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      {% if resources is defined %}
      storage: "{{ resources.get('storage') or '10Gi' }}"
      {% else %}
      storage: "10Gi"
      {% endif %}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ global.release }}"
  labels:
    taskDef: "{{ global.taskDef }}"
    release: "{{ global.release }}"
    application: "{{ global.application }}"
    creator: "{{ global.creator }}"
    mltask: mltask
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        taskDef: "{{ global.taskDef }}"
        release: "{{ global.release }}"
        application: "{{ global.application }}"
        mltask: mltask
    spec:
      restartPolicy: Never
      containers:
        - name: "{{ global.release }}"
          tty: true
          image: "{{ image.repository }}:{{ image.tag }}"
          imagePullPolicy: "{{ image.pullPolicy or 'Always'}}"
          env:
            - name: STORAGE
              value: "/mount/{{ global.release }}/"
          {% if configs is defined %}
            - name: GLOBALCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: globalCfg
            - name: PROCESSCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: processCfg
            - name: DATASETCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: datasetCfg
          {% endif %}
          volumeMounts:
          - name: "{{ global.release }}"
            mountPath: "/mount/{{ global.release }}/"
          {% if resources is defined %}
          resources:
            limits:
              cpu: "{{ resources.get('cpu') or '500m' }}"
              memory: "{{ resources.get('memory') or '512Mi' }}"
          {% endif %}
      nodeSelector:
        kops.k8s.io/ml-instance: "true"
      volumes:
      - name: "{{ global.release }}"
        persistentVolumeClaim:
          claimName: "{{ global.release }}"
```

Basically, the task template example declares three kubernetes resources will be created. A `ConfigMap`, a `PersistentVolumeClaim` and a `Job`. And it accepts a dict values with keys `global`, `configs`, `image`,  `resources`. The purpose of `ConfigMap` resources is mapping configs that passed by Restful API to docker container. `PersistentVolumeClaim` (PVC) plays in role a default storage for the job with default capacity about 10Gi (Amazon Elastic FileSystem internal). You can use it to store model checkpoints, store large dataset... anything you need to run your job. `STORAGE` environment variable in the container describes mounting path of PVC.

### List current task templates

- `countdown-task`:

  * Template:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ global.release }}"
  labels:
    taskDef: "{{ global.taskDef }}"
    release: "{{ global.release }}"
    application: "{{ global.application }}"
    creator: "{{ global.creator }}"
    mltask: mltask
spec:
  backoffLimit: 0
  template:
    metadata:
      name: "{{ global.release }}"
    spec:
      containers:
      - name: counter
        tty: true
        image: "{{ image.repository }}:{{ image.tag }}"
        command:
         - "bin/bash"
         - "-c"
         - "for i in $(seq {{ process.train.number }} -1 1); do echo $i ; done"
      restartPolicy: Never
```

  * Required dict values:

```json
{
    "global": {
        "release": "Release of job",
        "taskDef": "Name of task definition",
        "application": "Application of job",
        "creator": "Job's creator"
    },
    "process": {
        "train":{
            "number": "Upperbound number to countdown"
        }
    }
}
```

- `non-distributed-task`:

  * Template: same as the first example

  * Required dict values:

```json
{
    "global": {
        "release": "Release of job",
        "taskDef": "Name of task definition",
        "application": "Application of job",
        "creator": "Job's creator"
    },
    "process":{},
    "configs": "Automatically build in Restful API",
    "resources": "Optional"
}
```

- `single-gpu-task`:
  
  * Template: 

```yaml
{% if configs is defined %}
apiVersion: v1
kind: ConfigMap
metadata:
  name: "{{ global.release }}"
  labels:
    mltask: mltask
data:
  globalCfg: '{{ configs.get("global") }}'
  datasetCfg: '{{ configs.get("dataset") }}'
  processCfg: '{{ configs.get("process") }}'
---
{% endif %}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: "{{ global.release }}"
spec:
  storageClassName: aws-efs
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      {% if resources is defined and resources %}
      storage: "{{ resources.get('storage') or '10Gi' }}"
      {% else %}
      storage: "10Gi"
      {% endif %}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ global.release }}"
  labels:
    taskDef: "{{ global.taskDef }}"
    release: "{{ global.release }}"
    application: "{{ global.application }}"
    creator: "{{ global.creator }}"
    mltask: mltask
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        taskDef: "{{ global.taskDef }}"
        release: "{{ global.release }}"
        application: "{{ global.application }}"
        mltask: mltask
    spec:
      restartPolicy: Never
      containers:
        - name: "{{ global.release }}"
          tty: true
          image: "{{ image.repository }}:{{ image.tag }}"
          imagePullPolicy: "{{ image.pullPolicy or 'Always'}}"
          env:
            - name: STORAGE
              value: "/mount/{{ global.release }}/"
          {% if configs is defined %}
            - name: GLOBALCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: globalCfg
            - name: PROCESSCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: processCfg
            - name: DATASETCFG
              valueFrom:
                configMapKeyRef:
                  name: "{{ global.release }}"
                  key: datasetCfg
          {% endif %}
          volumeMounts:
          - name: "{{ global.release }}"
            mountPath: "/mount/{{ global.release }}/"
          resources:
            limits:
              nvidia.com/gpu: 1
      nodeSelector:
        kops.k8s.io/dl-instance: "true"
      volumes:
      - name: "{{ global.release }}"
        persistentVolumeClaim:
          claimName: "{{ global.release }}"
```

   * Requires dict values
```json
{
    "global": {
        "release": "Release of job",
        "taskDef": "Name of task definition",
        "application": "Application of job",
        "creator": "Job's creator"
    },
    "process":{},
    "configs": "Automatically build in Restful API",
    "resources": "Optional"
}
```

## Task definition

Usually, you only need to focus task defintions and filled values. Because task templates can reuse for a lot of jobs and they is defined before when deploying web service. Task definition requires some fields to work:

1. `creator`: User create task definition

2. `name`: Name of task definition

3. `task_template_name`: Name of task template

4. `description`: Description of task definition

5. `user_values_description`: YAML format describes values need to fill. Each fields must begin by `(Required)` or `(Optional)`

6. `user_values_example`: YAML format, a example for values that need to fill

7. `system_values`: YAML format, it usually is defined of docker image location. Sometimes, it can contain some configs for task template.

8. `append_uuid`: Optional, if you set this field is true. Register task defintion API will append to name of task definition a unique string to ensure name of task definition isn't duplicate.

 A example of task definition.

- `creator`: "admin"

- `name`: "mnist_softmax_cpu"

- `description`: "Tensorflow mnist softmax training on cpu"

- `user_values_description`: 

```yaml
taskDef: (Required) Task definition name
description: (Required) Description of job
name: (Required) Application name
creator: (Required) Creator Id

endpoints:
  -
    endpoint: (Required) S3 folder endpoint stores train.tfrecords, test.tfrecords, validation.tfrecords
    type: (Required) dataset
  -
    endpoint: (Optional) A callback endpoint
    type: (Optional) event

process:
  train:
    maxStep: (Optional) Max step to train, default is 1000
    batchSize: (Optional) Batch size to train, default is 100
    saveStep: (Required) Number of step to save checkpoint
  eval:
    batchSize: (Optional) Batch size to eval, default is 100
  model:
  export: 
    modelVersion: (Optional) Model version to export to local path,
    signatureMap: (Required) Signature map for serving model

resources:
  cpu: (Optional) CPU resource default `500m`
  memory: (Optional) Memory resource default `512Mi`

```

- `user_values_example`:

```yaml
taskDef: mnist_softmax_cpu
description: Training tensorflow MNIST softmax on CPU
name: demo
creator: ducta

endpoints:
  -
    endpoint: s3://abc/xyz
    type: dataset
  -
    endpoint: http://localhost:8080/v1/callback
    type: event

process:
  train:
    maxStep: 2000
    batchSize: 256
    saveStep: 200
  eval:
    batchSize: 100
  model:
  export: 
    modelVersion: 1
    signatureMap: predict_images

resources:
  cpu: 2000m
  memory: 2048Mi
```

- `system_values`:

```yaml
image:
  pullPolicy: Always
  repository: 065056466896.dkr.ecr.us-east-1.amazonaws.com/chappieai-base
  tag: mnist-softmax-cpu
```

- `application`: "Production"

- `append_uuid`: 0

When call register API, all above fields compact to a json object as the following:

```json
{
	"creator": "admin",
	"name": "mnist_softmax_cpu",
	"task_template_name": "non-distributed-task",
	"description": "Tensorflow mnist softmax training on cpu",
	"user_values_description": "taskDef: (Required) Task definition name\r\ndescription: (Required) Description of job\r\nname: (Required) Application name\r\ncreator: (Required) Creator Id\r\n\r\nendpoints:\r\n  -\r\n    endpoint: (Required) S3 folder endpoint stores train.tfrecords, test.tfrecords, validation.tfrecords\r\n    type: (Required) dataset\r\n  -\r\n    endpoint: (Optional) A callback endpoint\r\n    type: (Optional) event\r\n\r\nprocess:\r\n  train:\r\n    maxStep: (Optional) Max step to train, default is 1000\r\n    batchSize: (Optional) Batch size to train, default is 100\r\n    saveStep: (Required) Number of step to save checkpoint\r\n  eval:\r\n    batchSize: (Optional) Batch size to eval, default is 100\r\n  model:\r\n  export: \r\n    modelVersion: (Optional) Model version to export to local path,\r\n    signatureMap: (Required) Signature map for serving model\r\n\r\nresources:\r\n  cpu: (Optional) CPU resource default `500m`\r\n  memory: (Optional) Memory resource default `512Mi`\r\n",
	"system_values": "image:\r\n  pullPolicy: Always\r\n  repository: 065056466896.dkr.ecr.us-east-1.amazonaws.com\/chappieai-base\r\n  tag: mnist-softmax-cpu",
	"user_values_example": "taskDef: mnist_softmax_cpu\r\ndescription: Training tensorflow MNIST softmax on CPU\r\nname: demo\r\ncreator: ducta\r\n\r\nendpoints:\r\n  -\r\n    endpoint: s3:\/\/abc\/xyz\r\n    type: dataset\r\n  -\r\n    endpoint: http:\/\/localhost:8080\/v1\/callback\r\n    type: event\r\n\r\nprocess:\r\n  train:\r\n    maxStep: 2000\r\n    batchSize: 256\r\n    saveStep: 200\r\n  eval:\r\n    batchSize: 100\r\n  model:\r\n  export: \r\n    modelVersion: 1\r\n    signatureMap: predict_images\r\n\r\nresources:\r\n  cpu: 2000m\r\n  memory: 2048Mi\r\n",
	"application": "Production",
	"append_uuid": 0
}
```

## Filled values

The last step is simple. You create a json object to define filled values as in `user_values_description` of task definition and call API to create job on kubernetes cluster. For example:


```json
{
  "taskDef": "mnist_softmax_cpu",
  "description": "Training tensorflow MNIST softmax on CPU",
  "name": "demo",
  "creator": "ducta",
  "endpoints":[
    {
      "endpoint": "s3://chappiebot/dataset/mnist-tfrecords/",
      "type": "dataset"
    },
    {
      "endpoint": "https://proud-squid-71.localtunnel.me/v1/ml/train/callback",
      "type": "event",
      "headers":{
        "x-api-key": "123qweasd"
      }
    }
  ],
  "process":{
    "train":{
      "maxStep": 100,
        "batchSize": 32,
        "saveStep": 20  
    },
    "eval":{
      "batchSize": 10  
    },
    "model":{},
    "export":{
      "modelVersion": 1,  
      "signatureMap": "predict_images"
    }
  },
  "resources":{
    "cpu": "200m",
    "memory": "512Mi"
  }
}
```

## Summary

```
(task template) <- Admin create task template
      |
      v
(task definition) <- User create task definition with a defined task template
      |
      v
(filled values) <- User define filled values and call create job API
```

## How to write a docker file

See `mnist_softmax_cpu.ipynb` notebook and all source code in `mnist_softmax_cpu` folder.