# Udacity Project - Heart failure - AutoMl and Hyperparameter tune.

In this project, We used heart failure dataset(https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data). Downloaded this dataset from the kaggle website and uploaded to Azure ML Studio and registered it. This data generally contains data about people and has health related data with the death column. Generally, we have to predict early detection of Heart rate failure using ML.

Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Reason for AutoML and Hyperparametertune - Need early detection and management wherein a machine learning model can be of great help.

## Dataset

### Overview
Dataset - Heart failure dataset(https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).
Columns in dataset are - age,	anaemia	creatinine_phosphokinase,	diabetes,	ejection_fraction,	high_blood_pressure	platelets,	serum_creatinine,	serum_sodium,	sex	smoking	time,	DEATH_EVENT.
This dataset contains records of medical data from 299 patients with heart failure.
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/20a94ee9-ddae-44e5-80a7-f0dc1a1c16e9">

<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/fbcff22d-2269-4083-8352-1c0a7355d2da">

### Task
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Reason for AutoML and Hyperparametertune - Need early detection and management wherein a machine learning model can be of great help.

### Access
We manually downloaded and registered the data from kaggle into Azure ML Studio, uploading the csv and registering it as an Azure ML dataset.

## Automated ML
Field to predict - DEATH_EVENT which is a classification problem, we choose accuracy to be our primary metric here, as data is already registered we have the training dataset ready to use AutoML on it. 

"primary_metric": "accuracy" = This specifies that the primary metric used to evaluate the performance of the models is "accuracy". Accuracy is the ratio of the number of correct predictions to the total number of predictions. It is commonly used for classification tasks where the model needs to predict discrete labels.

"experiment_timeout_minutes": 15 = This sets a time limit for the entire AutoML experiment, which is 15 minutes in this case. The experiment will run for a maximum of 15 minutes. If it doesn't finish within this time, it will stop regardless of whether it has tested all possible configurations or not. This is useful for ensuring experiments do not run indefinitely and helps in managing computational resources.

"max_concurrent_iterations": 5 = This specifies the maximum number of iterations (model training runs) that can be executed concurrently. Running multiple iterations in parallel can speed up the experiment by utilizing available computational resources efficiently. In this case, up to 5 iterations can run simultaneously.

<img width="435" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/81c845bb-ff62-4fe4-bcd5-5fa2d5d75dac">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/3bb6f869-adb9-43e9-8c3c-315c8c1b289c">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/16042318-78e0-4eca-bcee-ff820889aa36">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/1c3796cd-274b-4d26-a889-e14bae828c06">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/7b163399-6304-4dbb-b0b7-3e15ea5e9f6e">

### Results
The best Auto ML experiment model is a Voting Ensemble

Best Model Run ID: AutoML_d15fa56f-2b6b-4648-92d8-6786b8ede363_38

Best Model Metrics: {'average_precision_score_macro': 0.90127133237628, 'precision_score_micro': 0.8727586206896552, 'balanced_accuracy': 0.8526649046034057, 'log_loss': 0.35194985644068316, 'recall_score_micro': 0.8727586206896552, 'average_precision_score_micro': 0.9233454587652921, 'AUC_micro': 0.9217858369665741, 'accuracy': 0.8727586206896552, 'precision_score_macro': 0.8589201735320662, 'AUC_macro': 0.9103684604022133, 'precision_score_weighted': 0.8831468766173547, 'weighted_accuracy': 0.8853524375855703, 'norm_macro_recall': 0.7053298092068114, 'f1_score_macro': 0.8476878750460962, 'matthews_correlation': 0.7099962646709519, 'f1_score_micro': 0.8727586206896552, 'f1_score_weighted': 0.8711967419108847, 'AUC_weighted': 0.9103684604022133, 'recall_score_macro': 0.8526649046034057, 'recall_score_weighted': 0.8727586206896552, 'average_precision_score_weighted': 0.9251800674491077, 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_d15fa56f-2b6b-4648-92d8-6786b8ede363_38/accuracy_table', 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_d15fa56f-2b6b-4648-92d8-6786b8ede363_38/confusion_matrix'}

Best Model Accuracy: 0.8727586206896552

More details on best Model Pipeline(steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, is_cross_validation=True, working_dir='/mnt/batch/tasks/shared/LS_root/mounts/clusters/notebook262104/code/Users/odl_user_262104')),
                ('prefittedsoftvotingclassifier',
                 PreFittedSoftVotingClassifier(classification_labels=array([0, 1]), estimators=[('18', Pipeli..., ('randomforestclassifier', RandomForestClassifier(bootstrap=False, criterion='entropy', max_features=0.6, min_samples_leaf=0.01, min_samples_split=0.2442105263157895, n_estimators=200, n_jobs=1))]))], flatten_transform=False, weights=[0.2857142857142857, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]))])

Improvement can be done if we made in run for some more time and also used cross fold validations. Additionally, a different metric, maybe taking into account to see.   

## Hyperparameter Tuning
RandomParameterSampling - This is a strategy used to randomly sample hyperparameters for model training. During the experiment, the ML framework will randomly select values from the specified choices for each hyperparameter.

learning_rate: The rate at which the model learns during training. The choices given are 0.01, 0.05, 0.1, 0.25, 1.0.
n_estimators: The number of trees in an ensemble model (like a Random Forest or Gradient Boosting). The choices are 1, 5, 10, 25.

BanditPolicy = This is an early stopping policy that helps in terminating poorly performing runs early, based on performance compared to the best run.

evaluation_interval: Specifies how often (in terms of number of iterations) the policy should evaluate the performance of the runs. Here, it is set to 3, meaning the performance will be checked every 3 iterations.
slack_factor: This sets the tolerance level for how much worse a run can perform relative to the best run before being terminated. A slack_factor of 0.2 means that a run can perform up to 20% worse than the best performing run before it is stopped.

In this project we used Gradient boosting as it builds an ensemble of trees sequentially, where each tree tries to correct the errors of the previous one. This iterative process often results in high predictive accuracy.  It is especially useful when you need a model that can capture complex patterns in the data while still being interpretable and customizable.

<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/235b0178-617a-4d90-8694-e6a3579eee26">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/964edbe9-60ff-446e-aff2-99b486b126a2">


### Results
The best Gradient Boosting Classifier also reaches an accuracy of ~80% with its hyperparameters {'Learning Rate:': 0.1, 'Number Estimators:': 25}


## Model Deployment
We deployed the best model from the AutoML experiment as a web service endpoint and tested it with three randomly chosen samples from the dataset.

url = 'http://8e870f91-0fe0-4068-ae57-8f1a5c3f65ca.westus2.azurecontainer.io/score'

data =  {
  "data": [
    {
      "age": 18.0,
      "anaemia": 0,
      "creatinine_phosphokinase": 0,
      "diabetes": 1,
      "ejection_fraction": 0,
      "high_blood_pressure": 1,
      "platelets": 0.0,
      "serum_creatinine": 0.0,
      "serum_sodium": 0,
      "sex": 1,
      "smoking": 0,
      "time": 0
    }
  ]
}

Output - b'[1]'

<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/81eca2e2-ea0e-4811-91f2-18ae03d88895">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/9f051935-f84e-465f-992c-fae8042f61a7">
<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/4b57888b-3e25-4bfb-8a8d-2965271821d5">



## Screen Recording
https://drive.google.com/file/d/1ExUN4mSmdSZmgXu9vFynkYUhjMVdKqtr/view?usp=sharing

## Cluster and service delete

<img width="452" alt="image" src="https://github.com/mehul1419/nd00333-capstone/assets/51814570/64df62c0-1c97-47f8-88a5-21e693c400a4">


