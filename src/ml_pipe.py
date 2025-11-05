import pandas as pd
import  numpy as np
import  os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from mlflow.models.signature import infer_signature
from mlflow.sklearn import log_model
from mlflow import log_metric,log_param,set_tracking_uri,set_experiment,start_run,get_tracking_uri 
import sys 
import urllib.parse

os.environ["MLFLOW_TRACKING_URI"]="http://ec2-3-107-196-65.ap-southeast-2.compute.amazonaws.com:5000/"

def load_and_preprocess(data_path):
    data = pd.read_csv(data_path,sep=";")
    # basic preprocessing steps
    data = data.dropna()
    return data


def train(model_path,data_path):
    
    alpha = sys.argv[1] if len(sys.argv) >1 else 0.5
    l1_ratio = sys.argv[2] if len(sys.argv) >2 else 0.5

    model= ElasticNet(alpha=0.5,l1_ratio=0.5,random_state=42)
    data = load_and_preprocess(data_path)
    x= data.drop("quality",axis=1)
    y= data["quality"]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.fit_transform(y_train.values.reshape(-1,1))
    model.fit(x_train,y_train)
    x_test = scaler.fit_transform(x_test)
    y_test = scaler.fit_transform(y_test.values.reshape(-1,1))

    y_pred = model.predict(x_test)

    # evaluation of metrics
    rmse,mae,r2 = evaluate_model(y_test,y_pred)

    # mlflow connection to  AWS S3 bucket
    set_tracking_uri("http://ec2-3-107-196-65.ap-southeast-2.compute.amazonaws.com:5000/")
    set_experiment("ElasticNet_Wine_Quality")
    # log into MLFlow
    with start_run(): 
        #log_model(model,signature=infer_signature(x_train,y_train))
        log_param("alpha",0.5)
        log_param("l1_ratio",0.5)
        log_metric("rmse",rmse)
        log_metric("mae",mae)
        log_metric("r2",r2)
    # save the model locally
    model_save_path = os.path.join(model_path,"elasticnet_model.pkl")
    pd.to_pickle(model,model_save_path)
    print(f"Model saved at {model_save_path}")

    # remote server connection to AWS S3 bucket
    remote_tracking_uri = ""
    set_tracking_uri(remote_tracking_uri)
    tracking_url_file_store = urllib.parse.urlparse(get_tracking_uri()).scheme
    #if tracking_url_file_store != "file":
    #    log_model(model,"model",signature=infer_signature(x_train,y_train),registered_model_name="ElasticNetRegressor")
    #else:
    #    log_model(model,"model")



    return model_save_path





def evaluate_model(true,pred):
    rmse = np.sqrt(mean_squared_error(true,pred))
    mae = mean_absolute_error(true,pred)
    r2 = r2_score(true,pred)
    return rmse,mae,r2

if __name__ == "__main__":
    data_path = "data/winequality-white.csv"
    model_path = train("model",data_path)
    print(f"Trained model saved at {model_path}") 

