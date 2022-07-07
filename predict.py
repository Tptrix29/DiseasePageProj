import joblib
import numpy as np

model_path = '/'.join(str(__file__).split('/')[:-1]) + "/PredModel/"
PD_model_dict = {
        "Decision Tree": "Tree-PD.pkl",
        "Random Forest": "Forest-PD.pkl"
        }
AD_model_dict = {
    "AdaBoost": "AdaBoost-AD.pkl",
    "SVM": "SVM-AD.pkl",
    "Logistic Regression": "Logistics-AD.pkl"
}


def PD_predicit(model_type, params):
    print(__file__)
    model = joblib.load(model_path+PD_model_dict[model_type])
    input_vec = np.array(params['param1'].split(','), dtype=np.float).reshape((1, -1))
    result = model.predict(input_vec).item()
    result = '高风险患病' if result else '低风险患病'
    return result


def AD_predicit(model_type, params):
    model = joblib.load(model_path+AD_model_dict[model_type])
    input_vec = np.array([i for i in params.values()], dtype=np.float).reshape((1, -1))
    result = model.predict(input_vec).item()
    result = '高风险患病' if result else '低风险患病'
    return result

