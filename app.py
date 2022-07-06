from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy

from predict import AD_predicit, PD_predicit, AD_model_dict, PD_model_dict
from config import DB_CONFIG

app = Flask(__name__)
app.config["SECRET_KEY"] = '202207'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{ DB_CONFIG["DB_USER"] }:{ DB_CONFIG["DB_PASSWD"] }' \
                                        f'@{ DB_CONFIG["DB_IP"] }:{ DB_CONFIG["DB_PORT"] }/NeuroDisease'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


@app.route('/Home')
def index():
    return render_template("index.html")


@app.route('/DataViewer')
def viewer_load():
    AD_data = {}
    PD_data = {}
    db.reflect()
    all_table = {table_obj.name: table_obj for table_obj in db.get_tables_for_bind()}

    all_data = db.session.query(all_table['ADPred']).limit(20)
    for entity in all_data:
        AD_data[entity[0]] = [i for i in entity[1:]]
    all_data = db.session.query(all_table['PDAudioPred']).limit(20)
    for entity in all_data:
        vals = [i for i in entity[1:14]]
        vals.extend(["...", entity[-1]])
        PD_data[entity[0]] = vals
    content = {
        'AD_data': AD_data,
        'PD_data': PD_data
    }
    return render_template("DataPage.html", **content)


@app.route("/ADPrediction", methods=['get', 'post'])
def AD_load():
    param_n = 8
    status, params, model = get_params(param_n)
    data = {"models": [i for i in AD_model_dict.keys()]}
    if status:
        data['result'] = AD_predicit(model, params)
        data.update(params)
    else:
        data['result'] = ""
    data['model'] = model
    # print(data)
    return render_template("AD.html", **data)


@app.route("/PDPrediction", methods=['GET', 'POST'])
def PD_load():
    param_n = 1
    status, params, model = get_params(param_n)
    data = {"models": [i for i in PD_model_dict.keys()]}
    if status:
        data['result'] = PD_predicit(model, params)
        data.update(params)
    else:
        data['result'] = ""
    data['model'] = model
    # print(data)
    return render_template("PD.html", **data)


# TODO: Data Query from web
def get_params(n):
    model = request.form.get("model-selection")
    params = {}
    if_validate = True
    for label in range(n):
        label = "param" + str(label+1)
        val = request.form.get(label)
        if val:
            params[label] = val
        else:
            if_validate = False
            break
    return if_validate, params, model


if __name__ == '__main__':
    app.run()
