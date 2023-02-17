from wtforms import Form, TextAreaField, FloatField, IntegerField, SelectField, validators
from tensorflow import keras
import numpy as np
import pandas as pd
from Input_Function import eval_model_DNN, define_model


class RDSForm(Form):
    value_choices = [('0', 'No'), ('1', 'Yes')]
    value_choices1 = [('0', 'No Divorced'), ('1', 'Divorced')]

    uv_value1 = FloatField("Gestational Age [weeks], (22 ~ 42)")
    uv_value2 = FloatField('Blood Gas Analysiswithin first hour of life [PH], (6.8 ~ 7.5)')
    uv_value3 = FloatField('5-min Apgar Score (0 ~ 10)')
    uv_value4 = FloatField("Blood Gas Analysis Base Excess within first hour of life [mmol/L], (-14 ~ 10)")
    uv_value5 = SelectField("4th order or more of Multiple Gestation", choices=value_choices)
    uv_value6 = FloatField("Body Temperature at the Initial Admission [°C], (33.0 ~ 39.0)")
    uv_value7 = SelectField('3rd order of Multilple Gestatioin', choices=value_choices)
    uv_value8 = FloatField("Birth Head Circumference [cm], (13.0 ~ 38.0)")
    uv_value9 = SelectField('2nd order of Multiple Gestation', choices=value_choices)
    uv_value10 = SelectField("Medication Usage at the Initial Resuscitation", choices=value_choices)

    uv_value11 = SelectField("Singleton", choices=value_choices)
    uv_value12 = SelectField('1st order of Multiple Gestation', choices=value_choices)
    uv_value13 = SelectField('Marital Status (Divorced)', choices=value_choices1)
    uv_value14 = SelectField("Non Hospital of Birth Place", choices=value_choices)
    uv_value15 = SelectField("Marital Status (single)", choices=value_choices)
    uv_value16 = SelectField("Resuscitation at Birth", choices=value_choices)
    uv_value17 = SelectField('Positive Pressure Ventillation Usage at the Initial Resuscitation', choices=value_choices)
    uv_value18 = SelectField("Triplet Gestation", choices=value_choices)
    uv_value19 = SelectField('Chronic Hypertension', choices=value_choices)
    uv_value20 = FloatField("Birth Weight [g], (300 ~ 1500)")

def method_evaluation(input_eval_data):
    SavePath = './Web_RDS/Results_DNN/'
    input_feature_importance = 20

    model_dnn = define_model(input_size=input_feature_importance, learning_rate=1e-4)
    prob_dnn = eval_model_DNN(input_feature_importance, input_eval_data, model_dnn, SavePath)
    keras.backend.clear_session()  # Clear the model in memory.

    survival_rate = prob_dnn * 100
    return survival_rate

def processRDSForm(request):
    form = RDSForm(request.form)
    return form

def processRDSResult(request):
    form = RDSForm(request.form)

    uv_value1 = request.form['uv_value1']
    uv_value2 = request.form['uv_value2']
    uv_value3 = request.form['uv_value3']
    uv_value4 = request.form['uv_value4']
    uv_value5 = request.form['uv_value5']
    uv_value6 = request.form['uv_value6']
    uv_value7 = request.form['uv_value7']
    uv_value8 = request.form['uv_value8']
    uv_value9 = request.form['uv_value9']
    uv_value10 = request.form['uv_value10']

    uv_value11 = request.form['uv_value11']
    uv_value12 = request.form['uv_value12']
    uv_value13 = request.form['uv_value13']
    uv_value14 = request.form['uv_value14']
    uv_value15 = request.form['uv_value15']
    uv_value16 = request.form['uv_value16']
    uv_value17 = request.form['uv_value17']
    uv_value18 = request.form['uv_value18']
    uv_value19 = request.form['uv_value19']
    uv_value20 = request.form['uv_value20']

    ###==================================================================================###
    input_concat = np.concatenate([[uv_value1], [uv_value2], [uv_value3], [uv_value4], [uv_value5], [uv_value6], [uv_value7], [uv_value8], [uv_value9], [uv_value10],
                                   [uv_value11], [uv_value12], [uv_value13], [uv_value14], [uv_value15], [uv_value16], [uv_value17], [uv_value18], [uv_value19], [uv_value20]])
    input_data = np.asarray([np.NaN if input_concat[ii] == '' else input_concat[ii] for ii in range(0, 20)], dtype=np.float64)
    print('Input Data: ', input_data)

    input_data = input_data.reshape(1, -1)
    survival_rate = method_evaluation(input_data)
    print()
    print()

    # sarcopenia_rate = np.round(np.random.uniform(0.0, 100.0), 2)
    survival_rate = np.round(survival_rate, 2)
    if survival_rate < 50:
        can_surv = "Non-RDS"
    else:
        can_surv = 'RDS'

    input_names = ["Gestational Age", 'Blood Gas Analysis(PH)', '5-min Apgar Score', "Blood Gas Analysis Base Excess", "4th order or more of Multiple Gestation", 
                   "Body Temperature at the Initial Admission", '3rd order of Multilple Gestatioin', "Birth Head Circumference", '2nd order of Multiple Gestation',
                   "Medication Usage at the Initial Resuscitation", "Singleton", '1st order of Multiple Gestation', 'Marital Status (divosce)',
                   "Non Hospital of Birth Place", "Marital Status (single)", "Resuscitation at Birth",'Positive Pressure Ventillation Usage at the Initial Resuscitation',
                   "Triplet Gestation", 'Chronic Hypertension', 'Birth Weight']
    
    
    
    dfList = pd.DataFrame(zip(input_names, np.reshape(input_concat, (-1,))), columns=['feature_name', 'feature_value'])
    dfList = dfList.reindex(index=[ 0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19])   # 행 순서 변경
    dfList = dfList.replace(to_replace='0', value='No', regex=False)  # regex=> True: 부분이라도 같을 경우, False: 모두 같을 경우
    dfList = dfList.replace(to_replace='1', value='Yes', regex=False)
    dfList = dfList.values.tolist()
    return form, can_surv, survival_rate, dfList



